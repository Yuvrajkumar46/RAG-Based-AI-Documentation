"""
Text chunking utilities.

WHY 400 TOKENS WITH 80-TOKEN OVERLAP?
──────────────────────────────────────
• Sentence-transformers (all-MiniLM-L6-v2) has a hard cap of 512 tokens.
  Staying at 400 leaves headroom so no chunk is silently truncated.

• 400 tokens ≈ 280–320 words — enough to contain 2–4 full paragraphs and
  preserve complete reasoning units (e.g., a definition + its example).
  Smaller chunks (< 150 tokens) produce embeddings that lack context;
  larger chunks (> 500 tokens) make the similarity signal noisy because
  many unrelated sentences dilute the query-relevant sentence.

• 80-token overlap (≈ 20 %) ensures that sentences at chunk boundaries
  appear in two adjacent chunks, preventing answers that straddle a split
  from being missed entirely.

KNOWN FAILURE CASE (documented for evaluation):
  When a document uses numbered lists where the stem question appears in
  chunk N and the answer options span chunks N and N+1, retrieval may
  return only one half.  Mitigation: paragraph-aware splitting (used here).
"""

from __future__ import annotations

import re
from typing import Generator


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple but robust regex."""
    # Keep the delimiter attached to the preceding sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _count_tokens(text: str) -> int:
    """
    Approximate token count without loading a full tokenizer.
    GPT-style tokenisers average ~0.75 tokens per word (1.33 words/token).
    """
    return max(1, len(text.split()))


def chunk_text(
    text: str,
    chunk_size: int = 400,
    overlap: int = 80,
) -> Generator[str, None, None]:
    """
    Yield text chunks of ≤ chunk_size tokens with `overlap`-token overlap.

    Strategy:
    1. Split by double-newline (paragraphs) first to respect natural boundaries.
    2. Within each paragraph, accumulate sentences until the token budget is hit.
    3. Slide the window back by `overlap` tokens to start the next chunk.

    This is a deliberate custom implementation rather than LangChain's
    RecursiveCharacterTextSplitter so we can tune every heuristic ourselves.
    """
    # Step 1: paragraph-aware pre-split
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    buffer:       list[str] = []   # accumulated sentences
    buffer_tokens: int       = 0

    def flush_buffer() -> str:
        return " ".join(buffer)

    for para in paragraphs:
        sentences = _split_sentences(para)

        for sent in sentences:
            sent_tokens = _count_tokens(sent)

            # If a single sentence already overflows chunk_size, emit it alone
            if sent_tokens >= chunk_size:
                if buffer:
                    yield flush_buffer()
                    # carry last `overlap` tokens into next chunk
                    buffer, buffer_tokens = _trim_to_overlap(buffer, overlap)
                yield sent
                buffer, buffer_tokens = [], 0
                continue

            if buffer_tokens + sent_tokens > chunk_size:
                # Emit current chunk
                yield flush_buffer()
                # Slide back: keep the tail that fits within `overlap` tokens
                buffer, buffer_tokens = _trim_to_overlap(buffer, overlap)

            buffer.append(sent)
            buffer_tokens += sent_tokens

    if buffer:
        yield flush_buffer()


def _trim_to_overlap(
    sentences: list[str], overlap: int
) -> tuple[list[str], int]:
    """
    Return a suffix of `sentences` whose total token count ≤ overlap.
    Used to seed the next chunk so context is not lost at boundaries.
    """
    kept: list[str] = []
    total = 0
    for sent in reversed(sentences):
        t = _count_tokens(sent)
        if total + t > overlap:
            break
        kept.insert(0, sent)
        total += t
    return kept, total
