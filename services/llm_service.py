"""
LLM service using Groq API (Llama 3.3 70B).

Two LLM calls per query:
  1. generate_answer  — answers the question from context
  2. evaluate_answer  — strictly evaluates the answer against context
"""

from __future__ import annotations

import time

from groq import Groq

from utils.logger import get_logger

logger = get_logger("rag.llm")

# ── Hardcoded Groq config ─────────────────────────────────────────────────────
GROQ_API_KEY = "ADD_YOUR_API"
LLM_MODEL    = "llama-3.3-70b-versatile"

_client: Groq | None = None


def get_llm_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ── Prompt: Answer generation ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise document question-answering assistant.

RULES:
1. Answer the user's question using ONLY the context chunks provided below.
2. If the context does not contain sufficient information, respond with:
   "I don't have enough information in the provided documents to answer this question."
3. Be concise and factual. Do not add information not present in the context.
4. Reference the source document name(s) when possible (e.g., "According to report.pdf").
5. Use bullet points or numbered lists only when they improve readability.
"""

# ── Prompt: RAG Evaluator ─────────────────────────────────────────────────────
EVALUATOR_SYSTEM_PROMPT = """You are an expert evaluator of RAG-based QA systems.
Strictly evaluate the system answer ONLY based on the given document context.
Do NOT assume or add any external knowledge.

Follow these rules:
- If information is not explicitly present in the context, mark it as hallucination.
- Be strict and critical in evaluation.
- Keep explanation clear, concise, and factual.

Return output in this EXACT format (no extra text before or after):
Relevance: (High / Medium / Low)
Accuracy: (Correct / Partially Correct / Incorrect)
Completeness: (Complete / Partial / Poor)
Retrieval Quality: (Good / Average / Bad)
Hallucination: (Yes / No)
Final Verdict: (Good / Needs Improvement / Poor)
Explanation:
- 3-4 short lines
- Clearly mention:
  what is correct
  what is missing
  what is hallucinated (if any)
  how well retrieval matches the question

Important:
- Do not generalize
- Do not overpraise
- Do not add assumptions
- Be objective like a reviewer
"""


def generate_answer(
    question: str,
    context_chunks: list[tuple[str, str, float]],
) -> tuple[str, float]:
    """
    Generate an answer given retrieved context chunks.
    Returns: (answer, latency_ms)
    """
    context_parts = []
    for i, (text, filename, sim) in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Chunk {i} | Source: {filename} | Relevance: {sim:.2f}]\n{text}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    user_message = f"""CONTEXT DOCUMENTS:
{context_str}

---

QUESTION: {question}

Please answer based on the context above."""

    t0 = time.perf_counter()
    try:
        response = get_llm_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
        logger.info("LLM answer generated successfully")
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        answer = (
            "⚠️ LLM service is currently unavailable. "
            "Here are the most relevant excerpts from your documents:\n\n"
            + "\n\n".join(f"• [{fn}] {txt[:300]}…" for txt, fn, _ in context_chunks[:3])
        )

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("LLM answered in %.1f ms", latency_ms)
    return answer, latency_ms


def evaluate_answer(
    question: str,
    answer: str,
    context_chunks: list[tuple[str, str, float]],
) -> tuple[str, float]:
    """
    Second LLM call: strictly evaluates the generated answer against the
    retrieved context. Returns structured evaluation text + latency.
    Returns: (evaluation, latency_ms)
    """
    context_parts = []
    for i, (text, filename, sim) in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Chunk {i} | Source: {filename} | Relevance: {sim:.2f}]\n{text}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    eval_user_message = f"""DOCUMENT CONTEXT:
{context_str}

---

QUESTION ASKED:
{question}

---

SYSTEM ANSWER TO EVALUATE:
{answer}

---

Now strictly evaluate the system answer using ONLY the document context above.
"""

    t0 = time.perf_counter()
    try:
        response = get_llm_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user",   "content": eval_user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        evaluation = response.choices[0].message.content.strip()
        logger.info("RAG evaluation completed successfully")
    except Exception as exc:
        logger.error("Evaluation LLM call failed: %s", exc)
        evaluation = "⚠️ Evaluation unavailable due to LLM error."

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("Evaluation completed in %.1f ms", latency_ms)
    return evaluation, latency_ms