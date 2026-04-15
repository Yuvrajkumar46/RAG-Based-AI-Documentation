"""
Text extraction from supported document formats.

Supported formats: PDF, TXT
"""

from __future__ import annotations

import re
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("rag.extractor")


def extract_text(file_path: Path) -> str:
    """
    Dispatch to the correct extractor based on file suffix.
    Returns cleaned plain text.
    Raises ValueError for unsupported formats.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(file_path)
    elif suffix == ".txt":
        return _extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


# ── PDF ────────────────────────────────────────────────────────────────────────

def _extract_pdf(path: Path) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz).
    Falls back to pypdf if fitz is unavailable.
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        pages: list[str] = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"[Page {page_num + 1}]\n{text}")

        doc.close()
        raw = "\n\n".join(pages)

    except ImportError:
        logger.warning("PyMuPDF not found, falling back to pypdf")
        raw = _extract_pdf_fallback(path)

    return _clean_text(raw)


def _extract_pdf_fallback(path: Path) -> str:
    """pypdf fallback (less accurate for complex layouts)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text}")
    return "\n\n".join(pages)


# ── TXT ────────────────────────────────────────────────────────────────────────

def _extract_txt(path: Path) -> str:
    """Read a plain-text file, trying common encodings."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = path.read_text(encoding=encoding)
            return _clean_text(text)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path.name} with any supported encoding")


# ── Shared cleanup ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Normalise extracted text:
    - Collapse 3+ blank lines into 2
    - Strip non-printable control characters (but keep newlines/tabs)
    - Strip leading/trailing whitespace
    """
    # Remove non-printable chars except whitespace
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]', ' ', text)
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Normalise multiple spaces (but not newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()
