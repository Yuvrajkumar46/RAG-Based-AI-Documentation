"""
/health – system health check endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter

from models.schemas import DocumentStatus, HealthResponse
from services.document_service import list_documents
from utils.config import get_settings

router = APIRouter(tags=["Health"])


def _get_store():
    from main import store
    return store


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns system health including model info and index stats."""
    cfg   = get_settings()
    store = _get_store()
    docs  = list_documents()
    ready = sum(1 for d in docs if d.get("status") == DocumentStatus.READY)

    return HealthResponse(
        status="ok",
        embedding_model=cfg.embedding_model,
        llm_model=cfg.llm_model,
        indexed_chunks=store.total_chunks,
        documents_ready=ready,
    )
