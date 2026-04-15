"""
/query – question-answering endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from models.schemas import QueryRequest, QueryResponse
from services.query_service import run_query
from utils.logger import get_logger

logger = get_logger("rag.routes.query")

router = APIRouter(prefix="/query", tags=["Query"])


def _get_store():
    from main import store
    return store


@router.post("/", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    """
    Ask a question against uploaded documents.

    - `question`: your natural-language question (3–2000 chars)
    - `document_id`: (optional) restrict search to a single document
    - `top_k`: (optional) number of chunks to retrieve (1–20, default 5)

    Returns the LLM-generated answer plus source chunk references and latency metrics.
    """
    store = _get_store()

    if store.total_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload documents first.",
        )

    # If scoped to a specific doc, verify it exists and is ready
    if req.document_id:
        from services.document_service import get_document
        doc = get_document(req.document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        if doc["status"] != "ready":
            raise HTTPException(
                status_code=409,
                detail=f"Document is not ready (status: {doc['status']}). "
                       "Wait for processing to complete.",
            )
        if store.chunks_for_document(req.document_id) == 0:
            raise HTTPException(
                status_code=404,
                detail="No vectors found for this document.",
            )

    result = run_query(
        question=req.question,
        store=store,
        document_id=req.document_id,
        top_k=req.top_k,
    )
    return result
