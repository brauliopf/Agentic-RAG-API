from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from ....models.requests import QueryRequest
from ....models.responses import QueryResponse
from ....services.rag_service import RAGService
from ....core.logging import get_logger
from ...deps import get_rag_service

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Submit a RAG query and get the response."""
    try:
        result = await rag_service.query(
            question=request.question,
            max_docs=request.max_docs,
            section_filter=request.section_filter
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error("Query failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )
