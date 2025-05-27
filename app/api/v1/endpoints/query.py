from fastapi import APIRouter, Depends, HTTPException, status

from ....models.requests import QueryRequest
from ....models.responses import QueryResponse
from ....services.rag_service import RAGService
from ....services.rag_service_agentic import RAGServiceAgentic
from ...deps import get_rag_service, get_rag_service_agentic
from ....core.logging import get_logger

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

@router.post("/query-agentic", response_model=QueryResponse)
async def submit_query_agentic(
    request: QueryRequest,
    rag_service: RAGServiceAgentic = Depends(get_rag_service_agentic)
):
    """Submit a agentic RAG query and get the response."""
    try:
        result = await rag_service.query(
            question=request.question,
            thread_id=request.thread_id
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error("Query failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )