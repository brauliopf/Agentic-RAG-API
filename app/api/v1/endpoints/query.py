from fastapi import APIRouter, Depends, HTTPException, status

from ....models.requests import QueryRequest
from ....models.responses import QueryResponse
from ....services.rag_service_agentic import RAGServiceAgentic
from ....core.auth import get_current_user, get_current_user_id
from ...deps import get_rag_service_agentic
from ....core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest,
    current_user_id: str = Depends(get_current_user_id),
    rag_service_agentic: RAGServiceAgentic = Depends(get_rag_service_agentic)
):
    """Submit a RAG query and get the response. Use agentic mode if use_agentic is True."""
    logger.info("Query request", user_id=current_user_id, question=request.query[:100])
    
    try:
        # Use agentic service
        result = await rag_service_agentic.query(
            query=request.query,
            thread_id=request.thread_id,
            user_id=current_user_id
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error("Query failed", user_id=current_user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )