from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from ....models.requests import DocumentIngestRequest
from ....models.responses import DocumentResponse, ErrorResponse
from ....services.document_service import DocumentService
from ....core.logging import get_logger
from ...deps import get_document_service

router = APIRouter()
logger = get_logger(__name__)


@router.post("/documents/ingest/tako", response_model=List[DocumentResponse])

async def ingest_documents_tako(
    request: List[DocumentIngestRequest],
    document_service: DocumentService = Depends(get_document_service)
):
    """Ingest documents from a list of URLs. Returns the retrieved documents."""
    docs = []
    try:
        for req in request:
            if req.source_type != "url":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only URL sources are supported currently"
                )
            doc_id = await document_service.ingest_url_tako(
                url=req.content,
                metadata=req.metadata
            )
            # Get the document details
            doc_data = await document_service.get_document(doc_id)
            if not doc_data:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve ingested document"
                )
            docs.append(DocumentResponse(**doc_data))
        
        return docs
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document ingestion failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}"
        )


@router.post("/documents/ingest", response_model=List[DocumentResponse])
async def ingest_documents(
    request: List[DocumentIngestRequest],
    document_service: DocumentService = Depends(get_document_service)
):
    """Ingest documents from a list of URLs. Returns the retrieved documents."""
    docs = []
    try:
        for req in request:
            if req.source_type.split("_")[0] != "url":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only URL sources are supported currently"
                )
            
            doc_id = await document_service.ingest_url(
                url=req.content,
                metadata=req.metadata,
                url_type=req.source_type.split("_")[1]
            )
            
            # Get the document details
            doc_data = await document_service.get_document(doc_id)
            if not doc_data:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve ingested document"
                )
            docs.append(DocumentResponse(**doc_data))
        
        return docs
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document ingestion failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}"
        )
    

@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    document_service: DocumentService = Depends(get_document_service)
):
    """List all ingested documents."""
    try:
        documents = await document_service.list_documents()
        return [DocumentResponse(**doc) for doc in documents]
        
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """Get a specific document by ID."""
    try:
        doc_data = await document_service.get_document(doc_id)
        if not doc_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return DocumentResponse(**doc_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document", doc_id=doc_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """Delete a document by ID."""
    try:
        success = await document_service.delete_document(doc_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document", doc_id=doc_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        ) 