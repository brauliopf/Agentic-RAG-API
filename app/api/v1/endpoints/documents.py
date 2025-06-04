from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional, Dict, Any
import json

from ....models.requests import URLIngestRequest, DocumentDeleteRequest, DocumentDescribeRequest, DocumentUpdateRequest
from ....models.responses import DocumentResponse
from ....models.document import DocumentUpdate
from ....services.document_service import DocumentService
from ....core.logging import get_logger
from ....core.auth import get_current_user_id
from ...deps import get_document_service

router = APIRouter()
logger = get_logger(__name__)


@router.post("/documents/ingest_url", response_model=List[DocumentResponse])
async def ingest_urls(
    urls: List[URLIngestRequest],
    current_user_id: str = Depends(get_current_user_id),
    document_service: DocumentService = Depends(get_document_service)
):
    """Ingest documents from a list of URLs (url) or from a file (pdf, md) uploaded by the user. Returns the a list of retrieved documents."""
    docs = []
    logger.info("URL ingestion request", user_id=current_user_id, url_count=len(urls))
    
    try:
        for url_request in urls:
            # Add user context to metadata
            metadata = url_request.metadata or {}
            metadata["user_id"] = current_user_id
            
            doc_id = await document_service.ingest_url(
                url=url_request.url,
                user_id=current_user_id,
                description=url_request.description,
                metadata=metadata
            )
            
            # Get the document details
            doc_data = await document_service.get_document(doc_id)
            if not doc_data:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve ingested document"
                )
            docs.append(DocumentResponse(**doc_data.dict()))
        
        return docs
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document ingestion failed", user_id=current_user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}"
        )
    
@router.post("/documents/ingest_file", response_model=DocumentResponse)
async def ingest_file(
    file_content: UploadFile = File(None, description="File to ingest"),
    description: str = Form(None, description="Human-readable description of the document"),
    metadata: str = Form(None, description="Additional metadata as JSON string"),
    current_user_id: str = Depends(get_current_user_id),
    document_service: DocumentService = Depends(get_document_service)
):
    """Ingest documents from a file (pdf, md) uploaded by the user. Returns the retrieved document."""
    logger.info("File ingestion request", user_id=current_user_id, filename=file_content.filename)
    
    try:
        # Parse metadata from JSON string
        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON format for metadata"
                )
        
        # Add user context to metadata
        metadata_dict["user_id"] = current_user_id
        
        doc_id = await document_service.ingest_file(
            file_content=file_content,
            user_id=current_user_id,
            description=description,
            metadata=metadata_dict
        )
        
        doc_data = await document_service.get_document(doc_id)
        return DocumentResponse(**doc_data.dict())
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document ingestion failed", user_id=current_user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}"
        )

@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: str,
    current_user_id: str = Depends(get_current_user_id),
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
        
        # Check if user has access to this document
        if doc_data.metadata.get("user_id") != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this document"
            )
        
        return DocumentResponse(**doc_data.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document", doc_id=doc_id, user_id=current_user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@router.put("/documents/{doc_id}", response_model=DocumentResponse)
async def update_document(
    doc_id: str,
    update_request: DocumentUpdateRequest,
    current_user_id: str = Depends(get_current_user_id),
    document_service: DocumentService = Depends(get_document_service)
):
    """Update a document's metadata and description."""
    try:
        # First check if document exists and user has access
        doc_data = await document_service.get_document(doc_id)
        if not doc_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        if doc_data.metadata.get("user_id") != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this document"
            )
        
        update_data = DocumentUpdate(
            description=update_request.description,
            metadata=update_request.metadata
        )
        
        updated_doc = await document_service.update_document(doc_id, update_data)
        if not updated_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return DocumentResponse(**updated_doc.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update document", doc_id=doc_id, user_id=current_user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document: {str(e)}"
        )


@router.post("/documents/describe", response_model=None)
async def describe_documents(
    request: DocumentDescribeRequest,
    current_user_id: str = Depends(get_current_user_id),
    document_service: DocumentService = Depends(get_document_service)
):
    """Describe a list of documents with title and description.
    This takes a dictionary of document title and description and updates the documents object in the document service.
    Returns nothing."""
    try:
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document creation failed", user_id=current_user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document creation failed: {str(e)}"
        )


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    current_user_id: str = Depends(get_current_user_id),
    document_service: DocumentService = Depends(get_document_service)
):
    """List all ingested documents for the current user."""
    try:
        documents = await document_service.list_documents()
        
        # Filter documents by user_id
        user_documents = [
            doc for doc in documents 
            if doc.metadata.get("user_id") == current_user_id
        ]
        
        return [DocumentResponse(**doc.dict()) for doc in user_documents]
        
    except Exception as e:
        logger.error("Failed to list documents", user_id=current_user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )
    
@router.post("/documents/delete")
async def delete_document(
    request: DocumentDeleteRequest,
    current_user_id: str = Depends(get_current_user_id),
    document_service: DocumentService = Depends(get_document_service)
):
    """Delete a specific document by ID."""
    try:
        # First check if document exists and user has access
        doc_data = await document_service.get_document(request.doc_id)
        if not doc_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        if doc_data.metadata.get("user_id") != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this document"
            )
        
        await document_service.delete_document(request.doc_id, current_user_id)
        return {"message": "Document deleted successfully", "doc_id": request.doc_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document", doc_id=request.doc_id, user_id=current_user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )
