from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List, Optional, Dict, Any
import json

from ....models.requests import URLIngestRequest, DocumentDeleteRequest, DocumentDescribeRequest, DocumentUpdateRequest
from ....models.responses import DocumentResponse
from ....models.document import DocumentUpdate
from ....services.document_service import DocumentService
from ....core.logging import get_logger
from ...deps import get_document_service

router = APIRouter()
logger = get_logger(__name__)


@router.post("/documents/ingest_url", response_model=List[DocumentResponse])
async def ingest_urls(
    urls: List[URLIngestRequest],
    document_service: DocumentService = Depends(get_document_service)
):
    """Ingest documents from a list of URLs (url) or from a file (pdf, md) uploaded by the user. Returns the a list of retrieved documents."""
    docs = []
    print("INGEST URLS", urls)
    try:
        for url_request in urls:
            doc_id = await document_service.ingest_url(
                url=url_request.url,
                description=url_request.description,
                metadata=url_request.metadata
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
        logger.error("Document ingestion failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}"
        )
    
@router.post("/documents/ingest_file", response_model=DocumentResponse)
async def ingest_file(
    file_content: UploadFile = File(None, description="File to ingest"),
    description: str = Form(None, description="Human-readable description of the document"),
    metadata: str = Form(None, description="Additional metadata as JSON string"),
    document_service: DocumentService = Depends(get_document_service)
):
    """Ingest documents from a file (pdf, md) uploaded by the user. Returns the retrieved document."""
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
        
        doc_id = await document_service.ingest_file(
            file_content=file_content,
            description=description,
            metadata=metadata_dict
        )
        
        doc_data = await document_service.get_document(doc_id)
        return DocumentResponse(**doc_data.dict())
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document ingestion failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}"
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
        
        return DocumentResponse(**doc_data.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document", doc_id=doc_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@router.put("/documents/{doc_id}", response_model=DocumentResponse)
async def update_document(
    doc_id: str,
    update_request: DocumentUpdateRequest,
    document_service: DocumentService = Depends(get_document_service)
):
    """Update a document's metadata and description."""
    try:
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
        logger.error("Failed to update document", doc_id=doc_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document: {str(e)}"
        )


@router.post("/documents/describe", response_model=None)
async def describe_documents(
    request: DocumentDescribeRequest,
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
        logger.error("Document creation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document creation failed: {str(e)}"
        )


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    document_service: DocumentService = Depends(get_document_service)
):
    """List all ingested documents."""
    try:
        documents = await document_service.list_documents()
        return [DocumentResponse(**doc.dict()) for doc in documents]
        
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )
    
@router.post("/documents/delete")
async def delete_document(
    request: DocumentDeleteRequest,
    document_service: DocumentService = Depends(get_document_service)
):
    """Delete a specific document by ID."""
    try:
        await document_service.delete_document(request.doc_id)
        return {"message": "Document deleted successfully", "doc_id": request.doc_id}
    except Exception as e:
        logger.error("Failed to delete document", doc_id=request.doc_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )
