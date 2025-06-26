from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, Literal
from fastapi import UploadFile, File


class URLIngestRequest(BaseModel):
    """Request model for URL ingestion."""
    url: str = Field(default=None, description="URL to ingest")
    description: Optional[str] = Field(None, description="Human-readable description of the document")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DocumentDescribeRequest(BaseModel):
    """Request model for document description."""
    documents: Dict[str, Any] = Field(..., description="The document title and description to be provided to the retriever LLM")

class DocumentDeleteRequest(BaseModel):
    """Request model for document deletion."""
    doc_id: str = Field(..., description="The document ID to delete")


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., description="The query to ask")
    thread_id: Optional[str] = Field(default=None, description="The thread ID to use for the query")
    use_agentic: Optional[bool] = Field(default=True, description="Whether to use agentic mode")


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class DocumentUpdateRequest(BaseModel):
    """Request model for document updates."""
    description: Optional[str] = Field(None, description="Human-readable description of the document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional document metadata") 