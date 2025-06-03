from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, Literal
from fastapi import UploadFile, File


class URLIngestRequest(BaseModel):
    """Request model for URL ingestion."""
    url: str = Field(default=None, description="URL to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DocumentIngestRequest(BaseModel):
    """Request model for document ingestion. Document can be a URL (webpage) or a file (PDF or markdown)."""
    file_content: UploadFile = File(default=None, description="File to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DocumentDeleteRequest(BaseModel):
    """Request model for document deletion."""
    doc_id: str = Field(..., description="The document ID to delete")


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., description="The question to ask")
    thread_id: Optional[str] = Field(default=None, description="The thread ID to use for the query")
    use_agentic: bool = Field(default=False, description="Whether to use agentic mode")


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    ) 