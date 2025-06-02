from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, Literal


class DocumentIngestRequest(BaseModel):
    """Request model for document ingestion."""
    source_type: Literal["url", "url_tako"] = Field(..., description="Type of document source")
    content: str = Field(..., description="URL to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


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