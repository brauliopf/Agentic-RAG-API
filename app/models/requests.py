from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any, Literal


class DocumentIngestRequest(BaseModel):
    """Request model for document ingestion."""
    source_type: Literal["url"] = Field(..., description="Type of document source")
    content: str = Field(..., description="URL to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., description="The question to ask")
    max_docs: Optional[int] = Field(default=4, ge=1, le=10, description="Maximum number of documents to retrieve")
    section_filter: Optional[Literal["beginning", "middle", "end"]] = Field(
        default=None, 
        description="Filter documents by section"
    ) 