from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    classes: Optional[str] = Field(None, description="CSS classes for web scraping")
    custom_fields: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom metadata fields")


class Document(BaseModel):
    """Core document model representing a document in the system."""
    id: str = Field(..., description="Unique document identifier")
    source_type: Literal["url", "file"] = Field(..., description="Type of document source")
    status: DocumentStatus = Field(..., description="Processing status")
    created_at: datetime = Field(..., description="Creation timestamp")
    chunks_count: int = Field(default=0, description="Number of chunks created from this document")
    description: Optional[str] = Field(None, description="Human-readable description of the document")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional document metadata")
    vector_ids: Optional[List[str]] = Field(default_factory=list, description="Vector store IDs for document chunks")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class DocumentCreate(BaseModel):
    """Model for creating a new document."""
    source_type: Literal["url", "file"] = Field(..., description="Type of document source")
    description: Optional[str] = Field(None, description="Human-readable description of the document")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional document metadata")


class DocumentUpdate(BaseModel):
    """Model for updating an existing document."""
    description: Optional[str] = Field(None, description="Human-readable description of the document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional document metadata")
    status: Optional[DocumentStatus] = Field(None, description="Processing status") 