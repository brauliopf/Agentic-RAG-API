from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    id: str = Field(..., description="Unique document identifier")
    source_type: Literal["url", "file"] = Field(..., description="Type of document source")
    status: DocumentStatus = Field(..., description="Processing status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    created_at: datetime = Field(..., description="Creation timestamp")


class QueryResponse(BaseModel):
    """Response model for query operations."""
    id: str = Field(..., description="Unique query identifier")
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The generated answer")
    context: List[Dict[str, Any]] = Field(..., description="Retrieved context documents")
    processing_time: float = Field(..., description="Processing time in seconds")
    created_at: datetime = Field(..., description="Query timestamp")


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(..., description="Health check timestamp")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details") 