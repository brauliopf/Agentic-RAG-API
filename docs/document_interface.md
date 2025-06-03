# Document Interface and Schema

## Overview

The document system has been refactored to use a proper Pydantic-based interface with improved type safety and a new `description` field for better document management.

## Document Model

### Core Document Schema

```python
class Document(BaseModel):
    id: str                                    # Unique document identifier
    source_type: Literal["url", "file"]        # Type of document source
    status: DocumentStatus                     # Processing status
    created_at: datetime                       # Creation timestamp
    chunks_count: int                          # Number of chunks created
    description: Optional[str]                 # Human-readable description (NEW)
    metadata: Optional[Dict[str, Any]]         # Additional metadata
    vector_ids: Optional[List[str]]            # Vector store IDs for chunks
```

### Document Status

```python
class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
```

## API Changes

### URL Ingestion

**Before:**

```python
URLIngestRequest(
    url="https://example.com",
    metadata={"category": "docs"}
)
```

**After:**

```python
URLIngestRequest(
    url="https://example.com",
    description="Example documentation website",  # NEW
    metadata={"category": "docs"}
)
```

### File Ingestion

**Before:**

```bash
curl -X POST "/documents/ingest_file" \
  -F "file_content=@document.pdf" \
  -F "metadata={\"category\": \"manual\"}"
```

**After:**

```bash
curl -X POST "/documents/ingest_file" \
  -F "file_content=@document.pdf" \
  -F "description=User manual for the application" \
  -F "metadata={\"category\": \"manual\"}"
```

### Document Updates (NEW)

```python
# Update document description and metadata
PUT /documents/{doc_id}
{
    "description": "Updated description",
    "metadata": {"updated": true}
}
```

## Service Layer Changes

### DocumentService

- **Type Safety**: Now uses `Dict[str, Document]` instead of `Dict[str, Dict[str, Any]]`
- **New Methods**:
  - `update_document(doc_id, update_data)` - Update document metadata
- **Enhanced Methods**: All ingestion methods now accept `description` parameter

### Benefits

1. **Type Safety**: Proper Pydantic models prevent runtime errors
2. **Better Documentation**: Description field improves document discoverability
3. **Maintainability**: Centralized document schema in dedicated module
4. **Extensibility**: Easy to add new fields and validation rules
5. **API Consistency**: Uniform response format across all endpoints

## Migration Notes

- Existing documents will have `description=None` until updated
- All API responses now include the new fields
- The old dict-based document structure is fully replaced
- Backward compatibility maintained for existing metadata structure

## Example Usage

```python
from app.models import Document, DocumentStatus, DocumentUpdateRequest
from app.services.document_service import document_service

# Create document with description
doc = Document(
    id="my-doc",
    source_type="url",
    status=DocumentStatus.COMPLETED,
    created_at=datetime.now(timezone.utc),
    description="Important company documentation",
    metadata={"department": "engineering"}
)

# Update document description
update_request = DocumentUpdateRequest(
    description="Updated company documentation",
    metadata={"department": "engineering", "updated": True}
)

updated_doc = await document_service.update_document("my-doc", update_request)
```
