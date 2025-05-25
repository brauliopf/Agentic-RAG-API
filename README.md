# FastAPI RAG System

A production-ready FastAPI RAG (Retrieval-Augmented Generation) system implementing Phase 1 of the development plan.

## Features

- **Document Ingestion**: Ingest documents from URLs with automatic chunking and embedding
- **RAG Queries**: Ask questions and get AI-generated answers based on ingested documents
- **Streaming Responses**: Real-time streaming of query responses
- **Section Filtering**: Filter documents by section (beginning, middle, end)
- **Structured Logging**: Comprehensive logging with structlog
- **Health Monitoring**: Health check endpoints
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

## Architecture

```
agentic-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   └── logging.py         # Logging configuration
│   ├── models/
│   │   ├── requests.py        # Pydantic request models
│   │   └── responses.py       # Pydantic response models
│   ├── services/
│   │   ├── llm_service.py     # LLM and embeddings management
│   │   ├── document_service.py # Document ingestion and storage
│   │   └── rag_service.py     # RAG pipeline with LangGraph
│   ├── api/
│   │   ├── deps.py            # Dependency injection
│   │   └── v1/
│   │       ├── api.py         # API router aggregation
│   │       └── endpoints/
│   │           ├── health.py   # Health check endpoints
│   │           ├── documents.py # Document management endpoints
│   │           └── query.py    # RAG query endpoints
│   └── utils/
│       └── helpers.py         # Utility functions
├── requirements.txt           # Python dependencies
└── app.py                    # Legacy entry point (redirects to new structure)
```

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   Create a `.env` file with the following variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Application Configuration (optional)
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# RAG Configuration (optional)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCS_RETRIEVAL=4
```

## Usage

### Running the Application

```bash
# Using the new structure
python -m app.main

# Or using the legacy entry point
python app.py
```

The application will start on `http://localhost:8000`

### API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### Health Check

- `GET /api/v1/health` - Check application health

#### Document Management

- `POST /api/v1/documents/ingest` - Ingest a document from URL
- `GET /api/v1/documents` - List all ingested documents
- `GET /api/v1/documents/{doc_id}` - Get specific document
- `DELETE /api/v1/documents/{doc_id}` - Delete a document

#### RAG Queries

- `POST /api/v1/query` - Submit a RAG query
- `GET /api/v1/query/{query_id}` - Get query result by ID
- `POST /api/v1/query/stream` - Stream a RAG query response

### Example Usage

#### 1. Ingest a Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "url",
    "content": "https://example.com/article",
    "metadata": {"title": "Example Article"}
  }'
```

#### 2. Submit a Query

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the article?",
    "max_docs": 4,
    "section_filter": "beginning"
  }'
```

#### 3. Stream a Query

```bash
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Summarize the key points",
    "max_docs": 4
  }'
```

## Configuration

The application uses Pydantic Settings for configuration management. All settings can be configured via environment variables:

### Vector Store Configuration

The system supports multiple vector store backends through a factory pattern:

- **in_memory** (default): Fast, ephemeral storage for development
- **chroma**: Persistent vector database (requires `pip install langchain-chroma`)
- **faiss**: High-performance similarity search (requires `pip install faiss-cpu`)

To switch vector stores, set the `VECTOR_STORE_TYPE` environment variable:

```bash
# Use Chroma for persistent storage
export VECTOR_STORE_TYPE=chroma

# Use FAISS for high-performance search
export VECTOR_STORE_TYPE=faiss
```

### All Configuration Options

- `OPENAI_API_KEY`: Required OpenAI API key
- `DEBUG`: Enable debug mode (default: false)
- `LOG_LEVEL`: Logging level (default: INFO)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `CHUNK_OVERLAP`: Chunk overlap (default: 200)
- `MAX_DOCS_RETRIEVAL`: Maximum documents to retrieve (default: 4)
- `VECTOR_STORE_TYPE`: Vector store type: in_memory, chroma, faiss (default: in_memory)

## Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain**: Framework for developing applications with LLMs
- **LangGraph**: Library for building stateful, multi-actor applications
- **OpenAI**: LLM and embedding models
- **Pydantic**: Data validation and settings management
- **Structlog**: Structured logging
- **BeautifulSoup**: HTML parsing for web scraping
- **Vector Store Factory**: Configurable vector storage backends (InMemory, Chroma, FAISS)

## Limitations (Phase 1)

- Default in-memory vector storage (data is lost on restart - switch to Chroma/FAISS for persistence)
- Document deletion support varies by vector store implementation
- Basic error handling and validation
- No authentication or authorization
- No rate limiting
- No persistent storage

## Next Steps (Phase 2 & 3)

- Implement persistent vector database (Chroma, Pinecone, etc.)
- Add authentication and authorization
- Implement rate limiting and caching
- Add comprehensive monitoring and metrics
- Implement batch processing
- Add support for multiple document types
- Enhanced error handling and recovery
- Unit and integration tests
- Docker containerization
- CI/CD pipeline
