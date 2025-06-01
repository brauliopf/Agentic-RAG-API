import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import bs4
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..core.config import settings
from ..core.logging import get_logger
from ..models.responses import DocumentStatus
from .llm_service import llm_service
from .vector_store_factory import create_vector_store, get_supported_vector_stores

logger = get_logger(__name__)


class DocumentService:
    """Service for managing document ingestion and storage."""
    
    def __init__(self):
        self.vector_store = create_vector_store(llm_service.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True
        )
        self.documents: Dict[str, Dict[str, Any]] = {}
        logger.info("Initialized DocumentService")
    
    async def ingest_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Ingest a document from a file: load, split, and add to vector store."""
        doc_id = str(uuid.uuid4())
        
        try:
            logger.info("Starting document ingestion", doc_id=doc_id, file_path=file_path)

            # Store document metadata
            self.documents[doc_id] = {
                "id": doc_id,
                "source_type": "file",
                "source_file": file_path,
                "status": DocumentStatus.PROCESSING,
                "metadata": metadata or {},
                "created_at": datetime.utcnow(),
                "chunks_count": 0
            }

            # Get file extension and load document accordingly
            file_extension = file_path.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == 'md':
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
            docs = loader.load()

            # Split documents into chunks
            all_splits = self.text_splitter.split_documents(docs)
            
            # Add to vector store
            document_ids = self.vector_store.add_documents(documents=all_splits)
            
            # Update document status
            self.documents[doc_id].update({
                "status": DocumentStatus.COMPLETED,
                "chunks_count": len(all_splits),
                "vector_ids": document_ids
            })
            
            logger.info(
                "Document ingestion completed", 
                doc_id=doc_id, 
                chunks_count=len(all_splits)
            )
            
            return doc_id
            
        except Exception as e:
            logger.error("Document ingestion failed", doc_id=doc_id, error=str(e))
            if doc_id in self.documents:
                self.documents[doc_id]["status"] = DocumentStatus.FAILED
            raise
            
    
    async def ingest_url(self, url: str, metadata: Optional[Dict[str, Any]] = None, url_type: Optional[str] = None) -> str:
        """Ingest a document from a URL: load, split, and add to vector store."""
        doc_id = str(uuid.uuid4())
        
        try:
            logger.info("Starting document ingestion", doc_id=doc_id, url=url, url_type=url_type)
            
            # Store document metadata
            self.documents[doc_id] = {
                "id": doc_id,
                "source_type": f"url_{url_type}" if url_type else "url",
                "source_url": url,
                "status": DocumentStatus.PROCESSING,
                "metadata": metadata or {},
                "created_at": datetime.utcnow(),
                "chunks_count": 0
            }
            
            # Load document using WebBaseLoader with BeautifulSoup parsing
            # Get only content under HTML tags with the following classes (@SoupStrainer)
            if url_type == "tako":
                bs4_strainer = bs4.SoupStrainer(
                    class_=("termos-de-uso", "anexo", "header", "wrapper")
                )
            else:
                bs4_strainer = bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
                
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs={"parse_only": bs4_strainer}
            )
            docs = loader.load()
            
            if not docs:
                raise ValueError("No content could be extracted from URL")
            
            # Split documents into chunks
            all_splits = self.text_splitter.split_documents(docs)
            
            # Add to vector store
            document_ids = self.vector_store.add_documents(documents=all_splits)
            
            # Update document status
            self.documents[doc_id].update({
                "status": DocumentStatus.COMPLETED,
                "chunks_count": len(all_splits),
                "vector_ids": document_ids
            })
            
            logger.info(
                "Document ingestion completed", 
                doc_id=doc_id, 
                chunks_count=len(all_splits)
            )
            
            return doc_id
            
        except Exception as e:
            logger.error("Document ingestion failed", doc_id=doc_id, error=str(e))
            if doc_id in self.documents:
                self.documents[doc_id]["status"] = DocumentStatus.FAILED
            raise
    
    def _add_section_metadata(self, documents: List[Document]) -> None:
        """Add section metadata to document chunks."""
        total_documents = len(documents)
        third = total_documents // 3
        
        for i, document in enumerate(documents):
            if i < third:
                document.metadata["section"] = "beginning"
            elif i < 2 * third:
                document.metadata["section"] = "middle"
            else:
                document.metadata["section"] = "end"
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata by ID."""
        return self.documents.get(doc_id)
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested documents."""
        return list(self.documents.values())
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks from the vector store."""
        try:
            if doc_id not in self.documents:
                return False
            
            # Note: Some vector stores don't support deletion by ID
            # This is a limitation that varies by vector store implementation
            logger.warning(
                "Document deletion may not be fully supported by current vector store", 
                doc_id=doc_id,
                vector_store_type=settings.vector_store_type
            )
            
            # Remove from our tracking
            del self.documents[doc_id]
            logger.info("Document removed from tracking", doc_id=doc_id)
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete document", doc_id=doc_id, error=str(e))
            return False

    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the current vector store configuration."""
        return {
            "type": settings.vector_store_type,
            "supported_types": get_supported_vector_stores(),
            "class_name": self.vector_store.__class__.__name__
        }


# Global service instance
document_service = DocumentService() 