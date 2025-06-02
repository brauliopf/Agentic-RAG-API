import bs4
import tempfile
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import UploadFile
from ..core.config import settings
from ..core.logging import get_logger
from ..models.responses import DocumentStatus
from .llm_service import llm_service
from .vector_store_factory import load_vector_store

logger = get_logger(__name__)


class DocumentService:
    """Service for managing document ingestion and storage in the vector store."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True
        )
        self.vector_store = load_vector_store(llm_service.embeddings)
        self.documents: Dict[str, Dict[str, Any]] = {
            'CLT Normas Correlatas 6th Ed.pdf': {
                'id': 'CLT Normas Correlatas 6th Ed.pdf',
                'source_type': 'file',
                'source_file': 'CLT Normas Correlatas 6th Ed.pdf',
                'status': DocumentStatus.COMPLETED,
                'created_at': datetime.utcnow(),
            }
        }
        logger.info("Initialized DocumentService")

    async def ingest_file(self, file_content: UploadFile, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Ingest a document from a file uploaded by the user: load, split, and upsert to vector store in batches."""
        doc_id = file_content.filename
        temp_file_path = None
        
        try:
            logger.info("Starting document ingestion", source_type="file", doc_id=doc_id)

            # Store document metadata
            self.documents[doc_id] = {
                "id": doc_id,
                "source_type": "file",
                "status": DocumentStatus.PROCESSING,
                "created_at": datetime.utcnow(),
                "chunks_count": 0,
                "metadata": metadata
            }

            # Get file extension and create temporary file
            file_extension = file_content.filename.lower().split('.')[-1]
            
            # Create a temporary file to save the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                temp_file_path = temp_file.name
                # Read and write the uploaded file content to the temporary file
                content = await file_content.read()
                temp_file.write(content)
                temp_file.flush()
            
            # Load document based on file type using the temporary file path
            if file_extension == 'pdf':
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == 'md':
                loader = UnstructuredMarkdownLoader(temp_file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
            docs = loader.load()

            # Split documents into chunks
            all_splits = self.text_splitter.split_documents(docs)
            
            # Process documents in batches
            all_document_ids = []
            batch_size = settings.batch_size
            
            for i in range(0, len(all_splits), batch_size):
                batch = all_splits[i:i + batch_size]
                logger.info(
                    "Processing batch", 
                    doc_id=doc_id, 
                    batch_number=i // batch_size + 1,
                    batch_size=len(batch),
                    total_batches=(len(all_splits) + batch_size - 1) // batch_size
                )
                
                # Generate unique IDs for each document in the batch
                batch_ids = [f"{doc_id}_{i + j}" for j in range(len(batch))]
                
                # Upsert batch to vector store
                document_ids = self.vector_store.add_documents(
                    documents=batch, 
                    ids=batch_ids,
                    metadata=metadata
                )
                all_document_ids.extend(document_ids)
            
            # Update document status
            self.documents[doc_id].update({
                "status": DocumentStatus.COMPLETED,
                "chunks_count": len(all_splits),
                "vector_ids": all_document_ids
            })
            
            logger.info(
                "Document ingestion completed", 
                doc_id=doc_id, 
                chunks_count=len(all_splits),
                total_batches=(len(all_splits) + batch_size - 1) // batch_size,
                metadata=metadata
            )
            
            return doc_id
            
        except Exception as e:
            logger.error("Document ingestion failed", doc_id=doc_id, error=str(e))
            if doc_id in self.documents:
                self.documents[doc_id]["status"] = DocumentStatus.FAILED
            raise
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug("Cleaned up temporary file", temp_file_path=temp_file_path)
                except Exception as cleanup_error:
                    logger.warning("Failed to clean up temporary file", temp_file_path=temp_file_path, error=str(cleanup_error))

    async def ingest_url(self, url: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Ingest a document from a URL: load, split, and add to vector store."""
        doc_id = url
        
        try:
            logger.info("Starting document ingestion", source_type="url", doc_id=doc_id, url=url)
            
            # Store document metadata
            self.documents[doc_id] = {
                "id": doc_id,
                "source_type": "url",
                "status": DocumentStatus.PROCESSING,
                "created_at": datetime.utcnow(),
                "chunks_count": 0,
                "metadata": metadata
            }
            
            # Load document using WebBaseLoader with BeautifulSoup parsing
            # Get only content under HTML tags with the following classes (@SoupStrainer)
            bs4_strainer = bs4.SoupStrainer(
                class_=("termos-de-uso", "anexo", "header", "wrapper")
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
            
            # Generate unique IDs for each chunk based on the doc_id
            chunk_ids = [f"{doc_id}_{i}" for i in range(len(all_splits))]
            
            # Add to vector store with IDs for upsert behavior
            document_ids = self.vector_store.add_documents(
                documents=all_splits, 
                ids=chunk_ids,
                metadata=metadata
            )
            
            # Update document status
            self.documents[doc_id].update({
                "status": DocumentStatus.COMPLETED,
                "chunks_count": len(all_splits),
                "vector_ids": document_ids
            })
            
            logger.info(
                "Document ingestion completed", 
                doc_id=doc_id, 
                chunks_count=len(all_splits),
                metadata=metadata
            )
            
            return doc_id
            
        except Exception as e:
            logger.error("Document ingestion failed", doc_id=doc_id, error=str(e))
            if doc_id in self.documents:
                self.documents[doc_id]["status"] = DocumentStatus.FAILED
            raise
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata (file or URL) by ID."""
        return self.documents.get(doc_id)
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested documents."""
        # Default docs: CLT and Tako API
        docs = list(self.documents.values())
        return docs


# Global service instance
document_service = DocumentService() 