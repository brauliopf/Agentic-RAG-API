import bs4
import tempfile
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import UploadFile
from ..core.config import settings
from ..core.logging import get_logger
from ..models.document import Document, DocumentStatus, DocumentUpdate
from .llm_service import llm_service
from .vector_store_factory import load_vector_store
from pinecone import Pinecone

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
        self.documents: Dict[str, Document] = {}
        logger.info("Initialized DocumentService")

    async def ingest_file(self, file_content: UploadFile, metadata: Optional[Dict[str, Any]] = None, description: Optional[str] = None) -> str:
        """Ingest a document from a file uploaded by the user: load, split, and upsert to vector store in batches."""
        doc_id = file_content.filename
        temp_file_path = None
        
        try:
            logger.info("Starting document ingestion", source_type="file", doc_id=doc_id)

            # Store document metadata using the Document model
            document = Document(
                id=doc_id,
                source_type="file",
                status=DocumentStatus.PROCESSING,
                created_at=datetime.now(timezone.utc),
                chunks_count=0,
                description=description,
                metadata=metadata or {},
                vector_ids=[]
            )
            self.documents[doc_id] = document

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
            self.documents[doc_id].status = DocumentStatus.COMPLETED
            self.documents[doc_id].chunks_count = len(all_splits)
            self.documents[doc_id].vector_ids = all_document_ids
            
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
                self.documents[doc_id].status = DocumentStatus.FAILED
            raise
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug("Cleaned up temporary file", temp_file_path=temp_file_path)
                except Exception as cleanup_error:
                    logger.warning("Failed to clean up temporary file", temp_file_path=temp_file_path, error=str(cleanup_error))

    async def ingest_url(self, url: str, metadata: Optional[Dict[str, Any]] = None, description: Optional[str] = None) -> str:
        """Ingest a document from a URL: load, split, and add to vector store."""
        doc_id = url
        
        try:
            logger.info("Starting document ingestion", source_type="url", doc_id=doc_id, url=url)
            
            # Store document metadata using the Document model
            document = Document(
                id=doc_id,
                source_type="url",
                status=DocumentStatus.PROCESSING,
                created_at=datetime.now(timezone.utc),
                chunks_count=0,
                description=description,
                metadata=metadata or {},
                vector_ids=[]
            )
            self.documents[doc_id] = document
            
            # Load document using WebBaseLoader with BeautifulSoup parsing
            # Get only content under HTML tags with the following classes (@SoupStrainer)
            target_classes = metadata.get("classes") if metadata else None

            if target_classes:
                bs4_strainer = bs4.SoupStrainer(
                    class_=target_classes.split(",")
                )
                loader = WebBaseLoader(
                    web_paths=(url,),
                    bs_kwargs={"parse_only": bs4_strainer}
                )
            else:
                loader = WebBaseLoader(
                    web_paths=(url,)
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
            )
            
            # Update document status
            self.documents[doc_id].status = DocumentStatus.COMPLETED
            self.documents[doc_id].chunks_count = len(all_splits)
            self.documents[doc_id].vector_ids = document_ids
            
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
                self.documents[doc_id].status = DocumentStatus.FAILED
            raise
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document metadata (file or URL) by ID."""
        return self.documents.get(doc_id)
    
    async def update_document(self, doc_id: str, update_data: DocumentUpdate) -> Optional[Document]:
        """Update document metadata."""
        if doc_id not in self.documents:
            return None
        
        document = self.documents[doc_id]
        
        # Update fields if provided
        if update_data.description is not None:
            document.description = update_data.description
        if update_data.metadata is not None:
            document.metadata.update(update_data.metadata)
        if update_data.status is not None:
            document.status = update_data.status
        
        logger.info("Document updated", doc_id=doc_id, update_fields=update_data.dict(exclude_none=True))
        return document
    
    async def list_documents(self) -> List[Document]:
        """List all ingested documents."""
        return list(self.documents.values())
    
    async def delete_document(self, doc_id: str):
        """Delete a document by ID."""
        try:
            # Get all ids in pinecone
            pc = Pinecone(api_key=settings.pinecone_api_key)

            # To get the unique host for an index, 
            # see https://docs.pinecone.io/guides/manage-data/target-an-index
            index = pc.Index(host="https://clt-tako-rag-kueduco.svc.aped-4627-b74a.pinecone.io")

            # Collect all IDs with the document prefix
            all_ids = []
            try:
                # The list() method returns an iterator where each iteration yields a list of ID strings
                for ids_batch in index.list(prefix=doc_id, namespace=''):
                    # ids_batch is a list of ID strings, so we extend our all_ids list
                    all_ids.extend(ids_batch)
                
                logger.info(f"Found {len(all_ids)} vectors to delete for document", doc_id=doc_id)
                
                # Only delete if we found IDs
                if all_ids:
                    # Delete all ids in pinecone
                    index.delete(ids=all_ids, namespace='')
                    logger.info(f"Deleted {len(all_ids)} vectors from Pinecone", doc_id=doc_id)
                else:
                    logger.warning("No vectors found in Pinecone for document", doc_id=doc_id)
                    
            except Exception as pinecone_error:
                logger.error("Failed to delete vectors from Pinecone", doc_id=doc_id, error=str(pinecone_error))
                # Continue with local cleanup even if Pinecone deletion fails
            
            # Delete document from documents dict
            if doc_id in self.documents:
                del self.documents[doc_id]
                logger.info("Document deleted from local storage", doc_id=doc_id)
            else:
                logger.warning("Document not found in local storage", doc_id=doc_id)
                
        except Exception as e:
            logger.error("Document deletion failed", doc_id=doc_id, error=str(e))
            raise


# Global service instance
document_service = DocumentService() 