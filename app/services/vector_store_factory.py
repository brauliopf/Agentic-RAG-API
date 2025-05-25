from typing import Protocol, runtime_checkable
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from langchain_core.embeddings import Embeddings

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol defining the interface for vector stores."""
    
    def add_documents(self, documents, **kwargs):
        """Add documents to the vector store."""
        ...
    
    def similarity_search(self, query: str, k: int = 4, **kwargs):
        """Search for similar documents."""
        ...


def create_vector_store(embeddings: Embeddings) -> VectorStore:
    """
    Factory function to create a vector store based on configuration.
    
    This function provides an abstraction layer that makes the vector store type
    transparent to the consuming code. The specific implementation is determined
    by the vector_store_type setting.
    
    Args:
        embeddings: The embeddings model to use with the vector store
        
    Returns:
        VectorStore: An instance of the configured vector store type
        
    Raises:
        ValueError: If an unsupported vector store type is specified
    """
    store_type = settings.vector_store_type.lower()
    
    logger.info("Creating vector store", store_type=store_type)
    
    if store_type == "in_memory":
        return InMemoryVectorStore(embeddings)
    elif store_type == "chroma":
        # Future implementation for Chroma
        try:
            from langchain_chroma import Chroma
            return Chroma(embedding_function=embeddings)
        except ImportError:
            logger.error("Chroma not installed. Install with: pip install langchain-chroma")
            raise ValueError("Chroma vector store requires langchain-chroma package")
    elif store_type == "faiss":
        # Future implementation for FAISS
        try:
            from langchain_community.vectorstores import FAISS
            # FAISS requires an index to be created with documents
            # For now, we'll create an empty index and let the service add documents
            import faiss
            import numpy as np
            
            # Create a placeholder index - will be properly initialized when documents are added
            dimension = 1536  # OpenAI embedding dimension
            index = faiss.IndexFlatL2(dimension)
            return FAISS(embeddings, index, {}, {})
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            raise ValueError("FAISS vector store requires faiss-cpu package")
    else:
        supported_types = ["in_memory", "chroma", "faiss"]
        error_msg = f"Unsupported vector store type: {store_type}. Supported types: {supported_types}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def get_supported_vector_stores() -> list[str]:
    """Get a list of supported vector store types."""
    return ["in_memory", "chroma", "faiss"] 