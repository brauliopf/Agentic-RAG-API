from typing import Protocol, runtime_checkable, Optional
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

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


def load_vector_store(embeddings: Embeddings, user_id: Optional[str] = "__default__") -> VectorStore:
    """
    Factory function to create a vector store based on configuration.
    
    This function provides an abstraction layer that makes the vector store type
    transparent to the consuming code. The specific implementation is determined
    by the vector_store_type setting.
    
    Args:
        embeddings: The embeddings model to use with the vector store
        user_id: Optional user ID to use as namespace for user-specific operations
        
    Returns:
        VectorStore: An instance of the configured vector store type
        
    Raises:
        ValueError: If an unsupported vector store type is specified
    """
    store_type = settings.vector_store_type.lower()
    
    logger.info("Creating vector store", store_type=store_type, user_id=user_id)

    if store_type == "in_memory":
        return InMemoryVectorStore(embeddings)
    elif store_type == "pinecone":
        try:
            if not settings.pinecone_api_key:
                raise ValueError("PINECONE_API_KEY is required for Pinecone vector store")
            
            # Create/Get vector store with user-specific namespace if provided
            vector_store = PineconeVectorStore(
                index_name=settings.pinecone_index,
                embedding=embeddings,
                namespace=user_id,  # Use user_id as namespace for user isolation
                pinecone_api_key=settings.pinecone_api_key
            )
            logger.info("Successfully created Pinecone vector store", index_name=settings.pinecone_index, namespace=user_id)
            return vector_store
        except Exception as e:
            logger.error("Failed to create Pinecone vector store", error=str(e))
            raise
    else:
        supported_types = ["in_memory", "pinecone"]
        error_msg = f"Unsupported vector store type: {store_type}. Supported types: {supported_types}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def get_supported_vector_stores() -> list[str]:
    """Get a list of supported vector store types."""
    return ["in_memory", "pinecone"] 