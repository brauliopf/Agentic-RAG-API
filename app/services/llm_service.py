from typing import Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class LLMService:
    """Service for managing LLM and embedding models."""
    
    def __init__(self):
        self._llm: Optional[ChatOpenAI] = None
        self._embeddings_llm: Optional[OpenAIEmbeddings] = None
    
    # Lazy-loading. _llm variable inits as None, so the first time llm is called, it will be created.
    # After that, it will return the cached instance.
    # @property decorator makes the llm method callable as an attribute, not as a function.
    @property
    def llm(self) -> ChatOpenAI:
        """Get or create the LLM instance."""
        if self._llm is None:
            try:
                if not settings.openai_api_key:
                    raise ValueError(
                        "OPENAI_API_KEY is required. Please set it as an environment variable or in a .env file."
                    )
                self._llm = ChatOpenAI(
                    model=settings.openai_model,
                    api_key=settings.openai_api_key
                )
                logger.info("Initialized LLM", model=settings.openai_model)
            except Exception as e:
                logger.error("Failed to initialize LLM", error=str(e))
                raise
        return self._llm
    
    @property
    def embeddings_llm(self) -> OpenAIEmbeddings:
        """Get or create the embeddings instance."""
        if self._embeddings_llm is None:
            try:
                if not settings.openai_api_key:
                    raise ValueError(
                        "OPENAI_API_KEY is required. Please set it as an environment variable or in a .env file."
                    )
                self._embeddings_llm = OpenAIEmbeddings(
                    model=settings.embedding_model,
                    api_key=settings.openai_api_key
                )
                logger.info("Initialized embeddings", model=settings.embedding_model)
            except Exception as e:
                logger.error("Failed to initialize embeddings", error=str(e))
                raise
        return self._embeddings_llm


# Global service instance
llm_service = LLMService() 