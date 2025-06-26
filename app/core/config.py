from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env file BEFORE any other imports to ensure LangSmith variables are available
def load_env_file():
    """Load environment variables from .env file."""
    env_paths = [
        Path(".env"),
        Path("agentic-rag/.env"),
        Path(__file__).parent.parent.parent / ".env"
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            return str(env_path)
    return None

# Load environment variables immediately
load_env_file()


class Settings(BaseSettings):
    """Application settings with environment variable support.
    *** Settings class automatically maps MY_ENVVAR_KEYS environment variable to settings.my_envvar_key ***
    """
    
    # API Configuration
    app_name: str = "FastAPI RAG System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Authentication Configuration
    supabase_url: str = "https://zxwfmrccjrbejqxmmxrw.supabase.co"
    supabase_jwt_secret: Optional[str] = Field(None, description="Supabase JWT secret for token verification")
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_model: str = "gpt-4o-mini"
    # Document ingestion (splitting, embedding, and storage)
    embedding_model: str = "text-embedding-3-small" #1536 dimensions
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_docs_retrieval: int = 4
    batch_size: int = 100  # Batch size for document ingestion

    # vector store configuration
    pinecone_api_key: Optional[str] = Field(None, description="Pinecone API key")
    pinecone_index: str = "agentic-rag"
    vector_store_type: str = Field(
        default="pinecone", 
        description="Vector store type: in_memory, pinecone"
    )
    
    # Logging Configuration
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# Global settings instance
settings = Settings() 