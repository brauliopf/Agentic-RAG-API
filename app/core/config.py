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
    """Application settings with environment variable support."""
    
    # API Configuration
    app_name: str = "FastAPI RAG System"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    openai_model: str = "gpt-4o-mini"
    
    # Document ingestion (splitting, embedding, and storage)
    embedding_model: str = "text-embedding-ada-002"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_docs_retrieval: int = 4
    vector_store_type: str = Field(
        default="in_memory", 
        description="Vector store type: in_memory, chroma, faiss"
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