from ..services.rag_service import rag_service
from ..services.rag_service_agentic import rag_service_agentic
from ..services.document_service import document_service
from ..services.llm_service import llm_service


def get_rag_service_agentic():
    """Dependency to get agentic RAG service instance."""
    return rag_service_agentic

def get_rag_service():
    """Dependency to get RAG service instance."""
    return rag_service


def get_document_service():
    """Dependency to get document service instance."""
    return document_service


def get_llm_service():
    """Dependency to get LLM service instance."""
    return llm_service 