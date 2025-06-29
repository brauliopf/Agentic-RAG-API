from ..services.rag_service_agentic import rag_service_agentic
from ..services.document_service import document_service
from ..services.llm_service import llm_service
from ..core.auth import get_current_user, get_current_user_id


def get_rag_service_agentic():
    """Dependency to get agentic RAG service instance."""
    return rag_service_agentic


def get_document_service():
    """Dependency to get document service instance."""
    return document_service


def get_llm_service():
    """Dependency to get LLM service instance."""
    return llm_service


# Authentication dependencies
def get_authenticated_user():
    """Dependency to get authenticated user."""
    return get_current_user


def get_authenticated_user_id():
    """Dependency to get authenticated user ID."""
    return get_current_user_id 