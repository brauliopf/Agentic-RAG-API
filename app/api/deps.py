from typing import Generator

from ..services.rag_service import nonagentic_rag_service, agentic_rag_service
from ..services.document_service import document_service
from ..services.llm_service import llm_service


def get_agentic_rag_service():
    """Dependency to get agentic RAG service instance."""
    return agentic_rag_service

def get_rag_service():
    """Dependency to get RAG service instance."""
    return nonagentic_rag_service


def get_document_service():
    """Dependency to get document service instance."""
    return document_service


def get_llm_service():
    """Dependency to get LLM service instance."""
    return llm_service 