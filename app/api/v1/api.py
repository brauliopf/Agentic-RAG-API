from fastapi import APIRouter

from .endpoints import health, documents, query

api_router = APIRouter(prefix="/api/v1")

# Include all endpoint routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(query.router, tags=["query"]) 