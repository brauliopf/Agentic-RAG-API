from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Relative imports (requires module syntax: python -m app.main)
from .core.config import settings
from .core.logging import configure_logging, get_logger
from .api.v1.api import api_router
from .models.responses import ErrorResponse

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Python context manager that handles application lifecycle events in FastAPI.
# "Before yield" gets executed on startup
# "After yield" gets executed on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("Starting FastAPI RAG application", version=settings.app_version)
    # (services will be lazy-loaded)
    logger.info("Application startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI RAG application")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A production-ready FastAPI RAG (Retrieval-Augmented Generation) system",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "https://agentic-rag-app.vercel.app"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP_ERROR",
            message=exc.detail,
            detail=None
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions with consistent error response format."""
    logger.error("Unhandled exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message="An internal server error occurred",
            detail=str(exc) if settings.debug else None
        ).model_dump()
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "FastAPI RAG System",
        "version": settings.app_version,
        "docs": "/docs"
    }


def main():
    """Main entry point for running the application."""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main() 