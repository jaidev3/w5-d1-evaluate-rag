"""
Main FastAPI application for Medical AI Assistant.
Entry point for the production-ready medical RAG system with RAGAS evaluation.
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
import uvicorn

from app.config import get_config
from app.api.routes import router
from app.utils.logging import setup_logging

# Initialize configuration
config = get_config()

# Setup logging
setup_logging(config.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting Medical AI Assistant")
    logger.info(f"📋 Configuration loaded: {config.name} v{config.version}")
    logger.info(f"🤖 Using model: {config.openai.model}")
    logger.info(f"📊 RAGAS thresholds: Faithfulness={config.ragas.faithfulness_threshold}, "
                f"Precision={config.ragas.context_precision_threshold}")
    
    try:
        # Initialize components during startup
        logger.info("🔧 Initializing system components...")
        
        # Test OpenAI connection
        import openai
        openai.api_key = config.openai.api_key
        
        # Test vector store connection
        from app.core.vector_store import MedicalVectorStore
        vector_store = MedicalVectorStore()
        health_check = vector_store.health_check()
        
        if health_check.get("status") == "healthy":
            logger.info("✅ Vector store initialized successfully")
        else:
            logger.warning("⚠️ Vector store health check failed")
        
        logger.info("🎯 Medical AI Assistant started successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Medical AI Assistant")


# Create FastAPI application
app = FastAPI(
    title="Medical AI Assistant",
    description="""
    ## Medical Knowledge Assistant RAG Pipeline
    
    A production-ready Medical Knowledge Assistant RAG (Retrieval-Augmented Generation) pipeline 
    for healthcare professionals to query medical literature, drug interactions, and clinical guidelines 
    using OpenAI API with comprehensive RAGAS evaluation framework.
    
    ### Features
    
    - **🔍 Advanced RAG Pipeline**: Medical document ingestion → Vector DB → Retrieval → OpenAI generation
    - **📊 RAGAS Evaluation**: Context Precision, Context Recall, Faithfulness, Answer Relevancy
    - **🛡️ Safety System**: RAGAS-validated response filtering and medical disclaimers
    - **⚡ Performance**: Response latency p95 < 3 seconds
    - **🏥 Medical Focus**: Specialized for healthcare professionals
    
    ### RAGAS Metrics
    
    - **Faithfulness** (Target: >0.90): Ensures generated answers are grounded in context
    - **Context Precision** (Target: >0.85): Measures relevance of retrieved contexts
    - **Context Recall** (Target: >0.80): Measures completeness of retrieved information
    - **Answer Relevancy** (Target: >0.85): Measures how well answers address queries
    
    ### Safety Features
    
    - RAGAS-validated responses with quality thresholds
    - Automatic medical disclaimers
    - Harmful content detection and filtering
    - Professional medical advice recommendations
    
    ### Usage
    
    1. **Query Medical Knowledge**: Use `/query` endpoint for medical questions
    2. **Upload Documents**: Use `/documents/upload` to add medical literature
    3. **Batch Processing**: Use `/query/batch` for multiple queries
    4. **Evaluation**: Use `/evaluate` for RAGAS assessment
    5. **Monitoring**: Use `/health` and `/metrics` for system monitoring
    
    **⚠️ Important**: This system is designed for healthcare professionals and should not 
    replace professional medical judgment. Always consult with qualified healthcare providers 
    for medical decisions.
    """,
    version=config.version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Custom exception handler
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with logging."""
    logger.error(f"HTTP {exc.status_code} error on {request.url}: {exc.detail}")
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unexpected error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "details": str(exc) if config.debug else None
        }
    )


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url} - {response.status_code} - {process_time:.3f}s")
    
    return response


# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Medical AI Assistant"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": config.name,
        "version": config.version,
        "description": "Medical AI Assistant RAG Pipeline with RAGAS Evaluation",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
        "metrics": "/api/v1/metrics",
        "status": "operational"
    }


# Health check endpoint (also available at root level)
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": config.version
    }


# Metrics endpoint (also available at root level)
@app.get("/metrics")
async def metrics():
    """Simple metrics endpoint."""
    return {
        "application": config.name,
        "version": config.version,
        "model": config.openai.model,
        "ragas_enabled": config.ragas.evaluation_enabled,
        "safety_enabled": config.safety.filtering_enabled
    }


# Development server runner
if __name__ == "__main__":
    
    logger.info("🚀 Starting Medical AI Assistant in development mode")
    
    # Use standard asyncio event loop to avoid uvloop conflict with nest_asyncio
    uvicorn.run(
        "app.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_level=config.log_level.lower(),
        access_log=True,
        loop="asyncio"  # Use standard asyncio instead of uvloop
    ) 