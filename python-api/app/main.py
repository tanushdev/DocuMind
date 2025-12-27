"""
DocuMind API - Main Application Entry Point

Production-grade AI document intelligence system using RAG.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import documents, query, health
from app.services import RedisClientFactory


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Handles startup and shutdown events:
    - Startup: Initialize services, warm up models
    - Shutdown: Close connections, cleanup
    """
    # Startup
    settings = get_settings()
    print(f"🚀 Starting {settings.app_name}")
    print(f"📡 Vector Service: {settings.vector_service_url}")
    print(f"🔴 Redis: {settings.redis_url}")
    print(f"🤖 LLM: Groq / Gemini / Perplexity (auto-detected)")
    
    # Test Redis connection
    try:
        redis = RedisClientFactory.get_client()
        if await redis.ping():
            print("✅ Redis connected")
        else:
            print("⚠️  Redis unavailable - caching disabled")
    except Exception as e:
        print(f"⚠️  Redis connection failed: {e}")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    await RedisClientFactory.close()
    print("👋 Goodbye!")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI app instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="""
## DocuMind API

Production-grade AI document intelligence system using Retrieval-Augmented Generation (RAG).

### Features
- 📄 **Document Upload**: Upload PDF and TXT files for processing
- 🔍 **Semantic Search**: Find relevant content using vector similarity
- 🧠 **AI Answers**: Get intelligent answers with source citations
- ⚡ **Performance**: Caching, async processing, and latency tracking

### Architecture
- **Python (FastAPI)**: API orchestration, ML models
- **Go**: Custom vector search service with HNSW
- **Redis**: Caching and task queue
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routers
    app.include_router(health.router)
    app.include_router(documents.router, prefix="/api")
    app.include_router(query.router, prefix="/api")
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
