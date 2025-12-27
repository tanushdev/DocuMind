"""Configuration settings for DocuMind API."""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    app_name: str = "DocuMind API"
    debug: bool = False
    
    # Service URLs
    vector_service_url: str = "http://localhost:8001"
    redis_url: str = "redis://localhost:6379"
    
    # Embedding Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    
    # Chunking Settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Re-ranking Settings
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Context Settings
    max_context_tokens: int = 2048
    top_k_retrieval: int = 20  # Retrieve more for re-ranking
    top_k_final: int = 5       # Final after re-ranking
    
    # Cache Settings
    cache_ttl_embeddings: int = 86400  # 24 hours
    cache_ttl_queries: int = 3600      # 1 hour
    
    # Upload Settings
    max_file_size_mb: int = 50
    allowed_extensions: list = [".pdf", ".txt"]
    upload_dir: str = "./uploads"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
