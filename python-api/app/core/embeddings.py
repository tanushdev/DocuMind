"""
Embedding generation module for DocuMind.

This module provides embedding generation using sentence-transformers
with Redis caching for performance optimization.
"""
import hashlib
import json
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer
import numpy as np

from app.config import get_settings


class EmbeddingService:
    """
    Embedding generation service with caching.
    
    Uses sentence-transformers for generating dense vector representations
    of text. Embeddings are cached in Redis to avoid redundant computation.
    
    Design decisions:
    - Using all-MiniLM-L6-v2 (384 dims) for good balance of quality/speed
    - Batch processing for efficiency
    - Async wrapper around sync model for FastAPI compatibility
    - Redis caching with content hashing
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        redis_client=None,
        cache_ttl: int = 86400
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model
            redis_client: Redis client for caching (optional)
            cache_ttl: Cache time-to-live in seconds
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.cache_ttl = cache_ttl
        self.redis = redis_client
        
        # Load model (this is CPU-intensive, done once)
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimensions = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Dimensions: {self.dimensions}")
        
        # Thread pool for async execution
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key from text content."""
        # Use SHA256 hash of the text
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        return f"emb:{self.model_name}:{text_hash}"
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Uses Redis cache for previously computed embeddings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Check cache for each text
        cache_hits = {}
        texts_to_embed = []
        text_indices = []
        
        if self.redis:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                try:
                    cached = await self.redis.get(cache_key)
                    if cached:
                        cache_hits[i] = json.loads(cached)
                    else:
                        texts_to_embed.append(text)
                        text_indices.append(i)
                except Exception:
                    texts_to_embed.append(text)
                    text_indices.append(i)
        else:
            texts_to_embed = texts
            text_indices = list(range(len(texts)))
        
        # Generate embeddings for cache misses
        new_embeddings = {}
        if texts_to_embed:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self._executor,
                self._embed_sync,
                texts_to_embed
            )
            
            # Cache new embeddings
            for idx, text in zip(text_indices, texts_to_embed):
                emb = embeddings[text_indices.index(idx) if idx in text_indices else 0]
                new_embeddings[idx] = emb
                
                if self.redis:
                    cache_key = self._get_cache_key(text)
                    try:
                        await self.redis.setex(
                            cache_key,
                            self.cache_ttl,
                            json.dumps(emb)
                        )
                    except Exception:
                        pass  # Cache failures are non-critical
        
        # Combine cached and new embeddings in original order
        result = []
        for i in range(len(texts)):
            if i in cache_hits:
                result.append(cache_hits[i])
            elif i in new_embeddings:
                result.append(new_embeddings[i])
            else:
                # Fallback: embed individually
                emb = self._embed_sync([texts[i]])[0]
                result.append(emb)
        
        return result
    
    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronous embedding generation.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors as Python lists
        """
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=False
        )
        
        # Convert to Python lists for JSON serialization
        return embeddings.tolist()
    
    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.embed([text])
        return embeddings[0] if embeddings else []
    
    def get_dimensions(self) -> int:
        """Return the embedding dimensionality."""
        return self.dimensions


class EmbeddingServiceFactory:
    """
    Factory for creating embedding service instances.
    
    Ensures that the embedding model is loaded only once
    and shared across the application.
    """
    
    _instance: Optional[EmbeddingService] = None
    
    @classmethod
    def get_instance(cls, redis_client=None) -> EmbeddingService:
        """Get or create the singleton embedding service."""
        if cls._instance is None:
            cls._instance = EmbeddingService(redis_client=redis_client)
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton (useful for testing)."""
        cls._instance = None
