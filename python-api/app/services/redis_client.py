"""
Redis client for DocuMind.

This module provides async Redis operations for caching
embeddings, query results, and task status.
"""
from typing import Optional, Any
import json
import redis.asyncio as redis

from app.config import get_settings


class RedisClient:
    """
    Async Redis client wrapper.
    
    Provides high-level methods for common caching operations
    with automatic serialization/deserialization.
    """
    
    def __init__(self, url: Optional[str] = None):
        """
        Initialize the Redis client.
        
        Args:
            url: Redis connection URL
        """
        settings = get_settings()
        self.url = url or settings.redis_url
        self._client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Establish connection to Redis."""
        if self._client is None:
            self._client = redis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=True
            )
    
    async def close(self):
        """Close the Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
    
    async def get(self, key: str) -> Optional[str]:
        """Get a value by key."""
        await self.connect()
        return await self._client.get(key)
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None):
        """Set a value with optional TTL."""
        await self.connect()
        if ttl:
            await self._client.setex(key, ttl, value)
        else:
            await self._client.set(key, value)
    
    async def setex(self, key: str, ttl: int, value: str):
        """Set a value with TTL."""
        await self.connect()
        await self._client.setex(key, ttl, value)
    
    async def delete(self, key: str):
        """Delete a key."""
        await self.connect()
        await self._client.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        await self.connect()
        return await self._client.exists(key) > 0
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get a JSON-serialized value."""
        value = await self.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set_json(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a JSON-serialized value."""
        json_value = json.dumps(value)
        await self.set(key, json_value, ttl)
    
    # Hash operations for structured data
    async def hset(self, name: str, key: str, value: str):
        """Set a hash field."""
        await self.connect()
        await self._client.hset(name, key, value)
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get a hash field."""
        await self.connect()
        return await self._client.hget(name, key)
    
    async def hgetall(self, name: str) -> dict:
        """Get all fields in a hash."""
        await self.connect()
        return await self._client.hgetall(name)
    
    async def hmset(self, name: str, mapping: dict):
        """Set multiple hash fields."""
        await self.connect()
        await self._client.hset(name, mapping=mapping)
    
    # List operations for metrics
    async def lpush(self, key: str, value: str):
        """Push to the left of a list."""
        await self.connect()
        await self._client.lpush(key, value)
    
    async def ltrim(self, key: str, start: int, end: int):
        """Trim a list to the specified range."""
        await self.connect()
        await self._client.ltrim(key, start, end)
    
    async def lrange(self, key: str, start: int, end: int) -> list:
        """Get a range from a list."""
        await self.connect()
        return await self._client.lrange(key, start, end)
    
    async def incr(self, key: str) -> int:
        """Increment a counter."""
        await self.connect()
        return await self._client.incr(key)
    
    async def expire(self, key: str, seconds: int):
        """Set expiration on a key."""
        await self.connect()
        await self._client.expire(key, seconds)
    
    async def ping(self) -> bool:
        """Check if Redis is available."""
        try:
            await self.connect()
            await self._client.ping()
            return True
        except Exception:
            return False


class CacheService:
    """
    High-level caching service for DocuMind.
    
    Provides domain-specific caching methods for embeddings,
    query results, and task status.
    """
    
    def __init__(self, redis_client: RedisClient):
        """
        Initialize the cache service.
        
        Args:
            redis_client: The Redis client to use
        """
        self.redis = redis_client
        settings = get_settings()
        self.embedding_ttl = settings.cache_ttl_embeddings
        self.query_ttl = settings.cache_ttl_queries
    
    # Embedding cache
    async def get_embedding(self, text_hash: str) -> Optional[list]:
        """Get cached embedding by text hash."""
        key = f"emb:{text_hash}"
        return await self.redis.get_json(key)
    
    async def set_embedding(self, text_hash: str, embedding: list):
        """Cache an embedding."""
        key = f"emb:{text_hash}"
        await self.redis.set_json(key, embedding, self.embedding_ttl)
    
    # Query result cache
    async def get_query_result(self, query_hash: str) -> Optional[dict]:
        """Get cached query result."""
        key = f"query:{query_hash}"
        return await self.redis.get_json(key)
    
    async def set_query_result(self, query_hash: str, result: dict):
        """Cache a query result."""
        key = f"query:{query_hash}"
        await self.redis.set_json(key, result, self.query_ttl)
    
    # Task status
    async def get_task_status(self, task_id: str) -> Optional[dict]:
        """Get task status."""
        key = f"task:{task_id}"
        return await self.redis.get_json(key)
    
    async def set_task_status(self, task_id: str, status: dict, ttl: int = 3600):
        """Set task status."""
        key = f"task:{task_id}"
        await self.redis.set_json(key, status, ttl)
    
    # Metrics
    async def record_latency(self, stage: str, latency_ms: float):
        """Record a latency measurement."""
        key = f"metrics:latency:{stage}"
        await self.redis.lpush(key, str(latency_ms))
        await self.redis.ltrim(key, 0, 999)  # Keep last 1000
    
    async def get_latencies(self, stage: str, count: int = 100) -> list:
        """Get recent latency measurements."""
        key = f"metrics:latency:{stage}"
        values = await self.redis.lrange(key, 0, count - 1)
        return [float(v) for v in values]
    
    async def increment_counter(self, name: str) -> int:
        """Increment a counter (e.g., cache hits/misses)."""
        key = f"counter:{name}"
        return await self.redis.incr(key)
    
    async def get_counter(self, name: str) -> int:
        """Get a counter value."""
        key = f"counter:{name}"
        value = await self.redis.get(key)
        return int(value) if value else 0


class RedisClientFactory:
    """Factory for Redis client instances."""
    
    _instance: Optional[RedisClient] = None
    _cache_service: Optional[CacheService] = None
    
    @classmethod
    def get_client(cls) -> RedisClient:
        """Get or create the singleton Redis client."""
        if cls._instance is None:
            cls._instance = RedisClient()
        return cls._instance
    
    @classmethod
    def get_cache_service(cls) -> CacheService:
        """Get or create the singleton cache service."""
        if cls._cache_service is None:
            cls._cache_service = CacheService(cls.get_client())
        return cls._cache_service
    
    @classmethod
    async def close(cls):
        """Close the Redis connection."""
        if cls._instance:
            await cls._instance.close()
            cls._instance = None
            cls._cache_service = None
