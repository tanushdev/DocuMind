"""Health check and system status endpoints."""
from fastapi import APIRouter

from app.models import HealthResponse, MetricsResponse
from app.services import VectorServiceFactory, RedisClientFactory
from app.utils import get_metrics_collector

router = APIRouter(tags=["system"])

VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health of all services.
    
    Returns status of:
    - API server
    - Vector service (Go)
    - Redis cache
    """
    services = {
        "api": "ok"
    }
    overall_status = "ok"
    
    # Check vector service
    try:
        vector_client = VectorServiceFactory.get_instance()
        vector_health = await vector_client.health()
        services["vector_service"] = vector_health.get("status", "unknown")
        if services["vector_service"] != "ok":
            overall_status = "degraded"
    except Exception as e:
        services["vector_service"] = f"error: {str(e)}"
        overall_status = "degraded"
    
    # Check Redis
    try:
        redis_client = RedisClientFactory.get_client()
        redis_ok = await redis_client.ping()
        services["redis"] = "ok" if redis_ok else "unavailable"
        if not redis_ok:
            overall_status = "degraded"
    except Exception as e:
        services["redis"] = f"error: {str(e)}"
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=VERSION,
        services=services
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get performance metrics.
    
    Returns:
    - Latency percentiles per stage
    - Cache hit ratio
    - Vector count
    """
    metrics = get_metrics_collector()
    
    # Get all stage metrics
    all_metrics = metrics.get_all_metrics()
    stages = {}
    for stage, m in all_metrics.items():
        stages[stage] = {
            "count": m.count,
            "p50_ms": round(m.p50, 2),
            "p95_ms": round(m.p95, 2),
            "p99_ms": round(m.p99, 2),
            "mean_ms": round(m.mean, 2)
        }
    
    # Get cache hit ratio
    cache_hit_ratio = await metrics.get_cache_hit_ratio()
    
    # Get vector count
    vector_count = 0
    try:
        vector_client = VectorServiceFactory.get_instance()
        stats = await vector_client.stats()
        vector_count = stats.get("vector_count", 0)
    except Exception:
        pass
    
    return MetricsResponse(
        stages=stages,
        cache_hit_ratio=round(cache_hit_ratio, 3),
        vector_count=vector_count
    )


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "DocuMind API",
        "version": VERSION,
        "description": "Production-grade AI document intelligence system",
        "docs": "/docs",
        "health": "/health"
    }
