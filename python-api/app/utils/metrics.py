"""
Performance metrics and logging utilities for DocuMind.

This module provides timing context managers and metrics
collection for monitoring system performance.
"""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import statistics

from app.services.redis_client import RedisClientFactory


@dataclass
class LatencyMetrics:
    """Container for latency statistics."""
    stage: str
    count: int
    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float


class MetricsCollector:
    """
    Collects and reports performance metrics.
    
    Provides:
    - Latency tracking per stage
    - Cache hit/miss ratios
    - Request counting
    - Percentile calculations (p50, p95, p99)
    """
    
    def __init__(self):
        """Initialize the metrics collector."""
        self._local_latencies: Dict[str, List[float]] = {}
        self._cache = None
    
    async def _get_cache(self):
        """Get the cache service lazily."""
        if self._cache is None:
            self._cache = RedisClientFactory.get_cache_service()
        return self._cache
    
    @contextmanager
    def measure(self, stage: str):
        """
        Context manager for measuring latency.
        
        Usage:
            with metrics.measure("embedding"):
                result = await embed(text)
        
        Args:
            stage: Name of the stage being measured
            
        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            self._record_local(stage, latency_ms)
    
    def _record_local(self, stage: str, latency_ms: float):
        """Record latency in local memory."""
        if stage not in self._local_latencies:
            self._local_latencies[stage] = []
        self._local_latencies[stage].append(latency_ms)
        
        # Keep only last 1000 measurements
        if len(self._local_latencies[stage]) > 1000:
            self._local_latencies[stage] = self._local_latencies[stage][-1000:]
    
    async def record_latency(self, stage: str, latency_ms: float):
        """
        Record a latency measurement (async version).
        
        Args:
            stage: Name of the stage
            latency_ms: Latency in milliseconds
        """
        self._record_local(stage, latency_ms)
        
        try:
            cache = await self._get_cache()
            await cache.record_latency(stage, latency_ms)
        except Exception:
            pass  # Non-critical
    
    def get_percentiles(self, stage: str) -> Optional[LatencyMetrics]:
        """
        Calculate latency percentiles for a stage.
        
        Args:
            stage: Name of the stage
            
        Returns:
            LatencyMetrics or None if no data
        """
        if stage not in self._local_latencies or not self._local_latencies[stage]:
            return None
        
        values = sorted(self._local_latencies[stage])
        n = len(values)
        
        return LatencyMetrics(
            stage=stage,
            count=n,
            p50=values[n // 2],
            p95=values[int(n * 0.95)] if n >= 20 else values[-1],
            p99=values[int(n * 0.99)] if n >= 100 else values[-1],
            mean=statistics.mean(values),
            min=min(values),
            max=max(values)
        )
    
    def get_all_metrics(self) -> Dict[str, LatencyMetrics]:
        """Get metrics for all stages."""
        result = {}
        for stage in self._local_latencies:
            metrics = self.get_percentiles(stage)
            if metrics:
                result[stage] = metrics
        return result
    
    async def record_cache_hit(self):
        """Record a cache hit."""
        try:
            cache = await self._get_cache()
            await cache.increment_counter("cache_hits")
        except Exception:
            pass
    
    async def record_cache_miss(self):
        """Record a cache miss."""
        try:
            cache = await self._get_cache()
            await cache.increment_counter("cache_misses")
        except Exception:
            pass
    
    async def get_cache_hit_ratio(self) -> float:
        """Calculate the cache hit ratio."""
        try:
            cache = await self._get_cache()
            hits = await cache.get_counter("cache_hits")
            misses = await cache.get_counter("cache_misses")
            total = hits + misses
            return hits / total if total > 0 else 0.0
        except Exception:
            return 0.0
    
    def reset(self):
        """Reset all local metrics."""
        self._local_latencies.clear()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


@dataclass
class RequestTimer:
    """
    Timer for tracking latency across multiple stages of a request.
    
    Usage:
        timer = RequestTimer()
        with timer.stage("embedding"):
            embeddings = await embed(texts)
        with timer.stage("search"):
            results = await search(embedding)
        print(timer.summary())
    """
    stages: Dict[str, float] = field(default_factory=dict)
    start_time: float = field(default_factory=time.perf_counter)
    
    @contextmanager
    def stage(self, name: str):
        """Time a stage of the request."""
        stage_start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = (time.perf_counter() - stage_start) * 1000
    
    @property
    def total_ms(self) -> float:
        """Total time elapsed since timer creation."""
        return (time.perf_counter() - self.start_time) * 1000
    
    def summary(self) -> dict:
        """Get a summary of all stages."""
        return {
            **self.stages,
            "total_ms": self.total_ms
        }
    
    async def record_all(self, collector: MetricsCollector):
        """Record all stages to the metrics collector."""
        for stage, latency_ms in self.stages.items():
            await collector.record_latency(stage, latency_ms)
