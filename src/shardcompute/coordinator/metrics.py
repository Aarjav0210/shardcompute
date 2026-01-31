"""Metrics aggregation for the cluster."""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Metrics from a single inference request."""
    
    request_id: str
    timestamp: float
    total_time_ms: float
    compute_time_ms: float
    comm_time_ms: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float


@dataclass
class ClusterMetrics:
    """Aggregated cluster metrics."""
    
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_compute_time_ms: float = 0.0
    total_comm_time_ms: float = 0.0
    
    # Rolling window for recent stats
    recent_latencies: List[float] = field(default_factory=list)
    recent_throughputs: List[float] = field(default_factory=list)
    
    @property
    def avg_latency_ms(self) -> float:
        if not self.recent_latencies:
            return 0.0
        return statistics.mean(self.recent_latencies)
    
    @property
    def p99_latency_ms(self) -> float:
        if not self.recent_latencies:
            return 0.0
        return sorted(self.recent_latencies)[int(len(self.recent_latencies) * 0.99)]
    
    @property
    def avg_throughput(self) -> float:
        if not self.recent_throughputs:
            return 0.0
        return statistics.mean(self.recent_throughputs)
    
    @property
    def compute_fraction(self) -> float:
        total = self.total_compute_time_ms + self.total_comm_time_ms
        if total == 0:
            return 0.0
        return self.total_compute_time_ms / total


class MetricsAggregator:
    """
    Aggregates metrics from workers and inference requests.
    
    Responsibilities:
    - Collect metrics from worker heartbeats
    - Track inference request metrics
    - Compute aggregate statistics
    - Maintain rolling window for recent stats
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize MetricsAggregator.
        
        Args:
            window_size: Number of recent requests to keep for statistics
        """
        self.window_size = window_size
        
        # Cluster-level metrics
        self.cluster = ClusterMetrics()
        
        # Per-worker metrics
        self.worker_metrics: Dict[int, Dict[str, Any]] = {}
        
        # Recent requests (rolling window)
        self._recent_requests: deque = deque(maxlen=window_size)
        
        # Timing
        self._start_time = time.time()
    
    def record_inference(
        self,
        request_id: str,
        timing: Dict[str, float],
        input_tokens: int,
        output_tokens: int,
    ):
        """
        Record metrics from an inference request.
        
        Args:
            request_id: Request identifier
            timing: Timing dictionary from executor
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        total_time = timing.get("total_ms", 0)
        compute_time = timing.get("compute_ms", 0)
        comm_time = timing.get("comm_ms", 0)
        
        # Calculate throughput
        tokens_per_second = 0
        if total_time > 0:
            tokens_per_second = (output_tokens / total_time) * 1000
        
        # Create metrics record
        metrics = InferenceMetrics(
            request_id=request_id,
            timestamp=time.time(),
            total_time_ms=total_time,
            compute_time_ms=compute_time,
            comm_time_ms=comm_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
        )
        
        # Update cluster metrics
        self.cluster.total_requests += 1
        self.cluster.total_tokens_generated += output_tokens
        self.cluster.total_compute_time_ms += compute_time
        self.cluster.total_comm_time_ms += comm_time
        
        # Update rolling window
        self._recent_requests.append(metrics)
        self.cluster.recent_latencies = [r.total_time_ms for r in self._recent_requests]
        self.cluster.recent_throughputs = [r.tokens_per_second for r in self._recent_requests]
        
        logger.debug(
            f"Recorded inference metrics: {request_id}, "
            f"{total_time:.1f}ms, {tokens_per_second:.1f} tok/s"
        )
    
    def record_worker_metrics(self, rank: int, metrics: Dict[str, Any]):
        """
        Record metrics from a worker heartbeat.
        
        Args:
            rank: Worker rank
            metrics: Metrics dictionary from worker
        """
        self.worker_metrics[rank] = {
            **metrics,
            "timestamp": time.time(),
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        uptime = time.time() - self._start_time
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.cluster.total_requests,
            "total_tokens_generated": self.cluster.total_tokens_generated,
            "avg_latency_ms": self.cluster.avg_latency_ms,
            "p99_latency_ms": self.cluster.p99_latency_ms,
            "avg_throughput_tokens_per_sec": self.cluster.avg_throughput,
            "compute_fraction": self.cluster.compute_fraction,
            "requests_per_second": (
                self.cluster.total_requests / uptime if uptime > 0 else 0
            ),
        }
    
    def get_worker_summary(self) -> Dict[int, Dict[str, Any]]:
        """Get summary of per-worker metrics."""
        return self.worker_metrics.copy()
    
    def get_recent_requests(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent request metrics."""
        recent = list(self._recent_requests)[-limit:]
        return [
            {
                "request_id": r.request_id,
                "timestamp": r.timestamp,
                "total_time_ms": r.total_time_ms,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "tokens_per_second": r.tokens_per_second,
            }
            for r in recent
        ]
    
    def get_full_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        return {
            "summary": self.get_summary(),
            "workers": self.get_worker_summary(),
            "recent_requests": self.get_recent_requests(),
            "timing_breakdown": {
                "total_compute_ms": self.cluster.total_compute_time_ms,
                "total_comm_ms": self.cluster.total_comm_time_ms,
                "compute_fraction": self.cluster.compute_fraction,
            },
        }
