"""Health monitoring for workers."""

import asyncio
import logging
import time
from typing import Dict, Optional, Callable, Set
from dataclasses import dataclass

from shardcompute.coordinator.registry import WorkerRegistry

logger = logging.getLogger(__name__)


@dataclass
class HealthConfig:
    """Configuration for health monitoring.

    Default timing matches COMMUNICATION_OUTLINE.md:
    - Workers marked offline after 60s without heartbeat
    - Background cleanup loop runs every 30s
    """

    heartbeat_timeout: float = 60.0  # Changed from 15.0 to match COMMUNICATION_OUTLINE
    check_interval: float = 30.0  # Changed from 5.0 to match COMMUNICATION_OUTLINE
    failure_threshold: int = 2  # 2 missed checks * 30s = 60s total


class HealthMonitor:
    """
    Monitors health of worker nodes.
    
    Responsibilities:
    - Track heartbeat timestamps
    - Detect unhealthy/failed workers
    - Trigger callbacks on failure
    """
    
    def __init__(
        self,
        registry: WorkerRegistry,
        config: HealthConfig,
        on_failure: Optional[Callable[[int], None]] = None,
    ):
        """
        Initialize HealthMonitor.
        
        Args:
            registry: Worker registry
            config: Health configuration
            on_failure: Callback when worker fails (receives rank)
        """
        self.registry = registry
        self.config = config
        self.on_failure = on_failure
        
        # Track missed heartbeats
        self._missed_heartbeats: Dict[int, int] = {}
        
        # Currently unhealthy workers
        self._unhealthy: Set[int] = set()
        
        # Monitor task
        self._task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")
    
    async def stop(self):
        """Stop health monitoring."""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            await asyncio.sleep(self.config.check_interval)
    
    async def _check_health(self):
        """Check health of all workers."""
        current_time = time.time()
        
        for worker in self.registry.get_all_workers():
            rank = worker.rank
            time_since_heartbeat = current_time - worker.last_heartbeat
            
            if time_since_heartbeat > self.config.heartbeat_timeout:
                # Increment missed heartbeats
                self._missed_heartbeats[rank] = self._missed_heartbeats.get(rank, 0) + 1
                missed = self._missed_heartbeats[rank]
                
                if missed >= self.config.failure_threshold:
                    if rank not in self._unhealthy:
                        self._unhealthy.add(rank)
                        logger.error(
                            f"Worker {rank} failed: {missed} missed heartbeats, "
                            f"last seen {time_since_heartbeat:.1f}s ago"
                        )
                        self.registry.mark_worker_failed(rank)
                        
                        if self.on_failure:
                            try:
                                self.on_failure(rank)
                            except Exception as e:
                                logger.error(f"Failure callback error: {e}")
                else:
                    logger.warning(
                        f"Worker {rank} unhealthy: {missed} missed heartbeats"
                    )
            else:
                # Worker is healthy, reset counter
                if rank in self._missed_heartbeats:
                    del self._missed_heartbeats[rank]
                if rank in self._unhealthy:
                    self._unhealthy.remove(rank)
                    logger.info(f"Worker {rank} recovered")
    
    def record_heartbeat(self, rank: int):
        """
        Record a heartbeat from a worker.
        
        Called by the server when heartbeat is received.
        """
        if rank in self._missed_heartbeats:
            del self._missed_heartbeats[rank]
    
    def get_health_status(self) -> Dict:
        """Get current health status."""
        workers = self.registry.get_all_workers()
        current_time = time.time()
        
        statuses = {}
        for worker in workers:
            time_since = current_time - worker.last_heartbeat
            statuses[worker.rank] = {
                "status": worker.status,
                "last_heartbeat_ago": time_since,
                "healthy": worker.rank not in self._unhealthy,
                "missed_heartbeats": self._missed_heartbeats.get(worker.rank, 0),
            }
        
        return {
            "workers": statuses,
            "healthy_count": len(workers) - len(self._unhealthy),
            "total_count": len(workers),
        }
    
    def is_cluster_healthy(self) -> bool:
        """Check if all workers are healthy."""
        return len(self._unhealthy) == 0 and self.registry.is_cluster_ready
