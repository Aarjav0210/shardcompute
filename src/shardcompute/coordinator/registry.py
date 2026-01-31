"""Worker registry for tracking registered workers."""

import asyncio
import logging
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from shardcompute.protocol.messages import WorkerInfo

logger = logging.getLogger(__name__)


@dataclass
class RegistryConfig:
    """Configuration for worker registry."""
    
    expected_workers: int = 2
    registration_timeout: float = 300.0  # 5 minutes


class WorkerRegistry:
    """
    Registry for tracking worker nodes.
    
    Responsibilities:
    - Accept worker registrations
    - Track worker status
    - Notify when cluster is ready
    - Provide worker list to clients
    """
    
    def __init__(self, config: RegistryConfig):
        """
        Initialize WorkerRegistry.
        
        Args:
            config: Registry configuration
        """
        self.config = config
        
        # Worker storage
        self._workers: Dict[int, WorkerInfo] = {}
        self._lock = asyncio.Lock()
        
        # Cluster ready event
        self._cluster_ready = asyncio.Event()
        
        logger.info(f"WorkerRegistry initialized, expecting {config.expected_workers} workers")
    
    @property
    def worker_count(self) -> int:
        """Number of registered workers."""
        return len(self._workers)
    
    @property
    def is_cluster_ready(self) -> bool:
        """Check if all expected workers are registered."""
        return len(self._workers) >= self.config.expected_workers
    
    async def register(self, info: WorkerInfo) -> bool:
        """
        Register a worker.
        
        Args:
            info: Worker information
            
        Returns:
            True if registration successful
        """
        async with self._lock:
            # Check for duplicate rank
            if info.rank in self._workers:
                existing = self._workers[info.rank]
                logger.warning(
                    f"Rank {info.rank} already registered from {existing.host}:{existing.port}, "
                    f"updating with {info.host}:{info.port}"
                )
            
            # Validate rank
            if info.rank < 0 or info.rank >= self.config.expected_workers:
                logger.error(f"Invalid rank {info.rank}, expected 0-{self.config.expected_workers - 1}")
                return False
            
            # Register worker
            info.status = "registered"
            info.last_heartbeat = time.time()
            self._workers[info.rank] = info
            
            logger.info(
                f"Worker registered: rank {info.rank} at {info.host}:{info.port}"
            )
            
            # Check if cluster is ready
            if self.is_cluster_ready:
                self._cluster_ready.set()
                logger.info(
                    f"Cluster ready with {len(self._workers)} workers"
                )
            
            return True
    
    async def deregister(self, rank: int) -> bool:
        """
        Deregister a worker.
        
        Args:
            rank: Worker rank to deregister
            
        Returns:
            True if deregistration successful
        """
        async with self._lock:
            if rank not in self._workers:
                logger.warning(f"Cannot deregister unknown rank {rank}")
                return False
            
            del self._workers[rank]
            
            # Reset cluster ready if we lost a worker
            if not self.is_cluster_ready:
                self._cluster_ready.clear()
            
            logger.info(f"Worker deregistered: rank {rank}")
            return True
    
    async def update_heartbeat(self, rank: int, status: str) -> bool:
        """
        Update worker heartbeat.
        
        Args:
            rank: Worker rank
            status: Worker status
            
        Returns:
            True if update successful
        """
        async with self._lock:
            if rank not in self._workers:
                logger.warning(f"Heartbeat from unknown rank {rank}")
                return False
            
            self._workers[rank].last_heartbeat = time.time()
            self._workers[rank].status = status
            return True
    
    async def wait_for_cluster(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for cluster to be ready.
        
        Args:
            timeout: Maximum time to wait (None for config default)
            
        Returns:
            True if cluster is ready, False if timeout
        """
        if timeout is None:
            timeout = self.config.registration_timeout
        
        try:
            await asyncio.wait_for(
                self._cluster_ready.wait(),
                timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout waiting for cluster: "
                f"{len(self._workers)}/{self.config.expected_workers} workers registered"
            )
            return False
    
    def get_worker(self, rank: int) -> Optional[WorkerInfo]:
        """Get worker info by rank."""
        return self._workers.get(rank)
    
    def get_all_workers(self) -> List[WorkerInfo]:
        """Get all registered workers."""
        return list(self._workers.values())
    
    def get_workers_dict(self) -> List[Dict]:
        """Get all workers as dictionaries."""
        return [w.to_dict() for w in self._workers.values()]
    
    def mark_worker_failed(self, rank: int):
        """Mark a worker as failed."""
        if rank in self._workers:
            self._workers[rank].status = "failed"
            logger.warning(f"Worker {rank} marked as failed")
