"""Heartbeat client for worker health reporting."""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class HeartbeatConfig:
    """Configuration for heartbeat client.

    Default timing matches COMMUNICATION_OUTLINE.md:
    - 30s heartbeat interval
    - 2 consecutive failures allowed (60s total timeout)
    """

    interval_seconds: float = 30.0  # Changed from 5.0 to match COMMUNICATION_OUTLINE
    timeout_seconds: float = 10.0
    max_failures: int = 2  # 2 failures * 30s = 60s timeout


class HeartbeatClient:
    """
    Client for sending heartbeats to the coordinator.
    
    Responsibilities:
    - Send periodic heartbeats to coordinator
    - Report worker status and metrics
    - Handle coordinator connection failures
    - Trigger callbacks on status changes
    """
    
    def __init__(
        self,
        rank: int,
        coordinator_url: str,
        config: HeartbeatConfig,
        metrics_callback: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        """
        Initialize HeartbeatClient.
        
        Args:
            rank: This worker's rank
            coordinator_url: Base URL of coordinator
            config: Heartbeat configuration
            metrics_callback: Optional callback to collect metrics for heartbeat
        """
        self.rank = rank
        self.coordinator_url = coordinator_url.rstrip('/')
        self.config = config
        self.metrics_callback = metrics_callback
        
        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._consecutive_failures = 0
        self._last_success: Optional[float] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Status
        self.status = "initializing"
    
    async def start(self):
        """Start the heartbeat loop."""
        if self._running:
            logger.warning("Heartbeat already running")
            return
        
        self._running = True
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._heartbeat_loop())
        
        logger.info(f"Rank {self.rank} heartbeat client started")
    
    async def stop(self):
        """Stop the heartbeat loop."""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        if self._session:
            await self._session.close()
        
        logger.info(f"Rank {self.rank} heartbeat client stopped")
    
    async def _heartbeat_loop(self):
        """Main heartbeat loop."""
        while self._running:
            try:
                await self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                self._consecutive_failures += 1
                
                if self._consecutive_failures >= self.config.max_failures:
                    logger.error(
                        f"Rank {self.rank} lost connection to coordinator "
                        f"after {self._consecutive_failures} failures"
                    )
            
            await asyncio.sleep(self.config.interval_seconds)
    
    async def _send_heartbeat(self):
        """Send a single heartbeat.

        Uses the endpoint structure from COMMUNICATION_OUTLINE.md.
        """
        url = f"{self.coordinator_url}/workers/heartbeat"

        # Collect metrics if callback provided
        metrics = {}
        if self.metrics_callback:
            try:
                metrics = self.metrics_callback()
            except Exception as e:
                logger.warning(f"Failed to collect metrics: {e}")

        # Payload matches COMMUNICATION_OUTLINE.md WorkerHeartbeat model
        payload = {
            "worker_id": f"worker-{self.rank}",
            "rank": self.rank,
            "status": self.status,
            "tokens_served_since_last": metrics.get("tokens_served", 0),
            "metrics": metrics,
        }
        
        try:
            async with self._session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
            ) as response:
                if response.status == 200:
                    self._consecutive_failures = 0
                    self._last_success = time.time()
                else:
                    logger.warning(
                        f"Heartbeat failed with status {response.status}"
                    )
                    self._consecutive_failures += 1
                    
        except aiohttp.ClientError as e:
            logger.warning(f"Heartbeat network error: {e}")
            self._consecutive_failures += 1
    
    def set_status(self, status: str):
        """Update worker status for next heartbeat."""
        self.status = status
        logger.debug(f"Rank {self.rank} status set to: {status}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get heartbeat statistics."""
        return {
            "running": self._running,
            "status": self.status,
            "consecutive_failures": self._consecutive_failures,
            "last_success": self._last_success,
        }
