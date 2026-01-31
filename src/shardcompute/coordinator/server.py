"""Coordinator server for cluster management."""

import asyncio
import logging
import signal
import argparse
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from aiohttp import web

from shardcompute.coordinator.registry import WorkerRegistry, RegistryConfig
from shardcompute.coordinator.health import HealthMonitor, HealthConfig
from shardcompute.coordinator.metrics import MetricsAggregator
from shardcompute.protocol.messages import WorkerInfo, InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)


class CoordinatorServer:
    """
    Coordinator server for managing the ShardCompute cluster.
    
    Responsibilities:
    - Accept worker registrations
    - Share peer list for mesh formation
    - Monitor worker health
    - Accept inference requests from clients
    - Dispatch requests to workers
    - Aggregate metrics
    
    The coordinator is NOT in the data path during inference.
    It only manages cluster formation and accepts requests.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        config_path: Optional[str] = None,
    ):
        """
        Initialize CoordinatorServer.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            config_path: Path to config file
        """
        self.host = host
        self.port = port
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup components
        world_size = self.config.get("parallelism", {}).get("tensor_parallel_size", 2)
        heartbeat_timeout = self.config.get("coordinator", {}).get("heartbeat_timeout_seconds", 15)
        
        registry_config = RegistryConfig(expected_workers=world_size)
        self.registry = WorkerRegistry(registry_config)
        
        health_config = HealthConfig(heartbeat_timeout=heartbeat_timeout)
        self.health_monitor = HealthMonitor(
            self.registry,
            health_config,
            on_failure=self._handle_worker_failure,
        )
        
        self.metrics = MetricsAggregator()
        
        # Inference queue
        self._pending_requests: asyncio.Queue = asyncio.Queue()
        self._completed_requests: Dict[str, InferenceResponse] = {}
        self._request_events: Dict[str, asyncio.Event] = {}
        
        # HTTP app
        self.app = web.Application()
        self._setup_routes()
        
        # State
        self._running = False
        self._runner: Optional[web.AppRunner] = None
        
        logger.info(f"Coordinator initialized: {host}:{port}, world_size={world_size}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _setup_routes(self):
        """Setup HTTP routes."""
        # Worker management
        self.app.router.add_post("/api/workers/register", self._handle_register)
        self.app.router.add_get("/api/workers/list", self._handle_list_workers)
        self.app.router.add_post("/api/heartbeat", self._handle_heartbeat)
        
        # Inference
        self.app.router.add_post("/api/inference", self._handle_inference)
        self.app.router.add_get("/api/inference/poll", self._handle_poll)
        self.app.router.add_post("/api/inference/response", self._handle_response)
        self.app.router.add_get("/api/inference/{request_id}", self._handle_get_result)
        
        # Metrics and status
        self.app.router.add_get("/api/status", self._handle_status)
        self.app.router.add_get("/api/metrics", self._handle_metrics)
        self.app.router.add_get("/api/health", self._handle_health)
        
        # Root
        self.app.router.add_get("/", self._handle_root)
    
    async def start(self):
        """Start the coordinator server."""
        logger.info(f"Starting coordinator server on {self.host}:{self.port}")
        
        self._running = True
        
        # Start health monitor
        await self.health_monitor.start()
        
        # Start HTTP server
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Coordinator server started on http://{self.host}:{self.port}")
        
        # Wait for cluster
        logger.info("Waiting for workers to register...")
        ready = await self.registry.wait_for_cluster()
        
        if ready:
            logger.info("Cluster is ready for inference")
        else:
            logger.warning("Cluster formation timed out")
    
    async def stop(self):
        """Stop the coordinator server."""
        logger.info("Stopping coordinator server")
        
        self._running = False
        
        await self.health_monitor.stop()
        
        if self._runner:
            await self._runner.cleanup()
        
        logger.info("Coordinator server stopped")
    
    def _handle_worker_failure(self, rank: int):
        """Handle worker failure."""
        logger.error(f"Worker {rank} failed - cluster is degraded")
        # In a production system, we would:
        # - Reject new inference requests
        # - Attempt worker recovery
        # - Notify clients
    
    # === HTTP Handlers ===
    
    async def _handle_root(self, request: web.Request) -> web.Response:
        """Handle root endpoint."""
        return web.json_response({
            "service": "ShardCompute Coordinator",
            "status": "running" if self._running else "stopped",
            "cluster_ready": self.registry.is_cluster_ready,
            "workers": self.registry.worker_count,
        })
    
    async def _handle_register(self, request: web.Request) -> web.Response:
        """Handle worker registration."""
        try:
            data = await request.json()
            
            info = WorkerInfo(
                rank=data["rank"],
                host=data["host"],
                port=data["port"],
                collective_port=data.get("collective_port", data["port"]),
                device_info=data.get("device_info", {}),
            )
            
            success = await self.registry.register(info)
            
            if success:
                return web.json_response({
                    "status": "registered",
                    "rank": info.rank,
                    "world_size": self.registry.config.expected_workers,
                })
            else:
                return web.json_response(
                    {"error": "Registration failed"},
                    status=400,
                )
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_list_workers(self, request: web.Request) -> web.Response:
        """Handle worker list request."""
        return web.json_response({
            "workers": self.registry.get_workers_dict(),
            "cluster_ready": self.registry.is_cluster_ready,
        })
    
    async def _handle_heartbeat(self, request: web.Request) -> web.Response:
        """Handle worker heartbeat."""
        try:
            data = await request.json()
            rank = data["rank"]
            status = data.get("status", "unknown")
            metrics = data.get("metrics", {})
            
            await self.registry.update_heartbeat(rank, status)
            self.health_monitor.record_heartbeat(rank)
            
            if metrics:
                self.metrics.record_worker_metrics(rank, metrics)
            
            return web.json_response({"status": "ok"})
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_inference(self, request: web.Request) -> web.Response:
        """Handle inference request from client."""
        try:
            if not self.registry.is_cluster_ready:
                return web.json_response(
                    {"error": "Cluster not ready"},
                    status=503,
                )
            
            if not self.health_monitor.is_cluster_healthy():
                return web.json_response(
                    {"error": "Cluster unhealthy"},
                    status=503,
                )
            
            data = await request.json()
            
            # Create request
            request_id = str(uuid.uuid4())
            inference_request = InferenceRequest(
                request_id=request_id,
                input_ids=data["input_ids"],
                max_new_tokens=data.get("max_new_tokens", 100),
                temperature=data.get("temperature", 0.7),
                top_p=data.get("top_p", 0.9),
                stop_tokens=data.get("stop_tokens", [2]),  # Default to EOS token
            )
            
            # Create event for response
            self._request_events[request_id] = asyncio.Event()
            
            # Queue request
            await self._pending_requests.put(inference_request)
            
            logger.info(f"Inference request queued: {request_id}")
            
            # Wait for response (with timeout)
            timeout = data.get("timeout", 300)
            try:
                await asyncio.wait_for(
                    self._request_events[request_id].wait(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return web.json_response(
                    {"error": "Inference timeout"},
                    status=504,
                )
            
            # Get response
            response = self._completed_requests.pop(request_id, None)
            del self._request_events[request_id]
            
            if response is None:
                return web.json_response(
                    {"error": "Response not found"},
                    status=500,
                )
            
            if response.error:
                return web.json_response(
                    {"error": response.error},
                    status=500,
                )
            
            return web.json_response({
                "request_id": response.request_id,
                "output_ids": response.output_ids,
                "timing": response.timing,
            })
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_poll(self, request: web.Request) -> web.Response:
        """Handle worker polling for inference requests."""
        try:
            # Wait for a request with timeout
            try:
                inference_request = await asyncio.wait_for(
                    self._pending_requests.get(),
                    timeout=25.0,
                )
                return web.json_response({
                    "request": inference_request.to_dict(),
                })
            except asyncio.TimeoutError:
                return web.Response(status=204)  # No content
                
        except Exception as e:
            logger.error(f"Poll error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_response(self, request: web.Request) -> web.Response:
        """Handle inference response from worker."""
        try:
            data = await request.json()
            response = InferenceResponse.from_dict(data)
            
            request_id = response.request_id
            
            # Record metrics
            if response.timing and not response.error:
                self.metrics.record_inference(
                    request_id=request_id,
                    timing=response.timing,
                    input_tokens=len(response.output_ids) // 2,  # Approximate
                    output_tokens=len(response.output_ids),
                )
            
            # Store response and signal
            self._completed_requests[request_id] = response
            
            if request_id in self._request_events:
                self._request_events[request_id].set()
            
            return web.json_response({"status": "ok"})
            
        except Exception as e:
            logger.error(f"Response handling error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_get_result(self, request: web.Request) -> web.Response:
        """Handle request to get inference result."""
        request_id = request.match_info["request_id"]
        
        if request_id in self._completed_requests:
            response = self._completed_requests[request_id]
            return web.json_response({
                "status": "completed",
                "output_ids": response.output_ids,
                "timing": response.timing,
                "error": response.error,
            })
        elif request_id in self._request_events:
            return web.json_response({"status": "pending"})
        else:
            return web.json_response(
                {"error": "Request not found"},
                status=404,
            )
    
    async def _handle_status(self, request: web.Request) -> web.Response:
        """Handle status request."""
        return web.json_response({
            "status": "running" if self._running else "stopped",
            "cluster_ready": self.registry.is_cluster_ready,
            "cluster_healthy": self.health_monitor.is_cluster_healthy(),
            "workers": self.registry.worker_count,
            "expected_workers": self.registry.config.expected_workers,
        })
    
    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """Handle metrics request."""
        return web.json_response(self.metrics.get_full_report())
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle health check request."""
        health = self.health_monitor.get_health_status()
        
        status = 200 if self.health_monitor.is_cluster_healthy() else 503
        
        return web.json_response(health, status=status)


def main():
    """Entry point for coordinator server."""
    parser = argparse.ArgumentParser(description="ShardCompute Coordinator")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Create server
    server = CoordinatorServer(
        host=args.host,
        port=args.port,
        config_path=args.config,
    )
    
    # Setup event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def shutdown():
        await server.stop()
        loop.stop()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
    
    try:
        loop.run_until_complete(server.start())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(server.stop())
        loop.close()


if __name__ == "__main__":
    main()
