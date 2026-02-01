"""Coordinator server for cluster management.

This coordinator follows the communication style defined in COMMUNICATION_OUTLINE.md:
- REST API without /api prefix
- DHT bootstrap with multiaddr peer format
- 30s heartbeat interval, 60s timeout
"""

import asyncio
import logging
import signal
import argparse
import struct
import uuid
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from aiohttp import web

from shardcompute.coordinator.registry import WorkerRegistry, RegistryConfig
from shardcompute.coordinator.health import HealthMonitor, HealthConfig
from shardcompute.coordinator.metrics import MetricsAggregator
from shardcompute.coordinator.dht_bootstrap import SimpleDHT, DHTConfig
from shardcompute.protocol.messages import WorkerInfo, InferenceRequest, InferenceResponse, StreamingToken

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except Exception:
    HAS_TOKENIZER = False


class InferenceError(Exception):
    """Inference error that carries an HTTP status."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


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
        port: int = 8000,  # Changed from 8080 to match COMMUNICATION_OUTLINE
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
        heartbeat_timeout = self.config.get("coordinator", {}).get("heartbeat_timeout_seconds", 60)

        registry_config = RegistryConfig(expected_workers=world_size)
        self.registry = WorkerRegistry(registry_config)

        health_config = HealthConfig(heartbeat_timeout=heartbeat_timeout)
        self.health_monitor = HealthMonitor(
            self.registry,
            health_config,
            on_failure=self._handle_worker_failure,
        )

        self.metrics = MetricsAggregator()

        # DHT bootstrap for peer discovery (COMMUNICATION_OUTLINE style)
        coordinator_config = self.config.get("coordinator", {})
        dht_config = DHTConfig(
            public_host=coordinator_config.get("public_host", "127.0.0.1"),
            dht_port=coordinator_config.get("dht_port", 31337),
            stale_timeout_seconds=120.0,
            cleanup_interval_seconds=coordinator_config.get("stale_cleanup_interval", 30),
        )
        self.dht = SimpleDHT(dht_config)

        # User and usage tracking (minimal implementation)
        self._users: Dict[str, Dict[str, Any]] = {}
        self._usage_log: List[Dict[str, Any]] = []
        self._start_time = time.time()

        # Optional tokenizer for text UI
        self.tokenizer = self._load_tokenizer()

        # Inference queue
        self._pending_requests: asyncio.Queue = asyncio.Queue()
        self._completed_requests: Dict[str, InferenceResponse] = {}
        self._request_events: Dict[str, asyncio.Event] = {}

        # Streaming token queues (one per streaming request)
        self._streaming_queues: Dict[str, asyncio.Queue] = {}

        # WebSocket relay connections for collective operations (rank -> ws)
        self._ws_connections: Dict[int, web.WebSocketResponse] = {}

        # Track input lengths for metric calculation
        self._request_input_lengths: Dict[str, int] = {}

        # HTTP app
        self.app = web.Application()
        self._static_path = Path(__file__).parent / "static"
        self._index_path = self._static_path / "index.html"
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
        """Setup HTTP routes.

        Endpoint structure follows COMMUNICATION_OUTLINE.md:
        - No /api prefix
        - /network/* for network info, health, stats
        - /workers/* for worker management
        - /inference/* for inference
        - /users/* and /usage/* for tracking
        """
        # Root and dashboard
        self.app.router.add_get("/", self._handle_root)
        self.app.router.add_get("/dashboard", self._handle_dashboard)
        self.app.router.add_get("/ui", self._handle_ui)

        # Network info (COMMUNICATION_OUTLINE style)
        self.app.router.add_get("/network/info", self._handle_network_info)
        self.app.router.add_get("/network/health", self._handle_health)
        self.app.router.add_get("/network/stats", self._handle_network_stats)

        # Worker management
        self.app.router.add_post("/workers/register", self._handle_register)
        self.app.router.add_get("/workers", self._handle_list_workers)
        self.app.router.add_post("/workers/heartbeat", self._handle_heartbeat)
        self.app.router.add_delete("/workers/{worker_id}", self._handle_unregister)

        # Inference
        self.app.router.add_post("/inference", self._handle_inference)
        self.app.router.add_post("/inference/text", self._handle_inference_text)
        self.app.router.add_post("/inference/text/stream", self._handle_inference_text_stream)
        self.app.router.add_get("/inference/poll", self._handle_poll)
        self.app.router.add_post("/inference/response", self._handle_response)
        self.app.router.add_post("/inference/stream/token", self._handle_streaming_token)
        self.app.router.add_get("/inference/{request_id}", self._handle_get_result)

        # Metrics
        self.app.router.add_get("/metrics", self._handle_metrics)

        # User management (minimal)
        self.app.router.add_post("/users/register", self._handle_user_register)
        self.app.router.add_get("/users/{user_id}", self._handle_get_user)

        # Usage tracking (minimal)
        self.app.router.add_post("/usage/log", self._handle_usage_log)
        self.app.router.add_get("/usage/recent", self._handle_usage_recent)

        # WebSocket relay for collective operations
        self.app.router.add_get("/ws/collective/{rank}", self._handle_ws_collective)

        # Static files
        if self._static_path.exists():
            self.app.router.add_static("/static", self._static_path)
    
    async def start(self):
        """Start the coordinator server."""
        logger.info(f"Starting coordinator server on {self.host}:{self.port}")

        self._running = True
        self._start_time = time.time()

        # Start DHT bootstrap
        await self.dht.start()

        # Start health monitor
        await self.health_monitor.start()

        # Start HTTP server
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()

        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

        logger.info(f"Coordinator server started on http://{self.host}:{self.port}")
        logger.info(f"Dashboard available at http://{self.host}:{self.port}/dashboard")

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

        # Stop DHT
        await self.dht.stop()

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
        """Handle root endpoint (GET /).

        Returns complete status info for dashboard when JSON is requested.
        """
        accept = request.headers.get("Accept", "")
        wants_json = "application/json" in accept or request.query.get("json") == "1"
        if wants_json:
            return web.json_response({
                "service": "ShardCompute Coordinator",
                "status": "running" if self._running else "stopped",
                "cluster_ready": self.registry.is_cluster_ready,
                "cluster_healthy": self.health_monitor.is_cluster_healthy(),
                "workers": self.registry.worker_count,
                "expected_workers": self.registry.config.expected_workers,
            })
        return await self._handle_ui(request)

    async def _handle_ui(self, request: web.Request) -> web.Response:
        """Serve the UI."""
        if self._index_path.exists():
            return web.FileResponse(self._index_path)
        return web.Response(text="UI assets not found", status=404)
    
    async def _handle_register(self, request: web.Request) -> web.Response:
        """Handle worker registration.

        Returns LayerAssignment response with initial_peers in multiaddr format,
        following COMMUNICATION_OUTLINE.md style.
        """
        try:
            data = await request.json()

            # Support both old and new field names
            rank = data.get("rank", 0)
            host = data.get("host", request.remote or "127.0.0.1")
            port = data.get("port", 9000)
            collective_port = data.get("collective_port", port)
            worker_id = data.get("worker_id", f"worker-{rank}")

            info = WorkerInfo(
                rank=rank,
                host=host,
                port=port,
                collective_port=collective_port,
                device_info=data.get("device_info", {}),
            )

            success = await self.registry.register(info)

            if success:
                # Register peer in DHT with multiaddr format
                peer_id = self.dht.generate_peer_id(rank)
                await self.dht.register_peer(
                    peer_id=peer_id,
                    address=host,
                    port=collective_port,
                    rank=rank,
                    metadata={"worker_id": worker_id, "device_info": info.device_info},
                )

                # Get model info for response
                model_config = self.config.get("model", {})
                model_name = model_config.get("name", "unknown")
                total_layers = model_config.get("num_layers", 0)

                # For tensor parallelism, all workers get all layers
                # (different from Petals which assigns layer ranges)
                return web.json_response({
                    "status": "online",  # Auto-approved
                    "worker_id": worker_id,
                    "rank": rank,
                    "world_size": self.registry.config.expected_workers,
                    "layers_start": 0,
                    "layers_end": total_layers,
                    "model_name": model_name,
                    "initial_peers": self.dht.get_initial_peers(),
                    "is_idle": False,
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
        """Handle worker list request (GET /workers)."""
        return web.json_response({
            "workers": self.registry.get_workers_dict(),
            "cluster_ready": self.registry.is_cluster_ready,
        })

    async def _handle_heartbeat(self, request: web.Request) -> web.Response:
        """Handle worker heartbeat (POST /workers/heartbeat).

        Updates both registry and DHT, following COMMUNICATION_OUTLINE.md style.
        """
        try:
            data = await request.json()
            # Support both old (rank) and new (worker_id) field names
            rank = data.get("rank")
            worker_id = data.get("worker_id")
            status = data.get("status", "online")
            metrics = data.get("metrics", {})
            tokens_served = data.get("tokens_served_since_last", 0)

            if rank is None and worker_id:
                # Try to find rank from worker_id
                for r, worker in self.registry.get_workers_dict().items():
                    if worker.get("worker_id") == worker_id:
                        rank = int(r)
                        break

            if rank is not None:
                await self.registry.update_heartbeat(rank, status)
                self.health_monitor.record_heartbeat(rank)

                # Update DHT heartbeat
                await self.dht.heartbeat_by_rank(rank)

                if metrics:
                    self.metrics.record_worker_metrics(rank, metrics)

            # Return worker status and layer info
            model_config = self.config.get("model", {})
            return web.json_response({
                "status": "ok",
                "worker_status": "online",
                "layers_start": 0,
                "layers_end": model_config.get("num_layers", 0),
            })
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_unregister(self, request: web.Request) -> web.Response:
        """Handle worker unregistration (DELETE /workers/{worker_id})."""
        try:
            worker_id = request.match_info["worker_id"]

            # Find and remove worker by ID or rank
            rank = None
            try:
                rank = int(worker_id)
            except ValueError:
                # Try to find rank from worker_id
                for r, worker in self.registry.get_workers_dict().items():
                    if worker.get("worker_id") == worker_id:
                        rank = int(r)
                        break

            if rank is not None:
                await self.registry.deregister(rank)
                await self.dht.remove_peer_by_rank(rank)
                logger.info(f"Worker {worker_id} (rank {rank}) unregistered")
                return web.json_response({"status": "ok"})
            else:
                return web.json_response({"error": "Worker not found"}, status=404)
        except Exception as e:
            logger.error(f"Unregister error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_inference(self, request: web.Request) -> web.Response:
        """Handle inference request from client."""
        try:
            data = await request.json()
            response = await self._run_inference(
                input_ids=data["input_ids"],
                max_new_tokens=data.get("max_new_tokens", 100),
                temperature=data.get("temperature", 0.7),
                top_p=data.get("top_p", 0.9),
                stop_tokens=data.get("stop_tokens", [2]),
                timeout=data.get("timeout", 300),
            )
            return web.json_response({
                "request_id": response.request_id,
                "output_ids": response.output_ids,
                "timing": response.timing,
            })
        except InferenceError as e:
            return web.json_response({"error": str(e)}, status=e.status)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_inference_text(self, request: web.Request) -> web.Response:
        """Handle text inference request from UI clients."""
        try:
            data = await request.json()
            prompt = str(data.get("prompt", "")).strip()
            messages = data.get("messages")
            if not prompt and not messages:
                return web.json_response(
                    {"error": "Prompt or messages required"},
                    status=400,
                )

            input_ids = self._encode_prompt(prompt, messages)
            input_length = len(input_ids)

            response = await self._run_inference(
                input_ids=input_ids,
                max_new_tokens=data.get("max_new_tokens", 120),
                temperature=data.get("temperature", 0.7),
                top_p=data.get("top_p", 0.9),
                stop_tokens=data.get("stop_tokens", self._default_stop_tokens()),
                timeout=data.get("timeout", 300),
            )

            output_ids = response.output_ids
            generated_ids = output_ids[input_length:] if len(output_ids) > input_length else output_ids
            output_text = self._decode_output(generated_ids)

            return web.json_response({
                "request_id": response.request_id,
                "output_text": output_text,
                "output_ids": output_ids,
                "timing": response.timing,
            })
        except InferenceError as e:
            return web.json_response({"error": str(e)}, status=e.status)
        except Exception as e:
            logger.error(f"Text inference error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_inference_text_stream(self, request: web.Request) -> web.StreamResponse:
        """Handle streaming text inference request from UI clients using SSE."""
        import json

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        request_id = None
        try:
            data = await request.json()
            prompt = str(data.get("prompt", "")).strip()
            messages = data.get("messages")
            if not prompt and not messages:
                await response.write(b"event: error\ndata: {\"error\": \"Prompt or messages required\"}\n\n")
                await response.write_eof()
                return response

            if not self.registry.is_cluster_ready:
                await response.write(b"event: error\ndata: {\"error\": \"Cluster not ready\"}\n\n")
                await response.write_eof()
                return response

            if not self.health_monitor.is_cluster_healthy():
                await response.write(b"event: error\ndata: {\"error\": \"Cluster unhealthy\"}\n\n")
                await response.write_eof()
                return response

            input_ids = self._encode_prompt(prompt, messages)

            # Create streaming request
            request_id = str(uuid.uuid4())
            inference_request = InferenceRequest(
                request_id=request_id,
                input_ids=input_ids,
                max_new_tokens=data.get("max_new_tokens", 120),
                temperature=data.get("temperature", 0.7),
                top_p=data.get("top_p", 0.9),
                stop_tokens=data.get("stop_tokens", self._default_stop_tokens()),
                stream=True,  # Enable streaming
            )

            # Create streaming queue and events for this request
            self._streaming_queues[request_id] = asyncio.Queue()
            self._request_events[request_id] = asyncio.Event()
            self._request_input_lengths[request_id] = len(input_ids)  # Track input length

            # Queue the request
            await self._pending_requests.put(inference_request)
            logger.info(f"Streaming inference request queued: {request_id}")

            timeout = data.get("timeout", 300)

            # Stream tokens as they arrive
            while True:
                try:
                    # Wait for next token with timeout
                    token = await asyncio.wait_for(
                        self._streaming_queues[request_id].get(),
                        timeout=timeout,
                    )

                    if token.error:
                        error_data = json.dumps({"error": token.error})
                        await response.write(f"event: error\ndata: {error_data}\n\n".encode("utf-8"))
                        break

                    # Decode the token to text
                    token_text = self._decode_output([token.token_id])
                    event_data = json.dumps({"token": token_text, "token_id": token.token_id})
                    await response.write(f"event: token\ndata: {event_data}\n\n".encode("utf-8"))

                    if token.is_final:
                        break

                except asyncio.TimeoutError:
                    error_data = json.dumps({"error": "Streaming timeout"})
                    await response.write(f"event: error\ndata: {error_data}\n\n".encode("utf-8"))
                    break

            # Wait for final response to get timing info
            try:
                await asyncio.wait_for(
                    self._request_events[request_id].wait(),
                    timeout=5.0,  # Short timeout since tokens are done
                )
                final_response = self._completed_requests.pop(request_id, None)
                if final_response:
                    completion_data = json.dumps({
                        "request_id": request_id,
                        "timing": final_response.timing,
                    })
                    await response.write(f"event: done\ndata: {completion_data}\n\n".encode("utf-8"))
            except asyncio.TimeoutError:
                # Send done without timing if final response doesn't arrive
                completion_data = json.dumps({"request_id": request_id})
                await response.write(f"event: done\ndata: {completion_data}\n\n".encode("utf-8"))

            await response.write_eof()
            return response

        except Exception as e:
            logger.error(f"Streaming text inference error: {e}")
            error_data = json.dumps({"error": str(e)})
            await response.write(f"event: error\ndata: {error_data}\n\n".encode("utf-8"))
            await response.write_eof()
            return response
        finally:
            # Cleanup
            if request_id:
                self._streaming_queues.pop(request_id, None)
                self._request_events.pop(request_id, None)
                self._completed_requests.pop(request_id, None)
                self._request_input_lengths.pop(request_id, None)

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
                # Calculate actual generated tokens
                input_length = self._request_input_lengths.get(request_id, 0)
                total_tokens = len(response.output_ids)
                generated_tokens = total_tokens - input_length if total_tokens > input_length else total_tokens

                self.metrics.record_inference(
                    request_id=request_id,
                    timing=response.timing,
                    input_tokens=input_length,
                    output_tokens=generated_tokens,
                )

                # Clean up input length tracking
                self._request_input_lengths.pop(request_id, None)

            # Store response and signal
            self._completed_requests[request_id] = response

            if request_id in self._request_events:
                self._request_events[request_id].set()

            return web.json_response({"status": "ok"})

        except Exception as e:
            logger.error(f"Response handling error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_streaming_token(self, request: web.Request) -> web.Response:
        """Handle streaming token from worker."""
        try:
            data = await request.json()
            token = StreamingToken.from_dict(data)

            request_id = token.request_id

            # Put token in the streaming queue if it exists
            if request_id in self._streaming_queues:
                await self._streaming_queues[request_id].put(token)
                return web.json_response({"status": "ok"})
            else:
                # No streaming client waiting - this is ok, client may have disconnected
                logger.debug(f"No streaming queue for request {request_id}")
                return web.json_response({"status": "no_listener"})

        except Exception as e:
            logger.error(f"Streaming token handling error: {e}")
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
        health_status = self.health_monitor.get_health_status()
        return web.json_response({
            "status": "running" if self._running else "stopped",
            "cluster_ready": self.registry.is_cluster_ready,
            "cluster_healthy": self.health_monitor.is_cluster_healthy(),
            "workers": health_status["healthy_count"],
            "expected_workers": self.registry.config.expected_workers,
        })
    
    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """Handle metrics request."""
        return web.json_response(self.metrics.get_full_report())
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle health check request (GET /network/health)."""
        health = self.health_monitor.get_health_status()

        status = 200 if self.health_monitor.is_cluster_healthy() else 503

        return web.json_response(health, status=status)

    async def _handle_network_info(self, request: web.Request) -> web.Response:
        """Handle network info request (GET /network/info).

        Returns bootstrap_peers in multiaddr format, following COMMUNICATION_OUTLINE.md.
        """
        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "unknown")
        total_layers = model_config.get("num_layers", 0)
        quantization_config = self.config.get("quantization", {})

        return web.json_response({
            "model_name": model_name,
            "total_layers": total_layers,
            "bootstrap_peers": self.dht.get_initial_peers(),
            "online_workers": self.registry.worker_count,
            "coverage_percent": 100.0 if self.registry.is_cluster_ready else 0.0,
            "ready_for_inference": (
                self.registry.is_cluster_ready and
                self.health_monitor.is_cluster_healthy()
            ),
            "quantization_mode": quantization_config.get("mode"),
        })

    async def _handle_network_stats(self, request: web.Request) -> web.Response:
        """Handle network stats request (GET /network/stats)."""
        summary = self.metrics.get_summary()
        uptime = time.time() - self._start_time

        return web.json_response({
            "uptime_seconds": uptime,
            "total_requests": summary.get("total_requests", 0),
            "total_tokens_generated": summary.get("total_tokens_generated", 0),
            "avg_latency_ms": summary.get("avg_latency_ms", 0),
            "p99_latency_ms": summary.get("p99_latency_ms", 0),
            "avg_throughput_tokens_per_sec": summary.get("avg_throughput_tokens_per_sec", 0),
            "coverage_percent": 100.0 if self.registry.is_cluster_ready else 0.0,
        })

    async def _handle_dashboard(self, request: web.Request) -> web.Response:
        """Handle dashboard request (GET /dashboard)."""
        if self._index_path.exists():
            return web.FileResponse(self._index_path)
        return web.Response(text="Dashboard not found", status=404)

    async def _handle_user_register(self, request: web.Request) -> web.Response:
        """Handle user registration (POST /users/register)."""
        try:
            data = await request.json()
            user_id = data.get("user_id")
            if not user_id:
                return web.json_response({"error": "user_id required"}, status=400)

            self._users[user_id] = {
                "user_id": user_id,
                "display_name": data.get("display_name"),
                "total_prompts": 0,
                "total_tokens": 0,
                "created_at": time.time(),
            }
            return web.json_response({"status": "ok", "user_id": user_id})
        except Exception as e:
            logger.error(f"User registration error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_get_user(self, request: web.Request) -> web.Response:
        """Handle get user request (GET /users/{user_id})."""
        user_id = request.match_info["user_id"]
        if user_id not in self._users:
            return web.json_response({"error": "User not found"}, status=404)
        return web.json_response(self._users[user_id])

    async def _handle_usage_log(self, request: web.Request) -> web.Response:
        """Handle usage log request (POST /usage/log)."""
        try:
            data = await request.json()
            entry = {
                "user_id": data.get("user_id"),
                "prompt_tokens": data.get("prompt_tokens", 0),
                "completion_tokens": data.get("completion_tokens", 0),
                "worker_ids": data.get("worker_ids", []),
                "timestamp": time.time(),
            }
            self._usage_log.append(entry)

            # Update user stats if user exists
            user_id = entry["user_id"]
            if user_id and user_id in self._users:
                self._users[user_id]["total_prompts"] += 1
                self._users[user_id]["total_tokens"] += (
                    entry["prompt_tokens"] + entry["completion_tokens"]
                )

            return web.json_response({"status": "ok"})
        except Exception as e:
            logger.error(f"Usage log error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_usage_recent(self, request: web.Request) -> web.Response:
        """Handle recent usage request (GET /usage/recent)."""
        limit = int(request.query.get("limit", 15))
        recent = self._usage_log[-limit:] if self._usage_log else []
        return web.json_response({"entries": recent})

    async def _handle_ws_collective(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket relay for collective operations.

        Each worker connects once.  Binary frames carry a 16-byte envelope
        [sender_rank(4) | target_rank(4) | msg_type(4) | data_len(4)] followed
        by payload.  The coordinator reads target_rank and forwards the entire
        frame to the target worker's WebSocket — zero-copy, no deserialization.
        """
        rank = int(request.match_info["rank"])
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)

        logger.info(f"WebSocket collective connection from rank {rank}")
        self._ws_connections[rank] = ws

        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    if len(msg.data) < 16:
                        continue
                    target_rank = struct.unpack(">I", msg.data[4:8])[0]
                    target_ws = self._ws_connections.get(target_rank)
                    if target_ws is not None and not target_ws.closed:
                        await target_ws.send_bytes(msg.data)
                    else:
                        logger.warning(
                            f"Cannot relay rank {rank} -> rank {target_rank}: target not connected"
                        )
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                    break
        finally:
            self._ws_connections.pop(rank, None)
            logger.info(f"WebSocket collective connection from rank {rank} closed")

        return ws

    def _load_tokenizer(self):
        """Load tokenizer if configured."""
        tokenizer_path = (
            self.config.get("coordinator", {}).get("tokenizer_path")
            or self.config.get("model", {}).get("tokenizer_path")
        )
        if not tokenizer_path:
            return None
        if not HAS_TOKENIZER:
            logger.warning("Transformers not available; tokenizer disabled")
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"Tokenizer loaded from {tokenizer_path}")
            return tokenizer
        except Exception as e:
            logger.warning(f"Tokenizer load failed: {e}")
            return None

    def _default_stop_tokens(self) -> List[int]:
        """Resolve stop tokens for text inference."""
        if self.tokenizer and hasattr(self.tokenizer, "eos_token_id"):
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                return [int(eos_id)]
        return [2]

    def _normalize_input_ids(self, tokenized) -> List[int]:
        """Normalize tokenizer outputs into a list of ints."""
        if hasattr(tokenized, "input_ids"):
            tokenized = tokenized.input_ids

        if isinstance(tokenized, str):
            if not self.tokenizer:
                raise ValueError("Tokenizer is required to encode string input.")
            tokenized = self.tokenizer.encode(tokenized)

        if hasattr(tokenized, "tolist"):
            tokenized = tokenized.tolist()

        if isinstance(tokenized, (list, tuple)):
            if tokenized and isinstance(tokenized[0], (list, tuple)):
                tokenized = tokenized[0]
            return [int(x) for x in tokenized]

        raise TypeError(f"Unsupported tokenized type: {type(tokenized)}")

    def _encode_prompt(self, prompt: str, messages) -> List[int]:
        """Encode prompt or messages into token ids."""
        if self.tokenizer and messages and hasattr(self.tokenizer, "apply_chat_template"):
            cleaned = []
            if isinstance(messages, list):
                for message in messages:
                    if isinstance(message, dict) and "role" in message and "content" in message:
                        cleaned.append({
                            "role": str(message["role"]),
                            "content": str(message["content"]),
                        })
            if cleaned:
                tokenized = self.tokenizer.apply_chat_template(
                    cleaned,
                    add_generation_prompt=True,
                    return_tensors=None,
                )
                return self._normalize_input_ids(tokenized)

        if not prompt and messages:
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in messages
                if isinstance(m, dict)
            )

        if self.tokenizer:
            tokenized = self.tokenizer.encode(prompt)
            return self._normalize_input_ids(tokenized)

        return [ord(c) % 32000 for c in prompt]

    def _decode_output(self, output_ids: List[int]) -> str:
        """Decode generated tokens into text."""
        if self.tokenizer:
            return self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return "".join(chr(min(i, 127)) for i in output_ids)

    async def _run_inference(
        self,
        input_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_tokens: List[int],
        timeout: int,
    ) -> InferenceResponse:
        """Queue inference and await a response."""
        if not self.registry.is_cluster_ready:
            raise InferenceError("Cluster not ready", 503)

        if not self.health_monitor.is_cluster_healthy():
            raise InferenceError("Cluster unhealthy", 503)

        request_id = str(uuid.uuid4())
        inference_request = InferenceRequest(
            request_id=request_id,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_tokens=stop_tokens,
        )

        self._request_events[request_id] = asyncio.Event()
        self._request_input_lengths[request_id] = len(input_ids)  # Track input length
        await self._pending_requests.put(inference_request)
        logger.info(f"Inference request queued: {request_id}")

        try:
            await asyncio.wait_for(
                self._request_events[request_id].wait(),
                timeout=timeout,
            )
        except asyncio.TimeoutError as e:
            self._request_events.pop(request_id, None)
            self._request_input_lengths.pop(request_id, None)
            raise InferenceError("Inference timeout", 504) from e

        response = self._completed_requests.pop(request_id, None)
        self._request_events.pop(request_id, None)
        self._request_input_lengths.pop(request_id, None)

        if response is None:
            raise InferenceError("Response not found", 500)
        if response.error:
            raise InferenceError(response.error, 500)

        return response


def main():
    """Entry point for coordinator server."""
    parser = argparse.ArgumentParser(description="ShardCompute Coordinator")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
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
