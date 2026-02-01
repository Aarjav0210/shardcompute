"""Coordinator server for cluster management."""

import asyncio
import logging
import signal
import argparse
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from aiohttp import web

from shardcompute.coordinator.registry import WorkerRegistry, RegistryConfig
from shardcompute.coordinator.health import HealthMonitor, HealthConfig
from shardcompute.coordinator.metrics import MetricsAggregator
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

        # Optional tokenizer for text UI
        self.tokenizer = self._load_tokenizer()
        
        # Inference queue
        self._pending_requests: asyncio.Queue = asyncio.Queue()
        self._completed_requests: Dict[str, InferenceResponse] = {}
        self._request_events: Dict[str, asyncio.Event] = {}

        # Streaming token queues (one per streaming request)
        self._streaming_queues: Dict[str, asyncio.Queue] = {}
        
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
        """Setup HTTP routes."""
        # Worker management
        self.app.router.add_post("/api/workers/register", self._handle_register)
        self.app.router.add_get("/api/workers/list", self._handle_list_workers)
        self.app.router.add_post("/api/heartbeat", self._handle_heartbeat)

        # Inference
        self.app.router.add_post("/api/inference", self._handle_inference)
        self.app.router.add_post("/api/inference/text", self._handle_inference_text)
        self.app.router.add_post("/api/inference/text/stream", self._handle_inference_text_stream)
        self.app.router.add_get("/api/inference/poll", self._handle_poll)
        self.app.router.add_post("/api/inference/response", self._handle_response)
        self.app.router.add_post("/api/inference/stream/token", self._handle_streaming_token)
        self.app.router.add_get("/api/inference/{request_id}", self._handle_get_result)
        
        # Metrics and status
        self.app.router.add_get("/api/status", self._handle_status)
        self.app.router.add_get("/api/metrics", self._handle_metrics)
        self.app.router.add_get("/api/health", self._handle_health)

        # Root
        self.app.router.add_get("/", self._handle_root)
        self.app.router.add_get("/ui", self._handle_ui)
        if self._static_path.exists():
            self.app.router.add_static("/static", self._static_path)
    
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
        accept = request.headers.get("Accept", "")
        wants_json = "application/json" in accept or request.query.get("json") == "1"
        if wants_json:
            return web.json_response({
                "service": "ShardCompute Coordinator",
                "status": "running" if self._running else "stopped",
                "cluster_ready": self.registry.is_cluster_ready,
                "workers": self.registry.worker_count,
            })
        return await self._handle_ui(request)

    async def _handle_ui(self, request: web.Request) -> web.Response:
        """Serve the UI."""
        if self._index_path.exists():
            return web.FileResponse(self._index_path)
        return web.Response(text="UI assets not found", status=404)
    
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
        """Handle health check request."""
        health = self.health_monitor.get_health_status()
        
        status = 200 if self.health_monitor.is_cluster_healthy() else 503
        
        return web.json_response(health, status=status)

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
        await self._pending_requests.put(inference_request)
        logger.info(f"Inference request queued: {request_id}")

        try:
            await asyncio.wait_for(
                self._request_events[request_id].wait(),
                timeout=timeout,
            )
        except asyncio.TimeoutError as e:
            self._request_events.pop(request_id, None)
            raise InferenceError("Inference timeout", 504) from e

        response = self._completed_requests.pop(request_id, None)
        self._request_events.pop(request_id, None)

        if response is None:
            raise InferenceError("Response not found", 500)
        if response.error:
            raise InferenceError(response.error, 500)

        return response


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
