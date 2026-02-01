"""Worker node for distributed tensor and pipeline parallel inference."""

import asyncio
import json
import logging
import signal
import sys
import traceback
import argparse
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import yaml
import aiohttp
import mlx.core as mx

from shardcompute.worker.peer_mesh import PeerMesh, MeshConfig
from shardcompute.worker.executor import ParallelExecutor, PipelineExecutor
from shardcompute.worker.heartbeat import HeartbeatClient, HeartbeatConfig
from shardcompute.parallel.transformer import ParallelTransformer, PipelineParallelTransformer
from shardcompute.model.loader import ModelLoader, PipelineModelLoader, detect_parallelism_mode
from shardcompute.model.config import ModelConfig, ParallelConfig
from shardcompute.protocol.messages import WorkerInfo, InferenceRequest, InferenceResponse, StreamingToken

logger = logging.getLogger(__name__)


class WorkerNode:
    """
    Worker node for distributed tensor parallel inference.
    
    Responsibilities:
    - Register with coordinator
    - Establish peer connections
    - Load model weight shards
    - Execute parallel inference
    - Report health via heartbeats
    - Handle inference requests (rank 0 receives, all workers execute)
    """
    
    def __init__(
        self,
        rank: int,
        coordinator_url: str,
        host: str = "0.0.0.0",
        collective_port: int = 9000,
        shard_dir: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize WorkerNode.
        
        Args:
            rank: This worker's rank
            coordinator_url: URL of the coordinator
            host: Host for peer connections
            collective_port: Port for peer connections
            shard_dir: Directory containing weight shards
            config_path: Path to config file
        """
        self.rank = rank
        self.coordinator_url = coordinator_url.rstrip('/')
        self.host = host
        self.collective_port = collective_port
        self.shard_dir = Path(shard_dir) if shard_dir else None
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Components (initialized during setup)
        self.peer_mesh: Optional[PeerMesh] = None
        self.executor: Optional[Union[ParallelExecutor, PipelineExecutor]] = None
        self.heartbeat: Optional[HeartbeatClient] = None
        self.model: Optional[Union[ParallelTransformer, PipelineParallelTransformer]] = None

        # Detect parallelism mode from config or shard directory
        parallelism_cfg = self.config.get("parallelism", {})
        self._parallelism_mode = parallelism_cfg.get("mode", "tensor")

        # State
        self._running = False
        if self._parallelism_mode == "pipeline":
            self._world_size = parallelism_cfg.get("pipeline_parallel_size", 2)
        else:
            self._world_size = parallelism_cfg.get("tensor_parallel_size", 2)
        self._inference_lock = asyncio.Lock()
        self._http_session: Optional[aiohttp.ClientSession] = None

        logger.info(
            f"WorkerNode initialized: rank {rank}, coordinator {coordinator_url}, "
            f"mode={self._parallelism_mode}"
        )
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    async def start(self):
        """Start the worker node."""
        logger.info(f"Rank {self.rank} starting worker node")
        
        self._running = True
        self._http_session = aiohttp.ClientSession()
        
        try:
            # Register with coordinator
            await self._register_with_coordinator()
            
            # Wait for all workers and get peer list
            peers = await self._wait_for_cluster()
            
            # Setup peer mesh
            await self._setup_peer_mesh(peers)
            
            # Load model
            await self._load_model()
            
            # Start heartbeat
            await self._start_heartbeat()
            
            # Synchronize with peers
            await self.peer_mesh.barrier()
            
            logger.info(f"Rank {self.rank} worker node started successfully")
            
            # Main loop
            if self.rank == 0:
                # Rank 0 listens for inference requests
                await self._inference_loop()
            else:
                # Other ranks wait for broadcasts from rank 0
                await self._follower_loop()
                
        except Exception as e:
            logger.error(f"Rank {self.rank} failed to start: {e}")
            raise
        finally:
            await self.stop()
    
    async def _register_with_coordinator(self):
        """Register this worker with the coordinator.

        Uses the endpoint structure from COMMUNICATION_OUTLINE.md.
        """
        url = f"{self.coordinator_url}/workers/register"

        payload = {
            "worker_id": f"worker-{self.rank}",
            "rank": self.rank,
            "host": self.host,
            "port": self.collective_port,
            "collective_port": self.collective_port,
            "hardware_type": "apple_silicon",
            "device_info": {
                "platform": "apple_silicon",
                "mlx_version": mx.__version__ if hasattr(mx, '__version__') else "unknown",
            },
        }

        logger.info(f"Rank {self.rank} registering with coordinator")

        async with self._http_session.post(url, json=payload) as response:
            if response.status != 200:
                error = await response.text()
                raise RuntimeError(f"Registration failed: {error}")

            data = await response.json()
            self._world_size = data.get("world_size", self._world_size)

            # Log initial_peers if available (multiaddr format)
            initial_peers = data.get("initial_peers", [])
            if initial_peers:
                logger.debug(f"Rank {self.rank} received initial_peers: {initial_peers}")

        logger.info(f"Rank {self.rank} registered successfully, world_size={self._world_size}")
    
    async def _wait_for_cluster(self) -> List[WorkerInfo]:
        """Wait for all workers to register and get peer list."""
        url = f"{self.coordinator_url}/workers"
        
        logger.info(f"Rank {self.rank} waiting for cluster formation")
        
        while self._running:
            try:
                async with self._http_session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        workers = data.get("workers", [])
                        
                        if len(workers) >= self._world_size:
                            peers = [WorkerInfo.from_dict(w) for w in workers]
                            logger.info(f"Rank {self.rank} cluster ready with {len(peers)} workers")
                            return peers
                        
                        logger.debug(
                            f"Rank {self.rank} waiting: {len(workers)}/{self._world_size} workers"
                        )
            except Exception as e:
                logger.warning(f"Error checking cluster status: {e}")
            
            await asyncio.sleep(1)
        
        raise RuntimeError("Worker stopped while waiting for cluster")
    
    async def _setup_peer_mesh(self, peers: List[WorkerInfo]):
        """Setup peer mesh and establish connections."""
        worker_config = self.config.get("worker", {})
        transport = worker_config.get("transport", "ws_relay")

        coordinator_ws_url = None
        if transport == "ws_relay":
            coordinator_ws_url = (
                self.coordinator_url
                .replace("http://", "ws://")
                .replace("https://", "wss://")
                + "/ws/collective"
            )

        config = MeshConfig(
            world_size=self._world_size,
            connection_timeout=worker_config.get("collective_timeout_seconds", 30),
            transport=transport,
            coordinator_ws_url=coordinator_ws_url,
        )

        self.peer_mesh = PeerMesh(
            rank=self.rank,
            host=self.host,
            port=self.collective_port,
            config=config,
        )

        self.peer_mesh.set_peers(peers)

        success = await self.peer_mesh.connect()
        if not success:
            raise RuntimeError("Failed to connect peer mesh")

        logger.info(f"Rank {self.rank} peer mesh connected via {transport}")
    
    async def _load_model(self):
        """Load model with weight shards."""
        # Override mode from shard directory if available
        if self.shard_dir and self.shard_dir.exists():
            detected_mode = detect_parallelism_mode(self.shard_dir)
            if detected_mode != self._parallelism_mode:
                logger.info(
                    f"Rank {self.rank}: shard directory mode '{detected_mode}' overrides "
                    f"config mode '{self._parallelism_mode}'"
                )
                self._parallelism_mode = detected_mode

        if self._parallelism_mode == "pipeline":
            await self._load_model_pipeline()
        else:
            await self._load_model_tensor()

    async def _load_model_tensor(self):
        """Load model for tensor parallelism (existing behavior)."""
        model_config = self.config.get("model", {})

        # Try to load config from shard directory first (has actual model config)
        if self.shard_dir and (self.shard_dir / "config.json").exists():
            with open(self.shard_dir / "config.json") as f:
                shard_config = json.load(f)
            if "model" in shard_config:
                config = ModelConfig.from_dict(shard_config["model"])
            else:
                config = ModelConfig.from_dict(shard_config)
            kv_info = f", {config.num_kv_heads} KV heads (GQA)" if config.num_kv_heads else ""
            logger.info(f"Rank {self.rank} loaded model config: {config.num_heads} heads{kv_info}")
        else:
            config = ModelConfig(
                vocab_size=model_config.get("vocab_size", 32000),
                hidden_size=model_config.get("hidden_size", 2048),
                num_layers=model_config.get("num_layers", 22),
                num_heads=model_config.get("num_heads", 32),
                num_kv_heads=model_config.get("num_kv_heads"),
                intermediate_size=model_config.get("intermediate_size", 5632),
                max_position_embeddings=model_config.get("max_position_embeddings", 2048),
                rms_norm_eps=model_config.get("rms_norm_eps", 1e-5),
                rope_theta=model_config.get("rope_theta", 10000.0),
            )

        # Check for quantized weights before creating model
        is_quantized = False
        quantization_bits = 4
        quantization_group_size = 64

        if self.shard_dir and self.shard_dir.exists():
            is_quantized = ModelLoader.detect_quantization(self.shard_dir, self.rank)
            if is_quantized:
                logger.info(f"Rank {self.rank}: Quantized weights detected, enabling quantized inference")
                if config.quantization:
                    quantization_bits = config.quantization.bits
                    quantization_group_size = config.quantization.group_size
                    logger.info(f"Rank {self.rank}: Using {quantization_bits}-bit quantization with group size {quantization_group_size}")

        # Create parallel model
        self.model = ParallelTransformer(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            world_size=self._world_size,
            rank=self.rank,
            communicator=self.peer_mesh.get_communicator(),
            num_kv_heads=config.num_kv_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_base=config.rope_theta,
            mlp_activation=config.hidden_act,
            use_gated_mlp=config.use_gated_mlp,
            tie_word_embeddings=config.tie_word_embeddings,
            use_quantized=is_quantized,
            quantization_bits=quantization_bits,
            quantization_group_size=quantization_group_size,
        )

        # Load weight shards if directory provided
        if self.shard_dir and self.shard_dir.exists():
            loader = ModelLoader(self.rank, self._world_size, quantized=is_quantized)
            await loader.load_shards(self.model, self.shard_dir)
            logger.info(f"Rank {self.rank} loaded model shards from {self.shard_dir}")
        else:
            logger.warning(f"Rank {self.rank} no shard directory, using uninitialized weights")

        # Create executor
        self.executor = ParallelExecutor(
            rank=self.rank,
            world_size=self._world_size,
            communicator=self.peer_mesh.get_communicator(),
            model=self.model,
        )

        logger.info(f"Rank {self.rank} model loaded (tensor): {self.model.num_parameters:,} parameters")

    async def _load_model_pipeline(self):
        """Load model for pipeline parallelism."""
        model_config = self.config.get("model", {})

        # Load config from shard directory
        if self.shard_dir and (self.shard_dir / "config.json").exists():
            with open(self.shard_dir / "config.json") as f:
                shard_config = json.load(f)
            if "model" in shard_config:
                config = ModelConfig.from_dict(shard_config["model"])
            else:
                config = ModelConfig.from_dict(shard_config)
        else:
            config = ModelConfig(
                vocab_size=model_config.get("vocab_size", 32000),
                hidden_size=model_config.get("hidden_size", 2048),
                num_layers=model_config.get("num_layers", 22),
                num_heads=model_config.get("num_heads", 32),
                num_kv_heads=model_config.get("num_kv_heads"),
                intermediate_size=model_config.get("intermediate_size", 5632),
                max_position_embeddings=model_config.get("max_position_embeddings", 2048),
                rms_norm_eps=model_config.get("rms_norm_eps", 1e-5),
                rope_theta=model_config.get("rope_theta", 10000.0),
            )

        # Compute layer assignment
        parallel_config = ParallelConfig(
            world_size=self._world_size,
            rank=self.rank,
            mode="pipeline",
            pipeline_parallel_size=self._world_size,
        )
        start_layer, end_layer = parallel_config.get_pipeline_stage_layers(
            self.rank, config.num_layers
        )

        logger.info(
            f"Rank {self.rank} pipeline stage: layers [{start_layer}, {end_layer}), "
            f"{config.num_layers} total layers across {self._world_size} workers"
        )

        # Create pipeline model
        self.model = PipelineParallelTransformer(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            world_size=self._world_size,
            rank=self.rank,
            communicator=self.peer_mesh.get_communicator(),
            num_kv_heads=config.num_kv_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_base=config.rope_theta,
            mlp_activation=config.hidden_act,
            use_gated_mlp=config.use_gated_mlp,
            tie_word_embeddings=config.tie_word_embeddings,
        )

        # Load pipeline weight shards
        if self.shard_dir and self.shard_dir.exists():
            loader = PipelineModelLoader(
                self.rank, self._world_size, start_layer, end_layer
            )
            await loader.load_shards(self.model, self.shard_dir)
            logger.info(f"Rank {self.rank} loaded pipeline model shards from {self.shard_dir}")
        else:
            logger.warning(f"Rank {self.rank} no shard directory, using uninitialized weights")

        # Create pipeline executor
        self.executor = PipelineExecutor(
            rank=self.rank,
            world_size=self._world_size,
            communicator=self.peer_mesh.get_communicator(),
            model=self.model,
        )

        logger.info(f"Rank {self.rank} model loaded (pipeline): {self.model.num_parameters:,} parameters")
    
    async def _start_heartbeat(self):
        """Start heartbeat client."""
        config = HeartbeatConfig(
            interval_seconds=self.config.get("coordinator", {}).get("heartbeat_interval_seconds", 5),
        )
        
        self.heartbeat = HeartbeatClient(
            rank=self.rank,
            coordinator_url=self.coordinator_url,
            config=config,
            metrics_callback=self._get_metrics,
        )
        
        await self.heartbeat.start()
        self.heartbeat.set_status("ready")
        
        logger.info(f"Rank {self.rank} heartbeat started")
    
    def _get_metrics(self) -> Dict[str, Any]:
        """Collect metrics for heartbeat."""
        metrics = {
            "rank": self.rank,
            "mesh_connected": self.peer_mesh.is_connected if self.peer_mesh else False,
        }
        
        if self.executor:
            timing = self.executor.get_timing_summary()
            if timing:
                metrics.update(timing)
        
        if self.peer_mesh:
            metrics["comm_stats"] = self.peer_mesh.get_stats()
        
        return metrics
    
    async def _inference_loop(self):
        """Main loop for rank 0 - receive and process inference requests."""
        url = f"{self.coordinator_url}/inference/poll"
        
        logger.info(f"Rank 0 starting inference loop")
        
        while self._running:
            try:
                # Poll for requests
                async with self._http_session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("request"):
                            request = InferenceRequest.from_dict(data["request"])
                            await self._process_inference(request)
                    elif response.status != 204:  # 204 = no content
                        logger.warning(f"Poll returned status {response.status}")
                        
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue polling
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Inference loop error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Rank 0 inference loop ended")
    
    async def _follower_loop(self):
        """Main loop for non-rank-0 workers - wait for broadcasts."""
        if self._parallelism_mode == "pipeline":
            await self._follower_loop_pipeline()
        else:
            await self._follower_loop_tensor()

    async def _follower_loop_tensor(self):
        """Follower loop for tensor parallelism — all workers run all layers."""
        logger.info(f"Rank {self.rank} starting tensor follower loop")

        while self._running:
            try:
                comm = self.peer_mesh.get_communicator()
                comm.flush_peers()

                # Wait for generation parameters broadcast from rank 0
                params = await comm.broadcast(None, root=0)

                if params.size == 0:
                    logger.warning(f"Rank {self.rank} received empty broadcast")
                    continue

                params_list = params.tolist()
                input_len = int(params_list[0])
                max_new_tokens = int(params_list[1]) if len(params_list) > 1 else 0
                temperature = float(params_list[2]) / 1000.0 if len(params_list) > 2 else 0.7
                top_p = float(params_list[3]) / 1000.0 if len(params_list) > 3 else 0.9
                num_stop_tokens = int(params_list[4]) if len(params_list) > 4 else 0

                stop_tokens = []
                if num_stop_tokens > 0:
                    stop_tokens_array = await comm.broadcast(None, root=0)
                    stop_tokens = stop_tokens_array.tolist()

                input_ids = await comm.broadcast(None, root=0)

                logger.info(f"Rank {self.rank} starting generation: {input_len} input tokens, "
                           f"{max_new_tokens} max new tokens, stop_tokens={stop_tokens}")

                if max_new_tokens > 0:
                    await self.executor.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop_tokens=stop_tokens,
                    )
                else:
                    await self.executor.forward(input_ids)

            except asyncio.CancelledError:
                break
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                logger.error(f"Follower loop error: {error_msg}")

        logger.info(f"Rank {self.rank} tensor follower loop ended")

    async def _follower_loop_pipeline(self):
        """
        Follower loop for pipeline parallelism.

        Non-rank-0 stages receive coordination parameters via broadcast,
        then participate in the pipeline forward pass. Hidden states flow
        through the pipeline via point-to-point send/recv inside the executor.
        """
        logger.info(f"Rank {self.rank} starting pipeline follower loop")

        while self._running:
            try:
                comm = self.peer_mesh.get_communicator()
                comm.flush_peers()

                # Wait for generation parameters broadcast from rank 0
                params = await comm.broadcast(None, root=0)

                if params.size == 0:
                    logger.warning(f"Rank {self.rank} received empty broadcast")
                    continue

                params_list = params.tolist()
                input_len = int(params_list[0])
                max_new_tokens = int(params_list[1]) if len(params_list) > 1 else 0
                temperature = float(params_list[2]) / 1000.0 if len(params_list) > 2 else 0.7
                top_p = float(params_list[3]) / 1000.0 if len(params_list) > 3 else 0.9
                num_stop_tokens = int(params_list[4]) if len(params_list) > 4 else 0

                stop_tokens = []
                if num_stop_tokens > 0:
                    stop_tokens_array = await comm.broadcast(None, root=0)
                    stop_tokens = stop_tokens_array.tolist()

                # Receive input_ids (needed for generate's broadcast token sync)
                input_ids = await comm.broadcast(None, root=0)

                logger.info(
                    f"Rank {self.rank} pipeline stage starting: {input_len} input tokens, "
                    f"{max_new_tokens} max new tokens"
                )

                # Pipeline executor handles point-to-point hidden state transfer
                if max_new_tokens > 0:
                    await self.executor.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stop_tokens=stop_tokens,
                    )
                else:
                    await self.executor.forward(input_ids=input_ids)

            except asyncio.CancelledError:
                break
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                logger.error(f"Pipeline follower loop error: {error_msg}")

        logger.info(f"Rank {self.rank} pipeline follower loop ended")
    
    async def _process_inference(self, request: InferenceRequest):
        """Process an inference request (rank 0 only)."""
        async with self._inference_lock:
            try:
                self.heartbeat.set_status("processing")

                # Convert input to tensor
                input_ids = mx.array([request.input_ids], dtype=mx.int32)

                # Flush stale data from previous failed inference cycles
                comm = self.peer_mesh.get_communicator()
                comm.flush_peers()

                # First broadcast number of stop tokens
                num_stop_tokens = len(request.stop_tokens) if request.stop_tokens else 0
                params = mx.array([
                    len(request.input_ids),
                    request.max_new_tokens,
                    int(request.temperature * 1000),  # Scale to int for easier transfer
                    int(request.top_p * 1000),
                    num_stop_tokens,
                ], dtype=mx.int32)
                await comm.broadcast(params, root=0)

                # Broadcast stop tokens if any
                if num_stop_tokens > 0:
                    stop_tokens_array = mx.array(request.stop_tokens, dtype=mx.int32)
                    await comm.broadcast(stop_tokens_array, root=0)

                # Broadcast input to all workers
                await comm.broadcast(input_ids, root=0)

                logger.info(f"Rank 0 starting generation: {len(request.input_ids)} input tokens, "
                           f"{request.max_new_tokens} max new tokens, "
                           f"stop_tokens={request.stop_tokens}, stream={request.stream}")

                # Create token callback for streaming if requested
                token_callback = None
                if request.stream:
                    async def send_streaming_token(token_id: int, token_index: int, is_final: bool):
                        """Send a streaming token to the coordinator."""
                        streaming_token = StreamingToken(
                            request_id=request.request_id,
                            token_id=token_id,
                            token_index=token_index,
                            is_final=is_final,
                        )
                        await self._send_streaming_token(streaming_token)

                    token_callback = send_streaming_token

                # Execute inference
                if request.max_new_tokens > 0:
                    output_ids = await self.executor.generate(
                        input_ids,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop_tokens=request.stop_tokens,
                        token_callback=token_callback,
                    )
                else:
                    # Just forward pass
                    logits = await self.executor.forward(input_ids)
                    output_ids = mx.argmax(logits, axis=-1)

                # Send final response (for non-streaming or as completion signal)
                response = InferenceResponse(
                    request_id=request.request_id,
                    output_ids=output_ids[0].tolist(),
                    timing=self.executor.get_timing_summary(),
                )

                await self._send_response(response)

                self.heartbeat.set_status("ready")

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                logger.error(f"Inference error: {error_msg}")
                if request.stream:
                    # Send error as streaming token
                    error_token = StreamingToken(
                        request_id=request.request_id,
                        token_id=-1,
                        token_index=-1,
                        is_final=True,
                        error=error_msg,
                    )
                    await self._send_streaming_token(error_token)
                response = InferenceResponse(
                    request_id=request.request_id,
                    output_ids=[],
                    error=error_msg,
                )
                await self._send_response(response)
                self.heartbeat.set_status("ready")
    
    async def _send_response(self, response: InferenceResponse):
        """Send inference response to coordinator."""
        url = f"{self.coordinator_url}/inference/response"

        async with self._http_session.post(url, json=response.to_dict()) as resp:
            if resp.status != 200:
                logger.error(f"Failed to send response: {await resp.text()}")

    async def _send_streaming_token(self, token: StreamingToken):
        """Send a streaming token to coordinator."""
        url = f"{self.coordinator_url}/inference/stream/token"

        try:
            async with self._http_session.post(
                url,
                json=token.to_dict(),
                timeout=aiohttp.ClientTimeout(total=2.0)  # Short timeout to avoid blocking
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to send streaming token: {await resp.text()}")
        except asyncio.TimeoutError:
            logger.warning(f"Timeout sending streaming token for request {token.request_id}")
        except Exception as e:
            logger.warning(f"Error sending streaming token: {e}")
    
    async def stop(self):
        """Stop the worker node."""
        logger.info(f"Rank {self.rank} stopping worker node")
        
        self._running = False
        
        if self.heartbeat:
            await self.heartbeat.stop()
        
        if self.peer_mesh:
            await self.peer_mesh.disconnect()
        
        if self._http_session:
            await self._http_session.close()
        
        logger.info(f"Rank {self.rank} worker node stopped")


def main():
    """Entry point for worker node."""
    parser = argparse.ArgumentParser(description="ShardCompute Worker Node")
    parser.add_argument("--rank", type=int, required=True, help="Worker rank")
    parser.add_argument("--coordinator-url", type=str, required=True, help="Coordinator URL")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for peer connections")
    parser.add_argument("--port", type=int, default=9000, help="Port for peer connections")
    parser.add_argument("--shard-dir", type=str, help="Directory containing weight shards")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Create worker
    worker = WorkerNode(
        rank=args.rank,
        coordinator_url=args.coordinator_url,
        host=args.host,
        collective_port=args.port,
        shard_dir=args.shard_dir,
        config_path=args.config,
    )
    
    # Setup signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def shutdown():
        await worker.stop()
        loop.stop()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))
    
    try:
        loop.run_until_complete(worker.start())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
