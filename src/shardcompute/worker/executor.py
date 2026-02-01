"""Parallel execution engine for distributed inference."""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Any, Tuple, Callable, Awaitable
from dataclasses import dataclass, field
import mlx.core as mx

from shardcompute.collectives.communicator import Communicator
from shardcompute.parallel.transformer import ParallelTransformer, PipelineParallelTransformer

# Type alias for token callback: (token_id, token_index, is_final) -> None
TokenCallback = Callable[[int, int, bool], Awaitable[None]]

logger = logging.getLogger(__name__)


@dataclass
class LayerTiming:
    """Timing information for a single layer."""
    
    layer_idx: int
    layer_type: str  # "embedding", "transformer", "lm_head"
    compute_ms: float
    comm_ms: float = 0.0
    
    @property
    def total_ms(self) -> float:
        return self.compute_ms + self.comm_ms


@dataclass
class ExecutionMetrics:
    """Metrics from a single forward pass."""
    
    total_time_ms: float
    layer_timings: List[LayerTiming]
    input_tokens: int
    output_shape: List[int]
    
    @property
    def compute_time_ms(self) -> float:
        return sum(t.compute_ms for t in self.layer_timings)
    
    @property
    def comm_time_ms(self) -> float:
        return sum(t.comm_ms for t in self.layer_timings)
    
    @property
    def compute_fraction(self) -> float:
        total = self.compute_time_ms + self.comm_time_ms
        return self.compute_time_ms / total if total > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_time_ms": self.total_time_ms,
            "compute_time_ms": self.compute_time_ms,
            "comm_time_ms": self.comm_time_ms,
            "compute_fraction": self.compute_fraction,
            "input_tokens": self.input_tokens,
            "output_shape": self.output_shape,
            "num_layers": len([t for t in self.layer_timings if t.layer_type == "transformer"]),
        }


class ParallelExecutor:
    """
    Executes parallel inference across distributed workers.
    
    All workers run the same code path, differing only in which
    weight shards they hold. Synchronization happens via collectives.
    
    Key responsibilities:
    - Coordinate forward pass across workers
    - Handle input broadcast from worker 0
    - Collect timing metrics
    - Support generation with KV cache
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        communicator: Communicator,
        model: ParallelTransformer,
    ):
        """
        Initialize ParallelExecutor.
        
        Args:
            rank: This worker's rank
            world_size: Total number of workers
            communicator: Communicator for collective operations
            model: Parallel transformer model
        """
        self.rank = rank
        self.world_size = world_size
        self.communicator = communicator
        self.model = model
        
        # Metrics from last execution
        self.last_metrics: Optional[ExecutionMetrics] = None
        
        # Generation state
        self._past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None
    
    async def forward(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        use_cache: bool = False,
    ) -> mx.array:
        """
        Execute full forward pass with tensor parallelism.
        
        Args:
            input_ids: Token IDs [batch, seq_len] - must be identical on all workers
            attention_mask: Optional attention mask
            use_cache: Whether to cache KV for generation
            
        Returns:
            Logits [batch, seq_len, vocab_size] - replicated on all workers
        """
        timings = []
        total_start = time.perf_counter()
        
        # Forward through model
        logits, cache = await self.model.forward(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=self._past_key_values,
            use_cache=use_cache,
        )
        
        # Update cache
        if use_cache:
            self._past_key_values = cache
        
        # Force evaluation
        mx.eval(logits)
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        # Create metrics
        self.last_metrics = ExecutionMetrics(
            total_time_ms=total_time,
            layer_timings=timings,
            input_tokens=input_ids.size,
            output_shape=list(logits.shape),
        )
        
        return logits
    
    async def forward_with_timing(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, ExecutionMetrics]:
        """
        Execute forward pass with detailed timing.
        
        Returns:
            (logits, metrics) tuple
        """
        timings = []
        total_start = time.perf_counter()
        
        # Embedding
        t0 = time.perf_counter()
        hidden_states = await self.model.embed_tokens.forward(input_ids)
        mx.eval(hidden_states)
        t1 = time.perf_counter()
        
        timings.append(LayerTiming(
            layer_idx=-1,
            layer_type="embedding",
            compute_ms=(t1 - t0) * 1000,
        ))
        
        # Transformer layers
        for idx, layer in enumerate(self.model.layers):
            t0 = time.perf_counter()
            hidden_states, _ = await layer.forward(hidden_states, attention_mask)
            mx.eval(hidden_states)
            t1 = time.perf_counter()
            
            # Estimate communication time from communicator stats
            comm_stats = self.communicator.get_stats()
            
            timings.append(LayerTiming(
                layer_idx=idx,
                layer_type="transformer",
                compute_ms=(t1 - t0) * 1000,
            ))
        
        # Final norm and LM head
        t0 = time.perf_counter()
        hidden_states = self.model.norm(hidden_states)
        
        if self.model.tie_word_embeddings:
            logits = hidden_states @ self.model.embed_tokens.weight.T
            logits = await self.communicator.all_gather(logits, dim=-1)
        else:
            logits = await self.model.lm_head.forward(hidden_states)
        
        mx.eval(logits)
        t1 = time.perf_counter()
        
        timings.append(LayerTiming(
            layer_idx=-2,
            layer_type="lm_head",
            compute_ms=(t1 - t0) * 1000,
        ))
        
        total_time = (time.perf_counter() - total_start) * 1000
        
        metrics = ExecutionMetrics(
            total_time_ms=total_time,
            layer_timings=timings,
            input_tokens=input_ids.size,
            output_shape=list(logits.shape),
        )
        
        self.last_metrics = metrics
        
        return logits, metrics
    
    async def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_tokens: Optional[List[int]] = None,
        token_callback: Optional[TokenCallback] = None,
    ) -> mx.array:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Initial token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stop_tokens: Token IDs that stop generation
            token_callback: Optional async callback called for each generated token
                           with (token_id, token_index, is_final)

        Returns:
            Generated token IDs [batch, seq_len + generated]
        """
        if stop_tokens is None:
            stop_tokens = []

        # Track total generation time
        generation_start = time.perf_counter()

        # Reset cache
        self._past_key_values = None

        # Process prompt
        logits = await self.forward(input_ids, use_cache=True)

        generated = []

        for token_idx in range(max_new_tokens):
            # Get next token logits (last position)
            next_logits = logits[:, -1, :]  # [batch, vocab]

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature

            # Sample
            if temperature == 0:
                # Greedy
                next_token = mx.argmax(next_logits, axis=-1)
            else:
                # Top-p sampling
                probs = mx.softmax(next_logits, axis=-1)

                # Sort probabilities
                sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]
                sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

                # Compute cumulative probs
                cumsum = mx.cumsum(sorted_probs, axis=-1)

                # Mask tokens above top_p
                mask = cumsum <= top_p
                mask = mx.concatenate([
                    mx.ones((mask.shape[0], 1), dtype=mx.bool_),
                    mask[:, :-1]
                ], axis=-1)

                masked_probs = mx.where(mask, sorted_probs, mx.zeros_like(sorted_probs))
                masked_probs = masked_probs / mx.sum(masked_probs, axis=-1, keepdims=True)

                # Sample from masked distribution
                next_token_idx = mx.random.categorical(mx.log(masked_probs + 1e-10))
                next_token = mx.take_along_axis(
                    sorted_indices,
                    next_token_idx[:, None],
                    axis=-1
                )[:, 0]

            mx.eval(next_token)
            token_id = int(next_token[0])
            generated.append(next_token)

            # Check for stop token
            is_stop = token_id in stop_tokens
            is_last = is_stop or (token_idx == max_new_tokens - 1)

            # Call the token callback if provided (rank 0 only sends tokens)
            if token_callback is not None:
                await token_callback(token_id, token_idx, is_last)

            if is_stop:
                break

            # Forward with cache
            next_input = next_token[:, None]  # [batch, 1]
            logits = await self.forward(next_input, use_cache=True)

        # Calculate total generation time
        generation_time = (time.perf_counter() - generation_start) * 1000  # ms

        # Update last_metrics with generation summary
        num_generated = len(generated)
        if num_generated > 0:
            self.last_metrics = ExecutionMetrics(
                total_time_ms=generation_time,
                layer_timings=[],
                input_tokens=num_generated,  # For metrics, this represents tokens generated
                output_shape=[1, num_generated],
            )

        # Concatenate generated tokens
        if generated:
            generated_ids = mx.stack(generated, axis=1)
            output_ids = mx.concatenate([input_ids, generated_ids], axis=1)
        else:
            output_ids = input_ids

        return output_ids
    
    def reset_cache(self):
        """Reset KV cache for new sequence."""
        self._past_key_values = None
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics from last execution."""
        if self.last_metrics:
            return self.last_metrics.to_dict()
        return None
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """Get summary of timing metrics."""
        if not self.last_metrics:
            return {}
        
        return {
            "total_ms": self.last_metrics.total_time_ms,
            "compute_ms": self.last_metrics.compute_time_ms,
            "comm_ms": self.last_metrics.comm_time_ms,
            "compute_fraction": self.last_metrics.compute_fraction,
            "tokens_per_second": (
                self.last_metrics.input_tokens /
                (self.last_metrics.total_time_ms / 1000)
            ),
        }


class PipelineExecutor:
    """
    Execution engine for pipeline parallelism.

    Instead of all workers running the same layers with partial weights,
    each worker runs different layers with full weights. Hidden states
    are passed between stages via point-to-point communication.

    Communication pattern per forward pass:
    - Stage 0: embedding + layers → send hidden states to stage 1
    - Stage k (middle): recv from k-1 → layers → send to k+1
    - Stage N-1 (last): recv from N-2 → layers + norm + lm_head → send logits to stage 0

    Total: (world_size - 1) point-to-point transfers per forward pass
    Compare with TP: 2 * num_layers all-reduces per forward pass
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        communicator: Communicator,
        model: PipelineParallelTransformer,
    ):
        self.rank = rank
        self.world_size = world_size
        self.communicator = communicator
        self.model = model

        self.last_metrics: Optional[ExecutionMetrics] = None
        self._past_key_values: Optional[List[Tuple[mx.array, mx.array]]] = None

        self._is_first_stage = model.has_embedding
        self._is_last_stage = model.has_lm_head

    async def forward(
        self,
        input_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        use_cache: bool = False,
    ) -> Optional[mx.array]:
        """
        Execute forward pass for this pipeline stage.

        First stage provides input_ids.
        Middle/last stages receive hidden states from the previous stage.
        Last stage returns logits and sends them back to stage 0.
        Non-last stages return None.

        Args:
            input_ids: Token IDs [batch, seq_len] — only used on first stage
            attention_mask: Optional attention mask
            use_cache: Whether to cache KV for generation

        Returns:
            logits on first stage (forwarded from last stage), None on others
        """
        total_start = time.perf_counter()

        hidden_states = None

        # Receive hidden states from previous stage
        if not self._is_first_stage:
            hidden_states = await self.communicator.recv(source=self.rank - 1)

        # Run local layers
        output, cache = await self.model.forward(
            input_ids=input_ids if self._is_first_stage else None,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=self._past_key_values,
            use_cache=use_cache,
        )

        if use_cache:
            self._past_key_values = cache

        mx.eval(output)

        # Send output to next stage or send logits back to stage 0
        if self._is_last_stage:
            # Last stage: output is logits — send to stage 0
            if self.rank != 0:
                await self.communicator.send(output, dest=0)
            logits = output
        else:
            # Send hidden states to next stage
            await self.communicator.send(output, dest=self.rank + 1)
            logits = None

        # First stage receives logits from last stage
        if self._is_first_stage and not self._is_last_stage:
            logits = await self.communicator.recv(source=self.world_size - 1)

        total_time = (time.perf_counter() - total_start) * 1000

        self.last_metrics = ExecutionMetrics(
            total_time_ms=total_time,
            layer_timings=[],
            input_tokens=input_ids.size if input_ids is not None else 0,
            output_shape=list(logits.shape) if logits is not None else [],
        )

        return logits

    async def generate(
        self,
        input_ids: mx.array,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_tokens: Optional[List[int]] = None,
        token_callback: Optional[TokenCallback] = None,
    ) -> mx.array:
        """
        Generate tokens autoregressively with pipeline parallelism.

        Only the first stage provides input_ids and performs sampling.
        All stages participate in the forward pass for each token.

        Args:
            input_ids: Initial token IDs [batch, seq_len] (used on all stages
                       for the initial broadcast, but only first stage provides
                       them to the model)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stop_tokens: Token IDs that stop generation
            token_callback: Async callback for each generated token

        Returns:
            Generated token IDs [batch, seq_len + generated] (on first stage)
            input_ids unchanged (on other stages)
        """
        if stop_tokens is None:
            stop_tokens = []

        generation_start = time.perf_counter()
        self._past_key_values = None

        # Prefill: process the full prompt
        logits = await self.forward(
            input_ids=input_ids if self._is_first_stage else input_ids,
            use_cache=True,
        )

        generated = []

        for token_idx in range(max_new_tokens):
            # Sampling happens on first stage; result is broadcast to all
            if self._is_first_stage and logits is not None:
                next_token = self._sample(logits, temperature, top_p)
                mx.eval(next_token)
                token_id = int(next_token[0])
            else:
                next_token = None
                token_id = -1

            # Broadcast the sampled token to all stages
            if self._is_first_stage:
                await self.communicator.broadcast(next_token, root=0)
            else:
                next_token = await self.communicator.broadcast(None, root=0)
                token_id = int(next_token[0])

            generated.append(next_token)

            # Check stop
            is_stop = token_id in stop_tokens
            is_last = is_stop or (token_idx == max_new_tokens - 1)

            if token_callback is not None and self._is_first_stage:
                await token_callback(token_id, token_idx, is_last)

            if is_stop:
                break

            # Forward with cache (single token)
            next_input = next_token[:, None]  # [batch, 1]
            logits = await self.forward(
                input_ids=next_input if self._is_first_stage else next_input,
                use_cache=True,
            )

        generation_time = (time.perf_counter() - generation_start) * 1000
        num_generated = len(generated)

        if num_generated > 0:
            self.last_metrics = ExecutionMetrics(
                total_time_ms=generation_time,
                layer_timings=[],
                input_tokens=num_generated,
                output_shape=[1, num_generated],
            )

        # Concatenate generated tokens (meaningful on first stage)
        if generated:
            generated_ids = mx.stack(generated, axis=1)
            output_ids = mx.concatenate([input_ids, generated_ids], axis=1)
        else:
            output_ids = input_ids

        return output_ids

    def _sample(
        self,
        logits: mx.array,
        temperature: float,
        top_p: float,
    ) -> mx.array:
        """Sample next token from logits."""
        next_logits = logits[:, -1, :]

        if temperature > 0:
            next_logits = next_logits / temperature

        if temperature == 0:
            return mx.argmax(next_logits, axis=-1)

        probs = mx.softmax(next_logits, axis=-1)
        sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumsum = mx.cumsum(sorted_probs, axis=-1)

        mask = cumsum <= top_p
        mask = mx.concatenate([
            mx.ones((mask.shape[0], 1), dtype=mx.bool_),
            mask[:, :-1],
        ], axis=-1)

        masked_probs = mx.where(mask, sorted_probs, mx.zeros_like(sorted_probs))
        masked_probs = masked_probs / mx.sum(masked_probs, axis=-1, keepdims=True)

        next_token_idx = mx.random.categorical(mx.log(masked_probs + 1e-10))
        next_token = mx.take_along_axis(
            sorted_indices, next_token_idx[:, None], axis=-1
        )[:, 0]

        return next_token

    def reset_cache(self):
        """Reset KV cache for new sequence."""
        self._past_key_values = None

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        if self.last_metrics:
            return self.last_metrics.to_dict()
        return None

    def get_timing_summary(self) -> Dict[str, Any]:
        if not self.last_metrics:
            return {}
        return {
            "total_ms": self.last_metrics.total_time_ms,
            "compute_ms": self.last_metrics.compute_time_ms,
            "comm_ms": self.last_metrics.comm_time_ms,
            "compute_fraction": self.last_metrics.compute_fraction,
            "tokens_per_second": (
                self.last_metrics.input_tokens
                / (self.last_metrics.total_time_ms / 1000)
            ) if self.last_metrics.total_time_ms > 0 else 0,
        }
