"""Ring all-reduce implementation for distributed tensor summation."""

import asyncio
import logging
import time
from typing import List, Optional, Dict
import mlx.core as mx

from shardcompute.collectives.point_to_point import PeerConnection
from shardcompute.collectives.topology import Topology, RingTopology, DirectTopology

logger = logging.getLogger(__name__)


class RingAllReduce:
    """
    Ring all-reduce implementation for distributed tensor summation.
    
    For 2 workers, simplifies to direct exchange.
    For N workers, uses ring algorithm to minimize peak bandwidth.
    
    The ring algorithm works in two phases:
    1. Reduce-scatter: Each worker ends with 1/N of the final result
    2. All-gather: Share reduced chunks so all workers have full result
    
    Total data transferred: 2 * (N-1) / N * tensor_size
    This is bandwidth-optimal for large tensors.
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        peers: Dict[int, PeerConnection],
        topology: Optional[Topology] = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.peers = peers
        
        # Create topology if not provided
        if topology is None:
            if world_size == 2:
                self.topology = DirectTopology(rank, world_size, [])
            else:
                self.topology = RingTopology(rank, world_size, [])
        else:
            self.topology = topology
        
        # Ring neighbors
        self.send_to = (rank + 1) % world_size
        self.recv_from = (rank - 1) % world_size
        
        # Timing stats
        self.last_op_time_ms: float = 0.0
        self.last_bytes_transferred: int = 0
    
    async def all_reduce(
        self,
        tensor: mx.array,
        op: str = "sum",
    ) -> mx.array:
        """
        Perform all-reduce across all workers.
        
        Args:
            tensor: Local tensor to reduce
            op: Reduction operation ('sum', 'mean', 'max', 'min')
            
        Returns:
            Reduced tensor (identical on all workers)
        """
        if self.world_size == 1:
            return tensor
        
        start_time = time.perf_counter()
        
        # Ensure tensor is evaluated before network operations
        mx.eval(tensor)
        
        if self.world_size == 2:
            result = await self._all_reduce_two_workers(tensor, op)
        else:
            result = await self._all_reduce_ring(tensor, op)
        
        self.last_op_time_ms = (time.perf_counter() - start_time) * 1000
        
        return result
    
    async def _all_reduce_two_workers(
        self,
        tensor: mx.array,
        op: str,
    ) -> mx.array:
        """
        Optimized all-reduce for exactly 2 workers.
        
        Simple bidirectional exchange followed by local reduction.
        """
        other_rank = 1 - self.rank
        peer = self.peers[other_rank]
        
        # Simultaneous send and receive
        other_tensor = await peer.send_recv_tensor(tensor)
        
        # Track bytes
        self.last_bytes_transferred = tensor.nbytes * 2
        
        # Reduce
        return self._reduce_op(tensor, other_tensor, op)
    
    async def _all_reduce_ring(
        self,
        tensor: mx.array,
        op: str,
    ) -> mx.array:
        """
        Ring all-reduce for N > 2 workers.
        
        Phase 1: Reduce-scatter
        - Divide tensor into N chunks
        - Each step: send chunk[i] to next, receive from prev and reduce into chunk[i-1]
        - After N-1 steps: each worker has 1/N of final reduced result
        
        Phase 2: All-gather
        - Each step: send reduced chunk to next, receive from prev
        - After N-1 steps: all workers have full reduced result
        """
        n = self.world_size
        
        # Flatten tensor for chunking
        original_shape = tensor.shape
        flat = tensor.reshape(-1)
        
        # Pad to be divisible by world_size
        padded_size = ((flat.shape[0] + n - 1) // n) * n
        if flat.shape[0] < padded_size:
            padding = mx.zeros((padded_size - flat.shape[0],), dtype=tensor.dtype)
            flat = mx.concatenate([flat, padding])
        
        # Chunk the tensor
        chunk_size = padded_size // n
        chunks = [flat[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]
        
        bytes_transferred = 0
        
        # Phase 1: Reduce-scatter
        for step in range(n - 1):
            # Determine which chunk to send/receive
            send_idx = (self.rank - step) % n
            recv_idx = (self.rank - step - 1) % n
            
            # Exchange with neighbors
            send_peer = self.peers[self.send_to]
            recv_peer = self.peers[self.recv_from]
            
            # Send and receive simultaneously
            _, received = await asyncio.gather(
                send_peer.send_tensor(chunks[send_idx]),
                recv_peer.recv_tensor(),
            )
            
            bytes_transferred += chunks[send_idx].nbytes + received.nbytes
            
            # Reduce into local chunk
            chunks[recv_idx] = self._reduce_op(chunks[recv_idx], received, op)
            mx.eval(chunks[recv_idx])
        
        # Phase 2: All-gather
        for step in range(n - 1):
            send_idx = (self.rank - step + 1) % n
            recv_idx = (self.rank - step) % n
            
            send_peer = self.peers[self.send_to]
            recv_peer = self.peers[self.recv_from]
            
            # Exchange
            _, chunks[recv_idx] = await asyncio.gather(
                send_peer.send_tensor(chunks[send_idx]),
                recv_peer.recv_tensor(),
            )
            
            bytes_transferred += chunks[send_idx].nbytes + chunks[recv_idx].nbytes
        
        self.last_bytes_transferred = bytes_transferred
        
        # Reassemble tensor
        result = mx.concatenate(chunks, axis=0)
        
        # Remove padding and reshape
        result = result[:tensor.size].reshape(original_shape)
        
        return result
    
    def _reduce_op(
        self,
        a: mx.array,
        b: mx.array,
        op: str,
    ) -> mx.array:
        """Apply reduction operation."""
        if op == "sum":
            return a + b
        elif op == "mean":
            return (a + b) / 2
        elif op == "max":
            return mx.maximum(a, b)
        elif op == "min":
            return mx.minimum(a, b)
        else:
            raise ValueError(f"Unknown reduction op: {op}")


class AllReduceManager:
    """
    Manages multiple concurrent all-reduce operations.
    
    Provides operation IDs for tracking and supports
    overlapping communication with computation.
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        peers: Dict[int, PeerConnection],
    ):
        self.rank = rank
        self.world_size = world_size
        self.peers = peers
        
        self._all_reduce = RingAllReduce(rank, world_size, peers)
        self._pending_ops: Dict[str, asyncio.Task] = {}
        self._op_counter = 0
    
    async def all_reduce(
        self,
        tensor: mx.array,
        op: str = "sum",
        op_id: Optional[str] = None,
    ) -> mx.array:
        """
        Perform all-reduce with optional operation ID.
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation
            op_id: Optional identifier for tracking
            
        Returns:
            Reduced tensor
        """
        return await self._all_reduce.all_reduce(tensor, op)
    
    def start_all_reduce(
        self,
        tensor: mx.array,
        op: str = "sum",
    ) -> str:
        """
        Start an async all-reduce and return operation ID.
        
        Use wait_all_reduce() to get the result.
        """
        self._op_counter += 1
        op_id = f"ar_{self.rank}_{self._op_counter}"
        
        task = asyncio.create_task(self._all_reduce.all_reduce(tensor, op))
        self._pending_ops[op_id] = task
        
        return op_id
    
    async def wait_all_reduce(self, op_id: str) -> mx.array:
        """Wait for a pending all-reduce to complete."""
        if op_id not in self._pending_ops:
            raise ValueError(f"Unknown operation ID: {op_id}")
        
        result = await self._pending_ops[op_id]
        del self._pending_ops[op_id]
        return result
    
    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics from last operation."""
        return {
            "last_op_time_ms": self._all_reduce.last_op_time_ms,
            "last_bytes_transferred": self._all_reduce.last_bytes_transferred,
        }
