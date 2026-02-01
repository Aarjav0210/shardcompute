"""All-gather implementation for distributed tensor concatenation."""

import asyncio
import logging
import time
from typing import List, Dict, Optional
import mlx.core as mx

from shardcompute.collectives.point_to_point import PeerConnection
from shardcompute.collectives.topology import Topology, RingTopology, DirectTopology

logger = logging.getLogger(__name__)


class AllGather:
    """
    All-gather implementation for distributed tensor concatenation.
    
    Each worker contributes a partition; all workers end with the
    full tensor containing all partitions concatenated.
    
    For column-parallel linear: each worker has output for different
    columns, all-gather produces complete output.
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
    
    async def all_gather(
        self,
        tensor: mx.array,
        dim: int = -1,
    ) -> mx.array:
        """
        Gather partitions from all workers and concatenate.
        
        Args:
            tensor: Local partition
            dim: Dimension along which to concatenate
            
        Returns:
            Full tensor with all partitions concatenated in rank order
        """
        if self.world_size == 1:
            return tensor
        
        start_time = time.perf_counter()
        
        # Ensure tensor is evaluated
        mx.eval(tensor)
        
        # Normalize dim
        if dim < 0:
            dim = len(tensor.shape) + dim
        
        if self.world_size == 2:
            result = await self._all_gather_two_workers(tensor, dim)
        else:
            result = await self._all_gather_ring(tensor, dim)
        
        self.last_op_time_ms = (time.perf_counter() - start_time) * 1000
        
        return result
    
    async def _all_gather_two_workers(
        self,
        tensor: mx.array,
        dim: int,
    ) -> mx.array:
        """
        Optimized all-gather for exactly 2 workers.
        
        Simple bidirectional exchange followed by concatenation.
        """
        other_rank = 1 - self.rank
        peer = self.peers[other_rank]
        
        # Simultaneous exchange
        other_tensor = await peer.send_recv_tensor(tensor)
        
        # Track bytes
        self.last_bytes_transferred = tensor.nbytes + other_tensor.nbytes
        
        # Concatenate in rank order
        if self.rank == 0:
            return mx.concatenate([tensor, other_tensor], axis=dim)
        else:
            return mx.concatenate([other_tensor, tensor], axis=dim)
    
    async def _all_gather_ring(
        self,
        tensor: mx.array,
        dim: int,
    ) -> mx.array:
        """
        Ring all-gather for N > 2 workers.
        
        Each step, each worker sends what it has to the next
        and receives a new partition from the previous.
        After N-1 steps, all workers have all partitions.
        """
        n = self.world_size
        
        # Initialize partitions array
        partitions: List[Optional[mx.array]] = [None] * n
        partitions[self.rank] = tensor
        
        bytes_transferred = 0
        
        # Current partition to send (starts with own)
        current_send_idx = self.rank
        
        for step in range(n - 1):
            send_peer = self.peers[self.send_to]
            recv_peer = self.peers[self.recv_from]
            
            # Compute index of received partition
            recv_idx = (self.rank - step - 1) % n

            # Send current partition, receive new one simultaneously
            _, partitions[recv_idx] = await asyncio.gather(
                send_peer.send_tensor(partitions[current_send_idx]),
                recv_peer.recv_tensor(),
            )
            
            bytes_transferred += (
                partitions[current_send_idx].nbytes +
                partitions[recv_idx].nbytes
            )
            
            # Next iteration sends what we just received
            current_send_idx = recv_idx
        
        self.last_bytes_transferred = bytes_transferred
        
        # Concatenate in rank order (partitions are already indexed by rank)
        return mx.concatenate(partitions, axis=dim)


class ReduceScatter:
    """
    Reduce-scatter: combines reduce and scatter operations.
    
    Each worker contributes a full tensor, and each ends up
    with 1/N of the reduced result.
    
    This is useful for 2D parallelism where you need to
    reduce across one dimension and scatter across another.
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
        
        self.send_to = (rank + 1) % world_size
        self.recv_from = (rank - 1) % world_size
        
        self.last_op_time_ms: float = 0.0
        self.last_bytes_transferred: int = 0
    
    async def reduce_scatter(
        self,
        tensor: mx.array,
        op: str = "sum",
        dim: int = -1,
    ) -> mx.array:
        """
        Reduce-scatter operation.
        
        Args:
            tensor: Full tensor to reduce-scatter
            op: Reduction operation
            dim: Dimension to scatter along
            
        Returns:
            Local partition of reduced result
        """
        if self.world_size == 1:
            return tensor
        
        start_time = time.perf_counter()
        mx.eval(tensor)
        
        # Normalize dim
        if dim < 0:
            dim = len(tensor.shape) + dim
        
        n = self.world_size
        
        # Split tensor into partitions along dim
        partition_size = tensor.shape[dim] // n
        partitions = []
        for i in range(n):
            start = i * partition_size
            end = start + partition_size
            # Use slicing along the specified dimension
            slices = [slice(None)] * len(tensor.shape)
            slices[dim] = slice(start, end)
            partitions.append(tensor[tuple(slices)])
        
        bytes_transferred = 0
        
        # Reduce-scatter phase (same as phase 1 of ring all-reduce)
        for step in range(n - 1):
            send_idx = (self.rank - step) % n
            recv_idx = (self.rank - step - 1) % n
            
            send_peer = self.peers[self.send_to]
            recv_peer = self.peers[self.recv_from]
            
            _, received = await asyncio.gather(
                send_peer.send_tensor(partitions[send_idx]),
                recv_peer.recv_tensor(),
            )
            
            bytes_transferred += partitions[send_idx].nbytes + received.nbytes
            
            # Reduce
            if op == "sum":
                partitions[recv_idx] = partitions[recv_idx] + received
            elif op == "max":
                partitions[recv_idx] = mx.maximum(partitions[recv_idx], received)
            elif op == "min":
                partitions[recv_idx] = mx.minimum(partitions[recv_idx], received)
            
            mx.eval(partitions[recv_idx])
        
        self.last_op_time_ms = (time.perf_counter() - start_time) * 1000
        self.last_bytes_transferred = bytes_transferred
        
        # Return our partition of the reduced result
        # After reduce-scatter, rank i has the reduced partition i
        my_partition_idx = (self.rank + 1) % n
        return partitions[my_partition_idx]


class AllGatherManager:
    """Manages all-gather operations with tracking and stats."""
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        peers: Dict[int, PeerConnection],
    ):
        self.rank = rank
        self.world_size = world_size
        self.peers = peers
        
        self._all_gather = AllGather(rank, world_size, peers)
        self._reduce_scatter = ReduceScatter(rank, world_size, peers)
    
    async def all_gather(
        self,
        tensor: mx.array,
        dim: int = -1,
    ) -> mx.array:
        """Perform all-gather."""
        return await self._all_gather.all_gather(tensor, dim)
    
    async def reduce_scatter(
        self,
        tensor: mx.array,
        op: str = "sum",
        dim: int = -1,
    ) -> mx.array:
        """Perform reduce-scatter."""
        return await self._reduce_scatter.reduce_scatter(tensor, op, dim)
    
    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        return {
            "all_gather_time_ms": self._all_gather.last_op_time_ms,
            "all_gather_bytes": self._all_gather.last_bytes_transferred,
            "reduce_scatter_time_ms": self._reduce_scatter.last_op_time_ms,
            "reduce_scatter_bytes": self._reduce_scatter.last_bytes_transferred,
        }
