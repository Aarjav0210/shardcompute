"""Main communicator interface for collective operations."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import mlx.core as mx

from shardcompute.collectives.point_to_point import PeerConnection, PeerConnectionServer
from shardcompute.collectives.all_reduce import RingAllReduce, AllReduceManager
from shardcompute.collectives.all_gather import AllGather, AllGatherManager, ReduceScatter
from shardcompute.collectives.topology import (
    Topology, RingTopology, DirectTopology, PeerInfo, create_topology
)
from shardcompute.protocol.serialization import TensorSerializer

logger = logging.getLogger(__name__)


@dataclass
class CommunicatorStats:
    """Statistics for communicator operations."""
    
    total_all_reduce_calls: int = 0
    total_all_gather_calls: int = 0
    total_all_reduce_time_ms: float = 0.0
    total_all_gather_time_ms: float = 0.0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_all_reduce_calls": self.total_all_reduce_calls,
            "total_all_gather_calls": self.total_all_gather_calls,
            "total_all_reduce_time_ms": self.total_all_reduce_time_ms,
            "total_all_gather_time_ms": self.total_all_gather_time_ms,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "avg_all_reduce_time_ms": (
                self.total_all_reduce_time_ms / self.total_all_reduce_calls
                if self.total_all_reduce_calls > 0 else 0
            ),
            "avg_all_gather_time_ms": (
                self.total_all_gather_time_ms / self.total_all_gather_calls
                if self.total_all_gather_calls > 0 else 0
            ),
        }


class Communicator:
    """
    Main interface for distributed collective operations.
    
    Provides high-level API for all-reduce, all-gather, and other
    collective operations needed for tensor parallelism.
    
    Usage:
        # Initialize
        comm = Communicator(rank=0, world_size=2)
        await comm.initialize(peer_infos)
        
        # Use collectives
        reduced = await comm.all_reduce(tensor)
        gathered = await comm.all_gather(partition, dim=-1)
        
        # Cleanup
        await comm.shutdown()
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        host: str = "0.0.0.0",
        port: int = 9000,
        topology_type: str = "ring",
    ):
        self.rank = rank
        self.world_size = world_size
        self.host = host
        self.port = port
        self.topology_type = topology_type
        
        # Will be initialized in initialize()
        self.peers: Dict[int, PeerConnection] = {}
        self.topology: Optional[Topology] = None
        self._all_reduce: Optional[RingAllReduce] = None
        self._all_gather: Optional[AllGather] = None
        self._reduce_scatter: Optional[ReduceScatter] = None
        
        # Server for accepting connections
        self._server: Optional[PeerConnectionServer] = None
        
        # Stats
        self.stats = CommunicatorStats()
        
        # State
        self._initialized = False
        self._barrier_counter = 0
    
    @property
    def is_initialized(self) -> bool:
        """Check if communicator is initialized."""
        return self._initialized
    
    async def initialize(
        self,
        peer_infos: List[PeerInfo],
        timeout: float = 30.0,
    ):
        """
        Initialize connections to all peers.
        
        Uses a connection protocol where lower-ranked workers
        connect to higher-ranked workers to avoid deadlock.
        
        Args:
            peer_infos: List of peer information indexed by rank
            timeout: Connection timeout in seconds
        """
        if self._initialized:
            logger.warning("Communicator already initialized")
            return
        
        logger.info(f"Rank {self.rank} initializing communicator with {self.world_size} workers")
        
        # Start server to accept connections from lower-ranked workers
        self._server = PeerConnectionServer(
            rank=self.rank,
            host=self.host,
            port=self.port,
            world_size=self.world_size,
        )
        await self._server.start()
        
        # Connect to higher-ranked workers (we initiate as client)
        # Accept connections from lower-ranked workers (we act as server)
        connect_tasks = []
        
        for other_rank in range(self.world_size):
            if other_rank == self.rank:
                continue
            
            peer_info = peer_infos[other_rank]
            connection = PeerConnection(
                local_rank=self.rank,
                peer_rank=other_rank,
                peer_info=peer_info,
                timeout=timeout,
            )
            self.peers[other_rank] = connection
            
            if other_rank > self.rank:
                # We connect to higher ranks
                connect_tasks.append(self._connect_to_peer(connection, timeout))
            else:
                # We accept from lower ranks
                connect_tasks.append(self._accept_from_peer(connection, timeout))
        
        # Wait for all connections
        await asyncio.gather(*connect_tasks)
        
        # Create topology
        self.topology = create_topology(
            self.topology_type,
            self.rank,
            self.world_size,
            peer_infos,
        )
        
        # Initialize collective operations
        self._all_reduce = RingAllReduce(self.rank, self.world_size, self.peers, self.topology)
        self._all_gather = AllGather(self.rank, self.world_size, self.peers, self.topology)
        self._reduce_scatter = ReduceScatter(self.rank, self.world_size, self.peers)
        
        self._initialized = True
        logger.info(f"Rank {self.rank} communicator initialized successfully")
    
    async def initialize_ws_relay(
        self,
        coordinator_ws_url: str,
        peer_infos: List[PeerInfo],
        timeout: float = 30.0,
    ):
        """
        Initialize connections via WebSocket relay through coordinator.

        Instead of direct TCP, each peer connection is a
        WebSocketRelayConnection that routes traffic through the
        coordinator's /ws/collective/{rank} endpoint.
        """
        if self._initialized:
            logger.warning("Communicator already initialized")
            return

        logger.info(f"Rank {self.rank} initializing communicator via WS relay")

        from shardcompute.collectives.ws_relay import WebSocketRelayManager

        self._relay_manager = WebSocketRelayManager(
            rank=self.rank,
            coordinator_ws_url=coordinator_ws_url,
            world_size=self.world_size,
            timeout=timeout,
        )
        await self._relay_manager.connect()

        # Create relay connections for each peer
        for other_rank in range(self.world_size):
            if other_rank == self.rank:
                continue
            self.peers[other_rank] = self._relay_manager.get_connection(other_rank)

        # Create topology
        self.topology = create_topology(
            self.topology_type,
            self.rank,
            self.world_size,
            peer_infos,
        )

        # Initialize collective operations (same as TCP path)
        self._all_reduce = RingAllReduce(self.rank, self.world_size, self.peers, self.topology)
        self._all_gather = AllGather(self.rank, self.world_size, self.peers, self.topology)
        self._reduce_scatter = ReduceScatter(self.rank, self.world_size, self.peers)

        self._initialized = True
        logger.info(f"Rank {self.rank} communicator initialized via WS relay")

    async def _connect_to_peer(
        self,
        connection: PeerConnection,
        timeout: float,
    ):
        """Connect to a higher-ranked peer."""
        success = await connection.connect()
        if not success:
            raise RuntimeError(f"Failed to connect to rank {connection.peer_rank}")
    
    async def _accept_from_peer(
        self,
        connection: PeerConnection,
        timeout: float,
    ):
        """Accept connection from a lower-ranked peer."""
        reader, writer = await self._server.get_connection(
            connection.peer_rank,
            timeout=timeout,
        )
        # Skip handshake since server already validated it
        await connection.accept(reader, writer, skip_handshake=True)
    
    async def all_reduce(
        self,
        tensor: mx.array,
        op: str = "sum",
    ) -> mx.array:
        """
        Perform all-reduce across all workers.
        
        After this operation, all workers have the same reduced tensor.
        
        Args:
            tensor: Local tensor to reduce
            op: Reduction operation ('sum', 'mean', 'max', 'min')
            
        Returns:
            Reduced tensor (identical on all workers)
        """
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        start_time = time.perf_counter()
        result = await self._all_reduce.all_reduce(tensor, op)
        elapsed = (time.perf_counter() - start_time) * 1000
        
        # Update stats
        self.stats.total_all_reduce_calls += 1
        self.stats.total_all_reduce_time_ms += elapsed
        self.stats.total_bytes_sent += self._all_reduce.last_bytes_transferred // 2
        self.stats.total_bytes_received += self._all_reduce.last_bytes_transferred // 2
        
        return result
    
    async def all_gather(
        self,
        tensor: mx.array,
        dim: int = -1,
    ) -> mx.array:
        """
        Gather partitions from all workers and concatenate.
        
        After this operation, all workers have the full concatenated tensor.
        
        Args:
            tensor: Local partition
            dim: Dimension along which to concatenate
            
        Returns:
            Full tensor with all partitions in rank order
        """
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        start_time = time.perf_counter()
        result = await self._all_gather.all_gather(tensor, dim)
        elapsed = (time.perf_counter() - start_time) * 1000
        
        # Update stats
        self.stats.total_all_gather_calls += 1
        self.stats.total_all_gather_time_ms += elapsed
        self.stats.total_bytes_sent += self._all_gather.last_bytes_transferred // 2
        self.stats.total_bytes_received += self._all_gather.last_bytes_transferred // 2
        
        return result
    
    async def reduce_scatter(
        self,
        tensor: mx.array,
        op: str = "sum",
        dim: int = -1,
    ) -> mx.array:
        """
        Reduce-scatter operation.
        
        Reduces tensor across workers and scatters the result,
        so each worker gets 1/N of the final reduced tensor.
        
        Args:
            tensor: Full tensor to reduce-scatter
            op: Reduction operation
            dim: Dimension to scatter along
            
        Returns:
            Local partition of reduced result
        """
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        return await self._reduce_scatter.reduce_scatter(tensor, op, dim)
    
    async def broadcast(
        self,
        tensor: Optional[mx.array],
        root: int = 0,
    ) -> mx.array:
        """
        Broadcast tensor from root to all workers.
        
        Args:
            tensor: Tensor to broadcast (only used on root)
            root: Rank of the broadcasting worker
            
        Returns:
            Broadcasted tensor (same on all workers)
        """
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        if self.rank == root:
            if tensor is None:
                raise ValueError("Root must provide tensor for broadcast")
            
            # Send to all other workers
            send_tasks = []
            for other_rank, peer in self.peers.items():
                send_tasks.append(peer.send_tensor(tensor))
            await asyncio.gather(*send_tasks)
            return tensor
        else:
            # Receive from root
            return await self.peers[root].recv_tensor()
    
    async def barrier(self):
        """
        Synchronization barrier - all workers must reach this point.
        
        Implemented using all-reduce of a dummy tensor.
        """
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        self._barrier_counter += 1
        dummy = mx.array([self._barrier_counter], dtype=mx.float32)
        await self.all_reduce(dummy, op="sum")
    
    async def send(
        self,
        tensor: mx.array,
        dest: int,
    ):
        """
        Point-to-point send.
        
        Args:
            tensor: Tensor to send
            dest: Destination rank
        """
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        if dest not in self.peers:
            raise ValueError(f"Invalid destination rank: {dest}")
        
        await self.peers[dest].send_tensor(tensor)
    
    async def recv(
        self,
        source: int,
    ) -> mx.array:
        """
        Point-to-point receive.
        
        Args:
            source: Source rank
            
        Returns:
            Received tensor
        """
        if not self._initialized:
            raise RuntimeError("Communicator not initialized")
        
        if source not in self.peers:
            raise ValueError(f"Invalid source rank: {source}")
        
        return await self.peers[source].recv_tensor()
    
    def flush_peers(self):
        """Flush all peer recv queues to discard stale data.

        Call at the start of each inference cycle to prevent
        desynchronization after a failed operation.
        """
        total = 0
        for peer in self.peers.values():
            if hasattr(peer, 'flush'):
                total += peer.flush()
        if total > 0:
            logger.info(f"Rank {self.rank} flushed {total} stale messages from peers")

    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return self.stats.to_dict()
    
    def reset_stats(self):
        """Reset communication statistics."""
        self.stats = CommunicatorStats()
    
    async def shutdown(self):
        """Shutdown communicator and close all connections."""
        logger.info(f"Rank {self.rank} shutting down communicator")

        # Close peer connections
        for peer in self.peers.values():
            await peer.close()
        self.peers.clear()

        # Stop TCP server if used
        if self._server:
            await self._server.stop()

        # Close WS relay manager if used
        if hasattr(self, '_relay_manager') and self._relay_manager:
            await self._relay_manager.close()
            self._relay_manager = None

        self._initialized = False
        logger.info(f"Rank {self.rank} communicator shutdown complete")


class ProcessGroup:
    """
    Process group for subgroup collectives.
    
    Allows collectives within subsets of workers (e.g., row-wise
    or column-wise collectives in 2D parallelism).
    """
    
    def __init__(
        self,
        ranks: List[int],
        local_rank: int,
        comm: Communicator,
    ):
        self.ranks = ranks
        self.local_rank = local_rank
        self.size = len(ranks)
        self.comm = comm
        
        # Map from group rank to global rank
        self.group_to_global = {i: r for i, r in enumerate(ranks)}
        self.global_to_group = {r: i for i, r in enumerate(ranks)}
        
        # My rank within this group
        self.group_rank = self.global_to_group[local_rank]
    
    async def all_reduce(
        self,
        tensor: mx.array,
        op: str = "sum",
    ) -> mx.array:
        """All-reduce within this process group."""
        # For now, use the full communicator
        # A real implementation would have subgroup-aware collectives
        return await self.comm.all_reduce(tensor, op)
    
    async def all_gather(
        self,
        tensor: mx.array,
        dim: int = -1,
    ) -> mx.array:
        """All-gather within this process group."""
        return await self.comm.all_gather(tensor, dim)
