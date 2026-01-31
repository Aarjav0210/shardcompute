"""Topology definitions for collective operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class PeerInfo:
    """Information about a peer in the topology."""
    
    rank: int
    host: str
    port: int
    
    def address(self) -> str:
        """Return host:port string."""
        return f"{self.host}:{self.port}"


class Topology(ABC):
    """Abstract base class for collective operation topologies."""
    
    def __init__(self, rank: int, world_size: int, peers: List[PeerInfo]):
        self.rank = rank
        self.world_size = world_size
        self.peers = peers
    
    @abstractmethod
    def get_send_peers(self, step: int) -> List[int]:
        """Get ranks to send to at given step."""
        pass
    
    @abstractmethod
    def get_recv_peers(self, step: int) -> List[int]:
        """Get ranks to receive from at given step."""
        pass
    
    @abstractmethod
    def num_steps(self) -> int:
        """Number of steps needed for the collective."""
        pass


class RingTopology(Topology):
    """
    Ring topology for collective operations.
    
    Workers are arranged in a logical ring:
    0 -> 1 -> 2 -> ... -> N-1 -> 0
    
    Each step, data flows one hop around the ring.
    Optimal for bandwidth-bound collectives.
    """
    
    def __init__(self, rank: int, world_size: int, peers: List[PeerInfo]):
        super().__init__(rank, world_size, peers)
        # Ring neighbors
        self.next_rank = (rank + 1) % world_size
        self.prev_rank = (rank - 1) % world_size
    
    def get_send_peers(self, step: int) -> List[int]:
        """In ring, always send to next."""
        return [self.next_rank]
    
    def get_recv_peers(self, step: int) -> List[int]:
        """In ring, always receive from prev."""
        return [self.prev_rank]
    
    def num_steps(self) -> int:
        """Ring needs world_size - 1 steps for reduce-scatter or all-gather."""
        return self.world_size - 1


class DirectTopology(Topology):
    """
    Direct topology for 2-worker case.
    
    Workers exchange directly with each other.
    Optimal for small world sizes.
    """
    
    def __init__(self, rank: int, world_size: int, peers: List[PeerInfo]):
        if world_size != 2:
            raise ValueError("DirectTopology only supports world_size=2")
        super().__init__(rank, world_size, peers)
        self.other_rank = 1 - rank
    
    def get_send_peers(self, step: int) -> List[int]:
        """Send to the other worker."""
        return [self.other_rank]
    
    def get_recv_peers(self, step: int) -> List[int]:
        """Receive from the other worker."""
        return [self.other_rank]
    
    def num_steps(self) -> int:
        """Direct exchange completes in 1 step."""
        return 1


class TreeTopology(Topology):
    """
    Binary tree topology for collective operations.
    
    Optimal for latency-bound collectives with small messages.
    Reduce: leaves -> root in log2(N) steps
    Broadcast: root -> leaves in log2(N) steps
    """
    
    def __init__(self, rank: int, world_size: int, peers: List[PeerInfo]):
        super().__init__(rank, world_size, peers)
        self._compute_tree_structure()
    
    def _compute_tree_structure(self):
        """Compute parent and children in binary tree."""
        if self.rank == 0:
            self.parent = None
        else:
            self.parent = (self.rank - 1) // 2
        
        self.children = []
        left = 2 * self.rank + 1
        right = 2 * self.rank + 2
        
        if left < self.world_size:
            self.children.append(left)
        if right < self.world_size:
            self.children.append(right)
        
        # Compute tree depth
        self.depth = 0
        n = self.world_size
        while n > 1:
            n = (n + 1) // 2
            self.depth += 1
    
    def get_send_peers(self, step: int) -> List[int]:
        """
        For reduce: send to parent after receiving from children.
        For broadcast: send to children after receiving from parent.
        """
        # Simplified: return parent for reduce phase
        if self.parent is not None:
            return [self.parent]
        return []
    
    def get_recv_peers(self, step: int) -> List[int]:
        """Receive from children (reduce) or parent (broadcast)."""
        return self.children
    
    def num_steps(self) -> int:
        """Tree depth determines number of steps."""
        return self.depth


class MeshTopology2D(Topology):
    """
    2D mesh topology for 2D tensor parallelism.
    
    Workers arranged in a grid, with row-wise and column-wise
    collectives possible within subgroups.
    
    For 4 workers in 2x2 grid:
    (0,0)  (0,1)
    (1,0)  (1,1)
    
    Row 0: ranks 0, 1
    Row 1: ranks 2, 3
    Col 0: ranks 0, 2
    Col 1: ranks 1, 3
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        peers: List[PeerInfo],
        rows: int,
        cols: int,
    ):
        if rows * cols != world_size:
            raise ValueError(f"rows * cols ({rows * cols}) != world_size ({world_size})")
        
        super().__init__(rank, world_size, peers)
        self.rows = rows
        self.cols = cols
        
        # Compute 2D coordinates
        self.row_idx = rank // cols
        self.col_idx = rank % cols
        
        # Compute row and column groups
        self._compute_groups()
    
    def _compute_groups(self):
        """Compute row and column peer groups."""
        # Row group: all ranks in same row
        row_start = self.row_idx * self.cols
        self.row_peers = [row_start + c for c in range(self.cols)]
        
        # Column group: all ranks in same column
        self.col_peers = [r * self.cols + self.col_idx for r in range(self.rows)]
        
        # Ring neighbors within row
        row_rank = self.col_idx  # Position within row
        self.row_next = self.row_peers[(row_rank + 1) % self.cols]
        self.row_prev = self.row_peers[(row_rank - 1) % self.cols]
        
        # Ring neighbors within column
        col_rank = self.row_idx  # Position within column
        self.col_next = self.col_peers[(col_rank + 1) % self.rows]
        self.col_prev = self.col_peers[(col_rank - 1) % self.rows]
    
    def get_row_ring_neighbors(self) -> Tuple[int, int]:
        """Get next and prev ranks in row-wise ring."""
        return (self.row_next, self.row_prev)
    
    def get_col_ring_neighbors(self) -> Tuple[int, int]:
        """Get next and prev ranks in column-wise ring."""
        return (self.col_next, self.col_prev)
    
    def get_send_peers(self, step: int) -> List[int]:
        """Default to row-wise ring."""
        return [self.row_next]
    
    def get_recv_peers(self, step: int) -> List[int]:
        """Default to row-wise ring."""
        return [self.row_prev]
    
    def num_steps(self) -> int:
        """Row-wise steps."""
        return self.cols - 1
    
    def get_coordinates(self) -> Tuple[int, int]:
        """Get (row, col) coordinates."""
        return (self.row_idx, self.col_idx)


def create_topology(
    topology_type: str,
    rank: int,
    world_size: int,
    peers: List[PeerInfo],
    **kwargs,
) -> Topology:
    """
    Factory function to create topology.
    
    Args:
        topology_type: "ring", "direct", "tree", or "mesh2d"
        rank: This worker's rank
        world_size: Total number of workers
        peers: List of peer info
        **kwargs: Additional args for specific topologies
        
    Returns:
        Topology instance
    """
    if topology_type == "direct" and world_size == 2:
        return DirectTopology(rank, world_size, peers)
    elif topology_type == "ring":
        return RingTopology(rank, world_size, peers)
    elif topology_type == "tree":
        return TreeTopology(rank, world_size, peers)
    elif topology_type == "mesh2d":
        rows = kwargs.get("rows", int(world_size ** 0.5))
        cols = kwargs.get("cols", world_size // rows)
        return MeshTopology2D(rank, world_size, peers, rows, cols)
    else:
        # Default to ring for general case, direct for 2 workers
        if world_size == 2:
            return DirectTopology(rank, world_size, peers)
        return RingTopology(rank, world_size, peers)
