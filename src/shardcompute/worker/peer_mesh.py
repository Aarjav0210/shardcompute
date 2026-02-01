"""Peer mesh for managing connections between workers."""

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from shardcompute.collectives.communicator import Communicator
from shardcompute.collectives.topology import PeerInfo
from shardcompute.protocol.messages import WorkerInfo

logger = logging.getLogger(__name__)


@dataclass
class MeshConfig:
    """Configuration for the peer mesh."""

    world_size: int
    connection_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 2.0
    transport: str = "ws_relay"  # "tcp" or "ws_relay"
    coordinator_ws_url: Optional[str] = None  # Required when transport="ws_relay"


class PeerMesh:
    """
    Manages the mesh of peer connections for tensor parallelism.
    
    Responsibilities:
    - Maintain list of peer workers
    - Coordinate connection establishment
    - Provide communicator for collective operations
    - Handle peer failures and recovery
    
    The mesh is fully connected - each worker has a direct connection
    to every other worker for efficient collective operations.
    """
    
    def __init__(
        self,
        rank: int,
        host: str,
        port: int,
        config: MeshConfig,
    ):
        """
        Initialize PeerMesh.
        
        Args:
            rank: This worker's rank
            host: Host address for incoming connections
            port: Port for incoming connections
            config: Mesh configuration
        """
        self.rank = rank
        self.host = host
        self.port = port
        self.config = config
        
        # Peer information
        self.peers: Dict[int, PeerInfo] = {}
        
        # Communicator (initialized after connections established)
        self.communicator: Optional[Communicator] = None
        
        # State
        self._connected = False
        self._shutting_down = False
    
    @property
    def is_connected(self) -> bool:
        """Check if mesh is fully connected."""
        return self._connected
    
    @property
    def world_size(self) -> int:
        """Get world size from config."""
        return self.config.world_size
    
    def set_peers(self, peer_infos: List[WorkerInfo]):
        """
        Set peer information from coordinator.
        
        Args:
            peer_infos: List of worker info from coordinator
        """
        self.peers.clear()
        
        for info in peer_infos:
            if info.rank != self.rank:
                self.peers[info.rank] = PeerInfo(
                    rank=info.rank,
                    host=info.host,
                    port=info.collective_port,
                )
        
        logger.info(f"Rank {self.rank} set {len(self.peers)} peers")
    
    async def connect(self) -> bool:
        """
        Establish connections to all peers.

        Returns:
            True if all connections successful
        """
        if self._connected:
            logger.warning("Mesh already connected")
            return True

        if self.config.transport == "ws_relay":
            return await self._connect_ws_relay()
        return await self._connect_tcp()

    async def _connect_tcp(self) -> bool:
        """Connect using direct TCP peer-to-peer connections."""
        logger.info(f"Rank {self.rank} connecting to peer mesh (TCP)")

        peer_info_list = [None] * self.config.world_size
        peer_info_list[self.rank] = PeerInfo(
            rank=self.rank,
            host=self.host,
            port=self.port,
        )
        for peer_rank, peer_info in self.peers.items():
            peer_info_list[peer_rank] = peer_info

        for attempt in range(self.config.retry_attempts):
            self.communicator = Communicator(
                rank=self.rank,
                world_size=self.config.world_size,
                host=self.host,
                port=self.port,
            )
            try:
                await self.communicator.initialize(
                    peer_info_list,
                    timeout=self.config.connection_timeout,
                )
                self._connected = True
                logger.info(f"Rank {self.rank} mesh connected successfully (TCP)")
                return True
            except Exception as e:
                logger.warning(
                    f"Rank {self.rank} connection attempt {attempt + 1} failed: {e}"
                )
                await self.communicator.shutdown()
                self.communicator = None
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)

        logger.error(f"Rank {self.rank} failed to connect to mesh (TCP)")
        return False

    async def _connect_ws_relay(self) -> bool:
        """Connect via WebSocket relay through coordinator."""
        if not self.config.coordinator_ws_url:
            raise ValueError("coordinator_ws_url required for ws_relay transport")

        logger.info(f"Rank {self.rank} connecting to peer mesh (WS relay)")

        peer_info_list = [None] * self.config.world_size
        peer_info_list[self.rank] = PeerInfo(
            rank=self.rank,
            host=self.host,
            port=self.port,
        )
        for peer_rank, peer_info in self.peers.items():
            peer_info_list[peer_rank] = peer_info

        for attempt in range(self.config.retry_attempts):
            self.communicator = Communicator(
                rank=self.rank,
                world_size=self.config.world_size,
                host=self.host,
                port=self.port,
            )
            try:
                await self.communicator.initialize_ws_relay(
                    coordinator_ws_url=self.config.coordinator_ws_url,
                    peer_infos=peer_info_list,
                    timeout=self.config.connection_timeout,
                )
                self._connected = True
                logger.info(f"Rank {self.rank} mesh connected successfully (WS relay)")
                return True
            except Exception as e:
                logger.warning(
                    f"Rank {self.rank} WS relay attempt {attempt + 1} failed: {e}"
                )
                await self.communicator.shutdown()
                self.communicator = None
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)

        logger.error(f"Rank {self.rank} failed to connect to mesh (WS relay)")
        return False
    
    async def barrier(self):
        """Synchronize all workers at a barrier."""
        if not self._connected:
            raise RuntimeError("Mesh not connected")
        
        await self.communicator.barrier()
    
    async def disconnect(self):
        """Disconnect from all peers."""
        if self._shutting_down:
            return
        
        self._shutting_down = True
        logger.info(f"Rank {self.rank} disconnecting from mesh")
        
        if self.communicator:
            await self.communicator.shutdown()
        
        self._connected = False
        logger.info(f"Rank {self.rank} mesh disconnected")
    
    def get_communicator(self) -> Communicator:
        """Get the communicator for collective operations."""
        if not self._connected or self.communicator is None:
            raise RuntimeError("Mesh not connected")
        return self.communicator
    
    def get_stats(self) -> Dict:
        """Get mesh statistics."""
        stats = {
            "rank": self.rank,
            "world_size": self.config.world_size,
            "connected": self._connected,
            "num_peers": len(self.peers),
        }
        
        if self.communicator:
            stats["communicator_stats"] = self.communicator.get_stats()
        
        return stats
