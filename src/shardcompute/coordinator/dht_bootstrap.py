"""Simple in-memory DHT bootstrap for peer discovery.

This module provides a lightweight DHT implementation that matches the
communication style from COMMUNICATION_OUTLINE.md. It provides initial peers
in multiaddr format: /ip4/{address}/tcp/{port}/p2p/{peer_id}

The DHT is used for peer discovery during worker registration. Workers receive
a list of bootstrap peers when they register, which they can use to connect
to the P2P network for collective operations.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DHTConfig:
    """Configuration for DHT bootstrap."""
    public_host: str = "127.0.0.1"
    dht_port: int = 31337
    stale_timeout_seconds: float = 120.0  # Remove peers after 120s idle
    cleanup_interval_seconds: float = 30.0


@dataclass
class DHTEntry:
    """Entry in the DHT peer registry."""
    peer_id: str
    address: str
    port: int
    rank: int
    last_seen: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


class SimpleDHT:
    """
    Simple in-memory peer registry for DHT bootstrap.

    Provides initial_peers in multiaddr format:
    /ip4/{address}/tcp/{port}/p2p/{peer_id}

    This is a simulated DHT for the MVP. In production, this could be replaced
    with a real hivemind.DHT implementation.
    """

    def __init__(self, config: Optional[DHTConfig] = None):
        """Initialize SimpleDHT.

        Args:
            config: DHT configuration. Uses defaults if not provided.
        """
        self.config = config or DHTConfig()
        self._peers: Dict[str, DHTEntry] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Generate a bootstrap peer ID
        self._bootstrap_peer_id = "QmBootstrap"

        logger.info(
            f"SimpleDHT initialized: {self.config.public_host}:{self.config.dht_port}"
        )

    async def start(self):
        """Start the DHT (cleanup loop)."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SimpleDHT started")

    async def stop(self):
        """Stop the DHT."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("SimpleDHT stopped")

    def generate_peer_id(self, rank: int) -> str:
        """Generate a unique peer ID for a worker.

        Args:
            rank: Worker rank

        Returns:
            Unique peer ID string
        """
        # Generate a deterministic but unique peer ID based on rank
        return f"QmWorker{rank}_{uuid.uuid4().hex[:8]}"

    async def register_peer(
        self,
        peer_id: str,
        address: str,
        port: int,
        rank: int,
        metadata: Optional[Dict] = None,
    ):
        """Register or update a peer in the DHT.

        Args:
            peer_id: Unique peer identifier
            address: IP address of the peer
            port: Port number for P2P communication
            rank: Worker rank
            metadata: Optional metadata about the peer
        """
        async with self._lock:
            self._peers[peer_id] = DHTEntry(
                peer_id=peer_id,
                address=address,
                port=port,
                rank=rank,
                last_seen=time.time(),
                metadata=metadata or {},
            )
            logger.info(f"DHT: Registered peer {peer_id} (rank {rank}) at {address}:{port}")

    async def remove_peer(self, peer_id: str):
        """Remove a peer from the DHT.

        Args:
            peer_id: Peer identifier to remove
        """
        async with self._lock:
            if peer_id in self._peers:
                entry = self._peers[peer_id]
                del self._peers[peer_id]
                logger.info(f"DHT: Removed peer {peer_id} (rank {entry.rank})")

    async def remove_peer_by_rank(self, rank: int):
        """Remove a peer by rank.

        Args:
            rank: Worker rank to remove
        """
        async with self._lock:
            to_remove = [
                peer_id for peer_id, entry in self._peers.items()
                if entry.rank == rank
            ]
            for peer_id in to_remove:
                del self._peers[peer_id]
                logger.info(f"DHT: Removed peer {peer_id} (rank {rank})")

    async def heartbeat(self, peer_id: str):
        """Update last_seen for a peer.

        Args:
            peer_id: Peer identifier
        """
        async with self._lock:
            if peer_id in self._peers:
                self._peers[peer_id].last_seen = time.time()

    async def heartbeat_by_rank(self, rank: int):
        """Update last_seen for a peer by rank.

        Args:
            rank: Worker rank
        """
        async with self._lock:
            for entry in self._peers.values():
                if entry.rank == rank:
                    entry.last_seen = time.time()
                    break

    def get_bootstrap_peer(self) -> str:
        """Get the bootstrap peer multiaddr.

        Returns:
            Bootstrap peer in multiaddr format
        """
        return f"/ip4/{self.config.public_host}/tcp/{self.config.dht_port}/p2p/{self._bootstrap_peer_id}"

    def get_initial_peers(self) -> List[str]:
        """Get all peers as multiaddr strings, including bootstrap.

        Returns:
            List of peer multiaddrs
        """
        peers = [self.get_bootstrap_peer()]

        for entry in self._peers.values():
            multiaddr = f"/ip4/{entry.address}/tcp/{entry.port}/p2p/{entry.peer_id}"
            peers.append(multiaddr)

        return peers

    def get_peer_count(self) -> int:
        """Get number of registered peers (excluding bootstrap).

        Returns:
            Number of registered worker peers
        """
        return len(self._peers)

    def get_peer_by_rank(self, rank: int) -> Optional[DHTEntry]:
        """Get peer entry by rank.

        Args:
            rank: Worker rank

        Returns:
            DHTEntry if found, None otherwise
        """
        for entry in self._peers.values():
            if entry.rank == rank:
                return entry
        return None

    async def _cleanup_loop(self):
        """Periodically clean up stale peers."""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                await self._cleanup_stale()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DHT cleanup error: {e}")

    async def _cleanup_stale(self):
        """Remove peers that haven't been seen recently."""
        current_time = time.time()
        stale = []

        async with self._lock:
            for peer_id, entry in self._peers.items():
                if current_time - entry.last_seen > self.config.stale_timeout_seconds:
                    stale.append(peer_id)

            for peer_id in stale:
                entry = self._peers[peer_id]
                del self._peers[peer_id]
                logger.info(
                    f"DHT: Cleaned up stale peer {peer_id} (rank {entry.rank}, "
                    f"idle for {current_time - entry.last_seen:.0f}s)"
                )
