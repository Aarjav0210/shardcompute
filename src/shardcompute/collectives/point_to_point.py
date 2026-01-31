"""Point-to-point communication primitives over TCP."""

import asyncio
import struct
import logging
from typing import Optional, Dict, Tuple
import mlx.core as mx

from shardcompute.protocol.serialization import TensorSerializer
from shardcompute.collectives.topology import PeerInfo

logger = logging.getLogger(__name__)


class PeerConnection:
    """
    Manages a TCP connection to a peer worker for tensor transfer.
    
    Supports both client mode (connects to peer) and server mode
    (accepts connection from peer).
    """
    
    HEADER_SIZE = 8  # 4 bytes type + 4 bytes length
    MSG_TYPE_TENSOR = 1
    MSG_TYPE_CONTROL = 2
    
    def __init__(
        self,
        local_rank: int,
        peer_rank: int,
        peer_info: Optional[PeerInfo] = None,
        buffer_size: int = 1024 * 1024,  # 1MB buffer
        timeout: float = 30.0,
    ):
        self.local_rank = local_rank
        self.peer_rank = peer_rank
        self.peer_info = peer_info
        self.buffer_size = buffer_size
        self.timeout = timeout
        
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.serializer = TensorSerializer()
        
        self._connected = False
        self._lock = asyncio.Lock()
        
    @property
    def is_connected(self) -> bool:
        """Check if connection is established."""
        return self._connected and self.writer is not None
    
    async def connect(self) -> bool:
        """
        Connect to peer as client.
        
        Lower rank connects to higher rank to avoid deadlock.
        """
        if self.peer_info is None:
            raise ValueError("peer_info required for client connection")
        
        try:
            logger.info(f"Rank {self.local_rank} connecting to rank {self.peer_rank} "
                       f"at {self.peer_info.host}:{self.peer_info.port}")
            
            self.reader, self.writer = await asyncio.wait_for(
                asyncio.open_connection(self.peer_info.host, self.peer_info.port),
                timeout=self.timeout,
            )
            
            # Send handshake
            await self._send_handshake()
            
            self._connected = True
            logger.info(f"Rank {self.local_rank} connected to rank {self.peer_rank}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to rank {self.peer_rank}: {e}")
            return False
    
    async def accept(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        skip_handshake: bool = False,
    ):
        """
        Accept connection from peer as server.
        
        Called by server accept handler.
        
        Args:
            reader: Stream reader
            writer: Stream writer
            skip_handshake: If True, skip handshake (already validated by server)
        """
        self.reader = reader
        self.writer = writer
        
        # Receive handshake only if not already validated
        if not skip_handshake:
            await self._recv_handshake()
        
        self._connected = True
        logger.info(f"Rank {self.local_rank} accepted connection from rank {self.peer_rank}")
    
    async def _send_handshake(self):
        """Send handshake to identify ourselves."""
        handshake = struct.pack(">II", self.local_rank, self.peer_rank)
        self.writer.write(handshake)
        await self.writer.drain()
    
    async def _recv_handshake(self):
        """Receive and validate handshake."""
        handshake = await self.reader.readexactly(8)
        sender_rank, target_rank = struct.unpack(">II", handshake)
        
        if sender_rank != self.peer_rank or target_rank != self.local_rank:
            raise ValueError(f"Handshake mismatch: expected from {self.peer_rank}, "
                           f"got from {sender_rank}")
    
    async def send_tensor(self, tensor: mx.array) -> int:
        """
        Send a tensor to the peer.
        
        Args:
            tensor: MLX array to send
            
        Returns:
            Number of bytes sent
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to peer")
        
        async with self._lock:
            # Serialize tensor
            data = self.serializer.serialize(tensor)
            
            # Send header
            header = struct.pack(">II", self.MSG_TYPE_TENSOR, len(data))
            self.writer.write(header)
            
            # Send data
            self.writer.write(data)
            await self.writer.drain()
            
            return len(data)
    
    async def recv_tensor(self) -> mx.array:
        """
        Receive a tensor from the peer.
        
        Returns:
            Received MLX array
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to peer")
        
        # Read header
        header = await asyncio.wait_for(
            self.reader.readexactly(self.HEADER_SIZE),
            timeout=self.timeout,
        )
        msg_type, data_len = struct.unpack(">II", header)
        
        if msg_type != self.MSG_TYPE_TENSOR:
            raise ValueError(f"Expected tensor message, got type {msg_type}")
        
        # Read data
        data = await asyncio.wait_for(
            self.reader.readexactly(data_len),
            timeout=self.timeout,
        )
        
        # Deserialize
        tensor, _ = self.serializer.deserialize(data)
        return tensor
    
    async def send_recv_tensor(
        self,
        send_tensor: mx.array,
    ) -> mx.array:
        """
        Simultaneously send and receive tensors.
        
        This is the core operation for all-reduce and all-gather.
        Uses concurrent tasks to overlap send and receive.
        
        Args:
            send_tensor: Tensor to send
            
        Returns:
            Received tensor
        """
        send_task = asyncio.create_task(self.send_tensor(send_tensor))
        recv_task = asyncio.create_task(self.recv_tensor())
        
        await send_task
        recv_tensor = await recv_task
        
        return recv_tensor
    
    async def close(self):
        """Close the connection."""
        if self.writer:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass
        self._connected = False
        logger.debug(f"Connection to rank {self.peer_rank} closed")


class PeerConnectionServer:
    """
    Server that accepts incoming peer connections.
    
    Each worker runs a server to accept connections from
    workers with lower rank.
    """
    
    def __init__(
        self,
        rank: int,
        host: str,
        port: int,
        world_size: int,
    ):
        self.rank = rank
        self.host = host
        self.port = port
        self.world_size = world_size
        
        self.server: Optional[asyncio.Server] = None
        self.pending_connections: Dict[int, Tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}
        self._accept_event = asyncio.Event()
    
    async def start(self):
        """Start the server."""
        self.server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )
        
        addr = self.server.sockets[0].getsockname()
        logger.info(f"Rank {self.rank} peer server listening on {addr}")
    
    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle incoming connection."""
        try:
            # Read handshake to identify peer
            handshake = await reader.readexactly(8)
            sender_rank, target_rank = struct.unpack(">II", handshake)
            
            if target_rank != self.rank:
                logger.error(f"Connection intended for rank {target_rank}, but we are {self.rank}")
                writer.close()
                return
            
            # Store for later retrieval
            self.pending_connections[sender_rank] = (reader, writer)
            self._accept_event.set()
            
            logger.debug(f"Rank {self.rank} received connection from rank {sender_rank}")
            
        except Exception as e:
            logger.error(f"Error handling incoming connection: {e}")
            writer.close()
    
    async def get_connection(self, peer_rank: int, timeout: float = 30.0) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """
        Get connection from a specific peer.
        
        Waits until the peer connects.
        """
        start = asyncio.get_event_loop().time()
        
        while peer_rank not in self.pending_connections:
            remaining = timeout - (asyncio.get_event_loop().time() - start)
            if remaining <= 0:
                raise TimeoutError(f"Timeout waiting for connection from rank {peer_rank}")
            
            self._accept_event.clear()
            try:
                await asyncio.wait_for(self._accept_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass
        
        return self.pending_connections.pop(peer_rank)
    
    async def stop(self):
        """Stop the server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
