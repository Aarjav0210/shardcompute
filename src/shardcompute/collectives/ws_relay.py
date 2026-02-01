"""WebSocket relay transport for collective operations through coordinator."""

import asyncio
import struct
import logging
from typing import Dict, Optional

import aiohttp
import mlx.core as mx

from shardcompute.protocol.serialization import TensorSerializer

logger = logging.getLogger(__name__)


class WebSocketRelayConnection:
    """
    Drop-in replacement for PeerConnection that routes traffic
    through the coordinator's WebSocket relay.

    Wire protocol per WS binary frame:
        [4B sender_rank][4B target_rank][4B msg_type][4B data_len][payload]
    """

    HEADER_SIZE = 16
    MSG_TYPE_TENSOR = 1
    MSG_TYPE_CONTROL = 2

    def __init__(
        self,
        local_rank: int,
        peer_rank: int,
        ws: aiohttp.ClientWebSocketResponse,
        send_lock: asyncio.Lock,
        timeout: float = 30.0,
    ):
        self.local_rank = local_rank
        self.peer_rank = peer_rank
        self._ws = ws
        self._send_lock = send_lock
        self.timeout = timeout
        self.serializer = TensorSerializer()
        self._connected = True
        self._recv_queue: asyncio.Queue = asyncio.Queue()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None and not self._ws.closed

    async def send_tensor(self, tensor: mx.array) -> int:
        if not self.is_connected:
            raise RuntimeError("Not connected to peer via relay")

        data = self.serializer.serialize(tensor)
        envelope = struct.pack(">IIII", self.local_rank, self.peer_rank,
                               self.MSG_TYPE_TENSOR, len(data))
        frame = envelope + data

        async with self._send_lock:
            await self._ws.send_bytes(frame)

        return len(data)

    async def recv_tensor(self) -> mx.array:
        if not self._connected:
            raise RuntimeError("Not connected to peer via relay")

        try:
            data = await asyncio.wait_for(
                self._recv_queue.get(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Timeout waiting for tensor from rank {self.peer_rank}"
            )

        if isinstance(data, Exception):
            raise data

        tensor, _ = self.serializer.deserialize(data)
        return tensor

    async def send_recv_tensor(self, send_tensor: mx.array) -> mx.array:
        _, recv_tensor = await asyncio.gather(
            self.send_tensor(send_tensor),
            self.recv_tensor(),
        )
        return recv_tensor

    def flush(self):
        """Drain all pending data from recv queue to discard stale messages."""
        discarded = 0
        while not self._recv_queue.empty():
            try:
                self._recv_queue.get_nowait()
                discarded += 1
            except asyncio.QueueEmpty:
                break
        if discarded > 0:
            logger.debug(
                f"Rank {self.local_rank} flushed {discarded} stale messages "
                f"from peer {self.peer_rank}"
            )
        return discarded

    async def close(self):
        self._connected = False


class WebSocketRelayManager:
    """
    Manages a single WebSocket connection to the coordinator and
    dispatches incoming messages to per-peer recv queues.
    """

    def __init__(
        self,
        rank: int,
        coordinator_ws_url: str,
        world_size: int,
        timeout: float = 30.0,
    ):
        self.rank = rank
        self.coordinator_ws_url = coordinator_ws_url
        self.world_size = world_size
        self.timeout = timeout
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._connections: Dict[int, WebSocketRelayConnection] = {}
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._send_lock = asyncio.Lock()

    async def connect(self):
        """Open WebSocket to coordinator and start dispatcher."""
        self._session = aiohttp.ClientSession()
        url = f"{self.coordinator_ws_url}/{self.rank}"
        logger.info(f"Rank {self.rank} connecting to WS relay at {url}")

        self._ws = await self._session.ws_connect(url, max_msg_size=0)
        self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        logger.info(f"Rank {self.rank} connected to WS relay")

    async def _dispatch_loop(self):
        """Read messages from WS and route to correct peer's recv_queue."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    if len(msg.data) < WebSocketRelayConnection.HEADER_SIZE:
                        logger.warning(f"Rank {self.rank} received short WS frame")
                        continue

                    sender_rank = struct.unpack(">I", msg.data[0:4])[0]
                    # Payload is everything after the 16-byte envelope header
                    payload = msg.data[WebSocketRelayConnection.HEADER_SIZE:]

                    conn = self._connections.get(sender_rank)
                    if conn is not None:
                        await conn._recv_queue.put(payload)
                    else:
                        logger.warning(
                            f"Rank {self.rank} got message from unknown rank {sender_rank}"
                        )
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                    logger.warning(f"Rank {self.rank} WS relay connection closed/error")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Rank {self.rank} WS dispatch loop error: {e}")
        finally:
            # Signal all pending recv operations
            err = ConnectionError("WebSocket relay disconnected")
            for conn in self._connections.values():
                conn._connected = False
                await conn._recv_queue.put(err)

    def get_connection(self, peer_rank: int) -> WebSocketRelayConnection:
        """Get or create a relay connection to a specific peer."""
        if peer_rank not in self._connections:
            conn = WebSocketRelayConnection(
                local_rank=self.rank,
                peer_rank=peer_rank,
                ws=self._ws,
                send_lock=self._send_lock,
                timeout=self.timeout,
            )
            self._connections[peer_rank] = conn
        return self._connections[peer_rank]

    async def close(self):
        """Close WebSocket and cleanup."""
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass

        for conn in self._connections.values():
            await conn.close()
        self._connections.clear()

        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

        logger.info(f"Rank {self.rank} WS relay manager closed")
