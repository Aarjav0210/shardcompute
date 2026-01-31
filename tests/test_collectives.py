"""Tests for collective operations."""

import pytest
import asyncio
import mlx.core as mx
import numpy as np

from shardcompute.collectives.all_reduce import RingAllReduce
from shardcompute.collectives.all_gather import AllGather


class MockPeerConnection:
    """Mock peer connection for testing collectives without network."""
    
    def __init__(self, partner: 'MockPeerConnection' = None):
        self.partner = partner
        self._send_queue = asyncio.Queue()
        self._recv_queue = asyncio.Queue()
    
    def set_partner(self, partner: 'MockPeerConnection'):
        """Set the partner connection."""
        self.partner = partner
        self._recv_queue = partner._send_queue
    
    async def send_tensor(self, tensor: mx.array):
        """Send tensor to partner."""
        await self._send_queue.put(tensor)
    
    async def recv_tensor(self) -> mx.array:
        """Receive tensor from partner."""
        return await self._recv_queue.get()
    
    async def send_recv_tensor(self, send_tensor: mx.array) -> mx.array:
        """Simultaneous send and receive."""
        send_task = asyncio.create_task(self.send_tensor(send_tensor))
        recv_task = asyncio.create_task(self.recv_tensor())
        
        await send_task
        return await recv_task


class TestAllReduceTwoWorkers:
    """Tests for all-reduce with 2 workers."""
    
    @pytest.fixture
    def two_worker_setup(self):
        """Create mock connections for 2-worker test."""
        # Create mock connections
        conn_0_to_1 = MockPeerConnection()
        conn_1_to_0 = MockPeerConnection()
        
        conn_0_to_1.set_partner(conn_1_to_0)
        conn_1_to_0.set_partner(conn_0_to_1)
        
        # Create all-reduce instances
        ar_0 = RingAllReduce(
            rank=0,
            world_size=2,
            peers={1: conn_0_to_1},
        )
        ar_1 = RingAllReduce(
            rank=1,
            world_size=2,
            peers={0: conn_1_to_0},
        )
        
        return ar_0, ar_1
    
    @pytest.mark.asyncio
    async def test_all_reduce_sum(self, two_worker_setup):
        """Test all-reduce with sum operation."""
        ar_0, ar_1 = two_worker_setup
        
        # Create test tensors
        tensor_0 = mx.array([1.0, 2.0, 3.0])
        tensor_1 = mx.array([4.0, 5.0, 6.0])
        expected = mx.array([5.0, 7.0, 9.0])
        
        # Run all-reduce concurrently
        result_0, result_1 = await asyncio.gather(
            ar_0.all_reduce(tensor_0, op='sum'),
            ar_1.all_reduce(tensor_1, op='sum'),
        )
        
        # Both should have the same summed result
        assert mx.allclose(result_0, expected).item()
        assert mx.allclose(result_1, expected).item()
    
    @pytest.mark.asyncio
    async def test_all_reduce_mean(self, two_worker_setup):
        """Test all-reduce with mean operation."""
        ar_0, ar_1 = two_worker_setup
        
        tensor_0 = mx.array([2.0, 4.0])
        tensor_1 = mx.array([6.0, 8.0])
        expected = mx.array([4.0, 6.0])
        
        result_0, result_1 = await asyncio.gather(
            ar_0.all_reduce(tensor_0, op='mean'),
            ar_1.all_reduce(tensor_1, op='mean'),
        )
        
        assert mx.allclose(result_0, expected).item()
        assert mx.allclose(result_1, expected).item()
    
    @pytest.mark.asyncio
    async def test_all_reduce_max(self, two_worker_setup):
        """Test all-reduce with max operation."""
        ar_0, ar_1 = two_worker_setup
        
        tensor_0 = mx.array([1.0, 5.0, 3.0])
        tensor_1 = mx.array([4.0, 2.0, 6.0])
        expected = mx.array([4.0, 5.0, 6.0])
        
        result_0, result_1 = await asyncio.gather(
            ar_0.all_reduce(tensor_0, op='max'),
            ar_1.all_reduce(tensor_1, op='max'),
        )
        
        assert mx.allclose(result_0, expected).item()
        assert mx.allclose(result_1, expected).item()
    
    @pytest.mark.asyncio
    async def test_all_reduce_2d_tensor(self, two_worker_setup):
        """Test all-reduce with 2D tensor."""
        ar_0, ar_1 = two_worker_setup
        
        tensor_0 = mx.array([[1.0, 2.0], [3.0, 4.0]])
        tensor_1 = mx.array([[5.0, 6.0], [7.0, 8.0]])
        expected = mx.array([[6.0, 8.0], [10.0, 12.0]])
        
        result_0, result_1 = await asyncio.gather(
            ar_0.all_reduce(tensor_0, op='sum'),
            ar_1.all_reduce(tensor_1, op='sum'),
        )
        
        assert result_0.shape == expected.shape
        assert mx.allclose(result_0, expected).item()
        assert mx.allclose(result_1, expected).item()


class TestAllGatherTwoWorkers:
    """Tests for all-gather with 2 workers."""
    
    @pytest.fixture
    def two_worker_setup(self):
        """Create mock connections for 2-worker test."""
        conn_0_to_1 = MockPeerConnection()
        conn_1_to_0 = MockPeerConnection()
        
        conn_0_to_1.set_partner(conn_1_to_0)
        conn_1_to_0.set_partner(conn_0_to_1)
        
        ag_0 = AllGather(
            rank=0,
            world_size=2,
            peers={1: conn_0_to_1},
        )
        ag_1 = AllGather(
            rank=1,
            world_size=2,
            peers={0: conn_1_to_0},
        )
        
        return ag_0, ag_1
    
    @pytest.mark.asyncio
    async def test_all_gather_1d(self, two_worker_setup):
        """Test all-gather with 1D tensors."""
        ag_0, ag_1 = two_worker_setup
        
        # Each worker has a partition
        tensor_0 = mx.array([1.0, 2.0])
        tensor_1 = mx.array([3.0, 4.0])
        expected = mx.array([1.0, 2.0, 3.0, 4.0])
        
        result_0, result_1 = await asyncio.gather(
            ag_0.all_gather(tensor_0, dim=0),
            ag_1.all_gather(tensor_1, dim=0),
        )
        
        assert result_0.shape == expected.shape
        assert mx.allclose(result_0, expected).item()
        assert mx.allclose(result_1, expected).item()
    
    @pytest.mark.asyncio
    async def test_all_gather_last_dim(self, two_worker_setup):
        """Test all-gather along last dimension."""
        ag_0, ag_1 = two_worker_setup
        
        # [batch, seq, local_hidden] -> [batch, seq, full_hidden]
        tensor_0 = mx.array([[[1.0, 2.0], [3.0, 4.0]]])  # [1, 2, 2]
        tensor_1 = mx.array([[[5.0, 6.0], [7.0, 8.0]]])  # [1, 2, 2]
        expected = mx.array([[[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]])  # [1, 2, 4]
        
        result_0, result_1 = await asyncio.gather(
            ag_0.all_gather(tensor_0, dim=-1),
            ag_1.all_gather(tensor_1, dim=-1),
        )
        
        assert result_0.shape == expected.shape
        assert mx.allclose(result_0, expected).item()
        assert mx.allclose(result_1, expected).item()


class TestAllReduceNumpyReference:
    """Test all-reduce against numpy reference implementation."""
    
    @pytest.fixture
    def setup(self):
        """Create mock connections."""
        conn_0_to_1 = MockPeerConnection()
        conn_1_to_0 = MockPeerConnection()
        
        conn_0_to_1.set_partner(conn_1_to_0)
        conn_1_to_0.set_partner(conn_0_to_1)
        
        ar_0 = RingAllReduce(rank=0, world_size=2, peers={1: conn_0_to_1})
        ar_1 = RingAllReduce(rank=1, world_size=2, peers={0: conn_1_to_0})
        
        return ar_0, ar_1
    
    @pytest.mark.asyncio
    async def test_random_tensors(self, setup):
        """Test with random tensors against numpy sum."""
        ar_0, ar_1 = setup
        
        # Generate random data
        np_0 = np.random.randn(10, 20).astype(np.float32)
        np_1 = np.random.randn(10, 20).astype(np.float32)
        
        tensor_0 = mx.array(np_0)
        tensor_1 = mx.array(np_1)
        
        # Expected from numpy
        expected = np_0 + np_1
        
        # All-reduce
        result_0, result_1 = await asyncio.gather(
            ar_0.all_reduce(tensor_0, op='sum'),
            ar_1.all_reduce(tensor_1, op='sum'),
        )
        
        # Compare
        np_result_0 = np.array(result_0)
        np_result_1 = np.array(result_1)
        
        np.testing.assert_allclose(np_result_0, expected, rtol=1e-5)
        np.testing.assert_allclose(np_result_1, expected, rtol=1e-5)
