# ShardCompute

**Distributed Tensor Parallelism for Apple Silicon**

ShardCompute is a proof-of-concept implementation of Megatron-style tensor parallelism across distributed Apple Silicon devices. Unlike pipeline parallelism where devices process sequentially, tensor parallelism splits individual weight matrices across devices so they compute in parallel.

## Key Features

- **True Tensor Parallelism**: Weight matrices split across devices, all computing simultaneously
- **MLX Native**: Built on Apple's MLX framework for Metal GPU acceleration
- **Megatron-Style Communication**: Minimized collectives using column-parallel → row-parallel pattern
- **Scalable Architecture**: Designed for 2D block parallelism extension

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Coordinator                               │
│  - Cluster bootstrap      - Health monitoring                    │
│  - Worker registration    - Metrics aggregation                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
    ┌──────────┐         ┌──────────┐         ┌──────────┐
    │ Worker 0 │◄───────►│ Worker 1 │◄───────►│ Worker N │
    │ (rank 0) │         │ (rank 1) │         │ (rank N) │
    │          │         │          │         │          │
    │ W[:,:H/N]│         │ W[:,H/N:]│         │    ...   │
    └──────────┘         └──────────┘         └──────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                     Peer-to-Peer Mesh
                   (All-Reduce, All-Gather)
```

### Tensor Parallelism Strategy

**Column Parallel Linear (no communication in forward):**
```
Input X (replicated) ──► W[:, local_cols] ──► Y_local (partitioned)
```

**Row Parallel Linear (all-reduce in forward):**
```
X_local (partitioned) ──► W[local_rows, :] ──► Y_partial ──► ALL-REDUCE ──► Y (replicated)
```

**Megatron MLP Pattern (1 all-reduce per MLP):**
```
X ──► ColPar(Up) ──► GeLU ──► RowPar(Down) ──► ALL-REDUCE ──► Y
```

## Installation

### Prerequisites

- Python 3.11+
- macOS with Apple Silicon (M1/M2/M3)
- Two or more Apple Silicon devices on the same network

### Install from source

```bash
git clone https://github.com/your-org/shardcompute.git
cd shardcompute
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download and Prepare Model

```bash
# Download TinyLlama (1.1B parameters)
python scripts/download_model.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output ./model_cache

# Shard weights for 2-way tensor parallelism
python scripts/shard_weights.py \
    --model ./model_cache \
    --output ./model_shards \
    --world-size 2
```

### 2. Start the Coordinator

On Machine A (or localhost for testing):

```bash
python scripts/start_coordinator.py \
    --host 0.0.0.0 \
    --port 8080 \
    --config config/default.yaml
```

### 3. Start Workers

**Worker 0** (on Machine A):
```bash
python scripts/start_worker.py \
    --rank 0 \
    --coordinator-url http://localhost:8080 \
    --port 9000 \
    --shard-dir ./model_shards
```

**Worker 1** (on Machine B):
```bash
python scripts/start_worker.py \
    --rank 1 \
    --coordinator-url http://MACHINE_A_IP:8080 \
    --port 9001 \
    --shard-dir ./model_shards
```

### 4. Run Inference

```bash
# Chat client
python -m shardcompute.client.chat \
    --coordinator-url http://localhost:8080 \
    --tokenizer ./model_cache

# Benchmark
python scripts/benchmark.py \
    --coordinator-url http://localhost:8080 \
    --num-requests 50
```

## Project Structure

```
shardcompute/
├── config/
│   └── default.yaml           # Configuration file
├── scripts/
│   ├── download_model.py      # Download HuggingFace models
│   ├── shard_weights.py       # Pre-shard weights for TP
│   ├── start_coordinator.py   # Launch coordinator
│   ├── start_worker.py        # Launch worker
│   └── benchmark.py           # Performance benchmark
├── src/shardcompute/
│   ├── coordinator/           # Cluster management
│   │   ├── server.py         # HTTP API server
│   │   ├── registry.py       # Worker registration
│   │   ├── health.py         # Health monitoring
│   │   └── metrics.py        # Metrics aggregation
│   ├── worker/               # Worker nodes
│   │   ├── node.py           # Worker main class
│   │   ├── executor.py       # Inference execution
│   │   ├── peer_mesh.py      # Peer connections
│   │   └── heartbeat.py      # Health reporting
│   ├── collectives/          # Collective operations
│   │   ├── communicator.py   # Main interface
│   │   ├── all_reduce.py     # Ring all-reduce
│   │   ├── all_gather.py     # Ring all-gather
│   │   ├── point_to_point.py # TCP send/recv
│   │   └── topology.py       # Network topologies
│   ├── parallel/             # Tensor parallel layers
│   │   ├── column_linear.py  # Column-parallel linear
│   │   ├── row_linear.py     # Row-parallel linear
│   │   ├── attention.py      # Parallel attention
│   │   ├── mlp.py            # Parallel MLP
│   │   ├── embedding.py      # Parallel embedding
│   │   └── transformer.py    # Full transformer
│   ├── model/                # Model loading
│   │   ├── loader.py         # Weight loading
│   │   ├── sharder.py        # Weight sharding
│   │   └── config.py         # Model config
│   ├── protocol/             # Network protocol
│   │   ├── messages.py       # Message types
│   │   └── serialization.py  # Tensor serialization
│   └── client/
│       └── chat.py           # Chat client
└── tests/                    # Test suite
```

## Communication Pattern

For a transformer with tensor parallelism:

| Operation | Communication | Pattern |
|-----------|---------------|---------|
| Embedding lookup | All-Gather | Column → Full |
| Attention QKV | None | Column parallel |
| Attention scores | None | Local per-head |
| Attention output | All-Reduce | Row → Sum |
| MLP up/gate | None | Column parallel |
| MLP activation | None | Local |
| MLP down | All-Reduce | Row → Sum |
| LM head | All-Gather | Column → Full |

**Total per layer:** 2 all-reduces (attention + MLP)

## Configuration

See `config/default.yaml` for all options:

```yaml
coordinator:
  host: "0.0.0.0"
  port: 8080
  heartbeat_interval_seconds: 5
  heartbeat_timeout_seconds: 15

worker:
  collective_port_base: 9000
  collective_timeout_seconds: 30

parallelism:
  tensor_parallel_size: 2
  topology: "ring"

model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  hidden_size: 2048
  intermediate_size: 5632
  num_heads: 32
  num_layers: 22
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_serialization.py
pytest tests/test_parallel_layers.py
pytest tests/test_collectives.py
pytest tests/test_integration.py

# Run with verbose output
pytest tests/ -v
```

## Performance Considerations

### Communication Overhead

For TinyLlama (22 layers) with 2 workers:
- 1 all-gather (embedding)
- 44 all-reduces (22 attention + 22 MLP)
- 1 all-gather (LM head)
- **Total: 46 collectives per forward pass**

### Optimization Tips

1. **Network**: Use fast local network (Thunderbolt Bridge ideal)
2. **Batch size**: Larger batches amortize communication
3. **Sequence length**: Longer sequences = more compute per communication
4. **MLX eval()**: Force evaluation between layers to prevent memory buildup

## Path to 2D Parallelism

The architecture supports extension to 2D block parallelism:

```python
# Current: 1D (2 workers)
# Worker 0: W[:, :H/2]
# Worker 1: W[:, H/2:]

# Future: 2D (4 workers in 2x2)
# Worker (0,0): W[:H/2, :W/2]
# Worker (0,1): W[:H/2, W/2:]
# Worker (1,0): W[H/2:, :W/2]
# Worker (1,1): W[H/2:, W/2:]
```

Extension points in code:
- `MeshTopology2D` in `topology.py`
- `ProcessGroup` in `communicator.py`
- Block sharding in `sharder.py`

## Known Limitations

- POC scope: 2 workers only (architecture ready for N)
- No KV cache for efficient generation
- Single inference request at a time
- No authentication/encryption
- Manual weight sharding (no automatic partitioning)

## Troubleshooting

### Connection refused
- Check coordinator is running and reachable
- Verify firewall allows ports 8080, 9000, 9001

### Timeout during collective
- Ensure both workers are started
- Check network connectivity between machines
- Increase `collective_timeout_seconds`

### Out of memory
- Reduce batch size
- Reduce sequence length
- Use smaller model

## License

MIT License

## References

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [MLX: An array framework for Apple Silicon](https://github.com/ml-explore/mlx)
- [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)
