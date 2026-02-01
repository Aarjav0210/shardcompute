#!/usr/bin/env python3
"""Shard model weights for tensor or pipeline parallelism."""

import argparse
import logging
from pathlib import Path

from shardcompute.model.config import ModelConfig, ParallelConfig
from shardcompute.model.sharder import (
    WeightSharder,
    PipelineWeightSharder,
    load_huggingface_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def shard_weights(
    model_dir: str,
    output_dir: str,
    world_size: int = 2,
    mode: str = "tensor",
):
    """
    Shard model weights for tensor or pipeline parallelism.

    Args:
        model_dir: Directory containing HuggingFace model
        output_dir: Output directory for sharded weights
        world_size: Number of parallel workers
        mode: Parallelism mode ("tensor" or "pipeline")
    """
    model_path = Path(model_dir)
    output_path = Path(output_dir)

    logger.info(f"Loading model from {model_path}")

    # Load model config
    model_config = ModelConfig.from_pretrained(str(model_path))
    logger.info(f"Model config: {model_config.num_layers} layers, "
                f"{model_config.hidden_size} hidden, {model_config.num_heads} heads")

    # Create parallel config
    if mode == "pipeline":
        parallel_config = ParallelConfig(
            world_size=world_size,
            mode="pipeline",
            pipeline_parallel_size=world_size,
        )
    else:
        parallel_config = ParallelConfig(
            world_size=world_size,
            mode="tensor",
            tensor_parallel_size=world_size,
        )

    # Validate
    parallel_config.validate(model_config)
    logger.info(f"Parallelism validated: {world_size}-way {mode} parallel")

    # Load weights
    logger.info("Loading weights...")
    weights = load_huggingface_weights(str(model_path))
    logger.info(f"Loaded {len(weights)} tensors")

    # Calculate total size
    total_params = sum(w.size for w in weights.values())
    total_bytes = sum(w.nbytes for w in weights.values())
    logger.info(f"Total parameters: {total_params:,} ({total_bytes / 1e9:.2f} GB)")

    # Create appropriate sharder
    if mode == "pipeline":
        sharder = PipelineWeightSharder(model_config, parallel_config)
        logger.info(f"Using pipeline sharding: {model_config.num_layers} layers across {world_size} workers "
                    f"({model_config.num_layers // world_size} layers per worker)")
    else:
        sharder = WeightSharder(model_config, parallel_config)

    # Shard and save
    logger.info(f"Sharding weights for {world_size} workers ({mode} mode)...")
    sharder.save_shards(weights, str(output_path))

    # Verify shards
    for rank in range(world_size):
        rank_dir = output_path / f"rank_{rank}"
        shard_files = list(rank_dir.glob("*.safetensors"))
        if shard_files:
            logger.info(f"Rank {rank}: {shard_files[0].name}")
        else:
            logger.error(f"Rank {rank}: No shards found!")

    logger.info(f"Sharding complete! Output: {output_path}")
    logger.info(f"Each worker should load from: {output_path}/rank_<rank>/")


def main():
    parser = argparse.ArgumentParser(
        description="Shard model weights for tensor or pipeline parallelism"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for sharded weights",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="tensor",
        choices=["tensor", "pipeline"],
        help="Parallelism mode: tensor (split weights) or pipeline (split layers)",
    )

    args = parser.parse_args()

    shard_weights(
        model_dir=args.model,
        output_dir=args.output,
        world_size=args.world_size,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
