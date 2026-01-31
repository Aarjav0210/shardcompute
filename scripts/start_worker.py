#!/usr/bin/env python3
"""Start a ShardCompute worker node."""

import argparse
import asyncio
import logging
import signal

from shardcompute.worker.node import WorkerNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_worker(
    rank: int,
    coordinator_url: str,
    host: str,
    port: int,
    shard_dir: str = None,
    config_path: str = None,
):
    """Run a worker node."""
    worker = WorkerNode(
        rank=rank,
        coordinator_url=coordinator_url,
        host=host,
        collective_port=port,
        shard_dir=shard_dir,
        config_path=config_path,
    )
    
    # Setup shutdown handler
    shutdown_event = asyncio.Event()
    
    def handle_shutdown():
        logger.info(f"Worker {rank}: Shutdown signal received")
        shutdown_event.set()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_shutdown)
    
    # Start worker in background
    worker_task = asyncio.create_task(worker.start())
    shutdown_task = asyncio.create_task(shutdown_event.wait())
    
    try:
        # Wait for either worker completion or shutdown
        done, pending = await asyncio.wait(
            [worker_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Check for worker errors
        if worker_task in done:
            try:
                worker_task.result()
            except Exception as e:
                logger.error(f"Worker error: {e}")
                raise
                
    finally:
        await worker.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Start ShardCompute Worker"
    )
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="Worker rank (0 to world_size-1)",
    )
    parser.add_argument(
        "--coordinator-url",
        type=str,
        required=True,
        help="URL of coordinator (e.g., http://localhost:8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for peer connections",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for peer connections (default: 9000 + rank)",
    )
    parser.add_argument(
        "--shard-dir",
        type=str,
        help="Directory containing weight shards",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Default port based on rank
    port = args.port if args.port else 9000 + args.rank
    
    # Update log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Starting ShardCompute Worker {args.rank}")
    logger.info(f"Coordinator: {args.coordinator_url}")
    logger.info(f"Collective port: {port}")
    
    asyncio.run(run_worker(
        rank=args.rank,
        coordinator_url=args.coordinator_url,
        host=args.host,
        port=port,
        shard_dir=args.shard_dir,
        config_path=args.config,
    ))


if __name__ == "__main__":
    main()
