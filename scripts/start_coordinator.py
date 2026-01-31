#!/usr/bin/env python3
"""Start the ShardCompute coordinator server."""

import argparse
import asyncio
import logging
import signal

from shardcompute.coordinator.server import CoordinatorServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def run_coordinator(
    host: str,
    port: int,
    config_path: str = None,
):
    """Run the coordinator server."""
    server = CoordinatorServer(
        host=host,
        port=port,
        config_path=config_path,
    )
    
    # Setup shutdown handler
    shutdown_event = asyncio.Event()
    
    def handle_shutdown():
        logger.info("Shutdown signal received")
        shutdown_event.set()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_shutdown)
    
    try:
        await server.start()
        
        logger.info(f"Coordinator running on http://{host}:{port}")
        logger.info("Waiting for workers to register...")
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
    finally:
        await server.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Start ShardCompute Coordinator"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on",
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
    
    # Update log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("Starting ShardCompute Coordinator")
    logger.info(f"Config: {args.config}")
    
    asyncio.run(run_coordinator(
        host=args.host,
        port=args.port,
        config_path=args.config,
    ))


if __name__ == "__main__":
    main()
