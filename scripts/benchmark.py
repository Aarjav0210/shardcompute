#!/usr/bin/env python3
"""Benchmark tensor parallel inference."""

import argparse
import asyncio
import logging
import time
import statistics
from typing import List, Dict
import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def run_inference(
    session: aiohttp.ClientSession,
    coordinator_url: str,
    input_ids: List[int],
    max_new_tokens: int,
) -> Dict:
    """Run a single inference request."""
    url = f"{coordinator_url}/api/inference"
    
    payload = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,  # Greedy for reproducibility
    }
    
    start = time.perf_counter()
    
    async with session.post(url, json=payload) as response:
        if response.status != 200:
            error = await response.text()
            return {"error": error, "latency_ms": 0}
        
        data = await response.json()
    
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    
    return {
        "latency_ms": latency_ms,
        "output_tokens": len(data.get("output_ids", [])),
        "timing": data.get("timing", {}),
    }


async def run_benchmark(
    coordinator_url: str,
    num_requests: int,
    input_length: int,
    output_length: int,
    warmup_requests: int = 3,
):
    """
    Run inference benchmark.
    
    Args:
        coordinator_url: Coordinator URL
        num_requests: Number of requests to run
        input_length: Number of input tokens
        output_length: Max output tokens
        warmup_requests: Warmup requests to discard
    """
    coordinator_url = coordinator_url.rstrip("/")
    
    # Check cluster status
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{coordinator_url}/api/status") as response:
            if response.status != 200:
                logger.error("Cannot connect to coordinator")
                return
            
            status = await response.json()
            if not status.get("cluster_ready"):
                logger.error("Cluster not ready")
                return
            
            logger.info(f"Cluster ready: {status.get('workers')} workers")
    
    # Generate sample input (dummy token IDs)
    input_ids = list(range(1, input_length + 1))
    
    logger.info(f"Running benchmark:")
    logger.info(f"  Input tokens: {input_length}")
    logger.info(f"  Max output tokens: {output_length}")
    logger.info(f"  Requests: {num_requests} (+ {warmup_requests} warmup)")
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        # Warmup
        logger.info("Warming up...")
        for i in range(warmup_requests):
            result = await run_inference(
                session, coordinator_url, input_ids, output_length
            )
            if "error" in result:
                logger.error(f"Warmup error: {result['error']}")
                return
            logger.info(f"  Warmup {i+1}: {result['latency_ms']:.1f}ms")
        
        # Benchmark
        logger.info("Running benchmark...")
        for i in range(num_requests):
            result = await run_inference(
                session, coordinator_url, input_ids, output_length
            )
            
            if "error" in result:
                logger.error(f"Request {i+1} error: {result['error']}")
                continue
            
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Completed {i+1}/{num_requests} requests")
    
    if not results:
        logger.error("No successful requests")
        return
    
    # Compute statistics
    latencies = [r["latency_ms"] for r in results]
    
    logger.info("\n" + "=" * 50)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 50)
    
    logger.info(f"\nLatency (ms):")
    logger.info(f"  Mean:   {statistics.mean(latencies):.1f}")
    logger.info(f"  Median: {statistics.median(latencies):.1f}")
    logger.info(f"  Std:    {statistics.stdev(latencies):.1f}" if len(latencies) > 1 else "  Std:    N/A")
    logger.info(f"  Min:    {min(latencies):.1f}")
    logger.info(f"  Max:    {max(latencies):.1f}")
    
    # Percentiles
    sorted_latencies = sorted(latencies)
    p50_idx = int(len(sorted_latencies) * 0.50)
    p90_idx = int(len(sorted_latencies) * 0.90)
    p99_idx = int(len(sorted_latencies) * 0.99)
    
    logger.info(f"\nPercentiles (ms):")
    logger.info(f"  P50:    {sorted_latencies[p50_idx]:.1f}")
    logger.info(f"  P90:    {sorted_latencies[p90_idx]:.1f}")
    logger.info(f"  P99:    {sorted_latencies[p99_idx]:.1f}")
    
    # Throughput
    total_tokens = sum(r["output_tokens"] for r in results)
    total_time = sum(latencies) / 1000  # seconds
    
    logger.info(f"\nThroughput:")
    logger.info(f"  Total tokens: {total_tokens}")
    logger.info(f"  Total time:   {total_time:.2f}s")
    logger.info(f"  Tokens/sec:   {total_tokens / total_time:.1f}")
    
    # Timing breakdown (if available)
    if results[0].get("timing"):
        compute_times = [r["timing"].get("compute_ms", 0) for r in results if r.get("timing")]
        comm_times = [r["timing"].get("comm_ms", 0) for r in results if r.get("timing")]
        
        if compute_times and comm_times:
            avg_compute = statistics.mean(compute_times)
            avg_comm = statistics.mean(comm_times)
            
            logger.info(f"\nTiming Breakdown:")
            logger.info(f"  Avg compute: {avg_compute:.1f}ms")
            logger.info(f"  Avg comm:    {avg_comm:.1f}ms")
            logger.info(f"  Compute %:   {avg_compute / (avg_compute + avg_comm) * 100:.1f}%")
    
    logger.info("\n" + "=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tensor parallel inference"
    )
    parser.add_argument(
        "--coordinator-url",
        type=str,
        default="http://localhost:8080",
        help="Coordinator URL",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Number of requests to run",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=64,
        help="Number of input tokens",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=32,
        help="Maximum output tokens",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup requests",
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(
        coordinator_url=args.coordinator_url,
        num_requests=args.num_requests,
        input_length=args.input_length,
        output_length=args.output_length,
        warmup_requests=args.warmup,
    ))


if __name__ == "__main__":
    main()
