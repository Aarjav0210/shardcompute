#!/usr/bin/env python3
"""Download and prepare model for ShardCompute."""

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def download_model(
    model_name: str,
    output_dir: str,
    include_tokenizer: bool = True,
):
    """
    Download model from HuggingFace Hub.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        output_dir: Local directory to save model
        include_tokenizer: Whether to also save tokenizer
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {model_name} to {output_dir}")
    
    # Download model files
    local_dir = snapshot_download(
        repo_id=model_name,
        local_dir=str(output_path),
        local_dir_use_symlinks=False,
    )
    
    logger.info(f"Model downloaded to {local_dir}")
    
    # Download tokenizer separately if needed
    if include_tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(str(output_path))
            logger.info("Tokenizer saved")
        except Exception as e:
            logger.warning(f"Could not save tokenizer: {e}")
    
    # List downloaded files
    files = list(output_path.glob("*"))
    logger.info(f"Downloaded files: {[f.name for f in files]}")
    
    # Check for model weights
    has_safetensors = any(f.suffix == ".safetensors" for f in files)
    has_pytorch = any(f.name.startswith("pytorch_model") for f in files)
    
    if has_safetensors:
        logger.info("Model has safetensors format (preferred)")
    elif has_pytorch:
        logger.info("Model has PyTorch format")
    else:
        logger.warning("No recognized model weight files found")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download model from HuggingFace Hub"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./model_cache",
        help="Output directory",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip tokenizer download",
    )
    
    args = parser.parse_args()
    
    download_model(
        model_name=args.model,
        output_dir=args.output,
        include_tokenizer=not args.no_tokenizer,
    )
    
    logger.info("Done! Next step: run shard_weights.py to prepare for tensor parallelism")


if __name__ == "__main__":
    main()
