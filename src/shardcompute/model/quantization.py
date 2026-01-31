"""MLX quantization utilities for efficient inference.

This module provides 4-bit and 8-bit quantization for linear layers using MLX's
native quantization support. Quantization significantly reduces memory usage
and improves inference speed on Apple Silicon.

MLX quantization works by:
1. Grouping weights into blocks of `group_size` elements
2. Computing scale and bias for each group
3. Storing weights as low-bit integers

For matmul operations, MLX uses optimized kernels that operate directly on
quantized weights without full dequantization.
"""

import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class QuantizedWeight:
    """Container for quantized weight data.
    
    Attributes:
        weight: Quantized weight tensor (low-bit integers packed into uint32)
        scales: Per-group scale factors
        biases: Per-group bias values
        group_size: Number of elements per quantization group
        bits: Quantization bit width (4 or 8)
        original_shape: Shape of the original weight tensor
    """
    weight: mx.array  # Quantized weights
    scales: mx.array  # Scale factors per group
    biases: mx.array  # Bias per group
    group_size: int
    bits: int
    original_shape: Tuple[int, ...]
    
    def dequantize(self) -> mx.array:
        """Dequantize weights back to full precision."""
        return mx.dequantize(
            self.weight,
            self.scales,
            self.biases,
            self.group_size,
            self.bits,
        )
    
    def matmul(self, x: mx.array, transpose: bool = True) -> mx.array:
        """Perform quantized matrix multiplication.
        
        Uses MLX's optimized quantized_matmul kernel which operates directly
        on quantized weights for maximum performance.
        
        Args:
            x: Input tensor [..., in_features]
            transpose: Whether to transpose the weight (usually True for linear layers)
            
        Returns:
            Output tensor [..., out_features]
        """
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=transpose,
            group_size=self.group_size,
            bits=self.bits,
        )
    
    @property
    def nbytes(self) -> int:
        """Total memory usage in bytes."""
        return self.weight.nbytes + self.scales.nbytes + self.biases.nbytes
    
    @property
    def compression_ratio(self) -> float:
        """Compression ratio compared to float32."""
        original_bytes = np.prod(self.original_shape) * 4  # float32
        return original_bytes / self.nbytes


def quantize_weight(
    weight: mx.array,
    group_size: int = 64,
    bits: int = 4,
) -> QuantizedWeight:
    """Quantize a weight tensor using MLX's quantization.
    
    Args:
        weight: Weight tensor to quantize [in_features, out_features]
        group_size: Number of elements per quantization group
        bits: Quantization bit width (4 or 8)
        
    Returns:
        QuantizedWeight containing quantized data
    """
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    
    original_shape = weight.shape
    
    # MLX quantize expects the last dimension to be divisible by group_size
    # For linear layers: weight is [in_features, out_features]
    # We quantize along the last dimension (out_features)
    
    # Ensure weight is contiguous and in the right format
    weight = mx.array(weight, dtype=mx.float32)
    
    # Quantize the weight
    quantized, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    
    return QuantizedWeight(
        weight=quantized,
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=bits,
        original_shape=original_shape,
    )


def quantize_weight_for_matmul(
    weight: mx.array,
    group_size: int = 64,
    bits: int = 4,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Quantize a weight tensor and return raw components.
    
    This is a convenience function that returns the raw quantization
    components instead of a QuantizedWeight object.
    
    Args:
        weight: Weight tensor to quantize
        group_size: Number of elements per quantization group
        bits: Quantization bit width (4 or 8)
        
    Returns:
        Tuple of (quantized_weight, scales, biases)
    """
    weight = mx.array(weight, dtype=mx.float32)
    return mx.quantize(weight, group_size=group_size, bits=bits)


class QuantizedLinear:
    """Linear layer with quantized weights.
    
    This is a drop-in replacement for standard linear layers that uses
    MLX's quantized_matmul for efficient inference.
    
    The forward pass uses quantized_matmul which operates directly on
    the quantized weights without full dequantization, providing both
    memory and speed benefits.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
    ):
        """Initialize QuantizedLinear.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to include bias
            group_size: Quantization group size
            bits: Quantization bit width (4 or 8)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.group_size = group_size
        self.bits = bits
        
        # Quantized weights (will be loaded later)
        self.quantized_weight: Optional[mx.array] = None
        self.scales: Optional[mx.array] = None
        self.biases_quant: Optional[mx.array] = None  # Quantization biases
        self.bias: Optional[mx.array] = None  # Linear layer bias
        
        logger.debug(
            f"QuantizedLinear: [{in_features}, {out_features}], "
            f"{bits}-bit, group_size={group_size}"
        )
    
    def load_quantized(
        self,
        quantized_weight: mx.array,
        scales: mx.array,
        biases: mx.array,
        linear_bias: Optional[mx.array] = None,
    ):
        """Load pre-quantized weights.
        
        Args:
            quantized_weight: Quantized weight tensor
            scales: Scale factors
            biases: Quantization biases
            linear_bias: Optional linear layer bias
        """
        self.quantized_weight = quantized_weight
        self.scales = scales
        self.biases_quant = biases
        self.bias = linear_bias
    
    def load_and_quantize(
        self,
        weight: mx.array,
        bias: Optional[mx.array] = None,
    ):
        """Load full-precision weights and quantize them.
        
        Args:
            weight: Full precision weight [in_features, out_features]
            bias: Optional bias vector
        """
        # Quantize the weight
        self.quantized_weight, self.scales, self.biases_quant = mx.quantize(
            weight.astype(mx.float32),
            group_size=self.group_size,
            bits=self.bits,
        )
        self.bias = bias
        
        # Log compression stats
        original_bytes = weight.size * 4  # float32
        quantized_bytes = (
            self.quantized_weight.nbytes + 
            self.scales.nbytes + 
            self.biases_quant.nbytes
        )
        ratio = original_bytes / quantized_bytes
        logger.debug(f"Quantized weight: {ratio:.1f}x compression")
    
    def forward(self, x: mx.array) -> mx.array:
        """Forward pass using quantized matmul.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        if self.quantized_weight is None:
            raise RuntimeError("Weights not loaded")
        
        # Use MLX's optimized quantized_matmul
        output = mx.quantized_matmul(
            x,
            self.quantized_weight,
            self.scales,
            self.biases_quant,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        
        if self.has_bias and self.bias is not None:
            output = output + self.bias
        
        return output
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        return self.forward(x)
    
    @property
    def num_parameters(self) -> int:
        """Number of parameters (in full precision equivalent)."""
        params = self.in_features * self.out_features
        if self.has_bias:
            params += self.out_features
        return params
    
    @property
    def memory_bytes(self) -> int:
        """Actual memory usage in bytes."""
        if self.quantized_weight is None:
            return 0
        total = (
            self.quantized_weight.nbytes +
            self.scales.nbytes +
            self.biases_quant.nbytes
        )
        if self.bias is not None:
            total += self.bias.nbytes
        return total


def save_quantized_weights(
    weights: Dict[str, QuantizedWeight],
    path: str,
):
    """Save quantized weights to a file.
    
    Saves in a format compatible with MLX's mx.save/mx.load.
    
    Args:
        weights: Dictionary of name -> QuantizedWeight
        path: Output file path
    """
    save_dict = {}
    metadata = {}
    
    for name, qw in weights.items():
        save_dict[f"{name}.weight"] = qw.weight
        save_dict[f"{name}.scales"] = qw.scales
        save_dict[f"{name}.biases"] = qw.biases
        metadata[name] = {
            "group_size": qw.group_size,
            "bits": qw.bits,
            "original_shape": list(qw.original_shape),
        }
    
    # Save weights using MLX
    mx.save(path, save_dict)
    
    # Save metadata separately (as JSON)
    import json
    meta_path = path.replace(".safetensors", "_meta.json").replace(".npz", "_meta.json")
    if not meta_path.endswith("_meta.json"):
        meta_path = path + "_meta.json"
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {len(weights)} quantized weights to {path}")


def load_quantized_weights(
    path: str,
) -> Dict[str, QuantizedWeight]:
    """Load quantized weights from a file.
    
    Args:
        path: Path to saved weights
        
    Returns:
        Dictionary of name -> QuantizedWeight
    """
    import json
    
    # Load weights
    data = mx.load(path)
    
    # Load metadata
    meta_path = path.replace(".safetensors", "_meta.json").replace(".npz", "_meta.json")
    if not meta_path.endswith("_meta.json"):
        meta_path = path + "_meta.json"
    
    with open(meta_path) as f:
        metadata = json.load(f)
    
    # Reconstruct QuantizedWeight objects
    weights = {}
    for name, meta in metadata.items():
        weights[name] = QuantizedWeight(
            weight=data[f"{name}.weight"],
            scales=data[f"{name}.scales"],
            biases=data[f"{name}.biases"],
            group_size=meta["group_size"],
            bits=meta["bits"],
            original_shape=tuple(meta["original_shape"]),
        )
    
    logger.info(f"Loaded {len(weights)} quantized weights from {path}")
    return weights


def estimate_quantized_memory(
    model_params: int,
    bits: int = 4,
    group_size: int = 64,
) -> Dict[str, float]:
    """Estimate memory usage for quantized model.
    
    Args:
        model_params: Number of model parameters
        bits: Quantization bits (4 or 8)
        group_size: Quantization group size
        
    Returns:
        Dictionary with memory estimates
    """
    # Original memory (float32)
    original_bytes = model_params * 4
    
    # Quantized weight memory
    # Each group of `group_size` elements becomes `group_size * bits / 8` bytes
    # Plus scales (float32) and biases (float32) per group
    num_groups = model_params / group_size
    weight_bytes = model_params * bits / 8
    scale_bytes = num_groups * 4  # float32 scales
    bias_bytes = num_groups * 4   # float32 biases
    
    quantized_bytes = weight_bytes + scale_bytes + bias_bytes
    
    return {
        "original_gb": original_bytes / 1e9,
        "quantized_gb": quantized_bytes / 1e9,
        "compression_ratio": original_bytes / quantized_bytes,
        "memory_savings_percent": (1 - quantized_bytes / original_bytes) * 100,
    }
