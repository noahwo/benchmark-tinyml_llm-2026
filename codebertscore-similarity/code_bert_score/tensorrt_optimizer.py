"""
TensorRT optimization wrapper for CodeBERT models.

This module provides utilities to compile and optimize CodeBERT models
using TensorRT for faster inference.
"""

import torch
import warnings
from typing import Optional

# Check if torch-tensorrt is available
try:
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    warnings.warn(
        "torch-tensorrt not found. Install with: "
        "pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121"
    )


def optimize_model_with_tensorrt(
    model,
    device: str = "cuda",
    use_fp16: bool = True,
    batch_size: int = 64,
    max_seq_length: int = 512,
    enable_optimization: bool = True,
):
    """
    Optimize a PyTorch model using TensorRT.
    
    Args:
        model: The PyTorch model to optimize
        device: Device to use ('cuda' or 'cpu')
        use_fp16: Whether to use FP16 precision (faster but slightly less accurate)
        batch_size: Expected batch size for inference
        max_seq_length: Maximum sequence length for input
        enable_optimization: Whether to enable TensorRT optimization
        
    Returns:
        Optimized model (or original if TensorRT unavailable or disabled)
    """
    
    if not enable_optimization:
        print("TensorRT optimization disabled")
        return model
    
    if not TENSORRT_AVAILABLE:
        print("TensorRT not available, using standard PyTorch model")
        return model
    
    if device != "cuda":
        print("TensorRT requires CUDA, using standard PyTorch model")
        return model
    
    try:
        print(f"Optimizing model with TensorRT (FP16: {use_fp16})...")
        
        model.eval()
        
        # Create example inputs
        example_input_ids = torch.randint(0, 30000, (batch_size, max_seq_length)).cuda()
        example_attention_mask = torch.ones(batch_size, max_seq_length).cuda()
        
        # Compile model with TensorRT
        if use_fp16:
            print("Using FP16 precision for ~2x speedup")
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=(1, 1),
                        opt_shape=(batch_size, max_seq_length),
                        max_shape=(batch_size, max_seq_length * 2),
                        dtype=torch.long,
                    ),
                    torch_tensorrt.Input(
                        min_shape=(1, 1),
                        opt_shape=(batch_size, max_seq_length),
                        max_shape=(batch_size, max_seq_length * 2),
                        dtype=torch.long,
                    ),
                ],
                enabled_precisions={torch.float, torch.half},  # Enable FP16
                truncate_long_and_double=True,
            )
        else:
            print("Using FP32 precision")
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=(1, 1),
                        opt_shape=(batch_size, max_seq_length),
                        max_shape=(batch_size, max_seq_length * 2),
                        dtype=torch.long,
                    ),
                    torch_tensorrt.Input(
                        min_shape=(1, 1),
                        opt_shape=(batch_size, max_seq_length),
                        max_shape=(batch_size, max_seq_length * 2),
                        dtype=torch.long,
                    ),
                ],
                enabled_precisions={torch.float},
            )
        
        print("✓ Model successfully optimized with TensorRT")
        return trt_model
        
    except Exception as e:
        print(f"TensorRT optimization failed: {e}")
        print("Falling back to standard PyTorch model")
        return model


def enable_cuda_optimizations():
    """
    Enable additional CUDA optimizations for better performance.
    """
    if torch.cuda.is_available():
        # Enable cudnn benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True
        
        # Enable TF32 on Ampere GPUs for faster matmul
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✓ Enabled TF32 for Ampere GPU")
        
        print(f"✓ CUDA optimizations enabled on {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available")


def benchmark_inference_speed(model, tokenizer, sample_text: str, num_runs: int = 100):
    """
    Benchmark the inference speed of a model.
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for the model
        sample_text: Sample text to use for benchmarking
        num_runs: Number of inference runs
        
    Returns:
        Average inference time in milliseconds
    """
    import time
    
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenize sample
    inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(**inputs)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time:.2f}ms ({1000/avg_time:.1f} inferences/sec)")
    return avg_time
