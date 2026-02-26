#!/usr/bin/env python3
"""
Benchmark script to compare inference speeds with different optimization levels.
Run this before and after applying TensorRT optimizations to measure speedup.
"""

import sys
import time
from pathlib import Path

# Add code_bert_score to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

import torch
import code_bert_score
from code_bert_score import BERTScorer


def benchmark_inference(scorer, cand_code, ref_code, num_runs=50, warmup_runs=5):
    """Run benchmark with warmup."""
    # Warmup
    for _ in range(warmup_runs):
        scorer.score([cand_code], [ref_code])
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        P, R, F1, F3 = scorer.score([cand_code], [ref_code])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    throughput = 1 / avg_time
    
    return avg_time * 1000, throughput  # ms, inferences/sec


def main():
    print("=" * 70)
    print("CodeBERT Inference Performance Benchmark")
    print("=" * 70)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Load sample reference and candidate
    ref_file = SCRIPT_DIR / "references" / "TFLite_detection_video.py"
    if not ref_file.exists():
        print(f"ERROR: Reference file not found: {ref_file}")
        print("Please run this script from codebertscore-similarity/ directory")
        return
    
    ref_code = ref_file.read_text()
    
    # Use a subset for faster benchmarking
    cand_code = ref_code[:len(ref_code)//2]  # Use half as candidate for quick test
    
    print(f"Reference code length: {len(ref_code)} chars")
    print(f"Candidate code length: {len(cand_code)} chars")
    print()
    
    # Benchmark configurations
    configs = [
        {
            "name": "Baseline (no optimizations)",
            "setup": lambda: None,
            "batch_size": 1,
        },
        {
            "name": "Basic CUDA optimizations",
            "setup": lambda: setattr(torch.backends.cudnn, 'benchmark', True),
            "batch_size": 1,
        },
        {
            "name": "Larger batch size",
            "setup": lambda: setattr(torch.backends.cudnn, 'benchmark', True),
            "batch_size": 32,
        },
    ]
    
    # Check for TensorRT
    try:
        import torch_tensorrt
        print(f"✓ torch-tensorrt {torch_tensorrt.__version__} detected")
        tensorrt_available = True
    except ImportError:
        print("⚠ torch-tensorrt not installed (install for best performance)")
        print("  pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121")
        tensorrt_available = False
    print()
    
    # Run benchmarks
    results = []
    baseline_time = None
    
    for i, config in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Testing: {config['name']}")
        print(f"    Batch size: {config['batch_size']}")
        
        # Setup
        if device == "cpu":
            print("    ⚠ Skipping (CUDA optimizations require GPU)")
            continue
            
        config["setup"]()
        
        # Create scorer
        try:
            scorer = BERTScorer(
                lang="python",
                device=device,
                batch_size=config["batch_size"],
                verbose=False,
            )
            
            # Benchmark
            avg_time, throughput = benchmark_inference(scorer, cand_code, ref_code)
            
            # Store results
            if baseline_time is None:
                baseline_time = avg_time
                speedup = 1.0
            else:
                speedup = baseline_time / avg_time
            
            results.append({
                "config": config["name"],
                "time_ms": avg_time,
                "throughput": throughput,
                "speedup": speedup,
            })
            
            print(f"    Time: {avg_time:.2f}ms")
            print(f"    Throughput: {throughput:.1f} inferences/sec")
            print(f"    Speedup: {speedup:.2f}x")
            print()
            
            # Clean up
            del scorer
            if device == "cuda":
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            print()
    
    # Summary
    if results:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Configuration':<40} {'Time (ms)':<12} {'Speedup':<10}")
        print("-" * 70)
        for r in results:
            print(f"{r['config']:<40} {r['time_ms']:>8.2f} ms   {r['speedup']:>6.2f}x")
        print("=" * 70)
        
        # Estimate time savings for full workload
        if len(results) > 1:
            best_speedup = max(r["speedup"] for r in results)
            total_files = 2000  # Approximate from your workload
            baseline_total = (baseline_time / 1000) * total_files
            optimized_total = baseline_total / best_speedup
            time_saved = baseline_total - optimized_total
            
            print()
            print(f"Estimated time for {total_files} files:")
            print(f"  Baseline: {baseline_total/60:.1f} minutes")
            print(f"  Optimized: {optimized_total/60:.1f} minutes")
            print(f"  Time saved: {time_saved/60:.1f} minutes ({best_speedup:.1f}x speedup)")
            print()
            
            if not tensorrt_available:
                print("💡 TIP: Install torch-tensorrt for up to 4-5x speedup!")
                print("   pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121")
    else:
        print("No benchmarks completed successfully")


if __name__ == "__main__":
    main()
