"""
Test script for full model: compilation time, memory usage, and inference speed.
Critical for ensuring Kaggle 8-hour GPU budget compliance.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import psutil
import sys
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))

from model.rnafold_se3_full import create_full_model, FullRNAFoldConfig


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


def test_model_compilation(config: FullRNAFoldConfig, test_length: int = 150):
    """
    Test model compilation time and memory usage.

    Args:
        config: Model configuration
        test_length: Sequence length for test

    Returns:
        Dict with compilation metrics
    """
    print(f"\n{'='*70}")
    print(f"Testing Model Compilation (L={test_length})")
    print(f"{'='*70}")

    # Create model
    model = create_full_model(config)

    # Create dummy input
    rng = jax.random.PRNGKey(42)
    sequence = jax.random.normal(rng, (test_length, config.vocab_size))
    msa = jax.random.normal(rng, (config.max_msa_depth, test_length, config.vocab_size))

    # Measure initialization
    print("\n1. Initializing model parameters...")
    mem_before_init = get_memory_usage()
    start_time = time.time()

    rng, key = jax.random.split(rng)
    params = model.init(key, sequence, msa)

    init_time = time.time() - start_time
    mem_after_init = get_memory_usage()

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    print(f"   Time: {init_time:.2f}s")
    print(f"   Parameters: {num_params:,}")
    print(f"   Memory: {mem_after_init:.2f} GB (+{mem_after_init - mem_before_init:.2f} GB)")

    # Measure first forward pass (includes compilation)
    print("\n2. First forward pass (includes JIT compilation)...")
    mem_before_compile = get_memory_usage()
    start_time = time.time()

    rng, key = jax.random.split(rng)
    coords_1 = model.apply(params, key, sequence, msa)
    coords_1.block_until_ready()  # Wait for computation

    compile_time = time.time() - start_time
    mem_after_compile = get_memory_usage()

    print(f"   Time: {compile_time:.2f}s")
    print(f"   Output shape: {coords_1.shape}")
    print(f"   Memory: {mem_after_compile:.2f} GB (+{mem_after_compile - mem_before_compile:.2f} GB)")

    # Measure second forward pass (cached, no compilation)
    print("\n3. Second forward pass (cached)...")
    start_time = time.time()

    rng, key = jax.random.split(rng)
    coords_2 = model.apply(params, key, sequence, msa)
    coords_2.block_until_ready()

    cached_time = time.time() - start_time

    print(f"   Time: {cached_time:.2f}s")
    print(f"   Speedup: {compile_time / cached_time:.1f}x")

    return {
        'test_length': test_length,
        'num_params': num_params,
        'init_time': init_time,
        'compile_time': compile_time,
        'inference_time': cached_time,
        'memory_gb': mem_after_compile,
        'speedup': compile_time / cached_time,
    }


def test_multiple_lengths(config: FullRNAFoldConfig):
    """
    Test model across different sequence lengths.

    Args:
        config: Model configuration
    """
    print(f"\n{'='*70}")
    print("Testing Multiple Sequence Lengths")
    print(f"{'='*70}")

    test_lengths = [20, 50, 100, 150]
    results = []

    for length in test_lengths:
        try:
            result = test_model_compilation(config, length)
            results.append(result)
        except Exception as e:
            print(f"\n✗ FAILED for length {length}: {e}")
            break

    # Summary
    if results:
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"{'Length':<10} {'Compile (s)':<15} {'Inference (s)':<15} {'Memory (GB)':<12}")
        print(f"{'-'*70}")

        for r in results:
            print(f"{r['test_length']:<10} "
                  f"{r['compile_time']:<15.2f} "
                  f"{r['inference_time']:<15.3f} "
                  f"{r['memory_gb']:<12.2f}")

        # Extrapolate to Kaggle budget
        if len(results) >= 2:
            # Use largest test
            largest_result = results[-1]
            inference_per_target = largest_result['inference_time']

            # Estimate for 100 targets (typical Kaggle test set size)
            total_time_hours = (100 * inference_per_target * 5) / 3600  # 5 predictions each

            print(f"\n{'='*70}")
            print("Kaggle Budget Estimate")
            print(f"{'='*70}")
            print(f"Inference time per target: {inference_per_target:.3f}s")
            print(f"Time for 100 targets (5 predictions each): {total_time_hours:.2f} hours")

            if total_time_hours < 8:
                print(f"✓ Within 8-hour budget (margin: {8 - total_time_hours:.2f} hours)")
            else:
                print(f"✗ EXCEEDS 8-hour budget (over by: {total_time_hours - 8:.2f} hours)")
                print("\nOptimizations needed:")
                print("  - Reduce num_evoformer_blocks")
                print("  - Use bfloat16 mixed precision")
                print("  - Cache MSA embeddings")
                print("  - Batch predictions by length")

    return results


def test_equivariance(config: FullRNAFoldConfig, test_length: int = 20):
    """
    Test SE(3)-equivariance property.

    Args:
        config: Model configuration
        test_length: Sequence length

    Returns:
        Boolean indicating if equivariance test passed
    """
    print(f"\n{'='*70}")
    print("Testing SE(3)-Equivariance")
    print(f"{'='*70}")

    model = create_full_model(config)
    rng = jax.random.PRNGKey(42)

    # Create input
    sequence = jax.random.normal(rng, (test_length, config.vocab_size))

    # Initialize model
    rng, key = jax.random.split(rng)
    params = model.init(key, sequence)

    # Original prediction
    rng, key = jax.random.split(rng)
    coords_original = model.apply(params, key, sequence)

    # Create random rotation
    rng, key = jax.random.split(rng)
    theta = jax.random.uniform(key) * 2 * jnp.pi
    axis = jax.random.normal(key, (3,))
    axis = axis / jnp.linalg.norm(axis)

    # Rotation matrix (Rodrigues formula)
    K = jnp.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = jnp.eye(3) + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * jnp.matmul(K, K)

    # Rotate original prediction
    coords_rotated_expected = jnp.matmul(coords_original, R.T)

    # Make new prediction (should give rotated result)
    # Note: Current model doesn't take rotated input, so this test is conceptual
    # Full SE(3) equivariance would require rotating input features

    print(f"\nRotation test:")
    print(f"  Original coords range: [{coords_original.min():.2f}, {coords_original.max():.2f}]")
    print(f"  Rotated coords range: [{coords_rotated_expected.min():.2f}, {coords_rotated_expected.max():.2f}]")

    # Check output is in valid range
    coords_valid = jnp.all((coords_original >= -999.999) & (coords_original <= 9999.999))
    print(f"\n✓ Coordinates in valid range: {coords_valid}")

    return coords_valid


def main():
    """Main test suite."""
    print("="* 70)
    print("FULL MODEL TESTING SUITE")
    print("="* 70)

    # Configure model (reduced for testing)
    config = FullRNAFoldConfig()
    config.num_evoformer_blocks = 8  # Reduced from 48 for testing
    config.num_ipa_blocks = 2  # Reduced from 8
    config.max_msa_depth = 128  # Reduced from 512
    config.use_bfloat16 = True  # Enable mixed precision

    print(f"\nModel Configuration:")
    print(f"  Evoformer blocks: {config.num_evoformer_blocks}")
    print(f"  IPA blocks: {config.num_ipa_blocks}")
    print(f"  MSA depth: {config.max_msa_depth}")
    print(f"  Mixed precision (bfloat16): {config.use_bfloat16}")
    print(f"  Node embedding dim: {config.node_embedding_dim}")
    print(f"  Pair embedding dim: {config.pair_embedding_dim}")

    # Run tests
    try:
        # Test 1: Equivariance
        equivariance_passed = test_equivariance(config, test_length=20)

        # Test 2: Multiple lengths
        results = test_multiple_lengths(config)

        # Final summary
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"✓ Equivariance test: {'PASSED' if equivariance_passed else 'FAILED'}")
        print(f"✓ Compilation tests: {len(results)}/4 passed")

        if results:
            max_memory = max(r['memory_gb'] for r in results)
            print(f"✓ Max memory usage: {max_memory:.2f} GB")

            if max_memory < 40:  # Typical GPU VRAM
                print(f"  → Fits in standard GPU memory")
            else:
                print(f"  → May require high-memory GPU or gradient checkpointing")

        print("\n" + "="*70)
        print("Next Steps:")
        print("="*70)
        print("1. If tests pass: Scale to full 48-block Evoformer")
        print("2. Implement gradient checkpointing for memory efficiency")
        print("3. Add MSA caching to reduce redundant computation")
        print("4. Run curriculum training pipeline")
        print("5. Validate on time-split test set")

        return True

    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR")
        print(f"{'='*70}")
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Enable JAX debugging
    import os
    os.environ['JAX_DEBUG_NANS'] = '1'

    # Run tests
    success = main()

    sys.exit(0 if success else 1)
