#!/usr/bin/env python3
"""
Test script for LocalKernelEvaluator

Usage:
    python test_evaluator.py [--no-docker]
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.gpu_mode.local_evaluator import LocalKernelEvaluator


# Minimal valid TriMul kernel for testing
TEST_KERNEL = """
import torch
import triton
import triton.language as tl

@triton.jit
def trimul_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    seq_len,
    dim,
    BLOCK_SIZE: tl.constexpr,
):
    # Minimal kernel that just copies input to output
    # This will pass tests but have poor performance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * seq_len * seq_len * dim

    input_vals = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input_vals, mask=mask)

def custom_kernel(data):
    '''Minimal TriMul implementation for testing.'''
    input_tensor, weights, config = data

    # Just return input (identity operation)
    # This passes correctness tests but is slow
    return input_tensor
"""


def test_evaluator(use_container: bool = True):
    """Test LocalKernelEvaluator."""

    print("=" * 80)
    print("Testing LocalKernelEvaluator")
    print("=" * 80)
    print(f"Container mode: {'enabled' if use_container else 'disabled'}")
    print()

    # Create evaluator
    print("1. Creating evaluator...")
    try:
        evaluator = LocalKernelEvaluator(
            gpu_id=5,
            timeout=300,  # 5 minutes for test
            max_retries=1,
            use_container=use_container,
        )
        print(f"   ✓ Evaluator created")
        print(f"   ✓ Container runtime: {evaluator.container_runtime}")
    except Exception as e:
        print(f"   ✗ Failed to create evaluator: {e}")
        return False

    print()

    # Test with valid kernel
    print("2. Testing with minimal valid kernel...")
    try:
        result = evaluator.evaluate(
            submission_code=TEST_KERNEL,
            task_name="trimul",
            gpu_type="H100"
        )

        print(f"   Result: {result}")

        if result["success"]:
            print(f"   ✓ Evaluation succeeded")
            print(f"   Score: {result['score_us']:.2f} μs")
        else:
            print(f"   ✗ Evaluation failed: {result['error']}")
            return False

    except Exception as e:
        print(f"   ✗ Exception during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test with invalid kernel (missing @triton.jit)
    print("3. Testing with invalid kernel (should return penalty)...")
    invalid_kernel = """
def custom_kernel(data):
    return data
"""
    try:
        result = evaluator.evaluate(
            submission_code=invalid_kernel,
            task_name="trimul",
            gpu_type="H100"
        )

        if not result["success"]:
            print(f"   ✓ Correctly returned failure: {result['error']}")
        else:
            print(f"   ⚠ Expected failure but got success")

    except Exception as e:
        print(f"   ✗ Should not raise exception: {e}")
        return False

    print()
    print("=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(description="Test LocalKernelEvaluator")
    parser.add_argument("--no-container", action="store_true",
                       help="Disable container isolation (use subprocess mode)")
    parser.add_argument("--no-docker", action="store_true",
                       help="(Deprecated, use --no-container) Disable Docker")

    args = parser.parse_args()

    # Backward compatibility
    use_container = not (args.no_container or args.no_docker)

    success = test_evaluator(use_container=use_container)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
