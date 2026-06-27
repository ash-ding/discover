#!/usr/bin/env python3
"""
Simple test for LocalKernelEvaluator (no container, subprocess mode)
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Set environment before import
os.environ["KERNEL_EVAL_USE_DOCKER"] = "false"  # Backward compatibility
os.environ["KERNEL_EVAL_USE_CONTAINER"] = "false"

from examples.gpu_mode.local_evaluator import LocalKernelEvaluator

# Minimal submission code (will fail tests but validates worker runs)
test_code = """
import torch

def custom_kernel(data):
    '''Minimal test - just returns input.'''
    if isinstance(data, tuple) and len(data) >= 1:
        return data[0]
    return data
"""

print("=" * 80)
print("Simple LocalKernelEvaluator Test (Subprocess Mode)")
print("=" * 80)
print()

# Create evaluator
print("1. Creating evaluator...")
evaluator = LocalKernelEvaluator(
    gpu_id=5,
    timeout=60,  # Short timeout for test
    max_retries=0,  # No retries for test
    use_container=False,
)
print(f"   ✓ Evaluator created")
print(f"   ✓ Runtime: {evaluator.container_runtime} (subprocess mode)")
print()

# Test evaluation
print("2. Running evaluation...")
print("   (This will likely fail correctness tests, but proves worker runs)")
result = evaluator.evaluate(
    submission_code=test_code,
    task_name="trimul",
    gpu_type="H100"
)

print()
print("3. Result:")
print(f"   Success: {result['success']}")
print(f"   Score: {result['score_us']}")
print(f"   Error: {result.get('error', 'None')}")

if not result['success']:
    print()
    print("   ⚠ Evaluation failed (expected - test code doesn't implement TriMul)")
    print("   ✓ But worker ran successfully and returned result!")
else:
    print()
    print("   ✓ Evaluation succeeded (unexpected but good!)")

print()
print("=" * 80)
print("Test completed successfully!")
print("Worker can run and return results.")
print("=" * 80)
