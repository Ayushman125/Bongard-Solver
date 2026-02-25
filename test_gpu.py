#!/usr/bin/env python3
"""Quick GPU diagnostic check."""

import torch
import sys

print("="*60)
print("GPU DIAGNOSTIC TEST")
print("="*60)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        cap = torch.cuda.get_device_capability(i)
        print(f"    Capability: {cap[0]}.{cap[1]}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    print("\nGPU Tensor Test:")
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        result = torch.matmul(test_tensor, test_tensor)
        print(f"  ✓ Successfully created and computed on: {test_tensor.device}")
        print(f"  ✓ GPU is working correctly!")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)
else:
    print("✗ CUDA not available - GPU will not be used")
    sys.exit(1)

print("="*60)
print("GPU TEST PASSED")
print("="*60)
