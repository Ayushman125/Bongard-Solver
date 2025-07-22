"""
CUDAStreamManager: overlap host<->device transfers with compute via PyTorch streams.
Version 0.1.0
Public API:
  - async_transfer(dst: torch.Tensor, src: torch.Tensor)
  - compute(fn, *args, **kwargs)
  - synchronize()
"""

__version__ = "0.1.0"
"""
CUDAStreamManager

Overlap host↔device transfers and compute using PyTorch streams:

  mgr = CUDAStreamManager()
  mgr.async_transfer(dst, src)
  mgr.compute(kernel_fn, args…)
  mgr.synchronize()
"""
    def async_transfer(self, dst: torch.Tensor, src: torch.Tensor):
        """
        Copy src→dst asynchronously on stream_in.
        """
        with torch.cuda.stream(self.stream_in):
            dst.copy_(src, non_blocking=True)

    def compute(self, fn, *args, **kwargs):
        """
        Execute fn(*args,**kwargs) on stream_compute.
        """
        with torch.cuda.stream(self.stream_compute):
            return fn(*args, **kwargs)

    def synchronize(self):
        """
        Wait for both streams to finish.
        """
        torch.cuda.synchronize()
"""
CUDAStreamManager: overlap host<->device transfers with compute.
Version: 0.1.0
"""

__version__ = "0.1.0"

import torch

class CUDAStreamManager:
    def __init__(self):
        self.stream_in = torch.cuda.Stream()
        self.stream_compute = torch.cuda.Stream()

    def async_transfer(self, tensor_dst, tensor_src):
        """
        Asynchronously copy tensor_src → tensor_dst on `stream_in`.
        """
        with torch.cuda.stream(self.stream_in):
            tensor_dst.copy_(tensor_src, non_blocking=True)

    def compute(self, fn, *args, **kwargs):
        """
        Run compute on `stream_compute` while host transfers may still be in-flight.
        """
        with torch.cuda.stream(self.stream_compute):
            return fn(*args, **kwargs)

    def synchronize(self):
        """Wait for both streams to finish."""
        torch.cuda.synchronize(self.stream_in)
        torch.cuda.synchronize(self.stream_compute)
