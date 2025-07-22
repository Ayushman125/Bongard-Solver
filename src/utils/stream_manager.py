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

    def async_transfer(self, dst: torch.Tensor, src: torch.Tensor):
        with torch.cuda.stream(self.stream_in):
            dst.copy_(src, non_blocking=True)

    def compute(self, fn, *args, **kwargs):
        with torch.cuda.stream(self.stream_compute):
            return fn(*args, **kwargs)

    def synchronize(self):
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
