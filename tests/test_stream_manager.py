import torch
from src.utils.stream_manager import CUDAStreamManager

def test_stream_overlap():
    mgr = CUDAStreamManager()
    dev = torch.device("cuda")
    a = torch.zeros(1000000, device=dev)
    b = torch.ones_like(a)
    mgr.async_transfer(a, b)  # enqueue copy
    # dummy compute
    def comp():
        return (a * 2).sum().item()
    res = mgr.compute(comp)
    mgr.synchronize()
    assert res == pytest.approx(b.sum().item()*2)
