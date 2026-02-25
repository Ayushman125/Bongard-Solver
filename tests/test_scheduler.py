from src.utils.scheduler import AdaptiveScheduler

def test_scheduler_order():
    order = []
    sched = AdaptiveScheduler()
    sched.submit(lambda: order.append("cpu1"), device="cpu")
    sched.submit(lambda: order.append("gpu1"), device="gpu")
    sched.submit(lambda: order.append("cpu2"), device="cpu")
    sched.run_all()
    assert order == ["cpu1","cpu2","gpu1"]
