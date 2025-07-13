# Folder: bongard_solver/
# File: pipeline_worker2.py
import time
import logging
import queue # For Queue.Full and Queue.Empty exceptions

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Dummy configuration for demonstration purposes
# In a real application, this would come from a config file or passed in.
class DummyConfig:
    def __init__(self):
        self.dali = {
            'backoff_base': 0.1,  # Initial backoff time in seconds
            'backoff_max': 5.0    # Maximum backoff time in seconds
        }
cfg = DummyConfig()

# Dummy DALI pipeline and queue for demonstration
# In a real scenario, 'pipe' would be an instance of a DALI pipeline
# and 'q' would be a multiprocessing.Queue or similar.
class DummyPipeline:
    def run(self):
        # Simulate batch processing time
        time.sleep(0.05)
        # Simulate occasional empty/full conditions for testing backoff logic
        if random.random() < 0.05: # 5% chance to simulate queue full
            raise queue.Full
        if random.random() < 0.02: # 2% chance to simulate queue empty
            raise queue.Empty
        return {"data": "dummy_batch"}

class DummyQueue:
    def __init__(self, maxsize=10):
        self._queue = collections.deque(maxlen=maxsize)
        self.maxsize = maxsize

    def put(self, item, timeout=None):
        if len(self._queue) >= self.maxsize:
            raise queue.Full
        self._queue.append(item)

    def get(self, timeout=None):
        if not self._queue:
            raise queue.Empty
        return self._queue.popleft()

pipe = DummyPipeline()
q = DummyQueue(maxsize=5) # A small queue to easily trigger overflow/underflow

def worker():
    """
    Worker function to process batches from a DALI pipeline and put them into a queue.
    Includes backoff logic for queue overflow and logging for underflow.
    """
    backoff = cfg.dali['backoff_base']
    logger.info("Worker started.")
    while True:
        try:
            batch = pipe.run() # Get batch from DALI pipeline
            q.put(batch, timeout=1) # Put batch into the queue
            backoff = cfg.dali['backoff_base'] # Reset backoff on successful operation
            logger.info("Batch processed and put into queue.")
        except queue.Full:
            logger.warning(f"Queue overflow, backing off for {backoff:.2f} seconds.")
            time.sleep(backoff)
            backoff = min(backoff * 2, cfg.dali['backoff_max']) # Exponential backoff
        except queue.Empty:
            logger.warning("Queue underflow, pipeline might be idle or empty.")
            # No backoff for underflow, just log and continue trying
            time.sleep(0.1) # Small sleep to prevent busy-waiting if pipeline is truly empty
        except Exception as e:
            logger.error(f"An unexpected error occurred in worker: {e}")
            time.sleep(1) # Sleep on unexpected errors to prevent rapid error looping

if __name__ == "__main__":
    import collections
    import random
    # You can run this directly to see the worker's behavior with dummy components
    # For a real application, this worker function would be run in a separate process/thread.
    logger.info("Running dummy worker for demonstration. Press Ctrl+C to stop.")
    try:
        worker()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user.")

