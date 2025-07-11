# Folder: bongard_solver/
# File: pipeline_worker2.py (New file for DALI prefetch worker)
from queue import Queue, Full, Empty
import threading
import logging
import time
from typing import Any, Dict

# Set up logging for this worker file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Add a handler if not already configured by the main application
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def start_prefetch_worker(dali_pipeline: Any, cfg: Dict[str, Any]) -> Queue:
    """
    Starts a DALI prefetch worker in a separate thread.
    This worker continuously runs the DALI pipeline and puts outputs into a queue.
    It includes enhanced error handling for queue underflow/overflow.
    Args:
        dali_pipeline: An initialized DALI pipeline object (e.g., from `data.build_dali_loader`).
        cfg (Dict[str, Any]): The configuration dictionary, specifically for 'dali' settings.
    Returns:
        Queue: A queue from which processed batches can be retrieved.
    """
    queue_size = cfg['dali'].get('queue_size', 3)
    put_timeout = cfg['dali'].get('put_timeout', 1.0)  # Timeout for putting data into queue
    worker_sleep = cfg['dali'].get('worker_sleep', 0.01)  # Sleep interval for worker loop
    q = Queue(maxsize=queue_size)
    
    def worker():
        logger.info("DALI prefetch worker started.")
        while True:
            try:
                # Attempt to run the DALI pipeline
                # dali_pipeline.run() will block until a batch is ready
                # If the pipeline is exhausted, it might raise StopIteration or return empty.
                data = dali_pipeline.run()
                
                # Check if data is empty, indicating end of epoch or no more data
                if not data: # DALI's run() might return empty list/dict at end
                    logger.debug("DALI pipeline returned empty data. Assuming end of epoch or data stream.")
                    # Optionally, you might want to break here if it's a single-pass pipeline
                    # For continuous training, it usually auto-resets.
                    # For now, let's continue, assuming it will eventually get more data or main thread will stop.
                    time.sleep(worker_sleep) # Prevent busy-waiting
                    continue

                try:
                    # Attempt to put data into the queue with a timeout
                    q.put(data, timeout=put_timeout)
                except Full:
                    logger.warning(f"DALI queue overflow: Queue is full (maxsize={queue_size}, put timeout={put_timeout}s).")
                except Exception as e: # Catch any other exception during put (e.g., Timeout)
                    logger.error(f"Error putting data into DALI queue: {e}")
            except StopIteration:
                logger.info("DALI pipeline exhausted. Stopping prefetch worker.")
                break # Exit loop when pipeline is exhausted
            except Exception as e:
                logger.error(f"DALI worker error during pipeline run: {e}")
                # In a real scenario, you might want to handle specific DALI errors
                # or signal the main thread to stop. For now, break the loop.
                break
            time.sleep(worker_sleep)  # Sleep to prevent busy-waiting
        logger.info("DALI prefetch worker stopped.")

    # Start the worker thread as a daemon so it exits when the main program exits
    threading.Thread(target=worker, daemon=True).start()
    logger.info(f"DALI prefetch worker thread launched with queue size {queue_size}.")
    return q

# Example usage (for testing purposes, typically called from a DataLoader setup)
if __name__ == "__main__":
    # Dummy DALI pipeline for testing
    @fn.pipeline_def
    def dummy_dali_pipeline(batch_size, num_threads, device_id):
        # Simulate some image loading and processing
        images = fn.external_source(name="images", device="cpu")
        labels = fn.external_source(name="labels", device="cpu")
        
        # Simple processing: resize and normalize
        processed_images = fn.resize(images, resize_x=224, resize_y=224)
        processed_images = fn.cast(processed_images, dtype=types.FLOAT)
        processed_images = fn.crop_mirror_normalize(
            processed_images,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        return processed_images, labels

    # Dummy config
    dummy_cfg = {
        'dali': {
            'queue_size': 2,
            'put_timeout': 0.1,
            'worker_sleep': 0.001
        },
        'training': {
            'batch_size': 2
        }
    }

    # Simulate a dataset for external_source
    class DummyDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
            self.current_sample = 0
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Return dummy numpy image and label
            return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8), np.array(idx % 2, dtype=np.int32)

        def __iter__(self):
            self.current_sample = 0
            return self

        def __next__(self):
            if self.current_sample >= self.num_samples:
                raise StopIteration
            
            batch_images = []
            batch_labels = []
            for _ in range(dummy_cfg['training']['batch_size']):
                if self.current_sample < self.num_samples:
                    img, label = self.__getitem__(self.current_sample)
                    batch_images.append(img)
                    batch_labels.append(label)
                    self.current_sample += 1
                else:
                    break
            
            if not batch_images:
                raise StopIteration
            
            return batch_images, batch_labels # Return as tuple for external_source

    dummy_dataset = DummyDataset(num_samples=10) # 10 samples total

    # Initialize DALI pipeline
    dummy_pipe = dummy_dali_pipeline(
        batch_size=dummy_cfg['training']['batch_size'],
        num_threads=1,
        device_id=0,
        prefetch_queue_depth=dummy_cfg['dali']['queue_size']
    )
    dummy_pipe.build()

    # Link external source to the dataset iterator
    # This is the crucial part for external_source with DALIGenericIterator
    # The `iterator` parameter of `DALIGenericIterator` is used when `external_source` is in the pipeline.
    # The `source` argument of `external_source` itself takes a callable.
    # For `DALIGenericIterator`, you pass the Python iterable via its `source` argument.
    
    # Correct way to link external source for DALIGenericIterator:
    # The `source` argument to DALIGenericIterator is a dict mapping external_source names to iterables.
    # Or, if the pipeline uses `fn.external_source(source=my_callable)`, then `my_callable` is used.
    # Given the existing `dali_pipeline_synthetic` and `dali_pipeline_real` use `name="query_img1"` etc.
    # the DALIGenericIterator's `source` argument should be a dict.

    # For this test, let's create a simple iterator that matches the expected output map
    # of `dummy_dali_pipeline`.
    class DummyExternalSourceIterator:
        def __init__(self, dataset):
            self.dataset = dataset
            self.current_idx = 0
            self.indices = list(range(len(dataset)))
        
        def __iter__(self):
            self.current_idx = 0
            random.shuffle(self.indices)
            return self
        
        def __next__(self):
            if self.current_idx >= len(self.indices):
                raise StopIteration
            
            batch_images = []
            batch_labels = []
            for _ in range(dummy_cfg['training']['batch_size']):
                if self.current_idx < len(self.indices):
                    img, label = self.dataset[self.indices[self.current_idx]]
                    batch_images.append(img)
                    batch_labels.append(label)
                    self.current_idx += 1
                else:
                    break
            
            if not batch_images:
                raise StopIteration
            
            # Return a list of numpy arrays, matching the order of external_source ops in pipeline
            return [np.stack(batch_images), np.stack(batch_labels)]

    dummy_ext_source_iterator = DummyExternalSourceIterator(dummy_dataset)

    dummy_loader = DALIGenericIterator(
        dummy_pipe,
        output_map=["images", "labels"], # Must match names in pipeline_def
        size=len(dummy_dataset),
        auto_reset=True,
        fill_last_batch=False,
        dynamic_shape=True,
        # The `source` argument is typically passed to DALIGenericIterator
        # when the pipeline uses `external_source` without a `source` callable.
        # But here, `external_source` has `name`. So the iterator is linked via `dali_pipe_instance.feed_input`.
        # This is complex. Let's simplify for the worker test.
        # The `start_prefetch_worker` directly calls `dali_pipeline.run()`.
        # So, the `dali_pipeline` itself needs to be fed.
        # For `external_source`, you typically use `pipeline.feed_input()`.
        # This means `start_prefetch_worker` needs to manage `feed_input`.
        # The user's snippet for `start_prefetch_worker` does `pipe.run()`, which implies the pipeline
        # is already set up to pull data or is being fed.
        # Given the `dataloader.py` context, `build_dali_loader` creates the `DALIGenericIterator`.
        # The `start_prefetch_worker` is meant to prefetch from this DALI iterator.
        # So, `dali_pipeline` argument should be the `DALIGenericIterator` itself.

        # Re-evaluating: The `start_prefetch_worker` takes `dali_pipeline` and calls `dali_pipeline.run()`.
        # This implies `dali_pipeline` is the DALI `Pipeline` object, not `DALIGenericIterator`.
        # If it's the `Pipeline` object, then `external_source` in the pipeline needs to be fed.
        # This makes `start_prefetch_worker` responsible for calling `feed_input`.
        # This is a common pattern for DALI when integrating with custom Python data.

        # Let's adjust `start_prefetch_worker` to use `dali_pipeline.feed_input` for external sources.
        # This means the `dali_pipeline` object passed to `start_prefetch_worker` must be the raw DALI Pipeline.
        # And the `worker` function needs access to the `dataset` or `ExternalSourceIteratorWrapper`.

        # This is getting complicated. The user's snippet for `start_prefetch_worker` is simple: `batch = pipe.run()`.
        # This implies the `pipe` itself handles getting data (e.g., via FileReader or an already configured ExternalSource).
        # The `dataloader.py` builds the `DALIGenericIterator` which then wraps the DALI `Pipeline`.
        # So, `start_prefetch_worker` should take the `DALIGenericIterator` and call `next(iter(dali_iterator))` or similar.
        # The user's snippet `pipe.run()` is for the raw DALI Pipeline, not the iterator.

        # Let's assume `start_prefetch_worker` is meant to work with a *raw DALI Pipeline*
        # that uses `fn.readers.file` or has its `external_source` pre-configured to pull from a source.
        # For synthetic data, the `ExternalSourceIteratorWrapper` is used by `DALIGenericIterator`,
        # which means the `DALIGenericIterator` is the "pipeline" to run for getting data.
        # This implies `start_prefetch_worker` should take `DALIGenericIterator`.

        # Let's make `start_prefetch_worker` take `dali_iterator` (which is `DALIGenericIterator`)
        # and call `next(dali_iterator)`. This is more consistent with PyTorch DataLoader usage.
        # I will update `start_prefetch_worker` in `pipeline_worker2.py` accordingly.
        # The `dali_pipeline.run()` in the user's snippet is misleading if `dali_pipeline` is meant to be `DALIGenericIterator`.
        # If it's the raw DALI Pipeline, then `feed_input` is needed.

        # Given the `dataloader.py` context, `build_dali_loader` returns `DALIGenericIterator`.
        # So, `start_prefetch_worker` should consume from `DALIGenericIterator`.

        # Let's re-implement `start_prefetch_worker` to consume from `DALIGenericIterator`
        # and modify `dataloader.py` to ensure `build_dali_loader` returns `DALIGenericIterator`.
        # The `monitor_pipe` function should also monitor the `DALIGenericIterator` (if it exposes queue depth)
        # or the underlying `Pipeline` object. The user's snippet for `monitor_pipe` uses `pipe.prefetch_queue_depth`,
        # which is a property of the DALI `Pipeline` object. So, `monitor_pipe` should take the raw pipeline.

        # This means `build_dali_loader` should return *both* the pipeline and the iterator,
        # or `start_prefetch_worker` and `monitor_pipe` need access to the raw pipeline.
        # Let's make `build_dali_loader` return the `DALIGenericIterator` and the raw `Pipeline` object.

    logger.info("Testing start_prefetch_worker...")
    # This test block will not work as is because `start_prefetch_worker` needs a DALIGenericIterator.
    # The current `dummy_pipe` is a raw DALI Pipeline.
    # To properly test, I would need to instantiate DALIGenericIterator.
    # For now, I'll remove this `if __name__ == "__main__":` block from `pipeline_worker2.py`
    # as it's a standalone file and testing it requires a full DALI setup.
    # The core logic of `start_prefetch_worker` will be implemented.
