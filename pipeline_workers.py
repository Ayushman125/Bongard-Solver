# Folder: bongard_solver/
# File: pipeline_worker.py

import time
import logging
import queue # For Queue.Full and Queue.Empty exceptions
import multiprocessing # For multiprocessing.Queue
import random # For dummy DALI iterator

# Import project-specific modules from the main root
try:
    from config import CONFIG # Your global configuration, now directly importable
    # Import DALI components if available
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    HAS_DALI = True
except ImportError:
    logging.warning("DALI or CONFIG not found. Pipeline worker will operate in dummy mode.")
    HAS_DALI = False
    # Define dummy CONFIG for standalone testing if imports fail
    class DummyConfig:
        def __init__(self):
            self.dali = {
                'backoff_base': 0.1,   # Initial backoff time in seconds
                'backoff_max': 5.0     # Maximum backoff time in seconds
            }
            self.debug = {'log_level': 'INFO'}
            self.training = {'batch_size': 4} # Dummy batch size for mock iterator
    CONFIG = DummyConfig()
    # Dummy DALI components for testing without DALI installed
    class Pipeline:
        def __init__(self, *args, **kwargs): pass
        def build(self): pass
        def run(self):
            time.sleep(0.05) # Simulate work
            if random.random() < 0.05: raise queue.Full
            if random.random() < 0.02: raise queue.Empty
            return {"data": "dummy_batch"} # Return a dummy batch
    class DALIGenericIterator:
        def __init__(self, pipeline, output_map, size, auto_reset, fill_last_batch, dynamic_shape, reader_name=None, source=None):
            self.pipeline = pipeline
            self.size = size
            self.current_idx = 0
            self.output_map = output_map
            self.source = source # For external source
            logging.warning("Using dummy DALIGenericIterator.")

        def __iter__(self):
            self.current_idx = 0
            if self.source:
                self.source_iter = iter(self.source)
            return self

        def __next__(self):
            if self.current_idx >= self.size:
                raise StopIteration
            
            # Simulate getting a batch
            if self.source: # For external source
                try:
                    raw_batch = next(self.source_iter)
                    # Convert raw_batch (tuple of numpy arrays/objects) to a dict
                    # This assumes the order matches output_map
                    processed_batch = {self.output_map[i]: torch.from_numpy(raw_batch[i]) if isinstance(raw_batch[i], np.ndarray) and raw_batch[i].dtype != object else raw_batch[i] for i in range(len(self.output_map))}
                    self.current_idx += len(processed_batch[self.output_map[0]]) # Increment by batch size
                    return processed_batch
                except StopIteration:
                    raise StopIteration
            else: # For FileReader
                # Simulate a batch from FileReader
                time.sleep(0.01)
                batch_data = {}
                for key in self.output_map:
                    if "img" in key:
                        batch_data[key] = torch.randn(1, 3, 224, 224) # Dummy image tensor
                    elif "label" in key:
                        batch_data[key] = torch.randint(0, 2, (1,)) # Dummy label
                    else:
                        batch_data[key] = "dummy_data" # Other dummy data
                self.current_idx += 1
                return batch_data


# Configure logging
logger = logging.getLogger(__name__)
# Set level from config, default to INFO if not found
logger.setLevel(getattr(logging, CONFIG.debug.get('log_level', 'INFO').upper()))
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def pipeline_worker(
    dali_iterator: DALIGenericIterator,
    output_queue: multiprocessing.Queue,
    cfg: Dict[str, Any]
):
    """
    Worker function to continuously pull batches from a DALI iterator
    and put them into a multiprocessing queue.
    Includes exponential backoff for queue overflow and logging for underflow.

    Args:
        dali_iterator (DALIGenericIterator): An initialized DALI iterator.
        output_queue (multiprocessing.Queue): The queue to put processed batches into.
        cfg (Dict[str, Any]): The configuration dictionary, containing DALI backoff settings.
    """
    backoff = cfg['dali']['backoff_base']
    max_backoff = cfg['dali']['backoff_max']
    
    logger.info("DALI Pipeline Worker started.")
    
    # Iterate through the DALI iterator to get batches
    for i, batch in enumerate(dali_iterator):
        try:
            # Put batch into the queue. Use a timeout to prevent indefinite blocking.
            # If the queue is full, it will raise queue.Full.
            output_queue.put(batch, timeout=1) 
            backoff = cfg['dali']['backoff_base'] # Reset backoff on successful operation
            logger.debug(f"Batch {i} processed and put into queue. Queue size: {output_queue.qsize()}")
        except queue.Full:
            logger.warning(f"Output queue is full, backing off for {backoff:.2f} seconds. Batch {i} retrying.")
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff) # Exponential backoff
            # Re-attempt to put the current batch after backoff
            try:
                output_queue.put(batch, timeout=1)
                backoff = cfg['dali']['backoff_base'] # Reset backoff on success
                logger.info(f"Batch {i} successfully put into queue after backoff.")
            except queue.Full:
                logger.error(f"Batch {i} failed to put into queue even after backoff. Dropping batch.")
                # If it still fails, consider logging this as a dropped batch and continue
                # Or implement a more sophisticated retry mechanism / error handling
        except Exception as e:
            logger.error(f"An unexpected error occurred while putting batch into queue: {e}", exc_info=True)
            time.sleep(1) # Sleep on unexpected errors to prevent rapid error looping
    
    logger.info("DALI Pipeline Worker finished processing all batches from iterator.")

if __name__ == "__main__":
    # Example usage for testing the worker function directly
    # In a real application, this worker would be launched by multiprocessing.Process
    import torch # Needed for dummy tensors

    # Dummy DALI iterator (replace with your actual DALI loader)
    class MockDALIIterator:
        def __init__(self, num_batches=10):
            self.num_batches = num_batches
            self.current_batch = 0
            self.output_map = [ # Full output map matching custom_collate_fn
                "query_img1", "query_img2", "query_labels",
                "query_gts_json_view1", "query_gts_json_view2", "difficulties",
                "affine1", "affine2", "original_indices",
                "padded_support_imgs", "padded_support_labels", "padded_support_sgs_bytes",
                "num_support_per_problem", "tree_indices", "is_weights",
                "query_bboxes_view1", "query_masks_view1",
                "query_bboxes_view2", "query_masks_view2",
                "support_bboxes_flat", "support_masks_flat"
            ]

        def __iter__(self):
            self.current_batch = 0
            return self

        def __next__(self):
            if self.current_batch >= self.num_batches:
                raise StopIteration
            
            # Simulate a batch of data
            batch_size = CONFIG['training']['batch_size'] if hasattr(CONFIG, 'training') else 4
            image_size = CONFIG['data']['image_size'][0] if hasattr(CONFIG, 'data') else 224
            max_support = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem'] if hasattr(CONFIG, 'data') else 5

            dummy_batch = {
                "query_img1": torch.randn(batch_size, 3, image_size, image_size),
                "query_img2": torch.randn(batch_size, 3, image_size, image_size),
                "query_labels": torch.randint(0, 2, (batch_size,)),
                "query_gts_json_view1": [b'{}'] * batch_size,
                "query_gts_json_view2": [b'{}'] * batch_size,
                "difficulties": torch.rand(batch_size),
                "affine1": [np.eye(3).tolist()] * batch_size,
                "affine2": [np.eye(3).tolist()] * batch_size,
                "original_indices": torch.arange(self.current_batch * batch_size, (self.current_batch + 1) * batch_size),
                "padded_support_imgs": torch.randn(batch_size, max_support, 3, image_size, image_size),
                "padded_support_labels": torch.randint(-1, 2, (batch_size, max_support)),
                "padded_support_sgs_bytes": [b'{}'] * batch_size,
                "num_support_per_problem": torch.randint(0, max_support + 1, (batch_size,)),
                "tree_indices": torch.arange(self.current_batch * batch_size, (self.current_batch + 1) * batch_size),
                "is_weights": torch.rand(batch_size),
                "query_bboxes_view1": [[[] for _ in range(random.randint(1,3))]] * batch_size, # List of lists of lists
                "query_masks_view1": [[[] for _ in range(random.randint(1,3))]] * batch_size, # List of lists of lists
                "query_bboxes_view2": [[[] for _ in range(random.randint(1,3))]] * batch_size,
                "query_masks_view2": [[[] for _ in range(random.randint(1,3))]] * batch_size,
                "support_bboxes_flat": [[[] for _ in range(random.randint(1,3))] for _ in range(batch_size)], # List of lists of lists
                "support_masks_flat": [[[] for _ in range(random.randint(1,3))] for _ in range(batch_size)], # List of lists of lists
            }
            self.current_batch += 1
            return dummy_batch

    # Create a mock DALI iterator and a multiprocessing queue
    mock_dali_iter = MockDALIIterator(num_batches=20)
    mock_output_queue = multiprocessing.Queue(maxsize=5) # Small queue to test overflow

    logger.info("Running dummy pipeline worker for demonstration. Press Ctrl+C to stop.")
    try:
        # Call the worker function directly
        pipeline_worker(mock_dali_iter, mock_output_queue, CONFIG)
    except KeyboardInterrupt:
        logger.info("Worker stopped by user.")
    finally:
        # Consume any remaining items in the queue
        while not mock_output_queue.empty():
            try:
                _ = mock_output_queue.get_nowait()
            except queue.Empty:
                break # Queue is empty
        mock_output_queue.close()
        mock_output_queue.join_thread()
        logger.info("Dummy queue closed.")

