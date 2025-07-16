import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import os
import logging
import numpy as np
import shutil # Added for dummy test cleanup

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleDaliPipeline(Pipeline):
    """
    A simple DALI pipeline for image loading, decoding, resizing, and normalization.
    This demonstrates basic DALI usage for fast data loading.
    """
    def __init__(self, image_dir, batch_size, num_threads, device_id, img_size):
        super().__init__(batch_size, num_threads, device_id, seed=12345)
        self.input = fn.readers.file(file_root=image_dir, random_shuffle=True)
        self.decode = fn.decoders.image(device="mixed", output_type=types.RGB)
        self.resize = fn.resize(
            self.decode,
            resize_x=img_size[0],
            resize_y=img_size[1],
            interp_type=types.INTERP_LINEAR
        )
        self.normalize = fn.crop_mirror_normalize(
            self.resize,
            dtype=types.FLOAT,
            output_layout=types.NCHW, # Channel-first for PyTorch
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], # ImageNet means
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255]   # ImageNet stds
        )
        logging.info(f"DALI Simple Pipeline initialized for images in {image_dir}")

    def define_graph(self):
        images, _ = self.input()
        images = self.decode(images)
        images = self.resize(images)
        output = self.normalize(images)
        return output

class YoloDaliPipeline(Pipeline):
    """
    A DALI pipeline for YOLO-style data loading, including images and bounding boxes.
    This is a more advanced example, demonstrating how to handle annotations.
    It includes decoding, resizing, and normalization.
    """
    def __init__(self, image_dir, label_dir, batch_size, num_threads, device_id, img_size, class_names):
        super().__init__(batch_size, num_threads, device_id, seed=12345)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.class_names = class_names

        # DALI readers for images and labels
        # Note: DALI's built-in readers for bounding boxes are typically for TFRecords or COCO format.
        # For YOLO .txt files, we often need a custom Python operator or pre-process into a DALI-friendly format.
        # For demonstration, we'll use file_reader for images and assume labels are handled externally or by a custom operator.
        # A more robust solution would involve a custom `ExternalSource` operator for parsing YOLO .txt files.
        self.jpegs = fn.readers.file(file_root=image_dir, random_shuffle=True, name="Reader")
        
        # This is a simplification. In a real YOLO DALI pipeline, you'd parse labels here too.
        # For now, we'll focus on image processing.
        logging.warning("YOLO DALI Pipeline: Label loading for .txt files is conceptual. Consider `ExternalSource` for production.")

        self.decode = fn.decoders.image(self.jpegs, device="mixed", output_type=types.RGB)
        self.resize = fn.resize(
            self.decode,
            resize_x=img_size[0],
            resize_y=img_size[1],
            interp_type=types.INTERP_LINEAR
        )
        self.normalize = fn.crop_mirror_normalize(
            self.resize,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
        )
        logging.info(f"DALI YOLO Pipeline initialized for images in {image_dir}")

    def define_graph(self):
        images, filenames = self.jpegs() # filenames are also returned by file reader
        images = self.decode(images)
        images = self.resize(images)
        output_images = self.normalize(images)
        
        # In a real YOLO pipeline, you'd also return normalized bounding boxes and class IDs here.
        # For this conceptual example, we just return images.
        return output_images #, bounding_boxes, class_labels

def prepare_dataset(output_dir: str, dali_config: dict):
    """
    Prepares the dataset using NVIDIA DALI pipelines.
    This function demonstrates how to initialize and run a DALI pipeline.
    Args:
        output_dir (str): The root directory of the processed dataset (e.g., ../data/processed).
        dali_config (dict): DALI configuration from config.yaml.
    """
    if not os.path.exists(output_dir):
        logging.error(f"Output directory for DALI not found: {output_dir}. Skipping DALI preparation.")
        return

    dali_enabled = dali_config.get('enabled', False)
    if not dali_enabled:
        logging.info("DALI integration is disabled in config. Skipping DALI preparation.")
        return

    pipeline_type = dali_config.get('pipeline_type', 'simple')
    batch_size = dali_config.get('batch_size', 16)
    num_threads = dali_config.get('num_threads', 4)
    device_id = 0 # Assuming single GPU, can be configured
    img_size = dali_config.get('image_size', [640, 640])
    iterations = dali_config.get('iterations', 100)

    train_img_dir = os.path.join(output_dir, 'images', 'train')
    train_lbl_dir = os.path.join(output_dir, 'labels', 'train') # For YOLO pipeline

    if not os.path.exists(train_img_dir):
        logging.warning(f"Training image directory not found for DALI: {train_img_dir}. Skipping DALI pipeline run.")
        return

    # Dummy class names for DALI YOLO pipeline initialization if needed
    class_names = ["object"] # Replace with actual class names if available

    logging.info(f"Initializing DALI pipeline of type '{pipeline_type}'...")
    if pipeline_type == 'simple':
        pipeline = SimpleDaliPipeline(
            image_dir=train_img_dir,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            img_size=img_size
        )
    elif pipeline_type == 'yolo':
        pipeline = YoloDaliPipeline(
            image_dir=train_img_dir,
            label_dir=train_lbl_dir, # This is conceptual for YOLO .txt
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            img_size=img_size,
            class_names=class_names # Pass actual class names if available
        )
    else:
        logging.error(f"Unknown DALI pipeline type: {pipeline_type}. Skipping DALI preparation.")
        return

    pipeline.build()
    logging.info("DALI pipeline built successfully.")

    # Run a few iterations to demonstrate data loading
    logging.info(f"Running {iterations} iterations of DALI pipeline to demonstrate data loading...")
    try:
        from tqdm import tqdm # Ensure tqdm is imported for this block
        for i in tqdm(range(iterations), desc="DALI Pipeline Iterations"):
            pipeline_output = pipeline.run()
            # print(f"Batch {i}: Images shape: {pipeline_output[0].as_tensor().shape}")
            # In a real scenario, you'd feed this batch to your model for training
    except Exception as e:
        logging.error(f"Error running DALI pipeline: {e}")
        import traceback
        traceback.print_exc()

    logging.info("DALI pipeline demonstration complete.")

# This file is intended to be imported and its functions called, not run directly.
if __name__ == '__main__':
    # Example usage for direct testing (not part of the main pipeline flow)
    # You would typically pass the actual output_dir from your main pipeline.
    print("This script is typically imported. Running a dummy test.")
    dummy_config = {
        'enabled': True,
        'pipeline_type': 'simple',
        'batch_size': 4,
        'num_threads': 2,
        'image_size': [224, 224],
        'iterations': 10
    }
    # Create a dummy image directory for testing DALI
    dummy_img_root = "dali_test_images"
    os.makedirs(dummy_img_root, exist_ok=True)
    # Create a few dummy images
    from PIL import Image
    for i in range(10):
        dummy_img = Image.new('RGB', (100, 100), color=(i*20 % 255, i*30 % 255, i*40 % 255))
        dummy_img.save(os.path.join(dummy_img_root, f'test_img_{i}.jpg'))
    
    # Call prepare_dataset with a dummy output_dir structure
    dummy_output_dir = "dali_test_output_structure"
    os.makedirs(os.path.join(dummy_output_dir, 'images', 'train'), exist_ok=True)
    # Move dummy images into the DALI-expected train folder
    for f in os.listdir(dummy_img_root):
        shutil.move(os.path.join(dummy_img_root, f), os.path.join(dummy_output_dir, 'images', 'train', f))

    prepare_dataset(dummy_output_dir, dummy_config)

    # Clean up dummy files
    shutil.rmtree(dummy_output_dir)
    shutil.rmtree(dummy_img_root) # This will be empty after move
    print("DALI dummy test complete and cleaned up.")
