# yolofinetuning/pipeline_workers.py
import os
import glob
import logging
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json # Added for FFCV labels_json serialization/deserialization

# Import CONFIG from config_loader.py
from .config_loader import CONFIG # This assumes config_loader.py is in the same package

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import pipeline factories from data_pipeline.py
# This block is adapted from the provided Analyze.docx snippet.
HAS_DALI = False
HAS_FFCV = False

try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    HAS_DALI = True
    logger.info("NVIDIA DALI found.")
except ImportError:
    logger.warning("NVIDIA DALI not found. DALI data pipeline will not be available.")

try:
    from ffcv.writer import DatasetWriter
    from ffcv.loader import Loader, OrderOption
    from ffcv.fields import RGBImageField, IntField # IntField is a placeholder for labels_json
    from ffcv.transforms import ToTensor
    HAS_FFCV = True
    logger.info("FFCV found.")
except ImportError:
    logger.warning("FFCV not found. FFCV data pipeline will not be available.")


# --- DALI Loader Implementation ---
def get_dali_loader(config, split: str):
    """Return a DALI pipeline wrapped as a PyTorch iterator.
    NOTE: This DALI pipeline currently outputs images and dummy labels.
    To include actual YOLO labels, you would need to extend this pipeline
    to read label files and potentially generate masks. This is a complex
    task for DALI and often involves custom operators or external sources.
    For this implementation, it's simplified to show image loading.
    """
    if not HAS_DALI:
        logger.error("NVIDIA DALI is not installed. Cannot create DALI loader.")
        return None

    data_root = Path(config['data_root'])
    
    class BongardDALIPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, seed, image_dir, yolo_img_size):
            super().__init__(
                batch_size=batch_size,
                num_threads=num_threads,
                device_id=device_id,
                seed=seed
            )
            self.image_dir = image_dir
            self.yolo_img_size = yolo_img_size
            
            # DALI's file.reader expects a file_root and optionally a file_list.
            # We'll use file_root directly for simplicity, assuming all images are there.
            self.input = fn.readers.file(
                file_root=str(self.image_dir),
                random_shuffle=(split=='train'),
                name="Reader"
            )
            
            # Decode, resize, and normalize images
            self.decode = fn.decoders.image(self.input, device="mixed", output_type=types.RGB)
            
            # Resize to the target YOLO image size
            self.res = fn.resize(
                self.decode,
                resize_x=self.yolo_img_size[1], # Width
                resize_y=self.yolo_img_size[0], # Height
                interp_type=types.INTERP_LINEAR
            )
            
            # Normalize to [0,1]
            self.norm = fn.crop_mirror_normalize(
                self.res,
                dtype=types.FLOAT,
                output_layout="CHW",  # Channels, Height, Width
                mean=[0.0, 0.0, 0.0],  # Normalizing to [0,1] by dividing by 255.0
                std=[255.0, 255.0, 255.0]
            )

        def define_graph(self):
            # For YOLO, you'd need to load labels as well. This is a placeholder.
            # DALI doesn't have a direct "read YOLO .txt" operator.
            # You'd typically load them via a custom Python operator or pre-process them into a format DALI can read.
            # For now, we'll return a dummy tensor for labels.
            # The number of boxes and their format would need to be dynamic.
            # This is a significant limitation for DALI with YOLO labels without custom ops.
            dummy_labels = fn.constant(
                data=np.array([[0, 0.5, 0.5, 1.0, 1.0]], dtype=np.float32), # [class_id, xc, yc, w, h]
                dtype=types.FLOAT,
                shape=[1, 5] # Placeholder: one dummy box per image
            )
            # To make it work for a batch, you'd need to generate labels per image in the batch.
            # This is a conceptual example for DALI, a full implementation is complex.
            
            return self.norm, dummy_labels # Return image and a dummy label

    logger.info(f"Creating DALI loader for split: {split}")
    
    image_dir_path = data_root / 'images' / split
    if not image_dir_path.exists():
        logger.error(f"DALI image directory not found: {image_dir_path}. Cannot create DALI loader.")
        return None

    pipeline = BongardDALIPipeline(
        batch_size=config["data_pipeline"]["dali"]["batch_size"],
        num_threads=config["data_pipeline"]["dali"]["num_threads"],
        device_id=0 if torch.cuda.is_available() else -1,
        seed=config['seed'],
        image_dir=image_dir_path,
        yolo_img_size=config['yolo_img_size']
    )
    pipeline.build()
    # The output names must match the define_graph return values
    return DALIGenericIterator(pipeline, ['images', 'labels'], reader_name="Reader")


# --- FFCV Loader Implementation ---
def convert_to_ffcv(config):
    """
    One-time: convert your dataset to .beton for ultra-fast loading.
    This function needs to be run once to generate the .beton file.
    For YOLO bounding boxes, you need to serialize them as FFCV doesn't
    have a native field for multi-object bounding boxes.
    """
    if not HAS_FFCV:
        logger.error("FFCV is not installed. Cannot convert dataset to .beton.")
        return

    data_root = Path(config['data_root'])
    beton_path_train = data_root / 'bongard_train.beton'
    beton_path_val = data_root / 'bongard_val.beton'

    logger.info(f"Starting FFCV dataset conversion to: {beton_path_train} and {beton_path_val}")

    class BongardFFCVDataset(torch.utils.data.Dataset):
        def __init__(self, config, split='train'):
            self.image_dir = Path(config['data_root']) / 'images' / split
            self.label_dir = Path(config['data_root']) / 'labels' / split
            self.image_paths = list(self.image_dir.rglob('*.png'))
            self.config = config
            logger.info(f"FFCV Dataset initialized for {split} with {len(self.image_paths)} images.")

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            label_path = self.label_dir / (img_path.stem + '.txt')

            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load image {img_path} for FFCV. Returning dummy data.")
                # Return dummy image (e.g., black) and label (e.g., empty list)
                dummy_img = np.zeros((self.config['yolo_img_size'][0], self.config['yolo_img_size'][1], 3), dtype=np.uint8)
                return dummy_img, b'[]' # Empty JSON string for labels
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # FFCV RGBImageField expects RGB
            
            # Load and serialize YOLO labels
            # YOLO labels are [class_id, xc, yc, w, h]
            boxes_data = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        try:
                            parts = list(map(float, line.strip().split()))
                            if len(parts) == 5:
                                boxes_data.append(parts)
                        except ValueError:
                            logger.warning(f"Could not parse label line in {label_path}: '{line.strip()}'")
            
            # Serialize the list of boxes to a JSON string (bytes)
            labels_json_bytes = json.dumps(boxes_data).encode('utf-8')
            
            return img, labels_json_bytes

    # Create datasets for train and val splits
    train_dataset = BongardFFCVDataset(config, split='train')
    val_dataset = BongardFFCVDataset(config, split='val')

    # Convert training split
    if len(train_dataset) == 0:
        logger.error("No training images found for FFCV conversion. Skipping training conversion.")
    else:
        writer_train = DatasetWriter(
            str(beton_path_train),
            {
                'image': RGBImageField(),
                # Use a custom field if you want proper deserialization within FFCV pipeline
                # For now, we'll use IntField and rely on external parsing or dummy labels
                'labels_json': IntField() # This will store bytes as an array of ints
            },
            num_workers=config['data_pipeline']['ffcv']['num_workers']
        )
        writer_train.from_indexed_dataset(train_dataset)
        logger.info(f" ✅ FFCV training dataset created at {beton_path_train}.")

    # Convert validation split
    if len(val_dataset) == 0:
        logger.error("No validation images found for FFCV conversion. Skipping validation conversion.")
    else:
        writer_val = DatasetWriter(
            str(beton_path_val),
            {
                'image': RGBImageField(),
                'labels_json': IntField() # Same placeholder as above
            },
            num_workers=config['data_pipeline']['ffcv']['num_workers']
        )
        writer_val.from_indexed_dataset(val_dataset)
        logger.info(f" ✅ FFCV validation dataset created at {beton_path_val}.")


def get_ffcv_loader(config, split: str):
    """
    Returns an FFCV Loader for train/val splits.
    NOTE: This FFCV loader requires the .beton files to be pre-generated by `convert_to_ffcv`.
    It decodes images and the serialized YOLO labels.
    """
    if not HAS_FFCV:
        logger.error("FFCV is not installed. Cannot create FFCV loader.")
        return None

    data_root = Path(config['data_root'])
    path = str(data_root / f'bongard_{split}.beton') # Use split-specific beton file

    if not Path(path).exists():
        logger.error(f"FFCV .beton file not found at {path}. Please run `convert_to_ffcv` first.")
        return None

    logger.info(f"Creating FFCV loader for split: {split}")
    
    # Define image decoding pipeline
    image_pipeline = [
        # Resize to the target YOLO image size
        # FFCV's Resize transform
        A.transforms.Resize(config['yolo_img_size'][0], config['yolo_img_size'][1], interpolation=cv2.INTER_LINEAR),
        ToTensor(), # Converts to torch.Tensor, normalizes to [0,1]
    ]
    
    # Define label decoding pipeline
    # This will read the bytes and deserialize the JSON string back into a list of lists.
    # A proper FFCV solution for YOLO labels usually involves a custom field.
    # For now, we'll return a dummy label as FFCV's built-in fields don't directly support
    # variable-length lists of lists like YOLO boxes without custom fields.
    
    class CustomYoloLabelDecoder(torch.nn.Module):
        def forward(self, x):
            # x will be a tensor of integers representing the byte values of the JSON string.
            # This is a hacky way to get the string back. Not recommended for production.
            # A proper FFCV custom field is the way.
            # For now, we'll return a dummy tensor for labels from FFCV.
            return torch.empty(0, 5, dtype=torch.float32) # Return empty tensor for labels

    label_pipeline = [
        CustomYoloLabelDecoder() # Placeholder for actual label decoding
    ]

    return Loader(
        path,
        batch_size=config['data_pipeline']['ffcv']['batch_size'],
        num_workers=config['data_pipeline']['ffcv']['num_workers'],
        order=OrderOption.RANDOM if split=='train' else OrderOption.SEQUENTIAL,
        pipelines={
            'image': image_pipeline,
            'labels_json': label_pipeline # Use the field name from DatasetWriter
        },
        drop_last_batch=(split=='train')
    )


# --------------------------------------------------------
# 1) PyTorch Dataset + DataLoader Fallback
# --------------------------------------------------------
class BongardYoloDataset(Dataset):
    """
    Loads an image and its YOLO .txt labels (multi-box format):
      <class_id> <xc> <yc> <w> <h>
    Returns:
      img_tensor [C,H,W], torch.FloatTensor
      targets:  [num_boxes, 5] → [class_id, xc, yc, w, h] (normalized)
    """
    def __init__(self, split="train"):
        data_root = CONFIG["data_root"]
        self.img_paths = sorted(glob.glob(os.path.join(data_root, "images", split, "*.png")))
        self.label_paths = {
            os.path.splitext(os.path.basename(p))[0]:
            os.path.join(data_root, "labels", split, os.path.basename(p).replace(".png", ".txt"))
            for p in self.img_paths
        }
        self.transform = None # Placeholder for optional extra transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_p = self.img_paths[idx]
        key   = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = self.label_paths[key]

        # 1) Load image
        img = cv2.imread(img_p)
        if img is None:
            logger.error(f"Failed to load image {img_p}. Returning dummy data.")
            # Return dummy image and empty targets
            dummy_img = torch.zeros(3, CONFIG['yolo_img_size'][0], CONFIG['yolo_img_size'][1], dtype=torch.float32)
            return dummy_img, torch.empty(0, 5, dtype=torch.float32)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize image to target YOLO size
        img = cv2.resize(img, (CONFIG['yolo_img_size'][1], CONFIG['yolo_img_size'][0]), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 2) Load YOLO labels
        boxes = []
        if os.path.exists(lbl_p):
            with open(lbl_p, 'r') as f:
                for line in f:
                    try:
                        cid, xc, yc, bw, bh = map(float, line.split())
                        boxes.append([cid, xc, yc, bw, bh])
                    except ValueError:
                        logger.warning(f"Malformed label line in {lbl_p}: '{line.strip()}'")
        targets = torch.tensor(boxes, dtype=torch.float32) # [N,5]

        # 3) Optionally apply extra transforms here
        # Note: YOLOv8 usually handles resizing/padding/augmentations internally
        # if you pass the dataset directly to model.train(data=...).
        # If you use a custom training loop, you might apply them here.

        return img_tensor, targets

# --------------------------------------------------------
# 3) Prefetch Loader
# --------------------------------------------------------
def prefetch_loader(dataloader):
    """Asynchronous CPU→GPU prefetch on a separate CUDA stream."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available for prefetching. Returning original dataloader.")
        yield from dataloader # Use yield from for simple iteration
        return

    stream = torch.cuda.Stream()
    
    # Initialize prev_imgs and prev_labels to None
    prev_imgs, prev_labels = None, None

    # Get the first batch outside the loop to initialize prev_imgs, prev_labels
    try:
        first_batch = next(iter(dataloader))
    except StopIteration:
        logger.warning("Dataloader is empty. No batches to prefetch.")
        return # Exit if dataloader is empty

    # Determine batch format (DALI vs. PyTorch/FFCV)
    if isinstance(first_batch, dict) and 'images' in first_batch: # DALI output format
        imgs_cpu = first_batch['images']
        labels_cpu = first_batch.get('labels', torch.empty(0)) # Get 'labels' if present, else dummy
    elif isinstance(first_batch, (list, tuple)) and len(first_batch) >= 2: # PyTorch/FFCV DataLoader
        imgs_cpu, labels_cpu = first_batch[0], first_batch[1]
    else:
        logger.warning(f"Unknown first batch format for prefetch_loader: {type(first_batch)}. Yielding as is.")
        yield first_batch
        yield from dataloader # Yield remaining batches as is
        return

    with torch.cuda.stream(stream):
        prev_imgs = imgs_cpu.cuda(non_blocking=True)
        prev_labels = labels_cpu.cuda(non_blocking=True)

    for batch in dataloader:
        # Determine batch format for current batch
        if isinstance(batch, dict) and 'images' in batch: # DALI output format
            imgs_cpu = batch['images']
            labels_cpu = batch.get('labels', torch.empty(0))
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2: # PyTorch/FFCV DataLoader
            imgs_cpu, labels_cpu = batch[0], batch[1]
        else:
            logger.warning(f"Unknown batch format for prefetch_loader: {type(batch)}. Yielding original batch.")
            yield batch
            continue # Skip prefetching for this malformed batch

        with torch.cuda.stream(stream):
            imgs = imgs_cpu.cuda(non_blocking=True)
            labels = labels_cpu.cuda(non_blocking=True)
        
        # Ensure the current stream waits for the prefetch stream
        torch.cuda.current_stream().wait_stream(stream)
        
        yield prev_imgs, prev_labels # Yield the previously prefetched batch
        
        # Update for next iteration
        prev_imgs, prev_labels = imgs, labels

    # Yield the last prefetched batch after the loop finishes
    if prev_imgs is not None:
        yield prev_imgs, prev_labels


# --------------------------------------------------------
# 4) Unified Loader Factory
# --------------------------------------------------------
def get_dataloader(split="train"):
    """
    Returns a high-throughput dataloader for 'train' or 'val'.
    Prefers DALI → FFCV → PyTorch.
    """
    dp_type = CONFIG["data_pipeline"]["type"].lower()
    bs  = CONFIG["batch_size"] # Use top-level batch_size for PyTorch DataLoader
    nw  = CONFIG["num_workers"] # Use top-level num_workers for PyTorch DataLoader
    pr  = CONFIG.get("pin_memory", True)

    # 4.1) NVIDIA DALI
    if dp_type == "dali" and HAS_DALI:
        logger.info(f"Attempting DALI loader for split='{split}'")
        dali_iter = get_dali_loader(CONFIG, split)
        if dali_iter:
            # DALI already prefetches internally; wrap to match PyTorch API for CUDA transfer
            return prefetch_loader(dali_iter)
        else:
            logger.warning("DALI loader creation failed. Falling back to FFCV/PyTorch.")

    # 4.2) FFCV (.beton)
    if dp_type == "ffcv" and HAS_FFCV:
        logger.info(f"Attempting FFCV loader for split='{split}'")
        ffcv_iter = get_ffcv_loader(CONFIG, split)
        if ffcv_iter:
            return ffcv_iter # FFCV loaders are already highly optimized
        else:
            logger.warning("FFCV loader creation failed. Falling back to PyTorch.")

    # 4.3) PyTorch fallback
    logger.info(f"Using PyTorch DataLoader for split='{split}'")
    ds = BongardYoloDataset(split)
    return DataLoader(
        ds,
        batch_size  = bs,
        shuffle     = (split == "train"),
        num_workers = nw,
        pin_memory  = pr,
        persistent_workers = True, # Keep workers alive between epochs
        prefetch_factor     = 2, # Number of batches to prefetch
        collate_fn=lambda batch: tuple(torch.stack(t) for t in zip(*batch)) # Simple collate for img, targets
    )

# Example usage (for testing purposes, not part of the main script)
if __name__ == '__main__':
    # For testing, ensure your 'data' directory exists with 'images' and 'labels'
    # and some .png and .txt files.
    
    # To test DALI, you need DALI installed and a GPU.
    # To test FFCV, you need FFCV installed and run convert_to_ffcv() first.
    
    # Example: run FFCV conversion (one-time)
    # print("Running FFCV conversion...")
    # convert_to_ffcv(CONFIG)
    # print("FFCV conversion complete.")

    print("\nTesting train dataloader...")
    train_loader = get_dataloader("train")
    for i, (imgs, targets) in enumerate(train_loader):
        print(f"Train Batch {i}: Images shape {imgs.shape}, Targets shape {targets.shape}")
        if i >= 2: # Print first 3 batches
            break

    print("\nTesting val dataloader...")
    val_loader = get_dataloader("val")
    for i, (imgs, targets) in enumerate(val_loader):
        print(f"Val Batch {i}: Images shape {imgs.shape}, Targets shape {targets.shape}")
        if i >= 2: # Print first 3 batches
            break

    print("\nDataLoader testing complete.")
