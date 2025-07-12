# yolofinetuning/data_pipeline.py
import os
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 1) NVIDIA DALI imports
try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    HAS_DALI = True
except ImportError:
    HAS_DALI = False
    logger.warning("NVIDIA DALI not found. DALI data pipeline will not be available.")

# 2) FFCV imports
# pip install ffcv
try:
    from ffcv.writer import DatasetWriter
    from ffcv.loader import Loader, OrderOption
    from ffcv.fields import RGBImageField, IntField
    from ffcv.transforms import ToTensor
    HAS_FFCV = True
except ImportError:
    HAS_FFCV = False
    logger.warning("FFCV not found. FFCV data pipeline will not be available.")


def get_dali_loader(config, split: str):
    """Return a DALI pipeline wrapped as a PyTorch iterator.
    
    NOTE: This DALI pipeline currently only outputs images.
    To include labels and masks, you would need to extend this pipeline
    to read label files and potentially generate masks, similar to
    the BongardDaliPipeline in my_data_utils.py.
    """
    if not HAS_DALI:
        logger.error("NVIDIA DALI is not installed. Cannot create DALI loader.")
        return None

    # Ensure data_root is correctly resolved
    data_root = Path(config['output_root']) # Assuming output_root is the base for images/labels
    
    class BongardDALIPipeline(Pipeline):
        def __init__(self):
            super().__init__(
                batch_size=config['data_pipeline']['dali']['batch_size'],
                num_threads=config['data_pipeline']['dali']['num_threads'],
                device_id=0 if torch.cuda.is_available() else -1, # Use GPU if available, else CPU
                seed=config['seed']
            )
            file_root = str(data_root / 'images' / split)
            # DALI's file_list expects a file containing paths relative to file_root
            # For simplicity, we'll assume a list file exists or generate one.
            # In your main.py, you generate dali_file_list_path which contains absolute paths.
            # Here, we need a list of image filenames relative to `file_root`.
            
            # For this simplified loader, let's assume images are directly in file_root
            # and we don't need a separate file_list for now.
            # If you need specific images, you'd generate a file_list.txt with relative paths.
            
            # This is a simplified reader. For full labels, you'd need a custom reader or
            # integrate with `my_data_utils`'s DALI pipeline.
            self.input = fn.readers.file(
                file_root=file_root,
                # file_list=os.path.join(data_root, 'lists', f"{split}.txt"), # This path needs to be correct
                random_shuffle=(split=='train'),
                name="Reader"
            )
            
            self.decode = fn.decoders.image(self.input, device="mixed", output_type=types.RGB)
            self.res   = fn.resize(self.decode, resize_shorter=config['yolo_img_size'][-1], interp_type=types.INTERP_LINEAR)
            
            # Normalize to [0,1] and then to standard ImageNet mean/std (or whatever your model expects)
            # YOLOv8 typically expects [0,1] or [0,255] directly, adjust mean/std if needed.
            self.norm  = fn.crop_mirror_normalize(
                self.res,
                dtype=types.FLOAT,
                output_layout="CHW", # Channels, Height, Width
                mean=[0.0, 0.0, 0.0], # Normalizing to [0,1] by dividing by 255.0
                std=[255.0, 255.0, 255.0]
            )

        def define_graph(self):
            # This simplified pipeline only returns images.
            # For labels, you would need to add a reader for label files and process them.
            return self.norm

    logger.info(f"Creating DALI loader for split: {split}")
    pipeline = BongardDALIPipeline()
    pipeline.build()
    # The output 'images' corresponds to the `self.norm` output
    return DALIGenericIterator(pipeline, ['images'], reader_name="Reader")

def convert_to_ffcv(config):
    """One-time: convert your dataset to .beton for ultra-fast loading.
    
    This function requires a custom `BongardDataset` that yields (img, label) pairs.
    It also assumes labels are simple integers for `IntField`. For YOLO bounding boxes,
    you'd need a custom FFCV field or serialize them.
    """
    if not HAS_FFCV:
        logger.error("FFCV is not installed. Cannot convert dataset to .beton.")
        return

    # Ensure data_root is correctly resolved
    data_root = Path(config['output_root'])
    beton_path = data_root / 'bongard.beton'

    logger.info(f"Starting FFCV dataset conversion to: {beton_path}")
    
    # You need to implement BongardDataset to yield (img, label) pairs
    # This `BongardDataset` should iterate over your images and labels.
    # For a YOLO dataset, this is more complex than simple image classification.
    # It would need to read images and their corresponding YOLO .txt files.
    
    # Placeholder for a conceptual BongardDataset that FFCV expects
    # This needs to be implemented to correctly read images and labels
    # from your generated dataset structure.
    class BongardFFCVDataset(torch.utils.data.Dataset):
        def __init__(self, config, split='train'):
            self.image_dir = Path(config['output_root']) / 'images' / split
            self.label_dir = Path(config['output_root']) / 'labels' / split
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
                # Return dummy image (e.g., black) and label (e.g., 0)
                return np.zeros((config['image_size'][0], config['image_size'][1], 3), dtype=np.uint8), 0
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # FFCV RGBImageField expects RGB
            
            # Load label (simplified: return first class_id found, or 0 if no labels)
            # For full YOLO labels, you'd need a custom FFCV field or serialize.
            label = 0 # Default label
            if label_path.exists():
                with open(label_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        try:
                            label = int(float(first_line.split(' ')[0])) # Get class_id
                        except ValueError:
                            logger.warning(f"Could not parse label from {label_path}. Using default 0.")
            
            return img, label

    # Create datasets for train, val, test splits
    train_dataset = BongardFFCVDataset(config, split='train')
    val_dataset = BongardFFCVDataset(config, split='val')
    test_dataset = BongardFFCVDataset(config, split='test')

    # FFCV requires a mapping from dataset index to (image, label)
    # This is a simplified example. For YOLO, labels are more complex.
    # You might need a custom field or serialize labels as JSON strings.
    
    # Example for writing a single split. You might need to write multiple .beton files
    # or combine all data if your FFCV setup allows.
    
    # For now, this will convert the training split.
    # If you need to convert all splits, you'd repeat this for 'val' and 'test'
    # or create a combined dataset.
    
    if len(train_dataset) == 0:
        logger.error("No training images found for FFCV conversion. Skipping conversion.")
        return

    writer = DatasetWriter(
        str(beton_path),
        {
          'image': RGBImageField(),
          'label': IntField() # For YOLO, this would need to be a custom field or serialized data
        },
        num_workers=config['data_pipeline']['ffcv']['num_workers']
    )
    
    writer.from_indexed_dataset(train_dataset) # Convert only the training dataset for now
    logger.info(f"✅ FFCV dataset created at {beton_path} (converted training split only).")

def get_ffcv_loader(config, split: str):
    """Return an FFCV Loader for train/val splits.
    
    NOTE: This FFCV loader currently only outputs images and a single integer label.
    To get YOLO bounding boxes and masks, you would need to:
    1. Modify `convert_to_ffcv` to write more complex label data into the .beton file
       (e.g., using a custom FFCV field or serializing labels).
    2. Adjust the pipelines here to decode that complex label data.
    """
    if not HAS_FFCV:
        logger.error("FFCV is not installed. Cannot create FFCV loader.")
        return None

    data_root = Path(config['output_root'])
    path = str(data_root / 'bongard.beton')

    if not Path(path).exists():
        logger.error(f"FFCV .beton file not found at {path}. Please run `convert_to_ffcv` first.")
        return None

    logger.info(f"Creating FFCV loader for split: {split}")

    # Pipelines for image and label processing
    image_pipeline = [ToTensor()] # FFCV's ToTensor handles normalization if needed
    label_pipeline = [ToTensor()]

    return Loader(
        path,
        batch_size=config['data_pipeline']['ffcv']['batch_size'],
        num_workers=config['data_pipeline']['ffcv']['num_workers'],
        order=OrderOption.RANDOM if split=='train' else OrderOption.SEQUENTIAL,
        pipelines={
          'image': image_pipeline,
          'label': label_pipeline
        },
        drop_last_batch=(split=='train') # Drop last batch for training to avoid uneven batches
    )

def prefetch_loader(dataloader):
    """Asynchronous CPU→GPU prefetch on a separate CUDA stream."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available for prefetching. Returning original dataloader.")
        return dataloader

    stream = torch.cuda.Stream()
    first = True
    prev_imgs, prev_labels = None, None

    # This prefetcher assumes the dataloader yields (imgs, labels)
    # For DALI, it yields [{'images': ..., 'yolo_labels': ..., 'annotations_json': ..., 'difficulty_score': ...}]
    # You'll need to adapt this prefetcher if using DALI directly.
    # For FFCV, it yields (imgs, labels) as expected here.

    for batch in dataloader:
        if isinstance(batch, dict) and 'images' in batch: # DALI output format
            imgs = batch['images']
            # DALI's DALIGenericIterator might not provide labels/masks directly in this simplified setup.
            # If you need them, they must be part of the DALI pipeline output.
            labels = torch.empty(0) # Dummy labels
            # gt_masks = torch.empty(0) # Dummy masks
            
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2: # FFCV or standard PyTorch DataLoader
            imgs, labels = batch[0], batch[1] # Assuming (images, labels)
            # gt_masks = None # No masks from FFCV by default
        else:
            logger.warning(f"Unknown batch format for prefetch_loader: {type(batch)}. Skipping prefetch for this batch.")
            yield batch # Yield original batch if format is unexpected
            continue

        with torch.cuda.stream(stream):
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # if gt_masks is not None:
            #     gt_masks = gt_masks.cuda(non_blocking=True)
        
        if not first:
            yield prev_imgs, prev_labels, None # Yield prev_gt_masks if available
        else:
            first = False
        
        torch.cuda.current_stream().wait_stream(stream)
        prev_imgs, prev_labels = imgs, labels
        # prev_gt_masks = gt_masks

    if prev_imgs is not None:
        yield prev_imgs, prev_labels, None # Yield prev_gt_masks if available
