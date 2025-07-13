# yolofinetuning/pipeline_workers.py
import os
import glob
import logging
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json # Added for reading JSON labels

# Import CONFIG and LABEL_DIRS_MAP from config_loader.py and prepare_yolo_dataset.py
from .config_loader import CONFIG # This assumes config_loader.py is in the same package
# We need LABEL_DIRS_MAP from prepare_yolo_dataset.py to correctly locate label types
try:
    from .prepare_yolo_dataset import LABEL_DIRS_MAP
    logger.info("LABEL_DIRS_MAP imported from prepare_yolo_dataset.py.")
except ImportError:
    logger.error("Could not import LABEL_DIRS_MAP from prepare_yolo_dataset.py. "
                 "Using default mapping. Ensure prepare_yolo_dataset.py is up-to-date.")
    LABEL_DIRS_MAP = {
        "boxes":       "labels",
        "masks":       "masks",
        "polygons":    "polygons",
        "programs":    "programs",
        "relations":   "relations",
        "topo":        "topo",
        "descriptors": "stats",
        "captions":    "captions"
    }


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
    # FFCV also needs albumentations for its transforms if used in pipelines
    import albumentations as A
    HAS_FFCV = True
    logger.info("FFCV found.")
except ImportError:
    HAS_FFCV = False
    logger.warning("FFCV not found. FFCV data pipeline will not be available.")
except Exception as e:
    HAS_FFCV = False
    logger.warning(f"Error importing FFCV or its dependencies (e.g., albumentations): {e}. FFCV will not be available.")


# --- DALI Loader Implementation (Simplified for images only) ---
def get_dali_loader(config, split: str):
    """Return a DALI pipeline wrapped as a PyTorch iterator.
    NOTE: This DALI pipeline currently outputs images and dummy labels.
    Integrating actual multi-modal YOLO labels (masks, polygons, programs, etc.)
    into DALI is complex and typically requires custom DALI operators or
    pre-processing into a format DALI can natively read (e.g., TFRecords with serialized data).
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
            
            self.input = fn.readers.file(
                file_root=str(self.image_dir),
                random_shuffle=(split=='train'),
                name="Reader"
            )
            
            self.decode = fn.decoders.image(self.input, device="mixed", output_type=types.RGB)
            
            self.res = fn.resize(
                self.decode,
                resize_x=self.yolo_img_size[1], # Width
                resize_y=self.yolo_img_size[0], # Height
                interp_type=types.INTERP_LINEAR
            )
            
            self.norm = fn.crop_mirror_normalize(
                self.res,
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=[0.0, 0.0, 0.0],
                std=[255.0, 255.0, 255.0]
            )

        def define_graph(self):
            # Placeholder for labels and other multi-modal data
            # A full implementation would need custom operators to load .txt, .json, etc.
            dummy_labels = fn.constant(
                data=np.array([[0, 0.5, 0.5, 1.0, 1.0]], dtype=np.float32),
                dtype=types.FLOAT,
                shape=[1, 5]
            )
            # For multi-modal data, you'd need to return multiple outputs here
            # e.g., return self.norm, dummy_labels, dummy_masks, dummy_programs, ...
            return self.norm, dummy_labels

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


# --- FFCV Loader Implementation (Simplified for images only) ---
def convert_to_ffcv(config):
    """
    One-time: convert your dataset to .beton for ultra-fast loading.
    NOTE: This conversion currently only handles images and serializes YOLO bounding
    boxes as a JSON string. For other multi-modal labels (masks, polygons, programs,
    relations, topo, stats, captions), you would need to extend this to serialize
    them into the .beton file, potentially requiring custom FFCV fields.
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

            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load image {img_path} for FFCV. Returning dummy data.")
                dummy_img = np.zeros((self.config['yolo_img_size'][0], self.config['yolo_img_size'][1], 3), dtype=np.uint8)
                return dummy_img, b'[]' # Empty JSON string for labels
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
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
            
            labels_json_bytes = json.dumps(boxes_data).encode('utf-8')
            
            return img, labels_json_bytes

    train_dataset = BongardFFCVDataset(config, split='train')
    val_dataset = BongardFFCVDataset(config, split='val')

    if len(train_dataset) == 0:
        logger.error("No training images found for FFCV conversion. Skipping training conversion.")
    else:
        writer_train = DatasetWriter(
            str(beton_path_train),
            {
                'image': RGBImageField(),
                'labels_json': IntField() # Placeholder for serialized bytes, needs custom field for proper parsing
            },
            num_workers=config['data_pipeline']['ffcv']['num_workers']
        )
        writer_train.from_indexed_dataset(train_dataset)
        logger.info(f" ✅ FFCV training dataset created at {beton_path_train}.")

    if len(val_dataset) == 0:
        logger.error("No validation images found for FFCV conversion. Skipping validation conversion.")
    else:
        writer_val = DatasetWriter(
            str(beton_path_val),
            {
                'image': RGBImageField(),
                'labels_json': IntField()
            },
            num_workers=config['data_pipeline']['ffcv']['num_workers']
        )
        writer_val.from_indexed_dataset(val_dataset)
        logger.info(f" ✅ FFCV validation dataset created at {beton_path_val}.")


def get_ffcv_loader(config, split: str):
    """
    Returns an FFCV Loader for train/val splits.
    NOTE: This FFCV loader requires the .beton files to be pre-generated by `convert_to_ffcv`.
    It currently decodes images and returns a dummy label.
    To fully utilize multi-modal labels, custom FFCV fields are required for parsing
    the serialized JSON data back into usable Python objects/tensors.
    """
    if not HAS_FFCV:
        logger.error("FFCV is not installed. Cannot create FFCV loader.")
        return None

    data_root = Path(config['data_root'])
    path = str(data_root / f'bongard_{split}.beton')

    if not Path(path).exists():
        logger.error(f"FFCV .beton file not found at {path}. Please run `convert_to_ffcv` first.")
        return None

    logger.info(f"Creating FFCV loader for split: {split}")
    
    image_pipeline = [
        A.transforms.Resize(config['yolo_img_size'][0], config['yolo_img_size'][1], interpolation=cv2.INTER_LINEAR),
        ToTensor(),
    ]
    
    class FFCVDummyLabelDecoder(torch.nn.Module):
        def forward(self, x):
            # This is a placeholder. A real implementation would deserialize the JSON bytes
            # and convert them to a usable format (e.g., padded tensor for boxes).
            return {
                "boxes": torch.empty(0, 5, dtype=torch.float32),
                "masks": torch.empty(0, config['yolo_img_size'][0], config['yolo_img_size'][1], dtype=torch.uint8),
                "polygons": [],
                "programs": [],
                "relations": {},
                "topo": {},
                "descriptors": {},
                "captions": ""
            }

    label_pipeline = [
        FFCVDummyLabelDecoder()
    ]

    return Loader(
        path,
        batch_size=config['data_pipeline']['ffcv']['batch_size'],
        num_workers=config['data_pipeline']['ffcv']['num_workers'],
        order=OrderOption.RANDOM if split=='train' else OrderOption.SEQUENTIAL,
        pipelines={
            'image': image_pipeline,
            'labels_json': label_pipeline
        },
        drop_last_batch=(split=='train')
    )


# --------------------------------------------------------
# PyTorch Dataset + DataLoader Fallback (Comprehensive Multi-Modal Loading)
# --------------------------------------------------------
class BongardYoloDataset(Dataset):
    """
    Loads an image and all its associated multi-modal labels:
      - YOLO bounding boxes (.txt)
      - Instance segmentation masks (.png)
      - Vector stroke polygons (.json)
      - Action-program sequences (.json)
      - Relational graphs (.json)
      - Topological descriptors (.json)
      - Shape descriptors (.json)
      - Natural-language captions (.txt)

    Returns a dictionary for each item:
      {
        "image": img_tensor [C,H,W],
        "boxes": targets_tensor [num_boxes, 5],
        "mask": mask_tensor [H,W],
        "polygons": list of polygon points,
        "program": list of action strings,
        "relations": dict (node-link graph data),
        "topo_features": dict,
        "descriptors": dict,
        "caption": str
      }
    """
    def __init__(self, split="train"):
        data_root = Path(CONFIG["data_root"])
        self.split = split
        self.img_paths = sorted(list((data_root / 'images' / split).glob("*.png")))
        
        # Pre-build paths for all label types
        self.label_paths = {}
        for label_type, folder_name in LABEL_DIRS_MAP.items():
            current_dir = data_root / folder_name / split
            self.label_paths[label_type] = {
                p.stem: current_dir / (p.stem + ('.txt' if label_type == 'captions' else '.json' if label_type != 'masks' else '.png'))
                for p in self.img_paths
            }
        
        self.transform = None # Placeholder for optional extra transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_p = self.img_paths[idx]
        stem = img_p.stem
        
        # 1) Load image
        img = cv2.imread(str(img_p))
        if img is None:
            logger.error(f"Failed to load image {img_p}. Returning dummy data.")
            # Return dummy data if image loading fails
            dummy_img_tensor = torch.zeros(3, CONFIG['yolo_img_size'][0], CONFIG['yolo_img_size'][1], dtype=torch.float32)
            return {
                "image": dummy_img_tensor,
                "boxes": torch.empty(0, 5, dtype=torch.float32),
                "mask": torch.zeros(CONFIG['yolo_img_size'][0], CONFIG['yolo_img_size'][1], dtype=torch.uint8),
                "polygons": [], "program": [], "relations": {}, "topo_features": {}, "descriptors": {}, "caption": ""
            }

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (CONFIG['yolo_img_size'][1], CONFIG['yolo_img_size'][0]), interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Initialize all label data
        item_data = {
            "image": img_tensor,
            "boxes": torch.empty(0, 5, dtype=torch.float32),
            "mask": torch.zeros(CONFIG['yolo_img_size'][0], CONFIG['yolo_img_size'][1], dtype=torch.uint8),
            "polygons": [],
            "program": [],
            "relations": {},
            "topo_features": {},
            "descriptors": {},
            "caption": ""
        }

        # 2) Load YOLO boxes
        box_path = self.label_paths["boxes"][stem]
        boxes_list = []
        if box_path.exists():
            with open(box_path, 'r') as f:
                for line in f:
                    try:
                        cid, xc, yc, bw, bh = map(float, line.split())
                        boxes_list.append([cid, xc, yc, bw, bh])
                    except ValueError:
                        logger.warning(f"Malformed label line in {box_path}: '{line.strip()}'")
            item_data["boxes"] = torch.tensor(boxes_list, dtype=torch.float32)

        # 3) Load mask
        mask_path = self.label_paths["masks"][stem]
        if mask_path.exists():
            mask_np = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_np is not None:
                mask_np = cv2.resize(mask_np, (CONFIG['yolo_img_size'][1], CONFIG['yolo_img_size'][0]), interpolation=cv2.INTER_NEAREST)
                item_data["mask"] = torch.from_numpy(mask_np).to(torch.uint8)

        # 4) Load polygons
        poly_path = self.label_paths["polygons"][stem]
        if poly_path.exists():
            try:
                with open(poly_path, 'r') as f:
                    item_data["polygons"] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading polygons from {poly_path}: {e}")

        # 5) Load program
        prog_path = self.label_paths["programs"][stem]
        if prog_path.exists():
            try:
                with open(prog_path, 'r') as f:
                    item_data["program"] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading program from {prog_path}: {e}")

        # 6) Load relations
        rel_path = self.label_paths["relations"][stem]
        if rel_path.exists():
            try:
                with open(rel_path, 'r') as f:
                    item_data["relations"] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading relations from {rel_path}: {e}")

        # 7) Load topological features
        topo_path = self.label_paths["topo"][stem]
        if topo_path.exists():
            try:
                with open(topo_path, 'r') as f:
                    item_data["topo_features"] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading topological features from {topo_path}: {e}")

        # 8) Load shape descriptors
        desc_path = self.label_paths["descriptors"][stem]
        if desc_path.exists():
            try:
                with open(desc_path, 'r') as f:
                    item_data["descriptors"] = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading shape descriptors from {desc_path}: {e}")

        # 9) Load caption
        cap_path = self.label_paths["captions"][stem]
        if cap_path.exists():
            try:
                with open(cap_path, 'r') as f:
                    item_data["caption"] = f.read().strip()
            except IOError as e:
                logger.warning(f"Error loading caption from {cap_path}: {e}")

        return item_data

# --------------------------------------------------------
# Custom Collate Function for Multi-Modal Data
# --------------------------------------------------------
def multi_modal_collate_fn(batch):
    """
    Custom collate function to handle batching of multi-modal data.
    It will stack images, boxes, and masks, and keep other JSON data as lists.
    """
    images = []
    boxes = [] # List of tensors, each [N, 5]
    masks = [] # List of tensors, each [H, W]
    polygons = [] # List of lists of points
    programs = [] # List of lists of strings
    relations = [] # List of dicts (node-link data)
    topo_features = [] # List of dicts
    descriptors = [] # List of dicts
    captions = [] # List of strings

    for item in batch:
        images.append(item["image"])
        boxes.append(item["boxes"])
        masks.append(item["mask"])
        polygons.append(item["polygons"])
        programs.append(item["program"])
        relations.append(item["relations"])
        topo_features.append(item["topo_features"])
        descriptors.append(item["descriptors"])
        captions.append(item["caption"])

    # Stack images into a single tensor
    images_batch = torch.stack(images, 0)

    # For boxes, we need to add a batch index to each box
    # If using Ultralytics YOLOv8, it expects targets in format [batch_idx, class, x, y, w, h]
    # So, we'll create a list of tensors and then concatenate them.
    batched_boxes = []
    for i, box_tensor in enumerate(boxes):
        if box_tensor.numel() > 0: # Check if tensor is not empty
            batch_idx_col = torch.full((box_tensor.shape[0], 1), i, dtype=torch.float32)
            batched_boxes.append(torch.cat((batch_idx_col, box_tensor), dim=1))
    
    if batched_boxes:
        boxes_batch = torch.cat(batched_boxes, 0)
    else:
        boxes_batch = torch.empty(0, 6, dtype=torch.float32) # [batch_idx, class, x, y, w, h]

    # Stack masks (assuming all masks are of the same H, W)
    masks_batch = torch.stack(masks, 0)

    # Other data types remain as lists of their respective objects/dicts/strings
    # as they are typically processed individually or require custom graph/text processing.

    return {
        "images": images_batch,
        "boxes": boxes_batch, # Format: [batch_idx, class_id, xc, yc, w, h]
        "masks": masks_batch,
        "polygons": polygons,
        "programs": programs,
        "relations": relations,
        "topo_features": topo_features,
        "descriptors": descriptors,
        "captions": captions
    }


# --------------------------------------------------------
# 4) Unified Loader Factory
# --------------------------------------------------------
def get_dataloader(split="train"):
    """
    Returns a high-throughput dataloader for 'train' or 'val'.
    Prefers DALI → FFCV → PyTorch.
    """
    dp_type = CONFIG["data_pipeline"]["type"].lower()
    bs  = CONFIG["batch_size"]
    nw  = CONFIG["num_workers"]
    pr  = CONFIG.get("pin_memory", True)

    # 4.1) NVIDIA DALI (Simplified for images only)
    if dp_type == "dali" and HAS_DALI:
        logger.info(f"Attempting DALI loader for split='{split}'. Note: DALI currently loads images and dummy labels only.")
        dali_iter = get_dali_loader(CONFIG, split)
        if dali_iter:
            return prefetch_loader(dali_iter)
        else:
            logger.warning("DALI loader creation failed. Falling back to FFCV/PyTorch.")

    # 4.2) FFCV (.beton) (Simplified for images only)
    if dp_type == "ffcv" and HAS_FFCV:
        logger.info(f"Attempting FFCV loader for split='{split}'. Note: FFCV currently loads images and dummy labels only.")
        ffcv_iter = get_ffcv_loader(CONFIG, split)
        if ffcv_iter:
            return ffcv_iter
        else:
            logger.warning("FFCV loader creation failed. Falling back to PyTorch.")

    # 4.3) PyTorch fallback (Comprehensive Multi-Modal Loading)
    logger.info(f"Using PyTorch DataLoader for split='{split}' with comprehensive multi-modal data loading.")
    ds = BongardYoloDataset(split)
    return DataLoader(
        ds,
        batch_size  = bs,
        shuffle     = (split == "train"),
        num_workers = nw,
        pin_memory  = pr,
        persistent_workers = True,
        prefetch_factor     = 2,
        collate_fn=multi_modal_collate_fn # Use the new custom collate function
    )

# Example usage (for testing purposes, not part of the main script)
if __name__ == '__main__':
    # For testing, ensure your 'data' directory exists with 'images' and all label subdirectories
    # and some .png and corresponding label files.
    
    # Example: run FFCV conversion (one-time)
    # print("Running FFCV conversion...")
    # convert_to_ffcv(CONFIG)
    # print("FFCV conversion complete.")

    print("\nTesting train dataloader (PyTorch fallback)...")
    train_loader = get_dataloader("train")
    for i, batch_data in enumerate(train_loader):
        print(f"Train Batch {i}:")
        print(f"  Images shape: {batch_data['images'].shape}")
        print(f"  Boxes shape: {batch_data['boxes'].shape}")
        print(f"  Masks shape: {batch_data['masks'].shape}")
        print(f"  Polygons count: {len(batch_data['polygons'])}")
        print(f"  Programs count: {len(batch_data['programs'])}")
        print(f"  Relations count: {len(batch_data['relations'])}")
        print(f"  Topo Features count: {len(batch_data['topo_features'])}")
        print(f"  Descriptors count: {len(batch_data['descriptors'])}")
        print(f"  Captions count: {len(batch_data['captions'])}")
        if i >= 1: # Print first 2 batches
            break

    print("\nTesting val dataloader (PyTorch fallback)...")
    val_loader = get_dataloader("val")
    for i, batch_data in enumerate(val_loader):
        print(f"Val Batch {i}:")
        print(f"  Images shape: {batch_data['images'].shape}")
        print(f"  Boxes shape: {batch_data['boxes'].shape}")
        print(f"  Masks shape: {batch_data['masks'].shape}")
        print(f"  Polygons count: {len(batch_data['polygons'])}")
        print(f"  Programs count: {len(batch_data['programs'])}")
        print(f"  Relations count: {len(batch_data['relations'])}")
        print(f"  Topo Features count: {len(batch_data['topo_features'])}")
        print(f"  Descriptors count: {len(batch_data['descriptors'])}")
        print(f"  Captions count: {len(batch_data['captions'])}")
        if i >= 1: # Print first 2 batches
            break

    print("\nDataLoader testing complete.")
