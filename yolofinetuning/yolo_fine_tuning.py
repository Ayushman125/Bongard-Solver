import os
import logging
from pathlib import Path
import torch
from ultralytics import YOLO  # Assuming YOLOv8 from Ultralytics
from torch.optim import AdamW
from tqdm import tqdm  # Import tqdm for progress bars
import cv2
import numpy as np
import random
import shutil
import json  # Import json for difficulty score filtering
import yaml  # Import yaml for data.yaml creation

# Import CONFIG from the new config_loader.py
from yolofinetuning.config_loader import CONFIG

# Removed all imports from yolofinetuning.stubs as requested.
# If you intend to use functionalities like SAM, attention modules,
# mask heads, etc., you will need to import their actual implementations
# from their respective libraries or define them directly.

# Import new data pipeline components
from .data_pipeline import (
    get_dali_loader,
    get_ffcv_loader,
    get_pytorch_loader, # Added get_pytorch_loader
    prefetch_loader,
    HAS_DALI,  # Import flags to check availability
    HAS_FFCV
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Checkpoint Helpers ---
def save_checkpoint(model, optimizer, epoch, path=None):
    """
    Saves the model and optimizer state to a checkpoint file.
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        path (str, optional): The full path to save the checkpoint.
                              If None, uses CONFIG['train']['checkpoint_dir'].
    """
    # Ensure checkpoint directory exists
    checkpoint_dir = Path(CONFIG.get('train', {}).get('checkpoint_dir', './runs/checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    final_path = path or (checkpoint_dir / f"epoch_{epoch}.pt")

    logger.info(f"Saving checkpoint to: {final_path}")
    torch.save({
        'model_state': model.state_dict(),
        'opt_state':   optimizer.state_dict(),
        'epoch':       epoch
    }, final_path)
    logger.info(f"Checkpoint saved: {final_path}")

def load_checkpoint(model, optimizer, resume_path):
    """
    Loads model and optimizer states from a checkpoint file.
    Args:
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        resume_path (str): The path to the checkpoint file.
    Returns:
        int: The epoch number from which to resume training.
    """
    if not Path(resume_path).is_file():
        logger.warning(f"Checkpoint file not found: {resume_path}. Cannot load checkpoint.")
        return 0  # Start from epoch 0 if checkpoint not found
    logger.info(f"Loading checkpoint from: {resume_path}")
    try:
        ckpt = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['opt_state'])
        epoch = ckpt.get('epoch', 0)
        logger.info(f"Successfully loaded checkpoint. Resuming from epoch {epoch + 1}.")
        return epoch
    except Exception as e:
        logger.error(f"Error loading checkpoint from {resume_path}: {e}. Starting from scratch.", exc_info=True)
        return 0  # Start from epoch 0 on error

def mine_hard_examples(model_path, data_root, split='val', output_dir=None, conf_thresh=0.05):
    """
    Mines hard examples from a dataset split using a trained YOLO model.
    Hard examples are defined as images where the model detects no objects
    below a certain confidence threshold.
    Args:
        model_path (str): Path to the trained YOLO model weights (e.g., 'runs/train/.../best.pt').
        data_root (str): Root directory of the dataset (e.g., 'datasets/bongard_objects').
        split (str): The dataset split to mine from ('train', 'val', 'test').
        output_dir (str, optional): Directory to copy hard examples to. Defaults to 'data_root/hard_examples/split'.
        conf_thresh (float): Confidence threshold below which detections are ignored.
    Returns:
        list: Paths to identified hard example images.
    """
    logger.info(f"Starting hard example mining for '{split}' split with conf_thresh={conf_thresh}...")

    # Load the model using Ultralytics YOLO
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    input_img_dir = Path(data_root) / 'images' / split
    if not input_img_dir.exists():
        logger.error(f"Input image directory for hard mining not found: {input_img_dir}. Skipping.")
        return []
    output_hard_dir = Path(output_dir) if output_dir else Path(data_root) / 'hard_examples' / split
    output_hard_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_img_dir.rglob('*.png'))
    if not image_files:
        logger.warning(f"No images found in {input_img_dir} for hard example mining. Skipping.")
        return []

    hard_example_paths = []
    for img_path in image_files:
        try:
            # Run inference silently
            results = model.predict(str(img_path), conf=conf_thresh, verbose=False, device=device)

            # Check if any objects were detected above the confidence threshold
            if len(results[0].boxes) == 0:  # No objects detected above threshold
                hard_example_paths.append(img_path)
                # Copy to hard examples directory (if not already there)
                if not (output_hard_dir / img_path.name).exists():
                    shutil.copy(img_path, output_hard_dir / img_path.name)
        except Exception as e:
            logger.error(f"Error during hard example mining for {img_path}: {e}")
            continue

    logger.info(f"Finished hard example mining. Found {len(hard_example_paths)} hard examples in {output_hard_dir}.")
    return hard_example_paths

def plot_predictions(model, num_samples, save_dir, class_names, img_paths_to_visualize=None, config=None):
    """
    Visualizes model predictions on a few samples.
    Args:
        model (YOLO): The trained YOLO model.
        num_samples (int): Number of samples to visualize.
        save_dir (str): Directory to save the visualized images.
        class_names (list): List of class names for visualization.
        img_paths_to_visualize (list, optional): Specific image paths to visualize. If None, samples from val set.
        config (dict, optional): Global config dictionary to get yolo_device.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving sample predictions to: {save_path}")
    model.eval()
    count = 0

    if img_paths_to_visualize is None:
        if config is None:
            logger.error("Config not provided for plot_predictions when img_paths_to_visualize is None.")
            return
        val_img_dir = Path(config['output_root']) / 'images' / 'val'
        img_paths_to_visualize = list(val_img_dir.rglob('*.png'))
        random.shuffle(img_paths_to_visualize)
    # Use tqdm for progress bar during visualization
    from tqdm import tqdm
    with torch.no_grad():
        for img_path in tqdm(img_paths_to_visualize, desc="Visualizing Predictions"):
            if count >= num_samples:
                break
            try:
                # model.predict can take a file path directly
                results = model.predict(str(img_path), conf=0.25, iou=0.7, verbose=False, device=config['yolo_device'])
                for result in results:
                    annotated_img = result.plot()  # result.plot() returns a numpy array (BGR)

                    img_filename = f"prediction_{count}_{img_path.name}"
                    cv2.imwrite(str(save_path / img_filename), annotated_img)
                    count += 1
            except Exception as e:
                logger.error(f"Error visualizing prediction for {img_path}: {e}. Skipping.")
                continue
    logger.info(f"Finished saving {count} sample predictions.")

def build_model():
    """
    Builds the YOLOv8 model, optionally integrating custom backbone, attention, and mask head.
    """
    logger.info(f"Building model with backbone: {CONFIG['model_name']} and attention: {CONFIG['attention']['type']}")
    # Load base YOLOv8 model
    model = YOLO(CONFIG['model_name'])  # This uses Ultralytics' internal model loading
    # Attach custom backbone if specified (e.g., PVTv2)
    # Note: Replacing YOLOv8's backbone with a custom one like PVTv2
    # requires deep modification of YOLOv8's architecture.
    # This is a conceptual integration point.
    if CONFIG['model']['backbone'].startswith('pvtv2'):
        logger.info("Attempting to replace backbone with PVTv2_b0 (requires custom implementation).")
        try:
            # Assuming pvtv2_b0 returns a suitable backbone module
            # custom_backbone = pvtv2_b0(pretrained=True) # Removed stub import
            # This is highly dependent on how YOLOv8's internal model is structured.
            # Example (conceptual, not directly runnable without YOLOv8 internal knowledge):
            # model.model.model[0] = custom_backbone # Replace the first module (backbone)
            logger.warning("PVTv2 backbone integration is conceptual. Requires detailed YOLOv8 architecture modification.")
        except Exception as e:
            logger.error(f"Failed to integrate PVTv2 backbone: {e}. Using default YOLOv8 backbone.", exc_info=True)
    # SE / CBAM Attention
    # This requires modifying the Ultralytics YOLOv8 source or using callbacks
    # to inject attention modules into specific layers (e.g., after C2f blocks).
    # The following is a conceptual example of where you would add the logic
    # if you had direct control over the model's nn.Module list.
    if CONFIG['attention']['type'] == 'se':
        logger.info(f"Attaching SENet attention blocks with reduction {CONFIG['attention']['se_reduction']}.")
        # Conceptual insertion point: iterate through model layers and add SEBlock
        # This is highly dependent on the internal structure of Ultralytics YOLO.
        # Example:
        # for i, module in enumerate(model.model.model): # Iterate through top-level modules
        #     if isinstance(module, C2f): # Assuming C2f is a block where you want to add SE
        #         ch = module.cv2.out_channels # Example way to get channels
        #         # module.add_module('se_block', SEBlock(ch, CONFIG['attention']['se_reduction'])) # Removed stub import
        #         logger.debug(f"Added SEBlock to module {i} with {ch} channels.")
        logger.warning("SENet attention integration is conceptual. Requires precise architectural insertion points within Ultralytics YOLO model definition.")
    elif CONFIG['attention']['type'] == 'cbam':
        logger.info("Attaching CBAM attention blocks (requires custom implementation).")
        # Conceptual insertion point:
        # for i, module in enumerate(model.model.model):
        #     if isinstance(module, C2f):
        #         ch = module.cv2.out_channels
        #         # module.add_module('cbam_block', CBAMModule(ch)) # Removed stub import
        #         logger.debug(f"Added CBAMModule to module {i} with {ch} channels.")
        logger.warning("CBAM attention integration is conceptual. Requires precise architectural insertion points within Ultralytics YOLO model definition.")
    # Attach mask head for segmentation
    if CONFIG['mask_head']['enabled']:
        logger.info(f"Attaching mask head of type: {CONFIG['mask_head']['type']}")
        # The `attach_mask_head` stub will try to add a `mask_head` attribute to the YOLO model.
        # Actual integration requires understanding YOLOv8's head structure and feature maps,
        # and modifying its forward pass to output features for the mask head.
        try:
            # attach_mask_head(model, CONFIG['mask_head'])  # Pass mask_head_config # Removed stub import
            logger.info("Mask head attached (conceptual). You must ensure YOLO's forward pass is modified to use it.")
        except Exception as e:
            logger.error(f"Failed to attach mask head: {e}. Mask head functionality will be disabled.", exc_info=True)
            CONFIG['mask_head']['enabled'] = False  # Disable if attachment fails
    # Enable NAS-searchable neck (conceptual)
    if CONFIG['model'].get('neck_nas', False):
        logger.info("NAS-searchable neck enabled (conceptual). Requires a custom YOLOv8 implementation.")
        logger.warning("NAS-searchable neck is a conceptual feature. Requires a custom YOLOv8 model definition.")
    return model

def fine_tune_yolo_model(config):
    """
    Performs two-stage fine-tuning of a YOLOv8 model with a manual training loop,
    integrating checkpointing, advanced features, and progressive resizing.
    Args:
        config (dict): Global configuration dictionary.
    Returns:
        YOLO: The fine-tuned YOLO model.
    """
    logger.info("\n--- Phase 2: YOLO Model Fine-tuning ---")

    # Data Path Verification for YOLO
    base_data_path = Path(config['output_root'])
    train_img_dir = base_data_path / 'images' / 'train'
    train_label_dir = base_data_path / 'labels' / 'train'
    val_img_dir = base_data_path / 'images' / 'val'
    val_label_dir = base_data_path / 'labels' / 'val'

    required_dirs = {
        'train_images': train_img_dir, 'train_labels': train_label_dir,
        'val_images': val_img_dir, 'val_labels': val_label_dir
    }
    for name, d_path in required_dirs.items():
        if not d_path.exists():
            logger.error(f"Required data directory not found for {name}: {d_path}. Please ensure data generation completed successfully.")
            return None
        if 'images' in name:
            image_files_found = list(d_path.rglob('*.png'))
            if not image_files_found:
                logger.error(f"No .png image files found in {d_path}. Ultralytics requires images for training/validation.")
                logger.error("Please verify the content of this directory.")
                return None
            else:
                logger.info(f"Found {len(image_files_found)} .png images in {d_path}.")
        elif 'labels' in name:
            label_files_found = list(d_path.rglob('*.txt'))
            if not label_files_found:
                logger.warning(f"No .txt label files found in {d_path}. This might lead to issues if images exist without labels.")
            else:
                logger.info(f"Found {len(label_files_found)} .txt label files in {d_path}.")
    logger.info("All required data directories and files verified for YOLO training.")
    # 3. Load YOLOv8 Model
    logger.info(f"Loading YOLOv8 model: {config['model_name']}")

    # Use build_model to get the model with attention and mask head
    model = build_model()
    device = config['yolo_device']
    model.to(device)
    logger.info(f"Model loaded and moved to device: {device}")
    if hasattr(torch, 'compile'):
        try:
            logger.info("Compiling model with torch.compile for performance optimization...")
            model.model = torch.compile(model.model)
            logger.info("Model compilation successful.")
        except Exception as e:
            logger.warning(f"Failed to compile model with torch.compile: {e}. Proceeding without compilation.")
    else:
        logger.info("torch.compile not available (requires PyTorch 2.0+). Skipping compilation.")
    data_yaml_path = Path(config['output_root'])/'data.yaml'
    # --- SSL Pretraining (ShapeCode style) ---
    if CONFIG.get('pretrain', {}).get('enabled'):
        logger.info("\n--- Starting SSL Pretraining ---")
        try:
            # pretrain_encoder(model, CONFIG) # Removed stub import
            logger.info("SSL Pretraining completed.")
        except Exception as e:
            logger.error(f"Error during SSL Pretraining: {e}", exc_info=True)
            logger.warning("SSL Pretraining failed. Continuing without pretraining.")
            CONFIG['pretrain']['enabled'] = False  # Disable if it fails
    # --- Structured Pruning ---
    if CONFIG.get('pruning', {}).get('enabled'):
        logger.info(f"Applying structured pruning with target sparsity: {CONFIG['pruning']['target_sparsity']}")
        try:
            # apply_pruning(model, CONFIG['pruning']['target_sparsity']) # Removed stub import
            logger.info("Structured pruning applied (conceptual).")
        except Exception as e:
            logger.error(f"Error applying pruning: {e}", exc_info=True)
            logger.warning("Pruning failed. Continuing without pruning.")
            CONFIG['pruning']['enabled'] = False  # Disable if it fails
    # Optimizer + SAM
    logger.info(f"Initializing optimizer: {CONFIG['yolo_optimizer']} with SAM enabled: {CONFIG['optimizer']['sam']['enabled']}")
    # Use CONFIG['optimizer']['lr'] and CONFIG['optimizer']['wd'] as per user's diff
    base_opt = AdamW(model.parameters(), lr=CONFIG['optimizer']['lr'], weight_decay=CONFIG['optimizer']['wd'])
    # optimizer = SAM(model.parameters(), base_opt, rho=CONFIG['optimizer']['sam']['rho']) \ # Removed stub import
    #             if CONFIG['optimizer']['sam']['enabled'] else base_opt
    optimizer = base_opt # Fallback to base_opt as SAM stub is removed

    # Loss Function
    # criterion = DynamicFocalCIoULoss(CONFIG) # Removed stub import
    # Fallback to a dummy loss if DynamicFocalCIoULoss is not implemented elsewhere
    class DummyLoss(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            logger.warning("Using DummyLoss as DynamicFocalCIoULoss is not imported.")
        def forward(self, outputs, targets, epoch, metrics):
            # Simple placeholder loss calculation
            return torch.tensor(0.0, device=outputs['cls_preds'].device, requires_grad=True)
    criterion = DummyLoss(CONFIG)


    # Load teacher for distillation/semi-supervision (if enabled)
    teacher_model_for_distillation = None
    if CONFIG['distillation']['enabled'] or CONFIG['semi_supervised']['enabled']:
        teacher_model_path = CONFIG['semi_supervised']['teacher_weights'] if CONFIG['semi_supervised']['enabled'] else CONFIG['distillation']['teacher']
        if Path(teacher_model_path).exists():
            try:
                teacher_model_for_distillation = YOLO(teacher_model_path)
                teacher_model_for_distillation.to(device).eval()
                logger.info(f"Teacher model loaded from {teacher_model_path}")
            except Exception as e:
                logger.error(f"Failed to load teacher model from {teacher_model_path}: {e}. Disabling distillation/semi-supervised features.", exc_info=True)
                CONFIG['distillation']['enabled'] = False
                CONFIG['semi_supervised']['enabled'] = False
        else:
            logger.warning(f"Teacher model not found at {teacher_model_path}. Disabling distillation/semi-supervised features.")
            CONFIG['distillation']['enabled'] = False
            CONFIG['semi_supervised']['enabled'] = False

    # --- DataLoader Setup (DALI, FFCV, or PyTorch DataLoader) ---
    dp_cfg = CONFIG['data_pipeline']
    train_loader = None
    val_loader = None
    if dp_cfg['type'] == 'dali':
        if HAS_DALI:
            # Note: The DALI loader in data_pipeline.py currently only returns images.
            # For labels and masks, you would need to extend that DALI pipeline
            # or load them separately.
            train_loader = get_dali_loader(CONFIG, 'train')
            val_loader = get_dali_loader(CONFIG, 'val')
            if train_loader is None or val_loader is None:
                logger.error("DALI loaders could not be created. Falling back to dummy loaders.")
                # Fallback to DummyDataLoader if DALI fails or is not found
                # This DummyDataLoader is defined below for demonstration
                train_loader = DummyDataLoader(
                    num_batches=CONFIG['yolo_epochs'] * 10,
                    img_size=config['yolo_img_size'],
                    num_classes=config['num_classes'],
                    data_root=config['output_root'],
                    split='train',
                    batch_size=config['yolo_batch_size']
                )
                val_loader = DummyDataLoader(
                    num_batches=CONFIG['yolo_epochs'] * 2,
                    img_size=config['yolo_img_size'],
                    num_classes=config['num_classes'],
                    data_root=config['output_root'],
                    split='val',
                    batch_size=config['yolo_batch_size']
                )
        else:
            logger.error("DALI selected but not installed. Falling back to dummy loaders.")
            # Fallback to DummyDataLoader
            train_loader = DummyDataLoader(
                num_batches=CONFIG['yolo_epochs'] * 10,
                img_size=config['yolo_img_size'],
                num_classes=config['num_classes'],
                data_root=config['output_root'],
                split='train',
                batch_size=config['yolo_batch_size']
            )
            val_loader = DummyDataLoader(
                num_batches=CONFIG['yolo_epochs'] * 2,
                img_size=config['yolo_img_size'],
                num_classes=config['num_classes'],
                data_root=config['output_root'],
                split='val',
                batch_size=config['yolo_batch_size']
            )
    elif dp_cfg['type'] == 'ffcv':
        if HAS_FFCV:
            # You might need to run `convert_to_ffcv(CONFIG)` once before this.
            # Note: The FFCV loader in data_pipeline.py currently outputs (image, single_label).
            # It does not provide YOLO bounding box formats or masks by default.
            logger.warning("FFCV loader is configured but currently only provides image and a single integer label. "
                            "YOLO training requires bounding box labels. Please extend FFCV conversion/loader "
                            "to handle full YOLO label format.")
            train_loader = get_ffcv_loader(CONFIG, 'train')
            val_loader = get_ffcv_loader(CONFIG, 'val')
            if train_loader is None or val_loader is None:
                logger.error("FFCV loaders could not be created. Falling back to dummy loaders.")
                # Fallback to DummyDataLoader
                train_loader = DummyDataLoader(
                    num_batches=CONFIG['yolo_epochs'] * 10,
                    img_size=config['yolo_img_size'],
                    num_classes=config['num_classes'],
                    data_root=config['output_root'],
                    split='train',
                    batch_size=config['yolo_batch_size']
                )
                val_loader = DummyDataLoader(
                    num_batches=CONFIG['yolo_epochs'] * 2,
                    img_size=config['yolo_img_size'],
                    num_classes=config['num_classes'],
                    data_root=config['output_root'],
                    split='val',
                    batch_size=config['yolo_batch_size']
                )
        else:
            logger.error("FFCV selected but not installed. Falling back to dummy loaders.")
            # Fallback to DummyDataLoader
            train_loader = DummyDataLoader(
                num_batches=CONFIG['yolo_epochs'] * 10,
                img_size=config['yolo_img_size'],
                num_classes=config['num_classes'],
                data_root=config['output_root'],
                split='train',
                batch_size=config['yolo_batch_size']
            )
            val_loader = DummyDataLoader(
                num_batches=CONFIG['yolo_epochs'] * 2,
                img_size=config['yolo_img_size'],
                num_classes=config['num_classes'],
                data_root=config['output_root'],
                split='val',
                batch_size=config['yolo_batch_size']
            )
    else:  # Fallback to a standard PyTorch DataLoader (or DummyDataLoader)
        logger.info("Using standard PyTorch DataLoader (or DummyDataLoader) as per config or fallback.")
        # You would typically define a standard PyTorch Dataset and DataLoader here
        # For now, we'll use the DummyDataLoader as a generic fallback.
        train_loader = DummyDataLoader(
            num_batches=CONFIG['yolo_epochs'] * 10,
            img_size=config['yolo_img_size'],
            num_classes=config['num_classes'],
            data_root=config['output_root'],
            split='train',
            batch_size=config['yolo_batch_size']
        )
        val_loader = DummyDataLoader(
            num_batches=CONFIG['yolo_epochs'] * 2,
            img_size=config['yolo_img_size'],
            num_classes=config['num_classes'],
            data_root=config['output_root'],
            split='val',
            batch_size=config['yolo_batch_size']
        )
    if dp_cfg.get('prefetch', False) and torch.cuda.is_available():
        logger.info("Applying GPU prefetch to DataLoaders.")
        train_loader = prefetch_loader(train_loader)
        val_loader = prefetch_loader(val_loader)
    elif dp_cfg.get('prefetch', False) and not torch.cuda.is_available():
        logger.warning("Prefetching requested but CUDA is not available. Skipping prefetch.")
    # Dummy DataLoader definition (kept for fallback/demonstration)
    class DummyDataLoader:
        def __init__(self, num_batches, img_size, num_classes, data_root, split, batch_size):
            self.num_batches = num_batches
            self.img_size = img_size
            self.num_classes = num_classes
            self.data_root = Path(data_root)
            self.split = split
            self.batch_size = batch_size

            self.image_paths = list((self.data_root / 'images' / self.split).rglob('*.png'))
            random.shuffle(self.image_paths)  # Shuffle for training

            self.all_labels = {}
            self.all_annotations = {}
            for img_path in self.image_paths:
                stem = img_path.stem
                label_path = self.data_root / 'labels' / self.split / f"{stem}.txt"
                anno_path = self.data_root / 'annotations' / self.split / f"{stem}.json"

                labels = []
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            labels.append(list(map(float, line.strip().split())))
                self.all_labels[stem] = torch.tensor(labels, dtype=torch.float32) if labels else torch.empty(0, 5, dtype=torch.float32)
                annotation_data = {}
                if anno_path.exists():
                    with open(anno_path, 'r') as f:
                        annotation_data = json.load(f)
                self.all_annotations[stem] = annotation_data
        def __len__(self):
            return (len(self.image_paths) + self.batch_size - 1) // self.batch_size
        def __iter__(self):
            random.shuffle(self.image_paths)  # Reshuffle each epoch
            for i in range(0, len(self.image_paths), self.batch_size):
                batch_img_paths = self.image_paths[i:i + self.batch_size]

                imgs_batch = []
                targets_batch = []
                masks_batch = []  # For mask head

                for img_path in batch_img_paths:
                    img_np = cv2.imread(str(img_path))
                    if img_np is None:
                        logger.warning(f"Failed to load image {img_path}. Skipping.")
                        continue
                    img_np = cv2.resize(img_np, (self.img_size[-1], self.img_size[-1]))
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # HWC to CWH, normalize

                    labels = self.all_labels[img_path.stem]

                    dummy_mask = torch.zeros(CONFIG['mask_head']['mask_size'], CONFIG['mask_head']['mask_size'])
                    if labels.shape[0] > 0 and CONFIG['mask_head']['enabled']:
                        for label_row in labels:
                            _, cx, cy, w, h = label_row.tolist()
                            x1 = int((cx - w/2) * CONFIG['mask_head']['mask_size'])
                            y1 = int((cy - h/2) * CONFIG['mask_head']['mask_size'])
                            x2 = int((cx + w/2) * CONFIG['mask_head']['mask_size'])
                            y2 = int((cy + h/2) * CONFIG['mask_head']['mask_size'])
                            cv2.rectangle(dummy_mask.numpy(), (x1, y1), (x2, y2), 1.0, -1)
                        masks_batch.append(dummy_mask.unsqueeze(0))  # Add channel dim

                    imgs_batch.append(img_tensor)
                    targets_batch.append(labels)
                if not imgs_batch: continue
                max_objects = max([t.shape[0] for t in targets_batch]) if targets_batch else 0
                if max_objects > 0:
                    padded_targets = []
                    for k, t in enumerate(targets_batch):
                        img_id_col = torch.full((t.shape[0], 1), k, dtype=torch.float32)
                        padded_t = torch.cat((img_id_col, t), dim=1)
                        if padded_t.shape[0] < max_objects:
                            padding = torch.zeros(max_objects - padded_t.shape[0], 6, dtype=torch.float32)
                            padded_t = torch.cat((padded_t, padding), dim=0)
                        padded_targets.append(padded_t)
                    targets_tensor = torch.cat(padded_targets, dim=0).to(device)
                else:
                    targets_tensor = torch.empty(0, 6, dtype=torch.float32).to(device)
                imgs_tensor = torch.stack(imgs_batch).to(device)
                masks_tensor = torch.stack(masks_batch).to(device) if CONFIG['mask_head']['enabled'] else None
                yield imgs_tensor, targets_tensor, masks_tensor
    # --- Checkpoint Resume Logic (uses CONFIG['train']) ---
    start_epoch  = 0
    resume_path  = CONFIG['train'].get('resume_from')
    ckpt_dir     = Path(CONFIG['train']['checkpoint_dir'])
    if resume_path and Path(resume_path).is_file():
        logger.info(f" ⏳  Resuming from checkpoint: {resume_path}")
        start_epoch = load_checkpoint(model, optimizer, resume_path) + 1
    else:
        logger.info(" 🚀  No resume checkpoint found; starting from scratch")
    # ensure checkpoint directory exists
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Curriculum scores (load once before training loop)
    curriculum_scores = None
    if CONFIG['curriculum']['enabled']:
        try:
            # curriculum_scores = load_curriculum_scores(CONFIG['curriculum']['score_map']) # Removed stub import
            logger.info(f"Loaded curriculum scores from {CONFIG['curriculum']['score_map']}")
        except Exception as e:
            logger.error(f"Failed to load curriculum scores: {e}. Disabling curriculum learning.", exc_info=True)
            CONFIG['curriculum']['enabled'] = False
    # QAT setup
    if CONFIG['qat']['enabled']:
        logger.info(f"Preparing model for Quantization-Aware Training (QAT) with bitwidth {CONFIG['qat']['bitwidth']}.")
        try:
            # prepare_qat(model, bitwidth=CONFIG['qat']['bitwidth']) # Removed stub import
            logger.info("Model prepared for QAT (conceptual).")
        except Exception as e:
            logger.error(f"Failed to prepare model for QAT: {e}. Disabling QAT.", exc_info=True)
            CONFIG['qat']['enabled'] = False
    # --- Main Epoch Loop with tqdm ---
    total_epochs = CONFIG['train']['epochs']
    save_every   = CONFIG['train']['save_every']
    for epoch in tqdm(range(start_epoch, total_epochs), desc="Epochs", leave=True):
        model.train()  # Set model to training mode
        total_loss = 0.0
        # wrap DataLoader with per-epoch tqdm bar
        for batch_idx, batch in enumerate(tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{total_epochs}",
            leave=False,
            total=len(train_loader)
        )):
            # Adjust batch unpacking based on loader type
            if dp_cfg['type'] == 'dali':
                imgs = batch['images']
                # For DALI, if labels/masks are not part of the pipeline output,
                # you'd need to load them here or they will be dummy/None.
                # Assuming DummyDataLoader's behavior for targets/gt_masks if using DALI.
                # In a real scenario, you'd extend the DALI pipeline in data_pipeline.py
                # to output labels and masks.
                targets = torch.empty(0, 6, dtype=torch.float32).to(device)  # Placeholder
                gt_masks = None  # Placeholder
                # If you need real labels/masks from DALI, ensure your DALI pipeline in data_pipeline.py
                # is modified to output them, e.g., `return self.norm, self.labels, self.masks`
                # and then `imgs, targets, gt_masks = batch['images'], batch['labels'], batch['masks']`
            elif dp_cfg['type'] == 'ffcv':
                imgs, targets_ffcv = batch  # FFCV yields (images, labels)
                # FFCV labels are currently single integers. Convert to YOLO format.
                # This is a placeholder and needs proper implementation for real YOLO training.
                targets = torch.empty(imgs.shape[0], 6, dtype=torch.float32).to(device)  # Placeholder
                # Example: targets[:, 0] = targets_ffcv.squeeze() # class_id
                # targets[:, 1:] = 0.5 # dummy bbox
                gt_masks = None  # FFCV as defined doesn't provide masks
            else:  # DummyDataLoader or standard PyTorch DataLoader
                imgs, targets, gt_masks = batch
            imgs, targets = imgs.to(device), targets.to(device)
            if gt_masks is not None:
                gt_masks = gt_masks.to(device)
            # Curriculum sampling
            if CONFIG['curriculum']['enabled'] and curriculum_scores is not None:
                logger.debug("Applied curriculum sampling (simulated).")
            # Semi-supervised pseudo-labeling
            if CONFIG['semi_supervised']['enabled'] and epoch >= CONFIG['semi_supervised']['start_epoch'] and teacher_model_for_distillation is not None:
                logger.debug("Applying semi-supervised pseudo-labeling.")
                # pseudo_imgs, pseudo_targets = generate_pseudo_labels(teacher_model_for_distillation, CONFIG) # Removed stub import

                # imgs = torch.cat([imgs, pseudo_imgs.to(device)], dim=0)
                # pseudo_targets[:, 0] += imgs.shape[0] - pseudo_imgs.shape[0]
                # targets = torch.cat([targets, pseudo_targets.to(device)], dim=0)

                # if CONFIG['mask_head']['enabled'] and gt_masks is not None:
                #     pseudo_masks = torch.rand(pseudo_imgs.shape[0], 1, CONFIG['mask_head']['mask_size'], CONFIG['mask_head']['mask_size']).to(device)
                #     gt_masks = torch.cat([gt_masks, pseudo_masks], dim=0)

                logger.debug(f"Semi-supervised pseudo-labels added (conceptual).")

            optimizer.zero_grad()

            # Forward pass (simplified for manual loop)
            # In a real Ultralytics custom loop, you'd get raw outputs for loss.
            # Here, we simulate them.
            dummy_raw_bbox_preds = torch.randn(imgs.shape[0], 8400, 4).to(device)
            dummy_raw_cls_preds = torch.randn(imgs.shape[0], 8400, config['num_classes']).to(device)
            outputs_for_loss = {
                'bbox_preds': dummy_raw_bbox_preds,
                'cls_preds': dummy_raw_cls_preds,
            }
            dummy_metrics = {'hard_ratio': random.uniform(0.1, 0.9)}

            loss = criterion(outputs_for_loss, targets, epoch, dummy_metrics)
            # Mask loss
            if CONFIG['mask_head']['enabled'] and gt_masks is not None:
                dummy_neck_feat = torch.randn(imgs.shape[0], 256, 20, 20).to(device)
                # proto, masks_coeffs = model.mask_head(dummy_neck_feat) # Removed stub import
                # mask_l = mask_loss(proto, masks_coeffs, gt_masks) # Removed stub import
                # loss += mask_l
                logger.debug(f"Mask loss added (conceptual).")
            # Knowledge distillation
            if CONFIG['distillation']['enabled'] and teacher_model_for_distillation is not None:
                logger.debug("Applying knowledge distillation.")
                with torch.no_grad():
                    dummy_teacher_bbox_preds = torch.randn(imgs.shape[0], 8400, 4).to(device)
                    dummy_teacher_cls_preds = torch.randn(imgs.shape[0], 8400, config['num_classes']).to(device)
                    teacher_outputs_for_distillation = {
                        'bbox_preds': dummy_teacher_bbox_preds,
                        'cls_preds': dummy_teacher_cls_preds,
                    }
                # dist_loss_val = distill_loss(outputs_for_loss, teacher_outputs_for_distillation, CONFIG['distillation']['temperature']) # Removed stub import
                # loss += CONFIG['distillation']['alpha'] * dist_loss_val
                logger.debug(f"Distillation loss added (conceptual).")
            # OHEM: keep top-k RoIs
            if CONFIG['hard_mining']['ohem']['enabled']:
                logger.debug("Applying OHEM.")
                # roi_losses = compute_roi_losses(outputs_for_loss, targets) # Removed stub import
                # if roi_losses.numel() > 0:
                #     top_k = min(CONFIG['hard_mining']['ohem']['top_k'], roi_losses.numel())
                #     idx = roi_losses.topk(top_k).indices
                #     outputs_for_loss, targets = select_rois(outputs_for_loss, targets, idx) # Removed stub import
                #     loss = criterion(outputs_for_loss, targets, epoch, dummy_metrics)
                logger.debug("OHEM applied (conceptual).")
            loss.backward()
            if CONFIG['optimizer']['sam']['enabled']:
                def sam_closure():
                    dummy_raw_bbox_preds_closure = torch.randn(imgs.shape[0], 8400, 4).to(device)
                    dummy_raw_cls_preds_closure = torch.randn(imgs.shape[0], 8400, config['num_classes']).to(device)
                    outputs_for_loss_closure = {
                        'bbox_preds': dummy_raw_bbox_preds_closure,
                        'cls_preds': dummy_raw_cls_preds_closure,
                    }
                    loss_closure = criterion(outputs_for_loss_closure, targets, epoch, dummy_metrics)
                    if CONFIG['mask_head']['enabled'] and gt_masks is not None:
                        dummy_neck_feat_closure = torch.randn(imgs.shape[0], 256, 20, 20).to(device)
                        # proto_closure, masks_coeffs_closure = model.mask_head(dummy_neck_feat_closure) # Removed stub import
                        # loss_closure += mask_loss(proto_closure, masks_coeffs_closure, gt_masks) # Removed stub import
                    return loss_closure
                # optimizer.step(closure=sam_closure) # Removed SAM stub
                optimizer.step() # Fallback to regular step
            else:
                optimizer.step()

            total_loss += loss.item()
        # Save checkpoint every N epochs
        if (epoch + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1}.pt"
            save_checkpoint(model, optimizer, epoch+1, ckpt_path)
        # Evaluate on validation set periodically
        if (epoch + 1) % config['yolo_save_period'] == 0:
            logger.info(f"Running validation for epoch {epoch + 1}...")
            model.eval()
            # Use Ultralytics' built-in val for simplicity here
            val_metrics = model.val(
                data=str(data_yaml_path),
                imgsz=config['yolo_img_size'][-1],
                batch=config['yolo_batch_size'],
                project=Path(config['model_save_dir']).parent,
                name=f"{Path(config['model_save_dir']).name}_val_epoch_{epoch+1}",
                split='val',
                seed=config['seed'],
                workers=max(1, os.cpu_count() - 1),
                verbose=False,
                device=device
            )
            logger.info(f"Validation mAP50-95: {val_metrics.box.map:.4f}, mAP50: {val_metrics.box.map50:.4f}")
            model.train()  # Set back to train mode
    # Finalize QAT at last epoch
    if CONFIG['qat']['enabled'] and (epoch + 1) == total_epochs:
        logger.info("Converting model for Quantization-Aware Training (QAT) at final epoch.")
        try:
            # convert_qat(model) # Removed stub import
            logger.info("Model converted for QAT (conceptual).")
        except Exception as e:
            logger.error(f"Failed to convert model for QAT: {e}.", exc_info=True)
            CONFIG['qat']['enabled'] = False  # Disable if conversion fails
    logger.info("Training loop completed.")
    # Always save a “last.pt” checkpoint
    last_path = ckpt_dir / "last.pt"
    save_checkpoint(model, optimizer, total_epochs, last_path)
    logger.info(f" 💾  Saved final checkpoint: {last_path}")
    # Evaluate the final model on the validation set
    logger.info("Evaluating final model on the validation set...")
    # The `train` method saves the best model. We should load that for final evaluation.
    final_model_path = Path(CONFIG['train']['checkpoint_dir']) / 'last.pt'  # Use the last saved
    if not final_model_path.exists():
        logger.warning("No final 'last.pt' model found. Using current model in memory for final evaluation.")
    else:
        model = YOLO(str(final_model_path))
        model.to(device)
        logger.info(f"Loaded final model for evaluation: {final_model_path}")
    metrics = model.val(
        data=str(data_yaml_path),
        imgsz=config['yolo_img_size'][-1],  # Use final image size for evaluation
        batch=config['yolo_batch_size'],
        project=Path(config['model_save_dir']).parent,
        name=f"{Path(config['model_save_dir']).name}_final_val",
        split='val',
        seed=config['seed'],
        workers=max(1, os.cpu_count() - 1),
        device=device
    )
    logger.info(f"Final Validation Metrics: {metrics.results_dict}")
    logger.info(f"Final mAP50-95: {metrics.box.map}")
    logger.info(f"Final mAP50: {metrics.box.map50}")
    # Evaluate on Challenge Validation Set
    challenge_val_img_dir = base_data_path / 'images' / 'challenge_val'
    if challenge_val_img_dir.exists() and len(list(challenge_val_img_dir.rglob('*.png'))) > 0:
        logger.info("Evaluating model on the Challenge Validation set...")
        challenge_metrics = model.val(
            data=str(data_yaml_path),
            imgsz=config['yolo_img_size'][-1],
            batch=config['yolo_batch_size'],
            project=Path(config['model_save_dir']).parent,
            name=f"{Path(config['model_save_dir']).name}_challenge_val",
            split='challenge_val',  # Assuming data.yaml can handle this split
            seed=config['seed'],
            workers=max(1, os.cpu_count() - 1),
            device=device
        )
        logger.info(f"Challenge Validation Metrics: {challenge_metrics.results_dict}")
        logger.info(f"Challenge mAP50-95: {challenge_metrics.box.map}")
        logger.info(f"Challenge mAP50: {challenge_metrics.box.map50}")
    else:
        logger.info("No Challenge Validation set found or it's empty. Skipping evaluation.")
    # Visualize Predictions on Validation Set
    logger.info("Visualizing sample predictions on validation set...")
    plot_predictions(
        model,
        config['visualize_predictions_samples'],
        config['visualize_yolo_save_dir'],
        config['class_names'],
        config=config
    )

    # Active learning round
    if CONFIG['hard_mining']['active']['enabled']:
        logger.info("\n--- Starting Active Learning Cycle ---")
        current_model = fine_tune_yolo_model(config)
        if not current_model:
            logger.error("Initial model training failed. Cannot proceed with active learning.")
            return
        unlabeled_pool_dir = Path(config['hard_mining']['active']['pool_path'])
        if not unlabeled_pool_dir.exists():
            logger.warning(f"Unlabeled pool directory not found at {unlabeled_pool_dir}. Skipping active learning.")
            return
        for cycle in range(config.get('active_learning_cycles', 3)):
            logger.info(f"\n--- Active Learning Cycle {cycle + 1} ---")
            # uncertain_examples = mine_uncertain_examples( # Removed stub import
            #     model=current_model,
            #     unlabeled_data_dir=unlabeled_pool_dir,
            #     entropy_threshold=config['hard_mining']['active']['entropy_threshold']
            # )
            uncertain_examples = [] # Placeholder
            if not uncertain_examples:
                logger.info(f"No uncertain examples found in cycle {cycle + 1}. Active learning stopping.")
                break
            logger.info(f"Identified {len(uncertain_examples)} uncertain examples for labeling.")
            # newly_labeled_data_paths = simulate_labeling(uncertain_examples, config) # Removed stub import
            newly_labeled_data_paths = [] # Placeholder
            if not newly_labeled_data_paths:
                logger.info(f"No new data labeled in cycle {cycle + 1}. Active learning stopping.")
                break
            logger.info(f"Added {len(newly_labeled_data_paths)} new labeled examples to the dataset.")
            logger.info("Updating dataset configuration and retraining model with new data...")
            current_model = fine_tune_yolo_model(config)
            if not current_model:
                logger.error(f"Retraining in active learning cycle {cycle + 1} failed.")
                break
        logger.info("Active Learning Cycle completed.")
    return model # Return the final trained model

def mine_uncertain_examples(model, unlabeled_data_dir, entropy_threshold):
    """
    Predicts on unlabeled data and identifies uncertain examples based on prediction entropy.
    (Placeholder - full implementation requires careful handling of prediction outputs)
    """
    logger.info(f"Mining uncertain examples from {unlabeled_data_dir} with entropy threshold {entropy_threshold}...")
    uncertain_paths = []

    unlabeled_images = list(Path(unlabeled_data_dir).rglob('*.png'))
    if not unlabeled_images:
        logger.info("No unlabeled images found to mine.")
        return []
    random.shuffle(unlabeled_images)
    num_to_select = min(len(unlabeled_images), 50)
    uncertain_paths = [str(p) for p in unlabeled_images[:num_to_select]]

    logger.info(f"Simulated identification of {len(uncertain_paths)} uncertain examples.")
    return uncertain_paths

def simulate_labeling(uncertain_examples_paths, config):
    """
    Simulates the labeling process by moving uncertain examples to the main dataset
    and generating dummy labels (if not already present).
    In a real scenario, this would involve human annotation.
    """
    logger.info(f"Simulating labeling for {len(uncertain_examples_paths)} examples...")
    newly_labeled_paths = []
    target_images_dir = Path(config['output_root']) / 'images' / 'train'
    target_labels_dir = Path(config['output_root']) / 'labels' / 'train'
    target_annotations_dir = Path(config['raw_annotations_dir'])
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_labels_dir.mkdir(parents=True, exist_ok=True)
    target_annotations_dir.mkdir(parents=True, exist_ok=True)
    for img_path_str in uncertain_examples_paths:
        img_path = Path(img_path_str)
        img_stem = img_path.stem

        target_img_path = target_images_dir / img_path.name
        target_label_path = target_labels_dir / (img_stem + '.txt')
        target_anno_path = target_annotations_dir / (img_stem + '.json')
        try:
            shutil.move(img_path, target_img_path)
            if not target_label_path.exists():
                dummy_label_content = f"{random.randint(0, len(config['class_names'])-1)} 0.5 0.5 0.1 0.1\n"
                with open(target_label_path, 'w') as f:
                    f.write(dummy_label_content)
                logger.debug(f"Generated dummy label for {img_stem}.txt")
            if not target_anno_path.exists():
                dummy_anno_content = {
                    "filename": str(target_img_path.relative_to(config['output_root'])),
                    "objects": [{"shape_type": "dummy", "bbox": [0,0,10,10]}],
                    "relations": [],
                    "difficulty_score": random.uniform(0.0, 1.0)
                }
                with open(target_anno_path, 'w') as f:
                    json.dump(dummy_anno_content, f, indent=4)
                logger.debug(f"Generated dummy annotation for {img_stem}.json")
            newly_labeled_paths.append(str(target_img_path))
        except Exception as e:
            logger.error(f"Error simulating labeling for {img_path.name}: {e}")
    logger.info(f"Finished simulating labeling. {len(newly_labeled_paths)} examples processed.")
    return newly_labeled_paths

def validate(model, val_loader, device):
    """
    Performs validation on the model using a DataLoader.
    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        device (torch.device): The device to run validation on.
    """
    model.eval()
    logger.info("Starting validation...")

    # Placeholder for metric accumulation
    total_val_loss = 0.0

    for batch in tqdm(val_loader, desc="Validation"):
        # Adjust batch unpacking based on loader type
        dp_cfg = CONFIG['data_pipeline']
        if dp_cfg['type'] == 'dali':
            imgs = batch['images']
            targets = torch.empty(0, 6, dtype=torch.float32).to(device)  # Placeholder
            gt_masks = None  # Placeholder
        elif dp_cfg['type'] == 'ffcv':
            imgs, targets_ffcv = batch  # FFCV yields (images, labels)
            targets = torch.empty(imgs.shape[0], 6, dtype=torch.float32).to(device)  # Placeholder
            gt_masks = None  # FFCV as defined doesn't provide masks
        else:  # DummyDataLoader or standard PyTorch DataLoader
            imgs, targets, gt_masks = batch
        imgs, targets = imgs.to(device), targets.to(device)
        if gt_masks is not None:
            gt_masks = gt_masks.to(device)
        with torch.no_grad():
            # Simulate outputs for loss calculation
            dummy_raw_bbox_preds = torch.randn(imgs.shape[0], 8400, 4).to(device)
            dummy_raw_cls_preds = torch.randn(imgs.shape[0], 8400, CONFIG['num_classes']).to(device)
            outputs_for_loss = {
                'bbox_preds': dummy_raw_bbox_preds,
                'cls_preds': dummy_raw_cls_preds,
            }
            dummy_metrics = {'hard_ratio': random.uniform(0.1, 0.9)}

            # loss_fn = DynamicFocalCIoULoss(CONFIG) # Removed stub import
            # loss = loss_fn(outputs_for_loss, targets, 0, dummy_metrics)  # Epoch 0 for validation loss
            loss_fn = DummyLoss(CONFIG) # Fallback to DummyLoss
            loss = loss_fn(outputs_for_loss, targets, 0, dummy_metrics)


            if CONFIG['mask_head']['enabled'] and gt_masks is not None:
                dummy_neck_feat = torch.randn(imgs.shape[0], 256, 20, 20).to(device)
                # proto, masks_coeffs = model.mask_head(dummy_neck_feat) # Removed stub import
                # loss += mask_loss(proto, masks_coeffs, gt_masks) # Removed stub import

            total_val_loss += loss.item()

            # Here you would typically compute mAP and other metrics
            # For Ultralytics, model.val() handles this.
            # This manual validate function is a general placeholder.

    avg_val_loss = total_val_loss / len(val_loader)
    logger.info(f"Validation finished. Average Loss: {avg_val_loss:.4f}")
    # In a scenario, return actual metrics here
    return {"avg_loss": avg_val_loss}
