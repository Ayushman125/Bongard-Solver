import os
import yaml
import logging
from pathlib import Path
import torch
import cv2
import numpy as np
import random
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- CONFIGURATION ---
CONFIG = {
    'data_root': './datasets/bongard_objects', # Root directory where phase1.py saved data
    'model_save_dir': './runs/train/yolov8_bongard', # Directory to save trained models
    'model_name': 'yolov8n.pt', # Base YOLOv8 model (n=nano, s=small, m=medium, l=large, x=extra-large)
    'num_classes': 1, # Only one class: 'object'
    'class_names': ['object'], # Name for the single class
    'img_size': 224, # Image size for YOLO training (must match DALI output or be handled by YOLO's internal resize)
    'epochs': 50, # Number of training epochs
    'batch_size': 32, # Batch size for DataLoader (can be different from DALI's batch_size)
    'learning_rate': 0.001, # Initial learning rate
    'device': 'cuda' if torch.cuda.is_available() else 'cpu', # Device for training
    'seed': 42, # Random seed for reproducibility
    'visualize_predictions_samples': 10, # Number of validation samples to visualize predictions for
    'visualize_save_dir': './runs/predict/yolov8_bongard_val_preds', # Directory to save prediction visualizations
    'resume': True # Set to True to attempt resuming from last checkpoint
}

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Custom Dataset for YOLO Fine-tuning ---
class BongardYOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=(224, 224), transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.transform = transform

        self.image_files = sorted(list(self.img_dir.rglob('*.png'))) # Assuming PNGs from generator
        self.label_files = sorted(list(self.label_dir.rglob('*.txt')))

        # Ensure image and label files match
        image_stems = {f.stem for f in self.image_files}
        label_stems = {f.stem for f in self.label_files}

        common_stems = sorted(list(image_stems.intersection(label_stems)))
        
        self.image_files = [self.img_dir / (s + '.png') for s in common_stems]
        self.label_files = [self.label_dir / (s + '.txt') for s in common_stems]

        if len(self.image_files) == 0:
            logger.warning(f"No image files found in {img_dir} or no matching label files in {label_dir}.")
        else:
            logger.info(f"Initialized dataset with {len(self.image_files)} samples from {img_dir}.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_files[idx]

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            return self.__getitem__(random.randint(0, len(self) - 1)) # Return a random sample if load fails
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB

        # Resize image if necessary (YOLOv8's internal transform will handle final sizing)
        # We ensure it's the expected input size for the model or let YOLO handle it.
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Load labels (YOLO format: class_id center_x center_y width height)
        labels = []
        try:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        labels.append(parts)
        except FileNotFoundError:
            logger.warning(f"Label file not found for {img_path}. Skipping.")
            return self.__getitem__(random.randint(0, len(self) - 1)) # Return a random sample if label fails

        # Convert labels to a PyTorch tensor (format: [class_id, x_center, y_center, width, height])
        # For YOLOv8 training, labels should be a tensor of shape (num_objects, 5)
        if len(labels) > 0:
            labels = np.array(labels, dtype=np.float32)
            # Ultralytics expects labels in shape (N, 5), where N is number of objects
            # and each row is [class_id, x_center, y_center, width, height]
            # The class_id should be an integer.
            labels[:, 0] = labels[:, 0].astype(int) # Ensure class_id is int
        else:
            labels = np.zeros((0, 5), dtype=np.float32) # No objects

        # Convert image to PyTorch tensor (HWC to CHW, uint8 to float32, normalize to 0-1)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1) # HWC to CHW

        # Apply any additional transforms (e.g., Albumentations) if provided
        if self.transform:
            # Note: For Albumentations, you'd typically convert to dict and back
            # For simplicity, if transform is a simple PyTorch transform, apply directly.
            # If using Albumentations, it would be:
            # transformed = self.transform(image=img.numpy().transpose(1, 2, 0), bboxes=labels[:, 1:])
            # img = torch.from_numpy(transformed['image']).permute(2, 0, 1)
            # labels[:, 1:] = np.array(transformed['bboxes'])
            pass # Currently no additional transforms implemented here, YOLO handles them

        return img, labels

# --- Utility Function to Create data.yaml ---
def create_data_yaml(data_root, class_names, output_path='./data.yaml'):
    """
    Creates a data.yaml file required by Ultralytics YOLO for training.
    """
    data_yaml_content = {
        'path': str(Path(data_root).resolve()), # Absolute path to the dataset root
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml_content, f, sort_keys=False)
    logger.info(f"Generated data.yaml at: {output_path}")

# --- Visualization Function ---
def plot_predictions(model, dataloader, num_samples, save_dir, class_names):
    """
    Visualizes model predictions on a few samples from the dataloader.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving sample predictions to: {save_path}")

    model.eval() # Set model to evaluation mode
    count = 0
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(tqdm(dataloader, desc="Visualizing Predictions")):
            if count >= num_samples:
                break

            # Move images to device
            imgs = imgs.to(CONFIG['device'])

            # Perform prediction
            # The predict method returns a list of Results objects
            results = model.predict(imgs, conf=0.25, iou=0.7) # Adjust confidence and iou thresholds

            for j, result in enumerate(results):
                if count >= num_samples:
                    break

                # Convert tensor to numpy array (HWC, uint8)
                img_np = result.orig_img # Original image from results object (HWC, BGR)
                
                # Draw bounding boxes and labels
                # result.plot() returns an annotated image (numpy array, BGR)
                annotated_img = result.plot() 
                
                # Save the annotated image
                img_filename = f"prediction_{count}_{i}_{j}.png"
                cv2.imwrite(str(save_path / img_filename), annotated_img)
                count += 1
    logger.info(f"Finished saving {count} sample predictions.")

# --- Main Fine-tuning Function ---
def main_yolo_finetune():
    logger.info("Starting YOLOv8 Fine-tuning Script...")
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['seed'])
        torch.cuda.manual_seed_all(CONFIG['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Set to False for reproducibility

    # 1. Create data.yaml for Ultralytics
    data_yaml_path = Path(CONFIG['data_root']) / 'data.yaml'
    create_data_yaml(CONFIG['data_root'], CONFIG['class_names'], data_yaml_path)

    # 2. Load YOLOv8 Model
    model_path = Path(CONFIG['model_save_dir']) / 'weights' / 'last.pt'
    if model_path.exists() and CONFIG.get('resume', False): # Check if resume is enabled in CONFIG
        logger.info(f"Resuming training from checkpoint: {model_path}")
        model = YOLO(str(model_path))
    else:
        logger.info(f"Loading base YOLOv8 model: {CONFIG['model_name']}")
        model = YOLO(CONFIG['model_name']) # Load a pre-trained YOLOv8n model

    # Set device
    model.to(CONFIG['device'])
    logger.info(f"Model loaded and moved to device: {CONFIG['device']}")

    # 3. Fine-tune the model
    logger.info(f"Starting model training for {CONFIG['epochs']} epochs...")
    results = model.train(
        data=str(data_yaml_path), # Path to the data.yaml file
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['img_size'],
        batch=CONFIG['batch_size'],
        lr0=CONFIG['learning_rate'], # Initial learning rate
        lrf=0.01, # Final learning rate (factor of lr0)
        optimizer='AdamW', # AdamW is a good choice for fine-tuning
        # Augmentations are handled internally by Ultralytics' train method
        # e.g., mosaic=1.0, mixup=0.0, copy_paste=0.0 (default values)
        # You can adjust these in the train call if needed.
        project=Path(CONFIG['model_save_dir']).parent, # Parent directory for runs
        name=Path(CONFIG['model_save_dir']).name, # Specific run name
        save=True, # Save checkpoints
        val=True, # Validate every epoch
        seed=CONFIG['seed'],
        workers=os.cpu_count() // 2, # Number of DataLoader workers
        # resume is now handled by explicit model loading
        # patience=50, # Optional: Early stopping patience
    )
    logger.info("Model training completed.")

    # 5. Evaluate the model on the validation set
    logger.info("Evaluating model on the validation set...")
    metrics = model.val(
        data=str(data_yaml_path),
        imgsz=CONFIG['img_size'],
        batch=CONFIG['batch_size'],
        project=Path(CONFIG['model_save_dir']).parent,
        name=f"{Path(CONFIG['model_save_dir']).name}_val",
        split='val', # Explicitly specify validation split
        seed=CONFIG['seed'],
        workers=os.cpu_count() // 2
    )
    logger.info(f"Validation Metrics: {metrics.results_dict}")
    logger.info(f"mAP50-95: {metrics.box.map}") # mAP50-95
    logger.info(f"mAP50: {metrics.box.map50}") # mAP50
    logger.info(f"Precision: {metrics.box.p}")
    logger.info(f"Recall: {metrics.box.r}")

    # 6. Visualize Predictions on Validation Set
    logger.info("Visualizing sample predictions on validation set...")
    val_dataset_for_vis = BongardYOLODataset(
        img_dir=Path(CONFIG['data_root']) / 'images' / 'val',
        label_dir=Path(CONFIG['data_root']) / 'labels' / 'val',
        img_size=CONFIG['img_size']
    )
    val_loader_for_vis = DataLoader(val_dataset_for_vis, batch_size=1, shuffle=True, num_workers=1, collate_fn=lambda x: tuple(zip(*x))) # Batch size 1 for easy visualization

    plot_predictions(
        model,
        val_loader_for_vis,
        CONFIG['visualize_predictions_samples'],
        CONFIG['visualize_save_dir'],
        CONFIG['class_names']
    )
    logger.info("YOLOv8 Fine-tuning script finished.")

if __name__ == "__main__":
    main_yolo_finetune()
