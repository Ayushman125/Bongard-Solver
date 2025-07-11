# Folder: bongard_solver/
# File: yolo_trainer.py
import logging
import os
from typing import Dict, Any, Optional
# Conditional import for ultralytics
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
    logger = logging.getLogger(__name__)
    logger.info("ultralytics (YOLO) found and enabled.")
except ImportError:
    HAS_ULTRALYTICS = False
    logger = logging.getLogger(__name__)
    logger.warning("ultralytics not found. YOLO fine-tuning will be disabled.")
logger = logging.getLogger(__name__)
def fine_tune_yolo(cfg: Dict[str, Any]) -> Optional[str]:
    """
    Fine-tunes a YOLO model on a custom dataset.
    
    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing YOLO training parameters.
                              Expected keys: 'yolo_pretrained', 'yolo_data_yaml',
                              'fine_tune_epochs', 'img_size', 'batch_size', 'lr'.
    Returns:
        Optional[str]: Path to the best fine-tuned model weights (e.g., 'runs/detect/train/weights/best.pt'),
                       or None if fine-tuning failed or not enabled.
    """
    if not HAS_ULTRALYTICS:
        logger.error("ultralytics library not found. Cannot fine-tune YOLO.")
        return None
    yolo_config = cfg  # The object_detector config is passed directly here
    
    pretrained_weights = yolo_config.get('yolo_pretrained', 'yolov8n.pt')
    data_yaml_path = yolo_config.get('yolo_data_yaml')
    epochs = yolo_config.get('fine_tune_epochs', 10)
    imgsz = yolo_config.get('img_size', 640)
    batch_size = yolo_config.get('batch_size', 16)
    lr0 = yolo_config.get('lr', 0.01)
    
    if not os.path.exists(data_yaml_path):
        logger.error(f"YOLO data YAML file not found: {data_yaml_path}. Cannot fine-tune.")
        return None
    logger.info(f"Starting YOLO fine-tuning with: "
                f"weights={pretrained_weights}, data={data_yaml_path}, epochs={epochs}, "
                f"imgsz={imgsz}, batch={batch_size}, lr0={lr0}")
    try:
        # Load a pretrained YOLO model
        model = YOLO(pretrained_weights)
        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=lr0,
            # Add other training arguments as needed, e.g., device, project, name
            # device='0' if torch.cuda.is_available() else 'cpu', # Ultralytics handles device automatically if not specified
            project='yolo_finetune',
            name='bongard_yolo_run',
            exist_ok=True  # Allow overwriting previous runs
        )
        
        # The `results` object contains information about the training run.
        # The best model path is typically stored in `results.save_dir / 'weights' / 'best.pt'`
        # or can be accessed via `results.best_model.path` or similar attributes depending on Ultralytics version.
        
        # Ultralytics typically saves to `runs/detect/trainX/weights/best.pt`
        # The `results` object after `train` has a `save_dir` attribute.
        # `results.save_dir` is a Path object, convert to string.
        best_weights_path = os.path.join(str(results.save_dir), 'weights', 'best.pt')
        
        if os.path.exists(best_weights_path):
            logger.info(f"YOLO fine-tuning successful. Best model saved to: {best_weights_path}")
            return best_weights_path
        else:
            logger.error(f"YOLO fine-tuning completed, but best weights not found at expected path: {best_weights_path}")
            return None
    except Exception as e:
        logger.error(f"An error occurred during YOLO fine-tuning: {e}", exc_info=True)
        return None
if __name__ == "__main__":
    # Example usage:
    # This requires a dummy config that mimics the structure expected by fine_tune_yolo
    dummy_config = {
        'yolo_pretrained': 'yolov8n.pt',  # Or path to a local .pt file
        'yolo_data_yaml': './data/yolo_dataset/data.yaml',  # Create this dummy file for testing
        'fine_tune_yolo': True,  # Set to True to actually run fine-tuning
        'fine_tune_epochs': 1,  # Keep low for quick test
        'img_size': 224,
        'batch_size': 2,
        'lr': 0.001
    }
    # Create a dummy data.yaml for YOLO training if it doesn't exist
    dummy_data_yaml_path = dummy_config['yolo_data_yaml']
    if not os.path.exists(dummy_data_yaml_path):
        os.makedirs(os.path.dirname(dummy_data_yaml_path), exist_ok=True)
        dummy_data_yaml_content = """
names:
  0: object # Example class name for generic objects
path: ../datasets/bongard/ # Root dataset directory
train: images/train/ # train images (relative to 'path')
val: images/val/ # val images (relative to 'path')
test: images/test/ # test images (optional)
        """
        with open(dummy_data_yaml_path, 'w') as f:
            f.write(dummy_data_yaml_content)
        logger.info(f"Created dummy YOLO data YAML at: {dummy_data_yaml_path}")
        logger.warning("You might need to create dummy image directories (e.g., data/datasets/bongard/images/train) for YOLO training to run without errors.")
        logger.warning("For a real test, populate data/datasets/bongard/images/train and val with images and labels.")
    best_weights = fine_tune_yolo(dummy_config)
    if best_weights:
        logger.info(f"Successfully fine-tuned YOLO. Best weights: {best_weights}")
    else:
        logger.error("YOLO fine-tuning failed.")
