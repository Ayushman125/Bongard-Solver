import os
import logging
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import random
import shutil
import json # Import json for difficulty score filtering

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

def fine_tune_yolo_model(config):
    """
    Performs two-stage fine-tuning of a YOLOv8 model.
    Stage 1: Train head with frozen backbone.
    Stage 2: Train full model with SAM optimizer and progressive resizing.
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
    ultralytics_cache_dir = Path.home() / '.cache' / 'ultralytics' / 'weights'
    model_weights_path = ultralytics_cache_dir / config['model_name']
    
    if model_weights_path.exists():
        logger.warning(f"Deleting existing model weights {model_weights_path} to force re-download and prevent potential corruption issues.")
        try:
            os.remove(model_weights_path)
            logger.info("Model weights deleted successfully.")
        except OSError as e:
            logger.error(f"Error deleting model weights: {e}. Please check permissions or delete manually if issue persists.")
    try:
        model = YOLO(config['model_name'])
    except Exception as e:
        logger.error(f"Failed to load YOLO model '{config['model_name']}': {e}")
        logger.error("This might be due to a corrupted model file, network issues during download, or an incompatible Ultralytics version.")
        logger.error(f"Please ensure '{config['model_name']}' is valid and try deleting '{model_weights_path}' manually if the issue persists.")
        return None
    model.to(config['yolo_device'])
    logger.info(f"Model loaded and moved to device: {config['yolo_device']}")

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

    # --- Two-Stage Freeze→Unfreeze Training + EMA + SAM ---
    # Stage 1: Freeze backbone, train head only
    logger.info("\n--- Stage 1: Training head with frozen backbone (10 epochs) ---")
    try:
        model.train(
            data=str(data_yaml_path),
            epochs=10,  # First 10 epochs for head only
            imgsz=config['yolo_img_size'][0],  # Start with smallest size
            batch=config['yolo_batch_size'],
            accumulate=config['yolo_accumulate'],
            freeze=10,  # Freeze backbone layers up to layer 9 (C2f.2) for YOLOv8s
            optimizer=config['yolo_optimizer'],
            lr0=config['yolo_learning_rate'],
            lrf=0.01,  # Linear scheduler for initial stage, so final LR is low
            scheduler='linear',  # Linear warmup for initial stage
            project=Path(config['model_save_dir']).parent,
            name=f"{Path(config['model_save_dir']).name}_stage1",
            save=True, save_period=config['yolo_save_period'],
            val=True, seed=config['seed'],
            workers=max(1, os.cpu_count() - 1),
            amp=True, ema=config['yolo_ema'], device=config['yolo_device'],
            patience=config['yolo_patience'],  # Early stopping for stage 1
            # Pass new regularization params
            label_smoothing=config['yolo_label_smoothing'],
            dropout=config['yolo_dropout'],
            weight_decay=config['yolo_weight_decay'],
        )
        logger.info("Stage 1 training completed.")
    except Exception as e:
        logger.error(f"An error occurred during Stage 1 training: {e}", exc_info=True)
        logger.error("Stage 1 training failed. Please review the logs for more details.")
        return None

    # Stage 2: Unfreeze full model and continue training with SAM
    logger.info("\n--- Stage 2: Training full model (remaining epochs) with SAM optimizer ---")
    try:
        # Load the best model from Stage 1 to continue training
        best_stage1_model_path = Path(config['model_save_dir'] + '_stage1') / 'weights' / 'best.pt'
        if best_stage1_model_path.exists():
            model = YOLO(str(best_stage1_model_path))
            model.to(config['yolo_device'])
            logger.info(f"Loaded best model from Stage 1: {best_stage1_model_path}")
        else:
            logger.warning("Best model from Stage 1 not found. Continuing with current model state.")
        
        # Install SAM optimizer if not present
        optimizer_name = config['yolo_optimizer']
        if optimizer_name == 'SAM':
            try:
                import sam_optimizer
                logger.info("SAM optimizer found.")
            except ImportError:
                logger.info("SAM optimizer not found. Attempting to install 'sam-optimizer'...")
                os.system("pip install sam-optimizer")
                try:
                    import sam_optimizer  # Try importing again
                    logger.info("SAM optimizer installed successfully.")
                except ImportError:
                    logger.warning("Failed to install SAM optimizer. Proceeding without SAM. Falling back to AdamW.")
                    optimizer_name = 'AdamW'  # Fallback to AdamW

        model.train(
            data=str(data_yaml_path),
            epochs=config['yolo_epochs'] - 10,  # Remaining epochs
            imgsz=config['yolo_img_size'],  # Progressive resize
            batch=config['yolo_batch_size'],
            accumulate=config['yolo_accumulate'],
            freeze=[],  # Unfreeze all layers
            optimizer=optimizer_name,
            lr0=config['yolo_learning_rate'],
            lrf=config['yolo_final_lr_factor'],
            scheduler='cosine',  # Cosine annealing scheduler
            project=Path(config['model_save_dir']).parent,
            name=f"{Path(config['model_save_dir']).name}_stage2",
            save=True, save_period=config['yolo_save_period'],
            val=True, seed=config['seed'],
            workers=max(1, os.cpu_count() - 1),
            amp=True, ema=config['yolo_ema'], device=config['yolo_device'],
            patience=config['yolo_patience'],  # Early stopping for stage 2
            # Pass new regularization params
            label_smoothing=config['yolo_label_smoothing'],
            dropout=config['yolo_dropout'],
            weight_decay=config['yolo_weight_decay'],
        )
        logger.info("Stage 2 training completed.")
    except Exception as e:
        logger.error(f"An error occurred during Stage 2 training: {e}", exc_info=True)
        logger.error("Stage 2 training failed. Please review the logs for more details.")
        return None

    # Evaluate the final model on the validation set
    logger.info("Evaluating final model on the validation set...")
    final_model_path = Path(config['model_save_dir'] + '_stage2') / 'weights' / 'best.pt'
    if final_model_path.exists():
        model = YOLO(str(final_model_path))
        model.to(config['yolo_device'])
        logger.info(f"Loaded best model for final evaluation: {final_model_path}")
    else:
        logger.warning("Best model from Stage 2 not found for final evaluation. Using current model state.")

    metrics = model.val(
        data=str(data_yaml_path),
        imgsz=config['yolo_img_size'][-1],  # Use final image size for evaluation
        batch=config['yolo_batch_size'],
        project=Path(config['model_save_dir']).parent,
        name=f"{Path(config['model_save_dir']).name}_final_val",
        split='val',
        seed=config['seed'],
        workers=max(1, os.cpu_count() - 1)
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
            workers=max(1, os.cpu_count() - 1)
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
    
    return model

def retrain_on_hard_examples(model, hard_examples_paths, config):
    """
    Augments and relabels hard examples, then retrains the model for a few epochs.
    Args:
        model (YOLO): The current trained YOLO model.
        hard_examples_paths (list): List of paths to hard example images.
        config (dict): Global configuration dictionary.
    Returns:
        YOLO: The model after retraining on hard examples.
    """
    if not hard_examples_paths:
        logger.info("No hard examples to retrain on. Skipping retraining phase.")
        return model
    logger.info(f"Retraining on {len(hard_examples_paths)} hard examples for {config['hard_mining_epochs']} epochs.")
    
    # Create a temporary file list for DALI to process only hard examples
    retrain_dali_file_list_path = Path(config['temp_generated_data_dir']) / 'retrain_hard_examples_dali_list.txt'
    retrain_dali_file_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(retrain_dali_file_list_path, 'w') as f:
        for img_path in hard_examples_paths:
            f.write(f"{img_path}\n")  # DALI reader will read image path

    # Import BongardDaliPipeline from my_data_utils within this scope
    # to avoid circular imports if my_data_utils imports yolo_fine_tuning.
    # Assuming my_data_utils is in the same parent directory as this script.
    from my_data_utils import BongardDaliPipeline, HAS_DALI, CLASS_ID, YOLO_CLASSES_LIST, DIFFICULTY_WEIGHTS
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy # Re-import for this function

    if not HAS_DALI:
        logger.error("DALI not available for hard example augmentation. Skipping retraining.")
        return model

    device_id = 0 if config['yolo_device'] == 'cuda' and torch.cuda.is_available() else -1
    hard_dali_pipeline = BongardDaliPipeline(
        file_root=retrain_dali_file_list_path.parent,
        file_list=retrain_dali_file_list_path,
        config=config,
        device_id=device_id,
        is_training=True  # Apply augmentations
    )
    hard_dali_pipeline.build()

    hard_dali_iterator = DALIGenericIterator(
        hard_dali_pipeline, 
        ['images', 'yolo_labels', 'annotations_json', 'difficulty_score'],
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL
    )
    
    retrain_temp_data_dir = Path(config['temp_generated_data_dir']) / 'retrain_hard_data'
    retrain_temp_data_dir.mkdir(parents=True, exist_ok=True)
    retrain_image_paths_for_yaml = []

    logger.info(f"Generating augmented versions of {len(hard_examples_paths)} hard examples via DALI for retraining...")
    from tqdm import tqdm  # Import tqdm here for local use
    for i, data in enumerate(tqdm(hard_dali_iterator, desc="Augmenting Hard Examples for Retraining")):
        images_batch = data[0]['images']
        yolo_labels_batch = data[0]['yolo_labels']
        # annotations_json_batch = data[0]['annotations_json'] # Not directly used for retraining YOLO
        # difficulty_score_batch = data[0]['difficulty_score'] # Not directly used for retraining YOLO

        for j in range(images_batch.shape[0]):
            img_tensor = images_batch[j]
            yolo_labels_np = yolo_labels_batch[j]
            
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Use a unique stem for each augmented hard example
            original_stem = Path(hard_examples_paths[i * config['dali_batch_size'] + j]).stem
            retrain_stem = f"{original_stem}_hard_retrain_aug_{j}"

            retrain_img_path = retrain_temp_data_dir / f"{retrain_stem}.png"
            retrain_label_path = retrain_temp_data_dir / f"{retrain_stem}.txt"

            cv2.imwrite(str(retrain_img_path), img_np)
            with open(retrain_label_path, 'w') as f:
                for label_row in yolo_labels_np:
                    f.write(f"{int(label_row[0])} {label_row[1]:.6f} {label_row[2]:.6f} {label_row[3]:.6f} {label_row[4]:.6f}\n")
            retrain_image_paths_for_yaml.append(retrain_img_path)
    
    hard_dali_pipeline.empty()
    del hard_dali_pipeline
    del hard_dali_iterator

    # Create a temporary data.yaml for retraining
    retrain_data_yaml_path = retrain_temp_data_dir / 'retrain_data.yaml'
    retrain_data_yaml_content = {
        'path': str(retrain_temp_data_dir.resolve()),
        'train': '.',  # Train on current directory
        'val': '.',  # Use same for val (or a small subset)
        'nc': config['num_classes'],
        'names': config['class_names']
    }
    yaml.dump(retrain_data_yaml_content, open(retrain_data_yaml_path,'w'), sort_keys=False)
    logger.info(f"Generated temporary data.yaml for retraining at: {retrain_data_yaml_path}")

    # Retrain the model for a few epochs
    logger.info(f"Starting retraining on hard examples for {config['hard_mining_epochs']} epochs...")
    try:
        model.train(
            data=str(retrain_data_yaml_path),
            epochs=config['hard_mining_epochs'],
            imgsz=config['yolo_img_size'][-1],  # Use final image size
            batch=config['yolo_batch_size'],
            accumulate=config['yolo_accumulate'],
            optimizer=config['yolo_optimizer'],
            lr0=config['yolo_learning_rate'] * 0.1,  # Slightly lower LR for fine-tuning
            lrf=0.01,
            scheduler='cosine',
            project=Path(config['model_save_dir']).parent,
            name=f"{Path(config['model_save_dir']).name}_hard_retrain",
            save=True, save_period=1,
            val=True, seed=config['seed'],
            workers=max(1, os.cpu_count() - 1),
            amp=True, ema=config['yolo_ema'], device=config['yolo_device'],
            patience=config['yolo_patience'],
            # Pass new regularization params
            label_smoothing=config['yolo_label_smoothing'],
            dropout=config['yolo_dropout'],
            weight_decay=config['yolo_weight_decay'],
        )
        logger.info("Retraining on hard examples completed.")
    except Exception as e:
        logger.error(f"An error occurred during hard example retraining: {e}", exc_info=True)
        logger.error("Hard example retraining failed. Please review the logs for more details.")

    # Clean up temporary hard example data
    logger.info(f"Cleaning up temporary hard example retraining data: {retrain_temp_data_dir}")
    if retrain_temp_data_dir.exists():
        shutil.rmtree(retrain_temp_data_dir)
    return model
