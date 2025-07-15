import os
import logging
import random
import yaml
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
from pathlib import Path
from PIL import Image
from collections import Counter
import glob
import shutil
import pandas as pd
import argparse

# Import modules from the same project structure
from .augmentations import get_train_transforms, custom_mixup, custom_cutmix, class_balanced_oversample, smote_oversample, visualize_augmentations, apply_augmentations
from .auto_labeling import auto_label, pseudo_labeling, active_learning_step, EmbedDetector # Assuming EmbedDetector is now in auto_labeling.py
from .copy_paste_synthesis import load_object_masks, paste_objects
from .data_preparation_utils import generate_synthetic_data, test_time_augmentation, audit_data_quality, detect_label_noise, correct_labels_with_model, curriculum_sampling, hard_negative_mining, continuous_validation, letterbox, make_dirs, parse_args, ensure_dirs, parse_classes, parse_annotations, convert_bbox, write_label_file, split_filenames, generate_yaml, generate_classes_file, process_dataset # Importing from data_preparation_utils
from .embedding_model import EmbedDetector # Re-importing EmbedDetector if it's truly in its own file, otherwise remove
from .fuse_graphs_with_yolo import load_graph, load_yolo, iou, nms, check_relation, label_quality, log_metadata as fg_log_metadata
from .logger import setup_logging, log_detection
from .metadata_logger import compute_metadata, log_metadata as meta_log_metadata
from .metrics import Evaluator
from .split_dataset import split_dataset, simple_split, stratified_split # Importing split_dataset functions

# --- DALI pipeline imports (placeholders if DALI is not fully integrated) ---
try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    DALI_AVAILABLE = True
except ImportError:
    logging.warning("NVIDIA DALI not found. DALI pipelines will not be available.")
    DALI_AVAILABLE = False
    # Define dummy fn and types if DALI is not available to prevent NameError
    class DummyFn:
        def readers(self): return self
        def file(self, **kwargs): return None
        def decoders(self): return self
        def image(self, **kwargs): return None
        def random(self): return self
        def coin_flip(self, **kwargs): return None
        def flip(self, **kwargs): return None
        def brightness_contrast(self, **kwargs): return None
        def uniform(self, **kwargs): return None
        def resize(self, **kwargs): return None
        def crop_mirror_normalize(self, **kwargs): return None
    fn = DummyFn()
    class DummyTypes:
        FLOAT = None
        RGB = None
        HWC = None
        INTERP_LINEAR = None
    types = DummyTypes()

# --- Config loading and seed setting ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
cfg = None
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def init_device_threads():
        if 'resources' in cfg:
            torch.set_num_threads(cfg['resources'].get('max_threads', 4))
            torch.set_num_interop_threads(cfg['resources'].get('max_threads', 4))
    init_device_threads()
else:
    logging.warning(f"No config.yaml found at {CONFIG_PATH}. Running with default settings and limited functionality.")

# --- MLflow setup (if enabled in config) ---
def setup_mlflow(cfg_param):
    if cfg_param and cfg_param.get('mlflow', {}).get('enable', False):
        if mlflow:
            mlflow.set_tracking_uri(cfg_param['mlflow'].get('tracking_uri', 'mlruns'))
            mlflow.set_experiment(cfg_param['mlflow'].get('experiment', 'dataset-prep'))
            return True
        else:
            logging.warning("MLflow is enabled in config but mlflow library is not installed.")
    return False

# --- DALI Pipeline Definitions ---
if DALI_AVAILABLE:
    @Pipeline.define
    class DALIAugPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, data_dir, image_size):
            super().__init__(batch_size, num_threads, device_id, seed=42)
            self.input = fn.readers.file(file_root=data_dir, random_shuffle=True)
            self.decode = fn.decoders.image(device="mixed", output_type=types.RGB)
            self.res = fn.resize(self.decode, resize_shorter=image_size, interp_type=types.INTERP_LINEAR)
            self.crop = fn.crop_mirror_normalize(
                self.res,
                dtype=types.FLOAT,
                output_layout="HWC",
                crop=(image_size, image_size),
                mirror=fn.random.coin_flip(),
                mean=[128,128,128],
                std=[58,57,57]
            )
        def define_graph(self):
            imgs = self.input
            images = self.decode(imgs)
            output = self.crop(images)
            return output

    @Pipeline.define
    class YOLOPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, image_dir, label_dir):
            super().__init__(batch_size, num_threads, device_id, seed=42)
            self.images, self.image_ids = fn.readers.file(file_root=image_dir, random_shuffle=True, name="ImageReader")
            self.labels = fn.readers.file(file_root=label_dir, random_shuffle=False, name="LabelReader")
            self.images = fn.decoders.image(self.images, device="mixed")

        def define_graph(self):
            images_out = self.images
            labels_out = self.labels

            # Basic augmentations without bbox operations (simplified for YOLO)
            flip_prob = fn.random.coin_flip(probability=0.5)
            images_out = fn.flip(images_out, horizontal=flip_prob)

            # Color augmentations
            images_out = fn.brightness_contrast(
                images_out,
                brightness=fn.random.uniform(range=(0.9, 1.1)),
                contrast=fn.random.uniform(range=(0.9, 1.1))
            )

            # Resize and normalize
            images_out = fn.resize(images_out, resize_x=640, resize_y=640)
            images_out = fn.crop_mirror_normalize(
                images_out,
                crop=(640, 640),
                mean=[0.485*255, 0.456*255, 0.406*255],
                std=[0.229*255, 0.224*255, 0.225*255],
                dtype=types.FLOAT
            )
            return images_out, labels_out

    # Simple image augmentation pipeline
    @Pipeline.define
    class SimpleAugmentPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, image_dir):
            super().__init__(batch_size, num_threads, device_id, seed=42)
            self.images = fn.readers.file(file_root=image_dir, random_shuffle=True, name="Reader")
            self.images = fn.decoders.image(self.images, device="mixed")

        def define_graph(self):
            images_out = self.images
            # Step 3: Rotation augmentation
            angles = fn.random.uniform(range=(-15.0, 15.0))
            images_out = fn.rotate(images_out, angle=angles, fill_value=0)

            # Step 4: Color augmentations
            brightness_vals = fn.random.uniform(range=(0.8, 1.2))
            contrast_vals = fn.random.uniform(range=(0.8, 1.2))
            images_out = fn.brightness_contrast(
                images_out,
                brightness=brightness_vals,
                contrast=contrast_vals
            )

            # Step 5: Geometric augmentations
            flip_prob = fn.random.coin_flip(probability=0.5)
            images_out = fn.flip(images_out, horizontal=flip_prob)

            # Step 6: Resize
            images_out = fn.resize(images_out, resize_x=640, resize_y=640)

            # Step 7: Normalize
            images_out = fn.crop_mirror_normalize(
                images_out,
                crop=(640, 640),
                mean=[0.485*255, 0.456*255, 0.406*255],
                std=[0.229*255, 0.224*255, 0.225*255],
                dtype=types.FLOAT
            )
            return (images_out,)

    def run_yolo_pipeline(image_dir, label_dir, iterations, batch_size=16):
        """Run the YOLO format pipeline."""
        pipe = YOLOPipeline(batch_size=batch_size, num_threads=4, device_id=0, image_dir=image_dir, label_dir=label_dir)
        pipe.build()
        logging.info(f"Running YOLO pipeline for {iterations} iterations...")
        for i in range(iterations):
            try:
                imgs, lbls = pipe.run()
                logging.info(f"YOLO batch {i+1}/{iterations}")
                imgs_np = imgs.as_cpu().as_array()
                logging.debug(f"   Image batch shape: {imgs_np.shape}")
            except Exception as e:
                logging.error(f"Error running YOLO pipeline batch {i+1}: {e}")
                break
        logging.info("YOLO pipeline completed successfully!")

    def run_simple_augment(image_dir, iterations, batch_size=16):
        """Run the simple augmentation pipeline."""
        pipe = SimpleAugmentPipeline(batch_size=batch_size, num_threads=4, device_id=0, image_dir=image_dir)
        pipe.build()
        logging.info(f"Running simple augmentation for {iterations} iterations...")
        for i in range(iterations):
            try:
                imgs, = pipe.run()   # Unpack single output
                logging.info(f"Simple augment batch {i+1}/{iterations}")
                imgs_np = imgs.as_cpu().as_array()
                logging.debug(f"   Augmented batch shape: {imgs_np.shape}")
            except Exception as e:
                logging.error(f"Error running simple augmentation batch {i+1}: {e}")
                break
        logging.info("Simple augmentation pipeline completed successfully!")
else: # DALI not available, define dummy functions
    def run_yolo_pipeline(image_dir, label_dir, iterations, batch_size=16):
        logging.warning("DALI is not available. Skipping YOLO pipeline execution.")
    def run_simple_augment(image_dir, iterations, batch_size=16):
        logging.warning("DALI is not available. Skipping simple augmentation pipeline execution.")


def run_full_dataset_preparation(args, cfg):
    if not cfg:
        logging.error("No config.yaml found, skipping advanced dataset prep.")
        return

    mlflow_enabled = setup_mlflow(cfg)
    if mlflow_enabled:
        mlflow.start_run(run_name="dataset-prep")
        mlflow.log_params(vars(args)) # Log CLI arguments

    # Initialize EmbedDetector and SAM Predictor for potential use in hooks
    # This assumes EmbedDetector and SamPredictor are imported from auto_labeling.py
    embed_detector_instance = None
    sam_predictor_instance = None
    if cfg.get('auto_labeling', {}).get('enabled', False) or \
       cfg.get('pseudo_labeling', {}).get('enabled', False) or \
       cfg.get('active_learning', {}).get('enabled', False) or \
       cfg.get('embedding_extraction', {}).get('enabled', False):
        try:
            embed_detector_instance = EmbedDetector(
                {'device': args.device, 'name': cfg['model']['name'], 'detection_threshold': cfg['model'].get('detection_threshold', 0.3)}
            )
            if SamPredictor: # Only try to initialize if SamPredictor import was successful
                sam_predictor_instance = SamPredictor(model=None) # Adjust as per your SAM setup
                logging.info("SAM Predictor initialized for full pipeline.")
            else:
                logging.warning("SAM Predictor not available, SAM-based auto-labeling/segmentation will be skipped.")
        except Exception as e:
            logging.error(f"Failed to initialize EmbedDetector or SAM Predictor for full pipeline: {e}")
            embed_detector_instance = None
            sam_predictor_instance = None

    # 1. Gather all images/labels
    raw_images_dir = cfg["paths"]["raw_images"]
    raw_labels_dir = cfg["paths"]["raw_labels"]

    if not os.path.exists(raw_images_dir) or not os.path.exists(raw_labels_dir):
        logging.error(f"Raw images directory ({raw_images_dir}) or raw labels directory ({raw_labels_dir}) not found. Exiting.")
        if mlflow_enabled: mlflow.end_run()
        return

    img_paths = sorted(glob.glob(raw_images_dir + "/*.jpg")) + sorted(glob.glob(raw_images_dir + "/*.png"))
    lbl_paths = sorted(glob.glob(raw_labels_dir + "/*.txt"))

    if not img_paths or not lbl_paths:
        logging.warning("No images or labels found in raw directories. Skipping full dataset prep.")
        if mlflow_enabled: mlflow.end_run()
        return

    # 2. Validate
    img_paths, lbl_paths = filter_valid_samples(img_paths, lbl_paths)
    continuous_validation(img_paths, lbl_paths, step_name="initial_validation")

    # 3. Data quality audit
    audit_report = audit_data_quality(lbl_paths, img_paths)
    logging.info(f"[AUDIT] Data quality report: {audit_report}")
    if mlflow_enabled:
        mlflow.log_dict(audit_report, "data_quality_audit.json")

    # 4. Label noise detection and correction
    label_issues = detect_label_noise(lbl_paths)
    if label_issues:
        logging.warning(f"[WARN] Label noise detected: {label_issues}")
        if mlflow_enabled:
            mlflow.log_dict(label_issues, "label_noise_detected.json")

        # Optionally: correct labels using model and human review
        if cfg.get('label_correction', {}).get('enable', False):
            logging.info("Attempting label correction...")
            for lbl in label_issues:
                img_path = lbl.replace('.txt', '.jpg') # Assuming .jpg, adjust if .png
                # Pass embed_detector_instance as the model
                correct_labels_with_model(lbl, img_path, model=embed_detector_instance, human_review=cfg['label_correction'].get('human_review', False))
            # Re-filter valid samples after correction
            img_paths, lbl_paths = filter_valid_samples(img_paths, lbl_paths)
            logging.info("Label correction attempted. Re-validating dataset.")
        else:
            # If correction is not enabled, remove problematic labels from paths
            initial_count = len(lbl_paths)
            lbl_paths = [l for l in lbl_paths if l not in label_issues]
            img_paths = [img_paths[i] for i, l in enumerate(lbl_paths) if l in lbl_paths] # Keep corresponding images
            logging.info(f"Removed {initial_count - len(lbl_paths)} problematic labels.")

    continuous_validation(img_paths, lbl_paths, step_name="post_label_correction")

    # 5. Class balancing (oversample rare classes)
    if cfg.get('class_balancing', {}).get('enable', False):
        logging.info("Applying class balancing (oversampling)...")
        img_paths, lbl_paths = class_balanced_oversample(img_paths, lbl_paths, min_count=cfg['class_balancing'].get("min_class_count", 100))
        if cfg.get('smote', {}).get('enable', False):
            smote_oversample(img_paths, lbl_paths, min_count=cfg['smote'].get("min_class_count", 100))
        continuous_validation(img_paths, lbl_paths, step_name="post_oversampling")

    # 6. Curriculum learning (progressive/meta-data sampling)
    if cfg.get('curriculum_learning', {}).get('enable', False):
        logging.info("Applying curriculum learning sampling...")
        # Pass embed_detector_instance as the model for curriculum sampling if it uses a model
        img_paths, lbl_paths = curriculum_sampling(img_paths, lbl_paths, model=embed_detector_instance, stages=cfg['curriculum_learning'].get('stages', 3))
        continuous_validation(img_paths, lbl_paths, step_name="post_curriculum")

    # 7. Hard negative mining (active retraining)
    if cfg.get('hard_negative_mining', {}).get('enable', False):
        logging.info("Performing hard negative mining...")
        # Pass embed_detector_instance as the model
        hard_negatives = hard_negative_mining(img_paths, lbl_paths, model=embed_detector_instance, threshold=cfg['hard_negative_mining'].get('threshold', 0.3))
        # Optionally: add hard negatives to training set
        if cfg['hard_negative_mining'].get('add_to_train', False):
            img_paths.extend(hard_negatives)
            # Need to get corresponding labels for hard negatives if they exist
            # This part might need more sophisticated logic depending on how hard negatives are generated/identified
            logging.warning("Adding hard negatives to training set. Ensure corresponding labels are handled.")
        continuous_validation(img_paths, lbl_paths, step_name="post_hard_negative")

    # 8. Synthetic data generation (GAN/Diffusion)
    if cfg.get('synthetic_data', {}).get('enabled', False):
        logging.info("Generating synthetic data...")
        generator_model = None # Placeholder: Load or instantiate your actual generator model here
        generate_synthetic_data(generator_model, n_samples=cfg['synthetic_data'].get('n_samples', 0), out_dir=cfg["paths"]["raw_images"])
        # After generating, re-scan and include new synthetic data in img_paths/lbl_paths
        img_paths = sorted(glob.glob(raw_images_dir + "/*.jpg")) + sorted(glob.glob(raw_images_dir + "/*.png"))
        lbl_paths = sorted(glob.glob(raw_labels_dir + "/*.txt")) # Assuming synthetic labels are also in raw_labels_dir
        continuous_validation(img_paths, lbl_paths, step_name="post_synthetic")

    # 9. Stats
    classes = open(cfg["paths"]["classes_file"]).read().splitlines()
    balance = compute_class_balance(lbl_paths, len(classes))
    mean, std = compute_mean_std(img_paths)
    logging.info(f"Class balance: {balance}")
    logging.info(f"Dataset mean: {mean}, std: {std}")
    if mlflow_enabled:
        mlflow.log_dict(balance, "class_balance.json")
        mlflow.log_dict({"mean": mean, "std": std}, "mean_std.json")

    # 10. Split (advanced stratified sampling for rare classes)
    logging.info("Splitting dataset...")
    if cfg["split"]["n_splits"] > 1:
        tr_imgs, tr_lbls, v_imgs, v_lbls = stratified_split(
            img_paths, lbl_paths, cfg["split"]["n_splits"], seed=cfg["seed"]
        )
    else:
        (tr_imgs, tr_lbls), (v_imgs, v_lbls) = simple_split(
            img_paths, lbl_paths, cfg["split"]["train_size"]
        )
    continuous_validation(tr_imgs, tr_lbls, step_name="train_split")
    continuous_validation(v_imgs, v_lbls, step_name="val_split")

    # 11. Save processed dataset
    logging.info("Saving processed dataset splits...")
    save_dataset(tr_imgs, tr_lbls, cfg["paths"]["out_train"], transforms=get_train_transforms(cfg))
    save_dataset(v_imgs, v_lbls, cfg["paths"]["out_val"], transforms=get_val_transforms(cfg)) # Pass cfg to get_val_transforms too

    # 12. Export YOLO yaml
    logging.info("Exporting YOLO data.yaml...")
    export_yolo_yaml(cfg)
    logging.info("[INFO] Dataset preparation complete. Train/val splits and data.yaml written.")
    if mlflow_enabled:
        mlflow.log_artifact("data.yaml")

    # 13. Test-Time Augmentation (TTA) for validation (example usage)
    if cfg.get('test_time_augmentation', {}).get('enable', False):
        logging.info("Running Test-Time Augmentation (TTA)...")
        # Placeholder: pass a trained model and validation images
        # tta_results = test_time_augmentation(image=..., model=embed_detector_instance, tta_transforms=...)
        logging.warning("TTA is enabled but requires a trained model and specific implementation.")

    # 14. Error analysis
    if cfg.get('error_analysis', {}).get('enable', False):
        logging.info("Running error analysis...")
        # Placeholder: pass a trained model and validation images/labels
        # error_report = error_analysis(v_imgs, v_lbls, model=embed_detector_instance)
        logging.warning("Error analysis is enabled but requires a trained model and specific implementation.")

    # 15. Visualization
    if cfg.get('visualization', {}).get('enable', False):
        logging.info("Generating visualizations...")
        # Example: visualize_augmentations(tr_imgs[0], tr_lbls[0], cfg=cfg)
        logging.warning("Visualization is enabled but requires specific implementation (e.g., image paths).")

    # 16. Auto labeling (if enabled in config)
    if cfg.get('auto_labeling', {}).get('enabled', False):
        logging.info("Running auto-labeling on specified images/directories...")
        # This typically runs on *unlabeled* data, separate from the main prepared dataset
        # You'd need to define `unlabeled_img_dir` and `auto_label_output_dir` in your config
        unlabeled_img_dir = cfg['auto_labeling'].get('unlabeled_img_dir')
        auto_label_output_dir = cfg['auto_labeling'].get('output_dir')
        if unlabeled_img_dir and auto_label_output_dir:
            os.makedirs(auto_label_output_dir, exist_ok=True)
            unlabeled_images = [os.path.join(unlabeled_img_dir, f) for f in os.listdir(unlabeled_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_path in tqdm(unlabeled_images, desc="Auto-labeling"):
                output_lbl_path = os.path.join(auto_label_output_dir, os.path.basename(img_path).replace(os.path.splitext(img_path)[1], '.txt'))
                auto_label(
                    image_path=img_path,
                    output_label_path=output_lbl_path,
                    embed_detector=embed_detector_instance,
                    sam_predictor_instance=sam_predictor_instance,
                    detection_prompts=cfg['auto_labeling'].get('prompts', ["object"]),
                    min_score=cfg['auto_labeling'].get('min_score', 0.3)
                )
        else:
            logging.warning("Auto-labeling enabled but 'unlabeled_img_dir' or 'output_dir' not specified in config.")

    # 17. Pseudo-labeling (if enabled in config)
    if cfg.get('pseudo_labeling', {}).get('enabled', False):
        logging.info("Running pseudo-labeling...")
        pseudo_label_img_dir = cfg['pseudo_labeling'].get('img_dir')
        pseudo_label_out_dir = cfg['pseudo_labeling'].get('output_dir')
        if pseudo_label_img_dir and pseudo_label_out_dir:
            pseudo_labeling(
                img_dir=pseudo_label_img_dir,
                embed_detector=embed_detector_instance,
                out_label_dir=pseudo_label_out_dir,
                detection_prompts=cfg['pseudo_labeling'].get('prompts', ["object"]),
                min_score=cfg['pseudo_labeling'].get('min_score', 0.3)
            )
        else:
            logging.warning("Pseudo-labeling enabled but 'img_dir' or 'output_dir' not specified in config.")

    # 18. Active learning (if enabled in config)
    if cfg.get('active_learning', {}).get('enabled', False):
        logging.info("Running active learning step...")
        active_learn_img_dir = cfg['active_learning'].get('img_dir')
        if active_learn_img_dir:
            selected_samples = active_learning_step(
                unlabeled_img_dir=active_learn_img_dir,
                embed_detector=embed_detector_instance,
                detection_prompts=cfg['active_learning'].get('prompts', ["object"]),
                selection_count=cfg['active_learning'].get('selection_count', 100)
            )
            logging.info(f"Active learning selected {len(selected_samples)} samples for annotation.")
        else:
            logging.warning("Active learning enabled but 'img_dir' not specified in config.")

    logging.info("All advanced dataset preparation steps completed.")
    if mlflow_enabled:
        mlflow.end_run()


# --- CLI entry point ---
if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Main Dataset Preparation Pipeline")
    parser.add_argument("--img_root", default="ShapeBongard_V2", help="Root directory of images (for DALI/general processing)")
    parser.add_argument("--output_root", default="dataset", help="Output root directory for processed data")
    parser.add_argument("--lbl_dir", default="data/annotations.json", help="Label file/directory path (used by some old functions, prefer config)")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations to run (for DALI pipelines)")
    parser.add_argument("--mode", default="simple", choices=["detection", "yolo", "simple"], help="Pipeline mode (for DALI examples)")
    parser.add_argument("--pseudo_label", action="store_true", help="Run pseudo-labeling (legacy, prefer config)")
    parser.add_argument("--active_learning", action="store_true", help="Run active learning step (legacy, prefer config)")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking (legacy, prefer config)")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Path to the main config.yaml file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for models (e.g., 'cuda', 'cpu')")

    args = parser.parse_args()

    logging.info(f"Main Dataset Preparation Pipeline Mode: {args.mode}")
    logging.info(f"Image Root: {args.img_root}")
    logging.info(f"Label Path: {args.lbl_dir}")
    logging.info(f"Output Root: {args.output_root}")
    logging.info(f"Iterations: {args.iters}")
    logging.info("-" * 50)

    # Load config dynamically based on CLI arg
    current_cfg = None
    if os.path.exists(args.config):
        with open(args.config) as f:
            current_cfg = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {args.config}")
    else:
        logging.warning(f"Config file not found at {args.config}. Proceeding with limited functionality and defaults.")

    # Run the full dataset preparation pipeline if config is loaded
    if current_cfg:
        run_full_dataset_preparation(args, current_cfg)
    else:
        logging.warning("No valid configuration found. Running only basic CLI-driven DALI examples if enabled.")
        # Fallback to direct DALI pipeline execution if no config is found but DALI is requested
        if DALI_AVAILABLE:
            if args.mode == "yolo":
                # These paths would need to be actual image/label directories
                logging.info(f"Running YOLO DALI pipeline on {args.img_root}...")
                run_yolo_pipeline(args.img_root, args.lbl_dir, args.iters)
            elif args.mode == "simple":
                logging.info(f"Running simple DALI augmentation pipeline on {args.img_root}...")
                run_simple_augment(args.img_root, args.iters)
            elif args.mode == "detection":
                logging.warning("Detection mode requires OWL-ViT and SAM setup, which is typically config-driven.")
        else:
            logging.error("DALI is not available, and no config was loaded to run other pipeline steps.")

    logging.info("Pipeline execution finished.")
