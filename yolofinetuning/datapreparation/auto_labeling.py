import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import Counter
import torch
import logging
from PIL import Image # Required for OWL-ViT's image input

# Import OWL-ViT components
try:
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    from torchvision.transforms import functional as TF
except ImportError:
    OwlViTProcessor = None
    OwlViTForObjectDetection = None
    TF = None
    logging.warning("Hugging Face Transformers or torchvision not installed. OWL-ViT will not be available.")

# Import SAM predictor
try:
    # Assuming 'predictor' is a SAM predictor instance from your 'sam' module
    # If 'sam' module provides a function to build/load the predictor, use that.
    # For now, we'll assume a direct import or placeholder for its availability.
    from sam import SamPredictor # Adjust this import based on your actual SAM setup
    # Placeholder for SAM predictor instance, will be initialized in main or passed
    sam_predictor_instance = None
except ImportError:
    SamPredictor = None
    logging.warning("SAM not installed. SAM functionality will not be available.")

# Optional: MLflow for logging
try:
    import mlflow
except ImportError:
    mlflow = None
    logging.warning("MLflow not installed. MLflow logging will be skipped.")


# --- EmbedDetector (OWL-ViT) Class - Copied for self-containment ---
# This class was previously in augmentations.py / split_dataset.py context.
# Included here for this script to be self-contained and runnable.
class EmbedDetector:
    """
    OWL-ViT detection and embedding extraction utility.
    Usage:
        det = EmbedDetector(model_cfg)
        boxes, scores, labels, embeddings = det.detect(image, prompts)
    """
    def __init__(self, model_cfg):
        if OwlViTProcessor is None or OwlViTForObjectDetection is None:
            raise ImportError("OWL-ViT dependencies (transformers, torchvision) are not installed.")
        try:
            self.device = torch.device(model_cfg.get('device', 'cpu'))
            self.processor = OwlViTProcessor.from_pretrained(model_cfg['name'])
            self.model = OwlViTForObjectDetection.from_pretrained(model_cfg['name'])
            self.model.to(self.device).eval()
            self.threshold = model_cfg.get('detection_threshold', 0.3)
            self.max_queries = model_cfg.get('max_queries', 10) # Not directly used in detect, but good to keep
            logging.info(f"OWL-ViT model '{model_cfg['name']}' loaded successfully on {self.device}.")
        except Exception as e:
            logging.error(f"[ERROR] Failed to load OWL-ViT model: {e}")
            self.processor = None
            self.model = None
            self.device = None
            raise

    def detect(self, image: Image.Image, prompts: list):
        """
        Detect objects and extract embeddings.
        Args:
            image: PIL.Image
            prompts: list of str
        Returns:
            boxes (np.array): [[x0,y0,x1,y1], ...]
            scores (np.array): [0.8, 0.5, ...]
            labels (np.array): [class_id, ...] (indices into prompts[])
            embeddings (list of np.array): CLIP embeddings for each detected crop
        """
        try:
            if self.model is None:
                raise RuntimeError("OWL-ViT model not loaded. Cannot perform detection.")
            if not isinstance(image, Image.Image):
                raise TypeError("Input 'image' must be a PIL.Image object.")

            inputs = self.processor(text=[prompts], images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            # Post-process detections to get xyxy boxes and scores
            target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.threshold
            )[0]

            boxes = results['boxes'].cpu().numpy()
            scores = results['scores'].cpu().numpy()
            labels = results['labels'].cpu().numpy() # These are indices into the prompts list

            # Extract CLIP embeddings for each crop
            clip_embeddings = []
            if TF is not None: # Ensure torchvision.transforms.functional is available
                for box in boxes:
                    x0, y0, x1, y1 = [int(v) for v in box]
                    # Ensure crop coordinates are within image bounds
                    x0 = max(0, x0)
                    y0 = max(0, y0)
                    x1 = min(image.width, x1)
                    y1 = min(image.height, y1)

                    if x1 > x0 and y1 > y0: # Ensure valid crop
                        crop = TF.crop(image, y0, x0, y1-y0, x1-x0)
                        pix = self.processor.images_processor(crop, return_tensors="pt").pixel_values.to(self.device)
                        clip_outputs = self.model.owlvit.vision_model(pix) # Corrected attribute access
                        clip_embeddings.append(clip_outputs.pooler_output.cpu().detach().numpy()[0])
                    else:
                        clip_embeddings.append(np.zeros(768)) # Placeholder for invalid crops
            else:
                logging.warning("torchvision.transforms.functional not available. Skipping CLIP embedding extraction.")
                clip_embeddings = [np.zeros(768) for _ in range(len(boxes))] # Return dummy embeddings

            return boxes, scores, labels, clip_embeddings
        except Exception as e:
            logging.error(f"[ERROR] Detection failed: {e}")
            return np.array([]), np.array([]), np.array([]), []


# --- Auto-labeling function ---
def auto_label(image_path, output_label_path, embed_detector: EmbedDetector, sam_predictor_instance, detection_prompts: list, min_score=0.3):
    """
    Auto-labels an image using OWL-ViT for detection and SAM for segmentation.
    Args:
        image_path: path to image
        output_label_path: path to save YOLO label
        embed_detector: An initialized EmbedDetector instance (OWL-ViT)
        sam_predictor_instance: An initialized SAM predictor instance
        detection_prompts: list of strings for OWL-ViT to detect
        min_score: minimum detection score for OWL-ViT
    """
    if embed_detector is None or sam_predictor_instance is None:
        logging.error("Required models (OWL-ViT or SAM) not available for auto-labeling.")
        return

    try:
        image_pil = Image.open(image_path).convert('RGB')
        image_np = cv2.imread(image_path) # For SAM, which often uses OpenCV format

        if image_np is None:
            logging.error(f"Could not read image: {image_path}")
            return

        # 1. OWL-ViT Detection
        boxes_xyxy, scores, label_indices, _ = embed_detector.detect(image_pil, detection_prompts)

        if len(boxes_xyxy) == 0:
            logging.info(f"No objects detected by OWL-ViT in {image_path} with prompts {detection_prompts}.")
            # Create an empty label file if no detections
            with open(output_label_path, 'w') as f:
                pass
            return

        # Filter boxes by min_score and convert to SAM's expected format (tensor)
        valid_boxes = []
        valid_scores = []
        valid_labels_text = [] # Store text labels for logging/mapping
        for i, score in enumerate(scores):
            if score >= min_score:
                valid_boxes.append(boxes_xyxy[i])
                valid_scores.append(score)
                valid_labels_text.append(detection_prompts[label_indices[i]])

        if not valid_boxes:
            logging.info(f"No objects detected above score {min_score} in {image_path}.")
            with open(output_label_path, 'w') as f:
                pass
            return

        # Convert valid_boxes to torch tensor for SAM
        input_boxes_sam = torch.tensor(valid_boxes, device=embed_detector.device)

        # 2. SAM Segmentation
        sam_predictor_instance.set_image(image_np)
        masks, _, _ = sam_predictor_instance.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes_sam,
            multimask_output=False
        )
        # masks shape: (num_boxes, 1, H, W) -> squeeze to (num_boxes, H, W)
        masks = masks.squeeze(1)

        h, w, _ = image_np.shape
        yolo_labels = []

        for i, (box, score, label_idx) in enumerate(zip(valid_boxes, valid_scores, label_indices)):
            # Convert box from xyxy to YOLO format (cx, cy, w, h)
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            # You might want to map the label_idx back to your custom class IDs if needed
            # For simplicity, we'll use the label_idx from OWL-ViT directly as the class ID
            # If you have a separate class mapping, apply it here:
            # class_id = map_owlvit_label_to_your_class_id(label_idx)
            class_id = int(label_idx) # Using OWL-ViT's label index as class ID

            yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # Optionally, save masks or use them for further analysis
            # Example: save_mask_png(masks[i].cpu().numpy(), f"{output_label_path.replace('.txt', f'_mask_{i}.png')}")

        with open(output_label_path, 'w') as f:
            f.write('\n'.join(yolo_labels))

        logging.info(f"Auto-labeled {image_path} -> {output_label_path} with {len(yolo_labels)} objects.")
        if mlflow:
            try:
                mlflow.log_param("auto_label_image", image_path)
                mlflow.log_param("auto_label_objects_count", len(yolo_labels))
            except Exception as e:
                logging.error(f"MLflow logging failed for auto_label: {e}")

    except Exception as e:
        logging.error(f"Auto-labeling failed for {image_path}: {e}")
        # Ensure an empty file is created even on error to avoid missing files
        with open(output_label_path, 'w') as f:
            pass


# --- Pseudo-labeling function ---
def pseudo_labeling(img_dir: str, embed_detector: EmbedDetector, out_label_dir: str, detection_prompts: list, min_score: float = 0.3):
    """
    Generate pseudo-labels for unlabeled images using an OWL-ViT model.
    Args:
        img_dir: Directory with unlabeled images (e.g., .jpg or .png)
        embed_detector: An initialized EmbedDetector instance (OWL-ViT)
        out_label_dir: Output directory for pseudo-labels (YOLO format .txt files)
        detection_prompts: list of strings for OWL-ViT to detect
        min_score: minimum detection score for pseudo-labeling
    """
    if embed_detector is None:
        logging.error("EmbedDetector (OWL-ViT) is not available for pseudo-labeling.")
        return

    os.makedirs(out_label_dir, exist_ok=True)
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    logging.info(f"Starting pseudo-labeling for {len(img_paths)} images in {img_dir}...")

    for img_path in tqdm(img_paths, desc="Pseudo-labeling images"):
        try:
            image_pil = Image.open(img_path).convert('RGB')
            h, w = image_pil.height, image_pil.width

            # Use OWL-ViT to detect objects
            boxes_xyxy, scores, label_indices, _ = embed_detector.detect(image_pil, detection_prompts)

            pseudo_labels = []
            for box, score, label_idx in zip(boxes_xyxy, scores, label_indices):
                if score >= min_score:
                    x1, y1, x2, y2 = box
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    class_id = int(label_idx) # Using OWL-ViT's label index as class ID
                    pseudo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            out_path = os.path.join(out_label_dir, os.path.basename(img_path).replace(os.path.splitext(img_path)[1], '.txt'))
            with open(out_path, 'w') as f:
                f.write('\n'.join(pseudo_labels))

            logging.debug(f"Saved pseudo-labels: {out_path} ({len(pseudo_labels)} objects)")
            if mlflow:
                try:
                    mlflow.log_param("pseudo_label_image", img_path)
                    mlflow.log_param("pseudo_label_objects_count", len(pseudo_labels))
                except Exception as e:
                    logging.error(f"MLflow logging failed for pseudo_labeling: {e}")

        except Exception as e:
            logging.error(f"Failed to pseudo-label {img_path}: {e}")
            # Ensure an empty file is created even on error
            out_path = os.path.join(out_label_dir, os.path.basename(img_path).replace(os.path.splitext(img_path)[1], '.txt'))
            with open(out_path, 'w') as f:
                pass


# --- Active learning utility ---
def active_learning_step(unlabeled_img_dir: str, embed_detector: EmbedDetector, detection_prompts: list, selection_count: int = 100):
    """
    Select most informative samples for annotation using OWL-ViT's uncertainty.
    Args:
        unlabeled_img_dir: Directory with unlabeled images
        embed_detector: An initialized EmbedDetector instance (OWL-ViT)
        detection_prompts: list of strings for OWL-ViT to detect
        selection_count: Number of samples to select
    Returns:
        List of selected image paths
    """
    if embed_detector is None:
        logging.error("EmbedDetector (OWL-ViT) is not available for active learning.")
        return []

    img_paths = [os.path.join(unlabeled_img_dir, f) for f in os.listdir(unlabeled_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    uncertainties = []

    logging.info(f"Starting active learning selection for {len(img_paths)} images...")

    for img_path in tqdm(img_paths, desc="Calculating uncertainties"):
        try:
            image_pil = Image.open(img_path).convert('RGB')
            # Use OWL-ViT to get scores; uncertainty can be derived from scores
            _, scores, _, _ = embed_detector.detect(image_pil, detection_prompts)

            # Simple uncertainty metric: standard deviation of scores, or average of (1 - max_score)
            if len(scores) > 0:
                # Using 1 - max_score for each detection, then average for image uncertainty
                # Or, more simply, just the average of (1 - score) for all detections
                uncertainty = np.mean(1 - scores) if len(scores) > 0 else 1.0 # High uncertainty if no detections
            else:
                uncertainty = 1.0 # Max uncertainty if no detections at all

            uncertainties.append((uncertainty, img_path))

        except Exception as e:
            logging.error(f"Failed to calculate uncertainty for {img_path}: {e}")
            uncertainties.append((1.0, img_path)) # Assume high uncertainty on error

    # Sort by uncertainty (descending) to get most informative
    uncertainties.sort(key=lambda x: x[0], reverse=True)
    selected = [p for _, p in uncertainties[:selection_count]]

    logging.info(f"Active learning selected {len(selected)} samples for annotation.")
    if mlflow:
        try:
            mlflow.log_param("active_learning_selected_count", len(selected))
            mlflow.log_param("active_learning_selected_paths", selected)
        except Exception as e:
            logging.error(f"MLflow logging failed for active_learning: {e}")

    return selected


# --- CLI entry point ---
if __name__ == "__main__":
    import argparse
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Auto-labeling and Dataset Utilities with OWL-ViT and SAM")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for auto_label
    auto_label_parser = subparsers.add_parser('auto_label', help='Auto-label a single image using OWL-ViT and SAM')
    auto_label_parser.add_argument('--img', required=True, help='Path to input image')
    auto_label_parser.add_argument('--out_lbl', required=True, help='Path to output YOLO label file (.txt)')
    auto_label_parser.add_argument('--prompts', required=True, nargs='+', help='Detection prompts for OWL-ViT (e.g., "person", "car")')
    auto_label_parser.add_argument('--min_score', type=float, default=0.3, help='Minimum detection score for OWL-ViT')
    auto_label_parser.add_argument('--owlvit_model', default='google/owlvit-base-patch32', help='OWL-ViT model name')
    auto_label_parser.add_argument('--device', default='cpu', help='Device to run OWL-ViT and SAM on (e.g., "cuda" or "cpu")')
    auto_label_parser.set_defaults(func='run_auto_label')

    # Subparser for pseudo_labeling
    pseudo_label_parser = subparsers.add_parser('pseudo_label', help='Generate pseudo-labels for a directory of images')
    pseudo_label_parser.add_argument('--img_dir', required=True, help='Directory with unlabeled images')
    pseudo_label_parser.add_argument('--out_lbl_dir', required=True, help='Output directory for pseudo-labels')
    pseudo_label_parser.add_argument('--prompts', required=True, nargs='+', help='Detection prompts for OWL-ViT')
    pseudo_label_parser.add_argument('--min_score', type=float, default=0.3, help='Minimum detection score for pseudo-labeling')
    pseudo_label_parser.add_argument('--owlvit_model', default='google/owlvit-base-patch32', help='OWL-ViT model name')
    pseudo_label_parser.add_argument('--device', default='cpu', help='Device to run OWL-ViT on')
    pseudo_label_parser.set_defaults(func='run_pseudo_label')

    # Subparser for active_learning_step
    active_learning_parser = subparsers.add_parser('active_learn', help='Select informative samples for annotation')
    active_learning_parser.add_argument('--img_dir', required=True, help='Directory with unlabeled images')
    active_learning_parser.add_argument('--selection_count', type=int, default=100, help='Number of samples to select')
    active_learning_parser.add_argument('--prompts', required=True, nargs='+', help='Detection prompts for OWL-ViT')
    active_learning_parser.add_argument('--owlvit_model', default='google/owlvit-base-patch32', help='OWL-ViT model name')
    active_learning_parser.add_argument('--device', default='cpu', help='Device to run OWL-ViT on')
    active_learning_parser.set_defaults(func='run_active_learn')

    args = parser.parse_args()

    # Initialize EmbedDetector once if a command requiring it is chosen
    embed_detector_instance = None
    if hasattr(args, 'owlvit_model'):
        try:
            embed_detector_instance = EmbedDetector(
                {'device': args.device, 'name': args.owlvit_model, 'detection_threshold': args.min_score}
            )
        except Exception as e:
            logging.critical(f"Failed to initialize EmbedDetector: {e}. Exiting.")
            exit(1)

    # Initialize SAM Predictor once if auto_label is chosen
    sam_predictor_instance = None
    if args.command == 'auto_label':
        if SamPredictor is not None:
            try:
                # Assuming SamPredictor can be initialized without specific model path here
                # You might need to adjust this based on how your 'sam' module works.
                # For example: sam_predictor_instance = SamPredictor(build_sam(checkpoint="sam_vit_h.pth"))
                # Or if it's simpler: sam_predictor_instance = SamPredictor()
                # For this example, we'll assume a simple instantiation or a global setup in 'sam' module.
                # If your SAM setup requires a model path, you'll need to add an argument for it.
                sam_predictor_instance = SamPredictor(model=None) # Placeholder, adjust as per your SAM setup
                logging.info("SAM Predictor initialized.")
            except Exception as e:
                logging.critical(f"Failed to initialize SAM Predictor: {e}. Auto-labeling will not work.")
                sam_predictor_instance = None
        else:
            logging.critical("SAM (SamPredictor) is not available. Auto-labeling requires SAM.")


    if args.command == 'run_auto_label':
        auto_label(
            image_path=args.img,
            output_label_path=args.out_lbl,
            embed_detector=embed_detector_instance,
            sam_predictor_instance=sam_predictor_instance,
            detection_prompts=args.prompts,
            min_score=args.min_score
        )
    elif args.command == 'run_pseudo_label':
        pseudo_labeling(
            img_dir=args.img_dir,
            embed_detector=embed_detector_instance,
            out_label_dir=args.out_lbl_dir,
            detection_prompts=args.prompts,
            min_score=args.min_score
        )
    elif args.command == 'run_active_learn':
        active_learning_step(
            unlabeled_img_dir=args.img_dir,
            embed_detector=embed_detector_instance,
            detection_prompts=args.prompts,
            selection_count=args.selection_count
        )
    else:
        parser.print_help()
