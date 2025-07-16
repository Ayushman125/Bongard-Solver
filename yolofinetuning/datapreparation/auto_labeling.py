import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import Counter
import torch
import logging
from PIL import Image # Required for OWL-ViT's image input
import json # For loading/saving class mappings

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
    # Assuming 'SamPredictor' is available from your 'sam' module
    from segment_anything import SamPredictor, sam_model_registry # Adjust based on your SAM setup
    # You might need to load a SAM model checkpoint here or pass it during initialization
    # Example: sam_model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    # sam_predictor_instance = SamPredictor(sam_model)
except ImportError:
    SamPredictor = None
    sam_model_registry = None
    logging.warning("SAM (segment_anything) not installed. SAM functionality will not be available.")

# Optional: MLflow for logging
try:
    import mlflow
except ImportError:
    mlflow = None
    logging.warning("MLflow not installed. MLflow logging will be skipped.")

# --- EmbedDetector (OWL-ViT) Class ---
class EmbedDetector:
    """
    OWL-ViT detection and embedding extraction utility.
    Usage:
        det = EmbedDetector(model_cfg)
        boxes, scores, labels, embeddings = det.detect(image, prompts)
    """
    def __init__(self, model_cfg: dict):
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
            raise # Re-raise to indicate critical failure

    def detect(self, image: Image.Image, prompts: list):
        """
        Detect objects and extract embeddings using OWL-ViT.
        Args:
            image (PIL.Image): Input image.
            prompts (list): List of strings representing detection prompts (e.g., ["a photo of a car"]).
        Returns:
            tuple: (boxes_xyxy, scores, label_indices, clip_embeddings)
                boxes_xyxy (np.ndarray): Detected bounding boxes in [x0,y0,x1,y1] format.
                scores (np.ndarray): Confidence scores for each detection.
                label_indices (np.ndarray): Indices into the `prompts` list for each detection.
                clip_embeddings (list of np.ndarray): CLIP embeddings for each detected crop.
        """
        try:
            if self.model is None:
                raise RuntimeError("OWL-ViT model not loaded. Cannot perform detection.")
            if not isinstance(image, Image.Image):
                logging.error("Input 'image' must be a PIL.Image object.")
                return np.array([]), np.array([]), np.array([]), []

            # Process inputs for OWL-ViT
            inputs = self.processor(text=[prompts], images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad(): # Ensure no gradient computation for inference
                outputs = self.model(**inputs)

            # Post-process detections to get xyxy boxes and scores
            target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.threshold
            )[0]

            boxes_xyxy = results['boxes'].cpu().numpy()         # [[x0,y0,x1,y1], ...]
            scores = results['scores'].cpu().numpy()           # [0.8, 0.5, ...]
            label_indices = results['labels'].cpu().numpy()    # [class_id, ...] (indices into prompts[])

            # Extract CLIP embeddings for each crop
            clip_embeddings = []
            if TF is not None: # Ensure torchvision.transforms.functional is available
                for box in boxes_xyxy:
                    x0, y0, x1, y1 = [int(v) for v in box]
                    # Ensure crop coordinates are within image bounds
                    x0 = max(0, x0)
                    y0 = max(0, y0)
                    x1 = min(image.width, x1)
                    y1 = min(image.height, y1)

                    if x1 > x0 and y1 > y0: # Ensure valid crop
                        crop = TF.crop(image, y0, x0, y1-y0, x1-x0)
                        # The images_processor expects a PIL Image or list of PIL Images
                        pix = self.processor.images_processor(crop, return_tensors="pt").pixel_values.to(self.device)
                        # Correct attribute access for CLIP vision model within OwlViTForObjectDetection
                        clip_outputs = self.model.owlvit.vision_model(pix) 
                        clip_embeddings.append(clip_outputs.pooler_output.cpu().detach().numpy()[0])
                    else:
                        # Append a zero vector for invalid crops
                        clip_embeddings.append(np.zeros(self.model.owlvit.config.text_config.hidden_size))
            else:
                logging.warning("torchvision.transforms.functional not available. Skipping CLIP embedding extraction.")
                # Return dummy embeddings of appropriate size (e.g., CLIP's default embedding size 768)
                clip_embeddings = [np.zeros(768) for _ in range(len(boxes_xyxy))]

            return boxes_xyxy, scores, label_indices, clip_embeddings
        except Exception as e:
            logging.error(f"[ERROR] OWL-ViT detection failed: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([]), np.array([]), []

# --- Class Mapping Utility ---
def load_class_map(class_map_path: str) -> dict:
    """
    Loads a class mapping from a JSON file.
    Expected format: {"owlvit_label_index": "your_class_id", ...}
    or {"owlvit_prompt_text": "your_class_id", ...}
    Args:
        class_map_path (str): Path to the JSON class mapping file.
    Returns:
        dict: The loaded class mapping.
    """
    if not os.path.exists(class_map_path):
        logging.warning(f"Class mapping file not found at {class_map_path}. Using OWL-ViT's raw label indices.")
        return {}
    try:
        with open(class_map_path, 'r') as f:
            class_map = json.load(f)
        logging.info(f"Loaded class mapping from {class_map_path}.")
        return class_map
    except Exception as e:
        logging.error(f"Error loading class map from {class_map_path}: {e}. Using empty map.")
        return {}

def map_owlvit_label_to_dataset_class_id(owlvit_label_index: int, owlvit_prompt_text: str, 
                                          class_map: dict, class_names: list) -> int:
    """
    Maps an OWL-ViT label (index or text) to a dataset's specific class ID.
    Prioritizes mapping by prompt text, then by index if text map is not found.
    If no mapping is found, returns -1 or a default class ID.
    
    Args:
        owlvit_label_index (int): The numerical label index returned by OWL-ViT.
        owlvit_prompt_text (str): The actual text prompt corresponding to the label index.
        class_map (dict): A dictionary for mapping. Can map "prompt_text" -> class_id
                          or "label_index" (as string) -> class_id.
        class_names (list): List of your dataset's class names, used to get class ID by name.
    Returns:
        int: The mapped class ID for your dataset. Returns -1 if no mapping is found.
    """
    # Try mapping by prompt text first
    if owlvit_prompt_text in class_map:
        mapped_id = class_map[owlvit_prompt_text]
        if isinstance(mapped_id, str): # If mapped to a class name, find its index
            try:
                return class_names.index(mapped_id)
            except ValueError:
                logging.warning(f"Mapped class name '{mapped_id}' not found in dataset class_names. Using -1.")
                return -1
        return int(mapped_id)

    # If not mapped by text, try mapping by index (converted to string key)
    if str(owlvit_label_index) in class_map:
        mapped_id = class_map[str(owlvit_label_index)]
        if isinstance(mapped_id, str):
            try:
                return class_names.index(mapped_id)
            except ValueError:
                logging.warning(f"Mapped class name '{mapped_id}' not found in dataset class_names. Using -1.")
                return -1
        return int(mapped_id)
    
    # If no explicit mapping, try to find the prompt text directly in class_names
    try:
        return class_names.index(owlvit_prompt_text)
    except ValueError:
        logging.warning(f"No explicit mapping or direct match found for OWL-ViT label '{owlvit_prompt_text}' (index {owlvit_label_index}). Using -1.")
        return -1


# --- Auto-labeling function ---
def auto_label(image_path: str, output_label_path: str, embed_detector: EmbedDetector, 
               sam_predictor_instance, detection_prompts: list, class_names: list, 
               class_map: dict = None, min_score: float = 0.3):
    """
    Auto-labels an image using OWL-ViT for detection and SAM for segmentation.
    Args:
        image_path (str): Path to image.
        output_label_path (str): Path to save YOLO label (.txt file).
        embed_detector (EmbedDetector): An initialized EmbedDetector instance (OWL-ViT).
        sam_predictor_instance: An initialized SAM predictor instance.
        detection_prompts (list): List of strings for OWL-ViT to detect.
        class_names (list): List of your dataset's class names (e.g., ['person', 'car']).
        class_map (dict, optional): A dictionary for mapping OWL-ViT labels to your dataset's class IDs.
        min_score (float): Minimum detection score for OWL-ViT to consider a box.
    """
    if embed_detector is None or sam_predictor_instance is None:
        logging.error("Required models (OWL-ViT or SAM) not available for auto-labeling.")
        # Ensure an empty label file is created on critical error
        with open(output_label_path, 'w') as f: pass
        return

    try:
        image_pil = Image.open(image_path).convert('RGB')
        image_np = cv2.imread(image_path) # For SAM, which often uses OpenCV format (BGR)

        if image_np is None:
            logging.error(f"Could not read image: {image_path}. Skipping auto-labeling.")
            with open(output_label_path, 'w') as f: pass
            return

        # 1. OWL-ViT Detection
        boxes_xyxy, scores, label_indices, _ = embed_detector.detect(image_pil, detection_prompts)

        if len(boxes_xyxy) == 0:
            logging.info(f"No objects detected by OWL-ViT in {image_path} with prompts {detection_prompts}. Creating empty label file.")
            with open(output_label_path, 'w') as f: pass
            return

        # Filter boxes by min_score and map labels
        final_boxes_xyxy = []
        final_scores = []
        final_class_ids = []

        for i, score in enumerate(scores):
            if score >= min_score:
                owlvit_label_index = label_indices[i]
                owlvit_prompt_text = detection_prompts[owlvit_label_index]
                
                mapped_class_id = map_owlvit_label_to_dataset_class_id(
                    owlvit_label_index, owlvit_prompt_text, class_map or {}, class_names
                )
                
                if mapped_class_id != -1: # Only include if a valid mapping was found
                    final_boxes_xyxy.append(boxes_xyxy[i])
                    final_scores.append(score)
                    final_class_ids.append(mapped_class_id)
                else:
                    logging.warning(f"Skipping detection for '{owlvit_prompt_text}' (index {owlvit_label_index}) in {image_path} due to no class mapping.")

        if not final_boxes_xyxy:
            logging.info(f"No objects detected above score {min_score} or with valid class mapping in {image_path}. Creating empty label file.")
            with open(output_label_path, 'w') as f: pass
            return

        # Convert final_boxes_xyxy to torch tensor for SAM
        input_boxes_sam = torch.tensor(final_boxes_xyxy, device=embed_detector.device)

        # 2. SAM Segmentation
        sam_predictor_instance.set_image(image_np) # SAM expects BGR numpy array
        masks, _, _ = sam_predictor_instance.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes_sam,
            multimask_output=False
        )
        # masks shape: (num_boxes, 1, H, W) -> squeeze to (num_boxes, H, W)
        masks = masks.squeeze(1).cpu().numpy() # Move to CPU and convert to numpy

        h, w, _ = image_np.shape
        yolo_labels = []

        for i, (box_xyxy, class_id) in enumerate(zip(final_boxes_xyxy, final_class_ids)):
            # Convert box from xyxy to YOLO format (cx, cy, w, h)
            x1, y1, x2, y2 = box_xyxy
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            
            # Ensure coordinates are within [0, 1] range after normalization
            cx = np.clip(cx, 0.0, 1.0)
            cy = np.clip(cy, 0.0, 1.0)
            bw = np.clip(bw, 0.0, 1.0)
            bh = np.clip(bh, 0.0, 1.0)

            yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # Optionally, save masks or use them for further analysis
            # For example, you could save each mask as a separate PNG
            # mask_output_path = output_label_path.replace('.txt', f'_mask_{i}.png')
            # cv2.imwrite(mask_output_path, (masks[i] * 255).astype(np.uint8))

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
        import traceback
        traceback.print_exc()
        # Ensure an empty file is created even on error to avoid missing files
        with open(output_label_path, 'w') as f: pass


# --- Pseudo-labeling function ---
def pseudo_labeling(img_dir: str, embed_detector: EmbedDetector, out_label_dir: str, 
                    detection_prompts: list, class_names: list, class_map: dict = None, 
                    min_score: float = 0.3):
    """
    Generate pseudo-labels for unlabeled images using an OWL-ViT model.
    Args:
        img_dir (str): Directory with unlabeled images (e.g., .jpg or .png).
        embed_detector (EmbedDetector): An initialized EmbedDetector instance (OWL-ViT).
        out_label_dir (str): Output directory for pseudo-labels (YOLO format .txt files).
        detection_prompts (list): List of strings for OWL-ViT to detect.
        class_names (list): List of your dataset's class names.
        class_map (dict, optional): A dictionary for mapping OWL-ViT labels to your dataset's class IDs.
        min_score (float): Minimum detection score for pseudo-labeling.
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
            w, h = image_pil.width, image_pil.height

            # Use OWL-ViT to detect objects
            boxes_xyxy, scores, label_indices, _ = embed_detector.detect(image_pil, detection_prompts)

            pseudo_labels = []
            for box_xyxy, score, owlvit_label_index in zip(boxes_xyxy, scores, label_indices):
                if score >= min_score:
                    owlvit_prompt_text = detection_prompts[owlvit_label_index]
                    mapped_class_id = map_owlvit_label_to_dataset_class_id(
                        owlvit_label_index, owlvit_prompt_text, class_map or {}, class_names
                    )
                    
                    if mapped_class_id != -1:
                        x1, y1, x2, y2 = box_xyxy
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        
                        # Ensure coordinates are within [0, 1] range after normalization
                        cx = np.clip(cx, 0.0, 1.0)
                        cy = np.clip(cy, 0.0, 1.0)
                        bw = np.clip(bw, 0.0, 1.0)
                        bh = np.clip(bh, 0.0, 1.0)

                        pseudo_labels.append(f"{mapped_class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    else:
                        logging.debug(f"Skipping pseudo-label for '{owlvit_prompt_text}' in {img_path} due to no class mapping.")

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
            import traceback
            traceback.print_exc()
            # Ensure an empty file is created even on error
            out_path = os.path.join(out_label_dir, os.path.basename(img_path).replace(os.path.splitext(img_path)[1], '.txt'))
            with open(out_path, 'w') as f: pass


# --- Active learning utility ---
def active_learning_step(unlabeled_img_dir: str, embed_detector: EmbedDetector, 
                         detection_prompts: list, class_names: list, class_map: dict = None, 
                         selection_count: int = 100):
    """
    Selects the most informative samples for annotation using OWL-ViT's uncertainty.
    A simple uncertainty metric (e.g., average of 1 - max_score per image) is used.
    Args:
        unlabeled_img_dir (str): Directory with unlabeled images.
        embed_detector (EmbedDetector): An initialized EmbedDetector instance (OWL-ViT).
        detection_prompts (list): List of strings for OWL-ViT to detect.
        class_names (list): List of your dataset's class names.
        class_map (dict, optional): A dictionary for mapping OWL-ViT labels to your dataset's class IDs.
        selection_count (int): Number of samples to select.
    Returns:
        List[str]: List of paths to selected image files.
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
            boxes_xyxy, scores, label_indices, _ = embed_detector.detect(image_pil, detection_prompts)

            # Filter scores based on class mapping and min_score (if desired for uncertainty calculation)
            filtered_scores = []
            for i, score in enumerate(scores):
                owlvit_label_index = label_indices[i]
                owlvit_prompt_text = detection_prompts[owlvit_label_index]
                mapped_class_id = map_owlvit_label_to_dataset_class_id(
                    owlvit_label_index, owlvit_prompt_text, class_map or {}, class_names
                )
                if mapped_class_id != -1 and score >= embed_detector.threshold: # Use detector's threshold
                    filtered_scores.append(score)

            # Uncertainty metric: average of (1 - score) for all confident, mapped detections
            # Or, if no detections, assume high uncertainty (1.0)
            uncertainty = np.mean(1 - np.array(filtered_scores)) if filtered_scores else 1.0 
            uncertainties.append((uncertainty, img_path))

        except Exception as e:
            logging.error(f"Failed to calculate uncertainty for {img_path}: {e}")
            import traceback
            traceback.print_exc()
            uncertainties.append((1.0, img_path)) # Assume high uncertainty on error

    # Sort by uncertainty (descending) to get most informative
    uncertainties.sort(key=lambda x: x[0], reverse=True)
    selected = [p for _, p in uncertainties[:selection_count]]

    logging.info(f"Active learning selected {len(selected)} samples for annotation.")
    if mlflow:
        try:
            mlflow.log_param("active_learning_selected_count", len(selected))
            # Log selected paths as an artifact if the list is long
            if len(selected) > 0:
                selected_paths_file = os.path.join(unlabeled_img_dir, "selected_for_annotation.txt")
                with open(selected_paths_file, 'w') as f:
                    for p in selected:
                        f.write(p + "\n")
                mlflow.log_artifact(selected_paths_file, "active_learning")
            else:
                mlflow.log_param("active_learning_selected_paths", "None")
        except Exception as e:
            logging.error(f"MLflow logging failed for active_learning: {e}")

    return selected

# --- CLI entry point (only for direct testing of this module's functions) ---
if __name__ == "__main__":
    import argparse
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Auto-labeling and Dataset Utilities with OWL-ViT and SAM")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Shared arguments for model initialization
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--owlvit_model', default='google/owlvit-base-patch32', help='OWL-ViT model name')
    parent_parser.add_argument('--device', default='cpu', help='Device to run OWL-ViT and SAM on (e.g., "cuda" or "cpu")')
    parent_parser.add_argument('--min_score', type=float, default=0.3, help='Minimum detection score')
    parent_parser.add_argument('--class_names', required=True, nargs='+', help='List of your dataset class names (e.g., "circle" "square")')
    parent_parser.add_argument('--class_map_path', default=None, help='Path to JSON file for OWL-ViT label to dataset class ID mapping')


    # Subparser for auto_label
    auto_label_parser = subparsers.add_parser('auto_label', parents=[parent_parser], help='Auto-label a single image using OWL-ViT and SAM')
    auto_label_parser.add_argument('--img', required=True, help='Path to input image')
    auto_label_parser.add_argument('--out_lbl', required=True, help='Path to output YOLO label file (.txt)')
    auto_label_parser.add_argument('--prompts', required=True, nargs='+', help='Detection prompts for OWL-ViT (e.g., "a photo of a person", "a car")')
    auto_label_parser.set_defaults(func='run_auto_label')

    # Subparser for pseudo_labeling
    pseudo_label_parser = subparsers.add_parser('pseudo_label', parents=[parent_parser], help='Generate pseudo-labels for a directory of images')
    pseudo_label_parser.add_argument('--img_dir', required=True, help='Directory with unlabeled images')
    pseudo_label_parser.add_argument('--out_lbl_dir', required=True, help='Output directory for pseudo-labels')
    pseudo_label_parser.add_argument('--prompts', required=True, nargs='+', help='Detection prompts for OWL-ViT')
    pseudo_label_parser.set_defaults(func='run_pseudo_label')

    # Subparser for active_learning_step
    active_learning_parser = subparsers.add_parser('active_learn', parents=[parent_parser], help='Select informative samples for annotation')
    active_learning_parser.add_argument('--img_dir', required=True, help='Directory with unlabeled images')
    active_learning_parser.add_argument('--selection_count', type=int, default=100, help='Number of samples to select')
    active_learning_parser.add_argument('--prompts', required=True, nargs='+', help='Detection prompts for OWL-ViT')
    active_learning_parser.set_defaults(func='run_active_learn')

    args = parser.parse_args()

    # Load class map if path is provided
    class_map_data = {}
    if args.class_map_path:
        class_map_data = load_class_map(args.class_map_path)

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
        if SamPredictor is not None and sam_model_registry is not None:
            try:
                # IMPORTANT: You need to specify a SAM model type and checkpoint path here.
                # Example: sam_model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
                # For this example, we'll use a placeholder.
                # In a real setup, you'd add CLI args for SAM model type and checkpoint.
                sam_model_type = "vit_b" # Example: vit_b, vit_l, vit_h
                sam_checkpoint_path = "path/to/your/sam_model.pth" # REPLACE WITH ACTUAL PATH
                
                if not os.path.exists(sam_checkpoint_path):
                    logging.critical(f"SAM checkpoint not found at {sam_checkpoint_path}. Auto-labeling will not work.")
                    sam_predictor_instance = None
                else:
                    sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
                    sam_model.to(device=args.device)
                    sam_predictor_instance = SamPredictor(sam_model)
                    logging.info("SAM Predictor initialized.")
            except Exception as e:
                logging.critical(f"Failed to initialize SAM Predictor: {e}. Auto-labeling will not work.")
                import traceback
                traceback.print_exc()
                sam_predictor_instance = None
        else:
            logging.critical("SAM (SamPredictor or sam_model_registry) is not available. Auto-labeling requires SAM.")


    if hasattr(args, 'func'):
        if args.func == 'run_auto_label':
            auto_label(
                image_path=args.img,
                output_label_path=args.out_lbl,
                embed_detector=embed_detector_instance,
                sam_predictor_instance=sam_predictor_instance,
                detection_prompts=args.prompts,
                class_names=args.class_names,
                class_map=class_map_data,
                min_score=args.min_score
            )
        elif args.func == 'run_pseudo_label':
            pseudo_labeling(
                img_dir=args.img_dir,
                embed_detector=embed_detector_instance,
                out_label_dir=args.out_lbl_dir,
                detection_prompts=args.prompts,
                class_names=args.class_names,
                class_map=class_map_data,
                min_score=args.min_score
            )
        elif args.func == 'run_active_learn':
            active_learning_step(
                unlabeled_img_dir=args.img_dir,
                embed_detector=embed_detector_instance,
                detection_prompts=args.prompts,
                class_names=args.class_names,
                class_map=class_map_data,
                selection_count=args.selection_count
            )
    else:
        parser.print_help()
