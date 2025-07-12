import cv2
import numpy as np
import json
import logging
from pathlib import Path
from copy import deepcopy

# Import helper functions from my_data_utils
# This assumes my_data_utils.py is in the same directory or accessible via PYTHONPATH
from my_data_utils import detect_contours, extract_attrs, compute_relations, CLASS_ID, YOLO_CLASSES_LIST, DIFFICULTY_WEIGHTS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def detect_labels_and_difficulty(image_path, config, class_id_map, yolo_classes_list):
    """
    Processes a single image to detect contours, extract attributes,
    compute relations, and calculate a difficulty score.
    Args:
        image_path (str or Path): Path to the image file.
        config (dict): Configuration dictionary containing parameters like cnt_block, min_area, etc.
        class_id_map (dict): Mapping from class name to ID.
        yolo_classes_list (list): List of YOLO class names.
    Returns:
        tuple: (list of YOLO labels, difficulty score, list of attributes, list of relations)
               Returns (None, None, None, None) if image cannot be read or processed.
    """
    try:
        image_np = cv2.imread(str(image_path))
        if image_np is None:
            logger.warning(f"Could not read image: {image_path}. Skipping.")
            return None, None, None, None

        img_height, img_width, _ = image_np.shape
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # 1. Detect Contours
        contours = detect_contours(
            image_gray=image_gray,
            block_size=config['cnt_block'],
            C=config['cnt_C'],
            min_area=config['min_area'],
            max_contours_to_find=config['max_cnt'],
            morphological_ops=config['morphological_ops']
        )

        yolo_labels = [] # List of (class_id, cx, cy, w, h)
        attributes_list = [] # List of attribute dicts for each object

        # 2. Extract Attributes and Generate YOLO Labels
        for contour in contours:
            attrs = extract_attrs(contour, image_np) # Pass original color image for color extraction
            shape_type = attrs.get('shape_type')
            if shape_type in class_id_map: # Ensure we only use known classes
                class_id = class_id_map[shape_type] # Use specific class ID
                yolo_box = cnt_to_yolo(contour, img_width, img_height, class_id, config['min_area'])
                if yolo_box:
                    yolo_labels.append(yolo_box)
                    attributes_list.append(attrs)
            else:
                logger.debug(f"Skipping contour with unknown shape_type: {shape_type}")

        # 3. Compute Relations
        relations = compute_relations(attributes_list)

        # 4. Calculate Difficulty Score
        num_objects = len(attributes_list)
        avg_complexity = np.mean([a.get('complexity', 0) for a in attributes_list]) if attributes_list else 0
        num_relations = len(relations)
        
        difficulty_score = (
            DIFFICULTY_WEIGHTS.get('num_objects', 0) * num_objects +
            DIFFICULTY_WEIGHTS.get('avg_complexity', 0) * avg_complexity +
            DIFFICULTY_WEIGHTS.get('num_relations', 0) * num_relations
        )
        
        max_possible_difficulty = (
            DIFFICULTY_WEIGHTS.get('num_objects', 0) * config['max_cnt'] +
            DIFFICULTY_WEIGHTS.get('avg_complexity', 0) * 100 + # Assuming max complexity around 100
            DIFFICULTY_WEIGHTS.get('num_relations', 0) * (config['max_cnt'] * (config['max_cnt'] - 1) / 2)
        )
        difficulty_score = difficulty_score / max_possible_difficulty if max_possible_difficulty > 0 else 0.0
        difficulty_score = np.clip(difficulty_score, 0.0, 1.0)

        return yolo_labels, float(difficulty_score), attributes_list, relations

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
        return None, None, None, None

def _detect_labels_and_difficulty_wrapper(args):
    """Wrapper for multiprocessing pool to pass multiple arguments."""
    image_path, config, class_id_map, yolo_classes_list = args
    return detect_labels_and_difficulty(image_path, config, class_id_map, yolo_classes_list)

def _detect_labels_and_difficulty_wrapper_for_save(args):
    """
    Wrapper for multiprocessing pool to process an image and save its annotation.
    This is used for the raw annotation pre-computation phase.
    """
    image_path, config, class_id_map, yolo_classes_list = args
    yolo_labels, difficulty_score, attributes_list, relations = detect_labels_and_difficulty(
        image_path, config, class_id_map, yolo_classes_list
    )

    if yolo_labels is not None:
        anno_filename = Path(image_path).stem + '.json'
        anno_path = Path(config['raw_annotations_dir']) / anno_filename

        annotations_dict = {
            'image_path': str(image_path),
            'objects': attributes_list,
            'relations': relations,
            'difficulty_score': difficulty_score,
            'yolo_labels': [list(label) for label in yolo_labels] # Convert numpy array to list for JSON
        }
        try:
            with open(anno_path, 'w') as f:
                json.dump(annotations_dict, f, indent=4)
            return True # Indicate success
        except Exception as e:
            logger.error(f"Error saving annotation for {image_path} to {anno_path}: {e}")
            return False # Indicate failure
    return False # Indicate failure if labels were None

