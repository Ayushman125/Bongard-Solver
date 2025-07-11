import os
import random
import json
import logging
import numpy as np
from pathlib import Path
from collections import Counter
import math  # For math.ceil
import cv2
import yaml
import torch
from perlin_noise import PerlinNoise

# --- DALI Imports ---
try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    HAS_DALI = True
    dali_logger = logging.getLogger("dali_logger")
     # Set DALI logger level to ERROR to suppress all INFO and WARNING messages
    dali_logger.setLevel(logging.ERROR) 
    if not dali_logger.handlers:
        dali_logger.addHandler(logging.StreamHandler())
    dali_logger.info("NVIDIA DALI found and imported.")
except ImportError:
    HAS_DALI = False
    dali_logger = logging.getLogger("dali_logger")
    dali_logger.setLevel(logging.WARNING)
    if not dali_logger.handlers:
        dali_logger.addHandler(logging.StreamHandler())
    dali_logger.warning("NVIDIA DALI not found. Falling back to PyTorch DataLoader only if implemented.")

# --- GLOBAL CONSTANTS (should be consistent with pipeline_workers) ---
YOLO_CLASSES_LIST = [
    'circle', 'square', 'triangle', 'rectangle', 'pentagon', 'hexagon', 'octagon', 'polygon'
]
CLASS_ID = {cls_name: i for i, cls_name in enumerate(YOLO_CLASSES_LIST)}

DIFFICULTY_WEIGHTS = { # Weights for calculating difficulty score
    'num_objects': 0.4,
    'avg_complexity': 0.3,
    'num_relations': 0.3,
}

# --- DALI PIPELINE ---
class BongardDaliPipeline(Pipeline):
    def __init__(self, file_root, file_list, config, device_id, is_training=True):
        super(BongardDaliPipeline, self).__init__(
            batch_size=config['dali_batch_size'],
            num_threads=config['dali_num_threads'],
            device_id=device_id,
            seed=config['seed'],
            prefetch_queue_depth=config['dali_prefetch_queue']
        )
        self.file_root = file_root
        self.file_list = file_list
        self.config = config
        self.is_training = is_training
        self.decode_device = "gpu" if device_id != -1 and not config['force_cpu_dali'] else "cpu"
        self.perlin_noise_seed = config['seed'] # Use global seed for Perlin noise

        # Initialize PerlinNoise here, but note that it's not directly Numba-compatible
        # when passed to fn.python_function. We'll handle this in the Python function.
        self.perlin_instance = PerlinNoise(octaves=self.config.get('fract_depth', 4), seed=self.perlin_noise_seed)

        # Albumentations and Torchvision transforms will be initialized in the Python function
        # to ensure they are re-initialized per worker process if needed.
        
        # Pass difficulty weights as a tuple for Numba compatibility
        self.difficulty_weights_tuple = (
            self.config['difficulty_weights'].get('num_objects', 0.0),
            self.config['difficulty_weights'].get('avg_complexity', 0.0),
            self.config['difficulty_weights'].get('num_relations', 0.0)
        )

    def define_graph(self):
        # Read both image paths and annotation paths from the file list
        # file_list_include_file=True tells DALI to treat each line as multiple columns
        # and return them as separate outputs.
        # The first output (images) will be the image file paths, second (anno_paths) will be annotation file paths.
        imgs, anno_paths = fn.readers.file(
            files=self.file_list,
            file_list_include_file=True, # Crucial for reading two columns
            random_shuffle=self.is_training,
            name="FileReader"
        )

        # Decode images
        decoded_images = fn.decoders.image(imgs, device=self.decode_device, output_type=types.RGB)
        
        # Resize images to a consistent size
        resized_images = fn.resize(decoded_images, resize_x=self.config['image_size'][1], resize_y=self.config['image_size'][0], 
                                   interp_type=types.INTERP_LINEAR, device=self.decode_device)

        # Use fn.python_function to read and parse the JSON annotations
        # The 'anno_paths' tensor contains byte strings, so decode them to utf-8
        # The output of this function will be a string (JSON content)
        annotations_json_str = fn.python_function(
            anno_paths,
            function=lambda p: open(p.decode('utf-8'), 'r').read(),
            num_outputs=1, # One output: the JSON string
            device="cpu" # File I/O should be on CPU
        )

        # Use fn.python_function to apply augmentations and re-calculate YOLO labels and difficulty
        # This function will receive the image tensor and the JSON string.
        # It needs to return the augmented image, new YOLO labels, and updated annotation JSON.
        # The output of the python_function needs to be a tuple of DALI tensors.
        # We need to ensure the `pipeline_workers` module is available in the DALI worker's environment.
        augmented_image, yolo_labels, updated_annotations_json_str, difficulty_score = fn.python_function(
            resized_images,
            annotations_json_str, # Pass the JSON string as well
            function=self._apply_augmentations_and_reannotate,
            num_outputs=4, # Returns image, yolo_labels (as string list), annotations_json_str, difficulty_score
            output_layouts=["HWC", "", "", ""], # HWC for image, empty for list/string/scalar
            output_types=[types.UINT8, types.STRING, types.STRING, types.FLOAT], # Types for outputs
            py_num_workers=self.config['dali_py_num_workers'], # Number of Python workers
            cpu_peak_threads=[os.cpu_count() -1], # Max threads for python function
            device="cpu" # Perform augmentations and re-annotation on CPU
        )
        
        # Convert image to float32 and normalize for model input
        augmented_image = fn.cast(augmented_image, dtype=types.FLOAT)
        augmented_image = augmented_image / 255.0

        return augmented_image, yolo_labels, updated_annotations_json_str, difficulty_score

    def _apply_augmentations_and_reannotate(self, img_np, anno_json_str_np):
        # img_np is a NumPy array (HWC, UINT8)
        # anno_json_str_np is a NumPy array of a single string, so extract the string
        anno_json_str = anno_json_str_np.item() # Extract the string from the 0-D array
        
        # Parse the initial annotation data
        initial_anno_data = json.loads(anno_json_str)
        
        # Re-initialize PerlinNoise instance within the worker process
        # This avoids pickling issues with global objects.
        perlin_instance = PerlinNoise(octaves=self.config.get('fract_depth', 4), seed=self.perlin_noise_seed)

        # Initialize augmenters within the worker process using the passed config dict
        # This ensures each worker has its own instances.
        alb_transform, augmix_transform, rand_augment, auto_augment = pipeline_workers.make_augmenters(self.config)

        # Extract initial YOLO labels and object attributes from the parsed JSON
        initial_yolo_labels = initial_anno_data.get('yolo_labels_raw', [])
        initial_objects_attrs = initial_anno_data.get('objects', [])
        initial_contours_objects = [] # DALI doesn't pass raw contours, so this part needs re-detection or simplification
                                    # For now, we'll assume initial_objects_attrs are sufficient or re-detect.
                                    # For fill_contour_variation, we'd need contours.
        
        # Re-detect contours if fill_contour_p is active and contours are needed
        if random.random() < self.config['fill_contour_p'] and initial_objects_attrs:
            # This is a simplification: to truly re-detect contours, we'd need the original
            # image or a way to reconstruct them from bboxes. For now, we'll skip
            # fill_contour_variation if original contours aren't available.
            # A more robust solution would involve passing contours through DALI or re-running
            # contour detection on the `img_np` if it's the raw image.
            # For this example, we'll use the bounding boxes to simulate a filled object.
            filled_img = img_np.copy()
            for obj_attr in initial_objects_attrs:
                x, y, w, h = obj_attr['box']
                # Create a simple filled rectangle based on bbox
                fill_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)) # Random color for simplicity
                cv2.rectangle(filled_img, (x, y), (x + w, y + h), fill_color, -1)
            img_np = filled_img # Apply this variant
            
        # Apply other augmentations (choose one randomly or apply all with probabilities)
        # For simplicity, let's pick one random augmentation for now for each call.
        # In a real pipeline, you'd apply them based on probabilities.
        applied_aug_name = "original"
        
        # Convert img_np to PIL Image for Torchvision transforms if needed
        pil_img_for_tv = Image.fromarray(img_np)
        
        # Apply augmentations based on their probabilities from config
        if random.random() < self.config['elastic_p'] or \
           random.random() < self.config['phot_blur_p'] or \
           random.random() < self.config['jpeg_p']:
            # Albumentations expects RGB and returns RGB
            # Prepare bboxes and class_labels for Albumentations
            alb_bboxes = []
            alb_class_labels = []
            for l in initial_yolo_labels:
                parts = list(map(float, l.split()))
                alb_class_labels.append(int(parts[0]))
                alb_bboxes.append(parts[1:]) # cx, cy, w, h
            
            try:
                augmented_alb = alb_transform(image=img_np, bboxes=alb_bboxes, class_labels=alb_class_labels)
                img_np = augmented_alb['image']
                initial_yolo_labels = [] # Clear old labels
                for bbox, class_label in zip(augmented_alb['bboxes'], augmented_alb['class_labels']):
                    cx, cy, w, h = bbox
                    new_w = max(0.0, w)
                    new_h = max(0.0, h)
                    if new_w == 0: new_w = 1e-6
                    if new_h == 0: new_h = 1e-6
                    initial_yolo_labels.append(f"{int(class_label)} {cx:.6f} {cy:.6f} {new_w:.6f} {new_h:.6f}")
                applied_aug_name = "albumentations"
            except ValueError as e:
                dali_logger.warning(f"Albumentations transform failed in DALI worker: {e}. Skipping this variant.")
            except Exception as e:
                dali_logger.error(f"An unexpected error occurred during Albumentations transform in DALI worker: {e}. Skipping this variant.")

        if random.random() < self.config['augmix_p']:
            img_np = augmix_transform(image=img_np)['image']
            applied_aug_name = "augmix"

        if random.random() < 0.5: # Example: apply RandAugment with 50% chance
            img_np = np.array(rand_augment(pil_img_for_tv))
            applied_aug_name = "rand_augment"
        
        # Apply Perlin Noise Background if chosen
        if random.random() < 0.3: # Example probability for Perlin background
            H, W, _ = img_np.shape
            # For Numba compatibility, pass a Numba-jitted function or re-implement Perlin logic
            # Here, we'll use a simple lambda that calls the PerlinNoise instance.
            # Numba's @njit(nopython=True) won't work directly with `perlin_instance`.
            # If `perlin_bg` is @njit, it will run in object mode for `perlin_noise_func`.
            @njit(cache=True)
            def _numba_perlin_noise_callable_for_dali(coords):
                return perlin_instance(coords) # This will run in object mode if perlin_instance is not Numba-compatible

            perlin_img_gray = pipeline_workers.perlin_bg(W, H, _numba_perlin_noise_callable_for_dali)
            perlin_img_bgr = cv2.cvtColor(perlin_img_gray, cv2.COLOR_GRAY2BGR)
            
            # Blend with original image (simple overlay for now)
            # Assuming img_np is HWC, RGB
            gray_fg = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray_fg, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            img_fg = cv2.bitwise_and(img_np, img_np, mask=mask)
            bg_part = cv2.bitwise_and(perlin_img_bgr, perlin_img_bgr, mask=mask_inv) # Ensure perlin_img_bgr is RGB
            img_np = cv2.add(img_fg, bg_part)
            applied_aug_name = "perlin_bg"

        # Re-detect objects and re-calculate attributes/relations/difficulty on the augmented image
        H_aug, W_aug, _ = img_np.shape
        gray_aug_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Need to pass the config for contour detection parameters
        contours_aug = pipeline_workers.detect_contours_tuned(gray_aug_img, self.config['cnt_block'], self.config['cnt_C'],
                                                              self.config['min_area'], self.config['max_cnt'], self.config['morphological_ops'])
        
        re_yolo_labels = []
        re_objects_attrs = []
        for c_aug in contours_aug:
            result_aug = pipeline_workers.cnt_to_yolo(c_aug, W_aug, H_aug, CLASS_ID, self.config['min_area'])
            if result_aug:
                shape_aug, yolo_box_str_aug, bbox_tuple_aug, _ = result_aug
                re_yolo_labels.append(yolo_box_str_aug)
                re_objects_attrs.append(pipeline_workers.extract_attrs(shape_aug, c_aug, gray_aug_img, bbox_tuple_aug, W_aug, H_aug))
        
        re_relations = pipeline_workers.compute_relations(re_objects_attrs)
        re_avg_complexity = np.mean([a.get('complexity', 0) for a in re_objects_attrs]) if re_objects_attrs else 0
        re_difficulty_score = pipeline_workers.compute_difficulty_score(len(re_objects_attrs), 
                                                                        re_avg_complexity,
                                                                        len(re_relations), 
                                                                        self.difficulty_weights_tuple) # Pass tuple
        
        # Update the annotation JSON with new information
        updated_anno_data = deepcopy(initial_anno_data)
        updated_anno_data['image_size'] = [H_aug, W_aug]
        updated_anno_data['yolo_labels_raw'] = re_yolo_labels # Store the re-detected YOLO labels
        updated_anno_data['objects'] = re_objects_attrs
        updated_anno_data['relations'] = re_relations
        updated_anno_data['difficulty_score'] = re_difficulty_score
        updated_anno_data['applied_augmentation'] = applied_aug_name # Track which aug was applied

        # Convert the updated annotation data back to a JSON string
        updated_annotations_json_str = json.dumps(updated_anno_data)
        
        # DALI expects outputs as NumPy arrays.
        # For yolo_labels, convert list of strings to a NumPy array of strings.
        # For annotations_json_str, convert string to a 0-D NumPy array of string.
        # For difficulty_score, convert float to a 0-D NumPy array of float.
        return img_np, np.array(re_yolo_labels, dtype=object), np.array(updated_annotations_json_str, dtype=object), np.array(re_difficulty_score, dtype=np.float32)

