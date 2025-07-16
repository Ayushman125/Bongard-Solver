# Folder: bongard_solver/src/data/
# File: generator.py

import random
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2 # For potential image processing, e.g., converting PIL to OpenCV format
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple, Optional, Union

# Import global configuration and Bongard rules from the project root
try:
    from config import CONFIG, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, \
                       ATTRIBUTE_FILL_MAP, ATTRIBUTE_SIZE_MAP, \
                       ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_TEXTURE_MAP, \
                       RELATION_MAP
    from src.bongard_rules import ALL_BONGARD_RULES # Assuming bongard_rules.py is in src/
    from src.utils.augment import augment_image # Import augmentation function
except ImportError as e:
    logging.error(f"Failed to import from config or src.bongard_rules/augment: {e}. Using dummy values.")
    CONFIG = {'data': {'image_size': [128, 128], 'synthetic_data_config': {'min_objects_per_image': 1, 'max_objects_per_image': 3, 'object_size_range': (20, 80), 'padding': 10, 'font_path': None, 'max_support_images_per_problem': 5}}, 'training': {'augmentation_config': {}}}
    ATTRIBUTE_SHAPE_MAP = {'circle':0, 'square':1, 'triangle':2}
    ATTRIBUTE_COLOR_MAP = {'red':0, 'blue':1}
    ATTRIBUTE_FILL_MAP = {'solid':0, 'outlined':1}
    ATTRIBUTE_SIZE_MAP = {'small':0, 'medium':1, 'large':2}
    ATTRIBUTE_ORIENTATION_MAP = {'upright':0}
    ATTRIBUTE_TEXTURE_MAP = {'flat':0}
    RELATION_MAP = {'above':0, 'left_of':1}
    ALL_BONGARD_RULES = [] # Empty list for dummy rules
    def augment_image(img): return img # Dummy augment function

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LogoGenerator:
    """
    Programmatically generates Bongard-LOGO problems, including images and
    their corresponding ground-truth DSL rules.
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg['data']['synthetic_data_config']
        self.image_size = cfg['data']['image_size'] # [height, width]
        self.canvas_width, self.canvas_height = self.image_size[1], self.image_size[0]
        self.min_obj_size, self.max_obj_size = self.cfg['object_size_range']
        self.padding = self.cfg['padding']
        self.font = None
        if self.cfg['font_path'] and os.path.exists(self.cfg['font_path']):
            try:
                self.font = ImageFont.truetype(self.cfg['font_path'], size=int(self.max_obj_size * 0.7))
            except Exception as e:
                logger.warning(f"Could not load font from {self.cfg['font_path']}: {e}. Text objects will not be generated.")

        self.all_shapes = list(ATTRIBUTE_SHAPE_MAP.keys())
        self.all_colors = list(ATTRIBUTE_COLOR_MAP.keys())
        self.all_fills = list(ATTRIBUTE_FILL_MAP.keys())
        self.all_sizes = list(ATTRIBUTE_SIZE_MAP.keys())
        self.all_orientations = list(ATTRIBUTE_ORIENTATION_MAP.keys())
        self.all_textures = list(ATTRIBUTE_TEXTURE_MAP.keys())
        self.all_relations = [k for k in RELATION_MAP.keys() if k != 'unrelated'] # Exclude 'unrelated'

        logger.info(f"LogoGenerator initialized with canvas size {self.image_size}.")

    def _get_random_color(self) -> Tuple[int, int, int]:
        """Returns a random RGB color."""
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def _get_random_position(self, obj_size: int) -> Tuple[int, int]:
        """Returns a random (x, y) position for an object of given size."""
        x = random.randint(self.padding + obj_size // 2, self.canvas_width - self.padding - obj_size // 2)
        y = random.randint(self.padding + obj_size // 2, self.canvas_height - self.padding - obj_size // 2)
        return x, y

    def _draw_single_object(self, draw: ImageDraw.ImageDraw, obj_info: Dict[str, Any]):
        """Draws a single object based on its attributes."""
        shape_type = obj_info['shape']
        obj_size = obj_info['size']
        pos_x, pos_y = obj_info['position']
        fill_type = obj_info['fill']
        color_rgb = obj_info['color_rgb']
        
        outline_color = (0, 0, 0) # Black outline
        fill_color = color_rgb if fill_type == 'solid' else None
        
        # Draw shape
        if shape_type == 'circle':
            bbox = [pos_x - obj_size // 2, pos_y - obj_size // 2, pos_x + obj_size // 2, pos_y + obj_size // 2]
            draw.ellipse(bbox, outline=outline_color, fill=fill_color)
        elif shape_type == 'square':
            bbox = [pos_x - obj_size // 2, pos_y - obj_size // 2, pos_x + obj_size // 2, pos_y + obj_size // 2]
            draw.rectangle(bbox, outline=outline_color, fill=fill_color)
        elif shape_type == 'triangle':
            # Equilateral triangle pointing up
            h = obj_size * (3**0.5 / 2)
            pts = [(pos_x, pos_y - h / 2),
                   (pos_x - obj_size / 2, pos_y + h / 2),
                   (pos_x + obj_size / 2, pos_y + h / 2)]
            draw.polygon(pts, outline=outline_color, fill=fill_color)
        elif shape_type == 'text_character' and self.font:
            char = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            text_width, text_height = draw.textsize(char, font=self.font)
            text_x = pos_x - text_width // 2
            text_y = pos_y - text_height // 2
            draw.text((text_x, text_y), char, font=self.font, fill=color_rgb)
        # Add more shapes as needed (pentagon, hexagon, star, etc.)
        else:
            logger.warning(f"Unsupported shape type for drawing: {shape_type}")
            # Fallback to square
            bbox = [pos_x - obj_size // 2, pos_y - obj_size // 2, pos_x + obj_size // 2, pos_y + obj_size // 2]
            draw.rectangle(bbox, outline=outline_color, fill=fill_color)

    def _generate_scene_graph(self, objects_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a simplified scene graph from object data.
        This is a placeholder; a real scene graph builder would be more complex.
        """
        scene_graph = {'objects': [], 'relations': []}
        for i, obj in enumerate(objects_data):
            # Basic object attributes
            scene_graph['objects'].append({
                'id': f"obj_{i}",
                'attributes': {
                    'shape': obj['shape'],
                    'color': obj['color'],
                    'size': obj['size'],
                    'fill': obj['fill'],
                    'position_h': 'left' if obj['position'][0] < self.canvas_width/3 else ('right' if obj['position'][0] > 2*self.canvas_width/3 else 'center_h'),
                    'position_v': 'top' if obj['position'][1] < self.canvas_height/3 else ('bottom' if obj['position'][1] > 2*self.canvas_height/3 else 'center_v'),
                }
            })
        
        # Add dummy relations for now
        if len(objects_data) >= 2:
            scene_graph['relations'].append({
                'type': 'left_of',
                'source': 'obj_0',
                'target': 'obj_1'
            })
        return scene_graph

    def _generate_image_and_sg(self, num_objects: int, target_feature: Optional[str] = None, target_value: Optional[str] = None, match: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generates a single image with objects and its corresponding scene graph.
        If target_feature/value are provided and match is True, ensures the feature is present.
        """
        img = Image.new('RGB', (self.canvas_width, self.canvas_height), (255, 255, 255)) # White background
        draw = ImageDraw.Draw(img)
        
        objects_data = []
        for i in range(num_objects):
            obj_size = random.randint(self.min_obj_size, self.max_obj_size)
            pos = self._get_random_position(obj_size)
            
            shape = random.choice(self.all_shapes)
            color = random.choice(self.all_colors)
            fill = random.choice(self.all_fills)
            size = random.choice(self.all_sizes) # This is a symbolic size, not pixel size
            orientation = random.choice(self.all_orientations)
            texture = random.choice(self.all_textures)

            # Convert symbolic color to RGB
            color_rgb_map = {
                'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0),
                'yellow': (255, 255, 0), 'black': (0, 0, 0), 'white': (255, 255, 255)
            }
            color_rgb = color_rgb_map.get(color, (128, 128, 128)) # Default to gray

            # Enforce target feature/value if 'match' is True
            if match and target_feature and target_value:
                if target_feature == 'shape': shape = target_value
                elif target_feature == 'color': color = target_value; color_rgb = color_rgb_map.get(color, (128, 128, 128))
                elif target_feature == 'fill': fill = target_value
                elif target_feature == 'size': size = target_value # This is symbolic, actual pixel size is random
                elif target_feature == 'orientation': orientation = target_value
                elif target_feature == 'texture': texture = target_value
                # Add logic for relations if needed
            
            objects_data.append({
                'shape': shape, 'color': color, 'fill': fill, 'size': size,
                'orientation': orientation, 'texture': texture,
                'position': pos, 'size_pixels': obj_size, 'color_rgb': color_rgb
            })
            self._draw_single_object(draw, objects_data[-1])

        # Apply augmentations after drawing
        augmented_img_np = augment_image(np.array(img)) # augment_image expects numpy array
        
        # Generate scene graph from the generated objects
        scene_graph = self._generate_scene_graph(objects_data)
        
        return augmented_img_np, scene_graph

    def make_problem(self, problem_id: int) -> Tuple[Any, ...]:
        """
        Generates a complete Bongard problem (positive and negative examples)
        and its ground-truth DSL rule.
        """
        # Randomly select a rule from ALL_BONGARD_RULES
        if not ALL_BONGARD_RULES:
            logger.warning("No Bongard rules defined in src.bongard_rules. Using a dummy rule.")
            # Fallback to a simple dummy rule
            rule_obj = type('Rule', (object,), {
                'description': "SHAPE(TRIANGLE)",
                'positive_features': {'shape': 'triangle'},
                'negative_features': {'shape': 'square'}
            })()
        else:
            rule_obj = random.choice(ALL_BONGARD_RULES)

        gt_rule = rule_obj.description
        target_feature = None
        target_value = None
        
        # Extract a simple target feature/value from the rule for generation
        # This is a simplification; real rule parsing would be more complex.
        if "SHAPE(" in gt_rule:
            target_feature = 'shape'
            target_value = gt_rule.split('SHAPE(')[1].split(')')[0].lower()
        elif "COLOR(" in gt_rule:
            target_feature = 'color'
            target_value = gt_rule.split('COLOR(')[1].split(')')[0].lower()
        # Add more parsing for other features/relations if your rules are complex

        positive_imgs_np = []
        positive_sgs = []
        for _ in range(6): # 6 positive examples
            num_objects = random.randint(self.cfg['min_objects_per_image'], self.cfg['max_objects_per_image'])
            img_np, sg = self._generate_image_and_sg(num_objects, target_feature, target_value, match=True)
            positive_imgs_np.append(img_np)
            positive_sgs.append(sg)

        negative_imgs_np = []
        negative_sgs = []
        for _ in range(6): # 6 negative examples
            num_objects = random.randint(self.cfg['min_objects_per_image'], self.cfg['max_objects_per_image'])
            img_np, sg = self._generate_image_and_sg(num_objects, target_feature, target_value, match=False) # Do not match rule
            negative_imgs_np.append(img_np)
            negative_sgs.append(sg)

        # Combine all images for the problem
        query_img1_np = positive_imgs_np[0] # First positive example as query_img1
        query_img2_np = negative_imgs_np[0] # First negative example as query_img2
        label = 1 # Always predict positive for query_img1, negative for query_img2 (or vice versa)
                  # In Bongard problems, the label is for the *problem*, not individual images.
                  # Here, 'label' refers to the ground truth label of the problem itself (e.g., 1 for rule holds, 0 for not)
                  # For synthetic data, we can define the problem as 'rule holds for positive set'.
                  # Let's simplify: `label` here is the ground truth for `query_img1` vs `query_img2` if they were a pair.
                  # For a Bongard problem, the label is usually implicit in the problem structure.
                  # If this `label` is for the overall problem, it's always 1 (rule holds for the positive set).
                  # For simplicity, let's assume `label` refers to the class of the problem (e.g., 1 for "rule is X", 0 for "rule is not X").
                  # For now, we'll use a dummy label.
        
        # In a real Bongard problem, the label is whether the *rule* applies to the test image.
        # For synthetic generation, we know the rule, so the "problem label" is 1 (it's a valid problem).
        # The `label` field in `data.py`'s dataset refers to the class of the problem, typically 0 or 1.
        # Let's set it to 1 for a valid synthetic problem.
        problem_label = 1 

        gt_json_view1 = json.dumps(positive_sgs[0]).encode('utf-8')
        gt_json_view2 = json.dumps(negative_sgs[0]).encode('utf-8')
        
        difficulty = random.random() # Dummy difficulty
        affine1 = np.eye(3).tolist() # Dummy affine
        affine2 = np.eye(3).tolist() # Dummy affine
        original_index = problem_id # Use problem_id as original_index
        
        # Support images: remaining positive and negative examples
        padded_support_imgs_np = positive_imgs_np[1:] + negative_imgs_np[1:]
        padded_support_labels = [1] * (len(positive_imgs_np) - 1) + [0] * (len(negative_imgs_np) - 1)

        # Pad support images/labels to max_support_images_per_problem
        max_support = self.cfg['max_support_images_per_problem']
        while len(padded_support_imgs_np) < max_support:
            padded_support_imgs_np.append(np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8))
            padded_support_labels.append(-1)  # Use -1 for padding labels
        
        padded_support_sgs_bytes = [json.dumps(sg).encode('utf-8') for sg in positive_sgs[1:] + negative_sgs[1:]]
        while len(padded_support_sgs_bytes) < max_support:
            padded_support_sgs_bytes.append(json.dumps({'objects': [], 'relations': []}).encode('utf-8'))

        num_support_per_problem_tensor = torch.tensor(len(positive_imgs_np) - 1 + len(negative_imgs_np) - 1, dtype=torch.long)
        tree_indices_tensor = torch.tensor(original_index, dtype=torch.long)
        is_weights_tensor = torch.tensor(1.0, dtype=torch.float)

        # New: Dummy detected bboxes and masks for query images (empty for now)
        dummy_query_bboxes_view1 = [[]]
        dummy_query_masks_view1 = [[]]
        dummy_query_bboxes_view2 = [[]]
        dummy_query_masks_view2 = [[]]

        # New: Dummy detected bboxes and masks for support images (empty for now)
        dummy_support_bboxes_flat = [[] for _ in range(max_support)]
        dummy_support_masks_flat = [[] for _ in range(max_support)]

        return (query_img1_np, query_img2_np, problem_label,
                gt_json_view1, gt_json_view2, difficulty, affine1, affine2, original_index,
                padded_support_imgs_np, padded_support_labels, padded_support_sgs_bytes,
                num_support_per_problem_tensor, tree_indices_tensor, is_weights_tensor,
                dummy_query_bboxes_view1, dummy_query_masks_view1,
                dummy_query_bboxes_view2, dummy_query_masks_view2,
                dummy_support_bboxes_flat, dummy_support_masks_flat)


class LogoDataset(Dataset):
    """
    A PyTorch Dataset for generated Bongard-LOGO problems.
    """
    def __init__(self, cfg: Dict[str, Any], num_samples: int, transform: Optional[T.Compose] = None):
        self.cfg = cfg
        self.num_samples = num_samples
        self.transform = transform
        self.generator = LogoGenerator(cfg)
        logger.info(f"LogoDataset initialized to generate {num_samples} problems.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: Union[int, Tuple[int, float]]) -> Tuple[Any, ...]:
        if isinstance(idx, tuple):
            original_idx, is_weight = idx
        else:
            original_idx = idx
            is_weight = 1.0
        
        problem_data = self.generator.make_problem(original_idx)
        
        # Convert problem_data (tuple) to list to allow modification
        problem_list = list(problem_data)
        
        # Apply transform to images if provided
        if self.transform is not None:
            # query_img1_np (index 0) and query_img2_np (index 1)
            problem_list[0] = self.transform(Image.fromarray(problem_list[0]))
            problem_list[1] = self.transform(Image.fromarray(problem_list[1]))
            
            # padded_support_imgs_np (index 9) - list of numpy arrays
            transformed_support_imgs = []
            for img_np in problem_list[9]:
                transformed_support_imgs.append(self.transform(Image.fromarray(img_np)))
            problem_list[9] = torch.stack(transformed_support_imgs) # Stack into a single tensor
        
        # Update the is_weights with the sampler's IS weight
        problem_list[14] = torch.tensor(is_weight, dtype=torch.float) 
        problem_list[8] = original_idx  # Ensure original_index is correct (used by collate_fn for tree_indices)
        problem_list[13] = torch.tensor(original_idx, dtype=torch.long)  # tree_indices maps to original_index in replay buffer
        
        return tuple(problem_list)

