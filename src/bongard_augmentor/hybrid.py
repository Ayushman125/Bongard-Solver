# ------------------------------------------------------------------
# Action-Based Augmentation Pipeline
# Processes Action Programs instead of Real Images  
# ------------------------------------------------------------------

import os
import sys
import numpy as np
import cv2
import time
import logging
import json
import tempfile
import turtle
import re
import math

import random
import itertools
import traceback
import hashlib
import shutil
import subprocess
import pickle
import matplotlib.pyplot as plt
import pickle
import hashlib
import math
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage import morphology as ski_morphology
from skimage.metrics import structural_similarity as ssim

# Add src to path for imports


# Ensure Bongard-LOGO/Bongard-LOGO is in sys.path for all bongard imports
BONGARD_LOGO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Bongard-LOGO', 'Bongard-LOGO'))
if BONGARD_LOGO_PATH not in sys.path:
    sys.path.insert(0, BONGARD_LOGO_PATH)


# --- Import Bongard-LOGO classes and logic directly ---

from bongard.bongard import (LineAction, ArcAction, OneStrokeShape,
        BongardImage, BongardProblem, BasicAction)

from bongard.bongard_painter import BongardImagePainter, BongardProblemPainter

# Try to import samplers - handle import errors gracefully
try:
    from bongard.sampler.basic_sampler import BasicSampler
    BASIC_SAMPLER_AVAILABLE = True
except ImportError:
    try:
        from bongard.basic_sampler import BasicSampler
        BASIC_SAMPLER_AVAILABLE = True
    except ImportError:
        BASIC_SAMPLER_AVAILABLE = False
        logging.warning("BasicSampler not available - using fallback logic")

try:
    from bongard.sampler.abstract_sampler import AbstractSampler
    ABSTRACT_SAMPLER_AVAILABLE = True
except ImportError:
    try:
        from bongard.abstract_sampler import AbstractSampler
        ABSTRACT_SAMPLER_AVAILABLE = True
    except ImportError:
        ABSTRACT_SAMPLER_AVAILABLE = False
        logging.warning("AbstractSampler not available - using fallback logic")

try:
    from bongard.sampler.freeform_sampler import FreeformSampler
    FREEFORM_SAMPLER_AVAILABLE = True
except ImportError:
    try:
        from bongard.freeform_sampler import FreeformSampler
        FREEFORM_SAMPLER_AVAILABLE = True
    except ImportError:
        FREEFORM_SAMPLER_AVAILABLE = False
        logging.warning("FreeformSampler not available - using fallback logic")

# Import data loading and parsing components
from src.data_pipeline.data_loader import load_action_programs
from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser, NVLABS_AVAILABLE

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Fast Spatial Grid for O(1) Collision Detection ---
from collections import defaultdict

class FastSpatialGrid:
    def __init__(self, canvas_size=512, cell_size=64):
        self.canvas_size = canvas_size
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    def _cells_covered(self, x, y, width, height):
        x0 = int(x // self.cell_size)
        y0 = int(y // self.cell_size)
        x1 = int((x + width) // self.cell_size)
        y1 = int((y + height) // self.cell_size)
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                yield (cx, cy)

    def check_collision(self, x, y, width, height):
        for cell in self._cells_covered(x, y, width, height):
            for bx, by, bw, bh in self.grid[cell]:
                if not (x + width <= bx or x >= bx + bw or y + height <= by or y >= by + bh):
                    return True
        return False

    def add_box(self, x, y, width, height):
        for cell in self._cells_covered(x, y, width, height):
            self.grid[cell].append((x, y, width, height))

def canvas_to_logo(canvas_x, canvas_y, canvas_size=512):
    # Map (0,512) to (-360,360)
    logo_x = (canvas_x * 720.0 / canvas_size) - 360.0
    logo_y = -((canvas_y * 720.0 / canvas_size) - 360.0)
    return logo_x, logo_y

def logo_to_canvas(logo_x, logo_y, canvas_size=512):
    # Map (-360,360) to (0,512)
    canvas_x = int((logo_x + 360.0) * canvas_size / 720.0)
    canvas_y = int((360.0 - logo_y) * canvas_size / 720.0)
    return canvas_x, canvas_y


class ActionMaskGenerator:
    def __init__(self, config=None):
        self.config = config
        # Optionally set attributes from config
        if config and isinstance(config, dict):
            for k, v in config.items():
                setattr(self, k, v)
        # Set defaults if not in config
        if not hasattr(self, 'canvas_size'):
            self.canvas_size = 512
        if not hasattr(self, 'coordinate_range'):
            self.coordinate_range = (-360, 360)
    def run_pipeline(self, action_programs_dir=None, output_path=None, problems_list=None, n_select=50):
        """
        Main entry point for action-based augmentation pipeline.
        Loads action programs, generates masks using Bongard-LOGO repo, and saves results.
        Uses config values if present, otherwise CLI arguments.
        """
        import pickle
        import cv2
        config = getattr(self, 'config', None)
        if config and isinstance(config, dict):
            action_programs_dir = action_programs_dir or config.get('action_programs_dir') or config.get('data', {}).get('action_programs_dir')
            output_path = output_path or config.get('output_path') or config.get('data', {}).get('output_path')
            problems_list = problems_list or config.get('problems_list') or config.get('data', {}).get('problems_list')
            n_select = n_select if n_select != 50 else config.get('n_select', 50)
        action_programs_dir = action_programs_dir or getattr(self, 'action_programs_dir', None)
        output_path = output_path or getattr(self, 'output_path', None)
        problems_list = problems_list or getattr(self, 'problems_list', None)
        n_select = n_select or getattr(self, 'n_select', 50)
        # Ensure logging level is DEBUG for full diagnostics
        logging.getLogger().setLevel(logging.DEBUG)

        if not action_programs_dir or not output_path:
            logging.critical("action_programs_dir and output_path must be specified.")
            raise ValueError("action_programs_dir and output_path must be specified.")

        # Load action programs
        from src.data_pipeline.data_loader import load_action_programs
        action_data = load_action_programs(action_programs_dir)
        # Filter action_data immediately after loading
        if problems_list and os.path.exists(problems_list):
            with open(problems_list, 'r') as f:
                filtered_problems = set(line.strip() for line in f if line.strip())
            action_data = {k: v for k, v in action_data.items() if k in filtered_problems}
        # Limit to n_select problems if needed
        if n_select and len(action_data) > n_select:
            action_data = dict(list(action_data.items())[:n_select])

        # Log the number of problems after filtering and limiting
        logging.info(f"Filtered action programs to {len(action_data)} problems for processing.")

        result = []
        for problem_id, problem_data in action_data.items():
            # Handle Bongard-LOGO format: [positive_examples, negative_examples]
            if isinstance(problem_data, list) and len(problem_data) == 2:
                for i, action_commands in enumerate(problem_data[0]):  # positive examples
                    entry = {
                        'problem_id': problem_id,
                        'image_id': f"{problem_id}_pos_{i}",
                        'action_commands': action_commands,
                    }
                    result.append(entry)
                for i, action_commands in enumerate(problem_data[1]):  # negative examples
                    entry = {
                        'problem_id': problem_id,
                        'image_id': f"{problem_id}_neg_{i}",
                        'action_commands': action_commands,
                    }
                    result.append(entry)
            else:
                # Fallback for dict/list formats
                if isinstance(problem_data, dict):
                    images = problem_data.get('images', [])
                elif isinstance(problem_data, list):
                    images = problem_data
                else:
                    images = []
                for img in images:
                    if isinstance(img, dict):
                        entry = {
                            'problem_id': problem_id,
                            'image_id': img.get('image_id'),
                            'action_commands': img.get('action_commands'),
                        }
                    elif isinstance(img, list) and len(img) == 2:
                        entry = {
                            'problem_id': problem_id,
                            'image_id': img[0],
                            'action_commands': img[1],
                        }
                    else:
                        continue
                    result.append(entry)

        logging.info(f"Processing {len(result)} images. Output masks will be saved to: {output_path}")
        output_records = []
        inspection_dir = os.path.join(os.path.dirname(output_path), "mask_inspection")
        os.makedirs(inspection_dir, exist_ok=True)
        from tqdm import tqdm
        for entry in tqdm(result, desc="Generating masks from action programs", mininterval=0.5):
            try:
                # Directly generate mask using canonical Bongard-LOGO workflow
                mask = self.generate_mask(entry['action_commands'], entry['problem_id'])
                record = {
                    'problem_id': entry['problem_id'],
                    'image_id': entry['image_id'],
                    'mask': mask
                }
                output_records.append(record)

                # Save mask as PNG for inspection
                mask_img_path = os.path.join(inspection_dir, f"{entry['image_id']}_mask.png")
                if isinstance(mask, np.ndarray):
                    cv2.imwrite(mask_img_path, (mask * 255).astype('uint8'))
                elif isinstance(mask, (list, tuple)):
                    mask_arr = np.array(mask)
                    cv2.imwrite(mask_img_path, (mask_arr * 255).astype('uint8'))
                elif isinstance(mask, (int, float)):
                    mask_arr = np.array([[mask]])
                    cv2.imwrite(mask_img_path, (mask_arr * 255).astype('uint8'))
                else:
                    logging.error(f"Mask for {entry['image_id']} is not a valid image array: type={type(mask)}")
            except Exception as e:
                logging.error(f"Failed to process entry {entry.get('image_id', 'unknown')}: {e}")
                continue
        # Save results as pickle file
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(output_records, f)
            logging.info("Augmentation pipeline completed successfully.")
        except Exception as e:
            logging.error(f"Failed to save output file {output_path}: {e}")
            raise

    def _robust_fallback_mask_generation(self, action_commands: list, problem_id: str, category: str):
        """
        Unified fallback mask generation with loosened constraints and better diagnostics.
        """
        logging.info(f"Using robust fallback for {category} problem: {problem_id}")
        bongard_image = BongardImage.import_from_action_string_list(action_commands)
        problem_painter = BongardProblemPainter()
        num_shapes = len(bongard_image.one_stroke_shapes)
        scaling_factors_range = getattr(problem_painter, 'scaling_factors_range', (150, 220))
        max_radius = 198
        
        # Loosened constraints for better success rate
        safe_boundary = 350  # Increased from 270
        min_sep_factor = 0.3  # Reduced from 0.7
        max_attempts = 100   # Reduced from 1000 for efficiency
        
        valid = False
        last_failure_reason = "Unknown"
        
        for attempt in range(max_attempts):
            try:
                # Sample positions and orientations
                start_coordinates, start_orientations = problem_painter.sample_start_coordinates_and_orientation(
                    num_shapes=num_shapes,
                    max_radius=max_radius
                )
                scaling_factors = [random.uniform(scaling_factors_range[0], scaling_factors_range[1]) for _ in range(num_shapes)]
                
                # Assign sampled values using Bongard-LOGO's method
                for i, shape in enumerate(bongard_image.one_stroke_shapes):
                    shape.start_coordinates = start_coordinates[i]
                    shape.start_orientation = start_orientations[i]
                    shape.set_consistent_scaling_factors(scaling_factors[i])

                # Manual assignment for missing attributes
                missing_manual = []
                for idx, shape in enumerate(bongard_image.one_stroke_shapes):
                    if not (hasattr(shape, 'start_coordinates') and shape.start_coordinates is not None):
                        shape.start_coordinates = (random.uniform(-max_radius, max_radius), random.uniform(-max_radius, max_radius))
                        missing_manual.append((idx, 'start_coordinates'))
                    if not (hasattr(shape, 'start_orientation') and shape.start_orientation is not None):
                        shape.start_orientation = random.uniform(0, 360)
                        missing_manual.append((idx, 'start_orientation'))
                    if not (hasattr(shape, 'scaling_factors') and shape.scaling_factors is not None):
                        shape.set_consistent_scaling_factors(random.uniform(scaling_factors_range[0], scaling_factors_range[1]))
                        missing_manual.append((idx, 'scaling_factor'))

                if missing_manual and attempt % 20 == 0:  # Log every 20th attempt
                    logging.debug(f"{category} manual assignment for {problem_id} on attempt {attempt+1}: {missing_manual}")
                
                # Loosened validity checks
                coords = [shape.start_coordinates for shape in bongard_image.one_stroke_shapes]
                
                # Check bounds - now more lenient
                out_of_bounds = [i for i, (x, y) in enumerate(coords) if abs(x) > safe_boundary or abs(y) > safe_boundary]
                if out_of_bounds:
                    last_failure_reason = f"Out of bounds: shapes {out_of_bounds}"
                    continue
                
                # Check minimum separation - now more lenient
                min_sep = max_radius * min_sep_factor
                violations = []
                valid_sep = True
                for i in range(num_shapes):
                    for j in range(i+1, num_shapes):
                        dist = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
                        if dist < min_sep:
                            violations.append((i, j, dist))
                            valid_sep = False
                
                if not valid_sep:
                    last_failure_reason = f"Separation violations: {violations[:3]}"  # Log first 3
                    continue
                
                # Try to paint
                temp_dir = tempfile.mkdtemp()
                ps_dir = os.path.join(temp_dir, "ps")
                png_dir = os.path.join(temp_dir, "png")
                os.makedirs(ps_dir, exist_ok=True)
                os.makedirs(png_dir, exist_ok=True)
                
                try:
                    problem_painter.save_bongard_images([bongard_image], "1", ps_dir, png_dir, auto_position=False)
                    valid = True
                    break
                except Exception as painter_exc:
                    import shutil
                    shutil.rmtree(temp_dir)
                    last_failure_reason = f"Painter error: {str(painter_exc)[:100]}"
                    continue
                    
            except Exception as attempt_exc:
                last_failure_reason = f"Attempt error: {str(attempt_exc)[:100]}"
                continue
        
        if not valid:
            logging.warning(f"{category} Monte Carlo positioning failed for {problem_id} after {max_attempts} attempts. Last failure: {last_failure_reason}")
            return self._generate_synthetic_mask(action_commands, problem_id)
        
        # Convert PS to PNG
        ps_file = os.path.join(ps_dir, "1", "0.ps")
        png_file = os.path.join(png_dir, "1", "0.png")
        
        if os.path.exists(ps_file):
            import subprocess
            gs_cmd = [
                "gswin64c" if os.name == "nt" else "gs",
                "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m",
                f"-sOutputFile={png_file}", ps_file
            ]
            try:
                subprocess.run(gs_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as gs_exc:
                logging.error(f"Ghostscript conversion failed: {gs_exc}")
                import shutil
                shutil.rmtree(temp_dir)
                return self._generate_synthetic_mask(action_commands, problem_id)
            
            if os.path.exists(png_file):
                from PIL import Image
                img = Image.open(png_file).convert('L')
                mask = (np.array(img) < 128).astype(np.uint8)
                import shutil
                shutil.rmtree(temp_dir)
                logging.info(f"Successfully generated {category} mask for {problem_id}")
                return mask
            else:
                import shutil
                shutil.rmtree(temp_dir)
                logging.error(f"Ghostscript did not produce PNG file for {problem_id}")
                return self._generate_synthetic_mask(action_commands, problem_id)
        else:
            import shutil
            shutil.rmtree(temp_dir)
            logging.error(f"PostScript file not found for {problem_id}")
            return self._generate_synthetic_mask(action_commands, problem_id)

    def generate_mask(self, action_commands: list, problem_id: str, output_size: tuple = None):
        """
        Generate a mask using a two-pass strategy for guaranteed placement.
        1. First pass calculates exact bounding boxes.
        2. Second pass uses this data for collision-aware placement.
        """
        import math

        # Define canvas_center and coord_range_max at the top so they are available everywhere
        canvas_size = self.canvas_size if hasattr(self, 'canvas_size') else 512
        canvas_center = canvas_size / 2
        coord_range_max = self.coordinate_range[1]

        shape_actions_filepath = os.path.join(BONGARD_LOGO_PATH, "data", "human_designed_shapes.tsv")
        shape_attributes_filepath = os.path.join(BONGARD_LOGO_PATH, "data", "human_designed_shapes_attributes.tsv")
        sampler = None
        sampler_type = None

        if problem_id.startswith('bd_') and BASIC_SAMPLER_AVAILABLE:
            sampler = BasicSampler(shape_actions_filepath, shape_attributes_filepath)
            sampler_type = 'BasicSampler'
        elif problem_id.startswith('hd_') and ABSTRACT_SAMPLER_AVAILABLE:
            sampler = AbstractSampler(shape_actions_filepath, shape_attributes_filepath)
            sampler_type = 'AbstractSampler'
        elif problem_id.startswith('ff_') and FREEFORM_SAMPLER_AVAILABLE:
            sampler = FreeformSampler(shape_actions_filepath, shape_attributes_filepath)
            sampler_type = 'FreeformSampler'
        else:
            raise RuntimeError(f"No sampler available for problem_id={problem_id}")

        bongard_image = BongardImage.import_from_action_string_list(action_commands)
        num_shapes = len(bongard_image.one_stroke_shapes)
        logging.info(f"Using {sampler_type} for mask generation and validation for {problem_id}")

        problem_painter = BongardProblemPainter()
        scaling_factors_range = getattr(problem_painter, 'scaling_factors_range', (150, 220))

        # --- Two-Pass Placement Strategy ---

        # 1. Pre-sample scaling factors
        if sampler and hasattr(sampler, 'sample_scaling_factors'):
            scaling_factors = sampler.sample_scaling_factors(num_shapes=num_shapes)
        else:
            scaling_factors = [random.uniform(scaling_factors_range[0], scaling_factors_range[1]) for _ in range(num_shapes)]

        # 2. First Pass: Calculate precise bounding box for each shape
        shape_bounds = []
        shape_offsets = []
        shape_canvas_sizes = []
        anchor_offsets = []
        for i, shape in enumerate(bongard_image.one_stroke_shapes):
            bounds = self._get_shape_bounds(shape, scaling_factors[i], problem_painter)
            if bounds is None:
                logging.warning(f"Could not calculate bounds for a shape in {problem_id}. Falling back.")
                return self._generate_synthetic_mask(action_commands, problem_id)
            width, height, cmin, rmin, large_canvas_size, anchor_offset_x, anchor_offset_y = bounds
            shape_bounds.append((width, height))
            shape_offsets.append((cmin, rmin))
            shape_canvas_sizes.append(large_canvas_size)
            anchor_offsets.append((anchor_offset_x, anchor_offset_y))

        # 3. Second Pass: Iterative placement with exact bounding boxes
        max_attempts = 500
        valid_placement_found = False
        final_placements = []
        margin = 15 # pixels

        # --- Improved adaptive placement strategy ---
        for attempt in range(max_attempts):
            placed_boxes = []
            is_valid_config = True
            box_positions = [None] * num_shapes
            
            # Calculate total area and adaptive spacing
            total_shape_area = sum(w * h for w, h in shape_bounds)
            canvas_area = canvas_size * canvas_size
            density_ratio = total_shape_area / canvas_area
            
            # Adaptive minimum separation based on object sizes and canvas density
            max_dimension = max(max(w, h) for w, h in shape_bounds)
            min_separation = max(20, int(max_dimension * 0.3), int(canvas_size * 0.1))
            
            if num_shapes == 1:
                # Single object: place in center with small random offset
                width, height = shape_bounds[0]
                center_x = canvas_size // 2 - width // 2
                center_y = canvas_size // 2 - height // 2
                offset_x = random.randint(-50, 50)
                offset_y = random.randint(-50, 50)
                x = max(margin, min(canvas_size - width - margin, center_x + offset_x))
                y = max(margin, min(canvas_size - height - margin, center_y + offset_y))
                box_positions[0] = {'x1': x, 'y1': y, 'x2': x + width, 'y2': y + height}
                placed_boxes = [box_positions[0]]
                
            elif num_shapes == 2:
                # Two objects: use adaptive diagonal or side-by-side placement
                width0, height0 = shape_bounds[0]
                width1, height1 = shape_bounds[1]
                
                # Determine if objects can fit side by side or need diagonal placement
                total_width = width0 + width1 + min_separation
                total_height = height0 + height1 + min_separation
                
                placement_strategies = []
                
                # Strategy 1: Side by side (horizontal)
                if total_width + 2 * margin <= canvas_size:
                    x0 = margin
                    x1 = x0 + width0 + min_separation
                    # Ensure both objects have valid Y positions within bounds
                    max_y0 = canvas_size - height0 - margin
                    max_y1 = canvas_size - height1 - margin
                    if max_y0 > margin and max_y1 > margin:
                        y0 = random.randint(margin, max_y0)
                        y1 = random.randint(margin, max_y1)
                        placement_strategies.append([(x0, y0), (x1, y1)])
                
                # Strategy 2: Vertical stack
                if total_height + 2 * margin <= canvas_size:
                    y0 = margin
                    y1 = y0 + height0 + min_separation
                    # Ensure both objects have valid X positions within bounds
                    max_x0 = canvas_size - width0 - margin
                    max_x1 = canvas_size - width1 - margin
                    if max_x0 > margin and max_x1 > margin:
                        x0 = random.randint(margin, max_x0)
                        x1 = random.randint(margin, max_x1)
                        placement_strategies.append([(x0, y0), (x1, y1)])
                
                # Strategy 3: Diagonal placement (always possible with smaller objects)
                for _ in range(5):  # Try multiple diagonal configurations
                    x0 = random.randint(margin, canvas_size - width0 - margin)
                    y0 = random.randint(margin, canvas_size - height0 - margin)
                    
                    # Place second object diagonally opposite with minimum separation
                    attempts_for_second = 0
                    while attempts_for_second < 50:
                        x1 = random.randint(margin, canvas_size - width1 - margin)
                        y1 = random.randint(margin, canvas_size - height1 - margin)
                        
                        # Check minimum separation (center-to-center distance)
                        center_dist = math.sqrt((x0 + width0/2 - x1 - width1/2)**2 + 
                                              (y0 + height0/2 - y1 - height1/2)**2)
                        if center_dist >= min_separation:
                            placement_strategies.append([(x0, y0), (x1, y1)])
                            break
                        attempts_for_second += 1
                
                # Try placement strategies
                found = False
                random.shuffle(placement_strategies)
                for strategy in placement_strategies:
                    (x0, y0), (x1, y1) = strategy
                    box0 = {'x1': x0, 'y1': y0, 'x2': x0 + width0, 'y2': y0 + height0}
                    box1 = {'x1': x1, 'y1': y1, 'x2': x1 + width1, 'y2': y1 + height1}
                    
                    # Check for overlap
                    if (box0['x2'] < box1['x1'] or box0['x1'] > box1['x2'] or 
                        box0['y2'] < box1['y1'] or box0['y1'] > box1['y2']):
                        box_positions[0] = box0
                        box_positions[1] = box1
                        placed_boxes = [box0, box1]
                        found = True
                        break
                
                if not found:
                    is_valid_config = False
            elif num_shapes == 3:
                # Three objects: use grid-based placement with better spacing
                widths = [shape_bounds[i][0] for i in range(3)]
                heights = [shape_bounds[i][1] for i in range(3)]
                
                # Calculate available space for each object
                available_width = canvas_size - 2 * margin
                available_height = canvas_size - 2 * margin
                
                # Try different arrangements: L-shape, triangle, line
                arrangements = []
                
                # Arrangement 1: L-shape (two on top, one on bottom)
                if sum(widths[:2]) + min_separation + 2 * margin <= canvas_size:
                    x0 = margin
                    x1 = margin + widths[0] + min_separation
                    y0 = y1 = margin
                    x2 = margin
                    y2 = margin + max(heights[0], heights[1]) + min_separation
                    if (x1 + widths[1] <= canvas_size - margin and 
                        y2 + heights[2] <= canvas_size - margin):
                        arrangements.append([(x0, y0), (x1, y1), (x2, y2)])
                
                # Arrangement 2: Triangle formation
                center_x = canvas_size // 2
                center_y = canvas_size // 2
                radius = min(canvas_size // 4, 100)
                
                for angle_offset in [0, 60, 120]:  # Try different triangle orientations
                    positions = []
                    valid_triangle = True
                    for i in range(3):
                        angle = (i * 120 + angle_offset) * math.pi / 180
                        x = center_x + radius * math.cos(angle) - widths[i] // 2
                        y = center_y + radius * math.sin(angle) - heights[i] // 2
                        
                        # Check bounds
                        if (x < margin or x + widths[i] > canvas_size - margin or
                            y < margin or y + heights[i] > canvas_size - margin):
                            valid_triangle = False
                            break
                        positions.append((int(x), int(y)))
                    
                    if valid_triangle:
                        arrangements.append(positions)
                
                # Arrangement 3: Vertical line
                total_height = sum(heights) + 2 * min_separation
                if total_height + 2 * margin <= canvas_size:
                    x_base = canvas_size // 2 - max(widths) // 2
                    y_current = margin
                    positions = []
                    for i in range(3):
                        x = x_base + (max(widths) - widths[i]) // 2
                        positions.append((x, y_current))
                        y_current += heights[i] + min_separation
                    arrangements.append(positions)
                
                # Try arrangements
                found = False
                random.shuffle(arrangements)
                for arrangement in arrangements:
                    boxes = []
                    overlap_found = False
                    
                    for i, (x, y) in enumerate(arrangement):
                        box = {'x1': x, 'y1': y, 'x2': x + widths[i], 'y2': y + heights[i]}
                        
                        # Check overlap with previously placed boxes
                        for existing_box in boxes:
                            if not (box['x2'] < existing_box['x1'] or box['x1'] > existing_box['x2'] or 
                                   box['y2'] < existing_box['y1'] or box['y1'] > existing_box['y2']):
                                overlap_found = True
                                break
                        
                        if overlap_found:
                            break
                        boxes.append(box)
                    
                    if not overlap_found and len(boxes) == 3:
                        for i, box in enumerate(boxes):
                            box_positions[i] = box
                        placed_boxes = boxes
                        found = True
                        break
                
                if not found:
                    is_valid_config = False
            else:
                # Greedy adaptive for 4+ objects with improved bounds checking
                shape_indices = list(range(num_shapes))
                shape_areas = [w * h for (w, h) in shape_bounds]
                sorted_indices = sorted(shape_indices, key=lambda i: -shape_areas[i])
                
                for idx in sorted_indices:
                    width, height = shape_bounds[idx]
                    found_spot_for_shape = False
                    
                    # Calculate valid placement bounds
                    max_x = canvas_size - width - margin
                    max_y = canvas_size - height - margin
                    
                    if max_x <= margin or max_y <= margin:
                        # Object too large for canvas
                        is_valid_config = False
                        break
                    
                    for _ in range(200):
                        x = random.randint(margin, max_x)
                        y = random.randint(margin, max_y)
                        current_box = {'x1': x, 'y1': y, 'x2': x + width, 'y2': y + height}
                        
                        # Check overlap with all previously placed boxes
                        is_overlapping = False
                        for other_box in placed_boxes:
                            if other_box is None:
                                continue
                            # Check overlap using bounding box intersection
                            if not (current_box['x2'] <= other_box['x1'] or 
                                    current_box['x1'] >= other_box['x2'] or 
                                    current_box['y2'] <= other_box['y1'] or 
                                    current_box['y1'] >= other_box['y2']):
                                is_overlapping = True
                                break
                        
                        # Also check minimum separation distance
                        if not is_overlapping:
                            current_center_x = x + width / 2
                            current_center_y = y + height / 2
                            
                            for other_box in placed_boxes:
                                if other_box is None:
                                    continue
                                other_center_x = (other_box['x1'] + other_box['x2']) / 2
                                other_center_y = (other_box['y1'] + other_box['y2']) / 2
                                
                                center_dist = math.sqrt((current_center_x - other_center_x)**2 + 
                                                      (current_center_y - other_center_y)**2)
                                if center_dist < min_separation:
                                    is_overlapping = True
                                    break
                        
                        if not is_overlapping:
                            box_positions[idx] = current_box
                            placed_boxes.append(current_box)
                            found_spot_for_shape = True
                            break
                    
                    if not found_spot_for_shape:
                        is_valid_config = False
                        break
            # Post-render diagnostic check: ensure all shapes are in bounds and do not overlap in the mask
            if is_valid_config:
                # Prepare BongardImage for this placement with improved coordinate transformation
                start_coords = []
                for i in range(num_shapes):
                    box = box_positions[i]
                    width, height = shape_bounds[i]
                    anchor_offset_x, anchor_offset_y = anchor_offsets[i]
                    
                    # Calculate the anchor point in canvas coordinates
                    anchor_x = box['x1'] + anchor_offset_x
                    anchor_y = box['y1'] + anchor_offset_y
                    
                    # Transform to Bongard-LOGO coordinate system (-360 to 360)
                    # Add bounds checking to prevent coordinates from going too far out
                    logo_x = (anchor_x - canvas_center) * (coord_range_max / canvas_center)
                    logo_y = -(anchor_y - canvas_center) * (coord_range_max / canvas_center)
                    
                    # Clamp coordinates to safe bounds to prevent objects from going off-canvas
                    safe_bound = coord_range_max * 0.9  # Use 90% of max range for safety
                    logo_x = max(-safe_bound, min(safe_bound, logo_x))
                    logo_y = max(-safe_bound, min(safe_bound, logo_y))
                    
                    start_coords.append((logo_x, logo_y))
                    
                if sampler and hasattr(sampler, 'sample_orientations'):
                    start_orients = sampler.sample_orientations(num_shapes=num_shapes)
                else:
                    start_orients = [random.uniform(0, 360) for _ in range(num_shapes)]
                norm_orients = [(o % 360) for o in start_orients]
                
                # Apply coordinates and orientations to shapes
                for i, shape in enumerate(bongard_image.one_stroke_shapes):
                    shape.start_coordinates = start_coords[i]
                    shape.start_orientation = norm_orients[i]
                    shape.set_consistent_scaling_factors(scaling_factors[i])
                # Render to temp mask
                temp_dir = tempfile.mkdtemp()
                ps_dir = os.path.join(temp_dir, "ps")
                png_dir = os.path.join(temp_dir, "png")
                os.makedirs(ps_dir, exist_ok=True)
                os.makedirs(png_dir, exist_ok=True)
                problem_painter.save_bongard_images([bongard_image], "1", ps_dir, png_dir, auto_position=False)
                ps_file = os.path.join(ps_dir, "1", "0.ps")
                png_file = os.path.join(png_dir, "1", "0.png")
                mask_valid = False
                if os.path.exists(ps_file):
                    import subprocess
                    gs_cmd = [
                        "gswin64c" if os.name == "nt" else "gs",
                        "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m",
                        f"-sOutputFile={png_file}", ps_file
                    ]
                    try:
                        subprocess.run(gs_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    except Exception:
                        import shutil
                        shutil.rmtree(temp_dir)
                        continue
                    if os.path.exists(png_file):
                        from PIL import Image
                        img = Image.open(png_file).convert('L')
                        mask = (np.array(img) < 128).astype(np.uint8)
                        # Check all nonzero pixels are within bounds
                        if np.any(np.where(mask)[0] < 0) or np.any(np.where(mask)[0] >= canvas_size) or \
                           np.any(np.where(mask)[1] < 0) or np.any(np.where(mask)[1] >= canvas_size):
                            import shutil
                            shutil.rmtree(temp_dir)
                            continue
                        # Check for overlap: label connected components, should be num_shapes
                        from scipy.ndimage import label
                        labeled, n = label(mask)
                        if n != num_shapes:
                            import shutil
                            shutil.rmtree(temp_dir)
                            continue
                        mask_valid = True
                import shutil
                shutil.rmtree(temp_dir)
                if mask_valid:
                    final_placements = box_positions
                    valid_placement_found = True
                    logging.debug(f"Found valid placement for {problem_id} on attempt {attempt + 1} (diagnostic check passed)")
                    break
            # If failed after many attempts, try reducing scaling and retry
            if (not valid_placement_found) and attempt == max_attempts - 1:
                # Reduce scaling and retry
                logging.warning(f"Placement failed for {problem_id} after {max_attempts} attempts, reducing scaling and retrying.")
                scaling_factors = [max(60, s * 0.8) for s in scaling_factors]
                # Recompute bounds with new scaling
                shape_bounds = []
                anchor_offsets = []
                for i, shape in enumerate(bongard_image.one_stroke_shapes):
                    bounds = self._get_shape_bounds(shape, scaling_factors[i], problem_painter)
                    if bounds is None:
                        logging.warning(f"Could not calculate bounds for a shape in {problem_id} after scaling reduction. Falling back.")
                        return self._generate_synthetic_mask(action_commands, problem_id)
                    width, height, cmin, rmin, large_canvas_size, anchor_offset_x, anchor_offset_y = bounds
                    shape_bounds.append((width, height))
                    anchor_offsets.append((anchor_offset_x, anchor_offset_y))

        if not valid_placement_found:
            logging.warning(f"Could not find a valid placement for {problem_id} after {max_attempts} attempts. Falling back to synthetic mask.")
            return self._generate_synthetic_mask(action_commands, problem_id)

        # 4. Final Rendering with calculated placements
        start_coords = []
        canvas_center = canvas_size / 2
        coord_range_max = self.coordinate_range[1]
        for i, box in enumerate(final_placements):
            width, height = shape_bounds[i]
            anchor_offset_x, anchor_offset_y = anchor_offsets[i]
            anchor_x = box['x1'] + anchor_offset_x
            anchor_y = box['y1'] + anchor_offset_y
            logo_x = (anchor_x - canvas_center) * (coord_range_max / canvas_center)
            logo_y = -(anchor_y - canvas_center) * (coord_range_max / canvas_center)
            start_coords.append((logo_x, logo_y))

        if sampler and hasattr(sampler, 'sample_orientations'):
            start_orients = sampler.sample_orientations(num_shapes=num_shapes)
        else:
            start_orients = [random.uniform(0, 360) for _ in range(num_shapes)]

        norm_orients = [(o % 360) for o in start_orients]
        for i, shape in enumerate(bongard_image.one_stroke_shapes):
            shape.start_coordinates = start_coords[i]
            shape.start_orientation = norm_orients[i]
            shape.set_consistent_scaling_factors(scaling_factors[i])

        temp_dir = tempfile.mkdtemp()
        ps_dir = os.path.join(temp_dir, "ps")
        png_dir = os.path.join(temp_dir, "png")
        os.makedirs(ps_dir, exist_ok=True)
        os.makedirs(png_dir, exist_ok=True)
        problem_painter.save_bongard_images([bongard_image], "1", ps_dir, png_dir, auto_position=False)
        ps_file = os.path.join(ps_dir, "1", "0.ps")
        png_file = os.path.join(png_dir, "1", "0.png")
        if os.path.exists(ps_file):
            import subprocess
            gs_cmd = [
                "gswin64c" if os.name == "nt" else "gs",
                "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m",
                f"-sOutputFile={png_file}", ps_file
            ]
            try:
                subprocess.run(gs_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logging.error(f"Ghostscript failed for {problem_id}: {e}. Falling back to synthetic mask.")
                import shutil
                shutil.rmtree(temp_dir)
                return self._generate_synthetic_mask(action_commands, problem_id)

            if os.path.exists(png_file):
                from PIL import Image
                img = Image.open(png_file).convert('L')
                mask = (np.array(img) < 128).astype(np.uint8)
                import shutil
                shutil.rmtree(temp_dir)
                logging.info(f"Successfully generated {sampler_type} mask for {problem_id}")
                return mask
            else:
                import shutil
                shutil.rmtree(temp_dir)
                logging.error(f"Ghostscript did not produce PNG for {problem_id}. Falling back to synthetic mask.")
                return self._generate_synthetic_mask(action_commands, problem_id)
        else:
            import shutil
            shutil.rmtree(temp_dir)
            logging.error(f"PostScript file not found for {problem_id}. Falling back to synthetic mask.")
            return self._generate_synthetic_mask(action_commands, problem_id)

    def _generate_synthetic_mask(self, action_commands: list, problem_id: str) -> np.ndarray:
        """
        Generate a synthetic mask when turtle graphics fails.
        Uses Bongard-LOGO normalization functions to create varied masks.
        """
        try:
            # Create BongardImage to get proper shape information
            bongard_image = BongardImage.import_from_action_string_list(action_commands)

            # Calculate total complexity based on actions
            total_actions = sum(len(shape.get_actions()) for shape in bongard_image.one_stroke_shapes)
            total_length = 0
            for shape in bongard_image.one_stroke_shapes:
                for action in shape.get_actions():
                    if hasattr(action, 'line_length'):
                        total_length += action.line_length
                    elif hasattr(action, 'arc_radius'):
                        total_length += action.arc_radius * 2  # approximate arc length
                    else:
                        total_length += 50  # default length

            base_size = min(max(int(total_length / 10), 100), 500)
            mask_size = (512, 512)  # Standard output size

            # Generate synthetic shape based on complexity
            mask = np.zeros(mask_size, dtype=np.uint8)
            center_x, center_y = mask_size[0] // 2, mask_size[1] // 2

            # Create varied patterns based on action types and lengths
            for i, shape in enumerate(bongard_image.one_stroke_shapes):
                shape_mask = np.zeros(mask_size, dtype=np.uint8)

                # Offset each shape to avoid overlap
                offset_x = (i % 3 - 1) * 80
                offset_y = (i // 3 - 1) * 80

                actions = shape.get_actions()
                for j, action in enumerate(actions):
                    if hasattr(action, 'line_length'):
                        length = int(action.line_length)
                        # Draw line-like pattern
                        cv2.line(shape_mask, 
                               (center_x + offset_x - length//2, center_y + offset_y),
                               (center_x + offset_x + length//2, center_y + offset_y + j*15),
                               1, thickness=3)
                    elif hasattr(action, 'arc_radius'):
                        radius = int(action.arc_radius)
                        # Draw arc-like pattern
                        cv2.circle(shape_mask,
                                 (center_x + offset_x, center_y + offset_y),
                                 radius, 1, thickness=3)

                # Combine shape masks
                mask = np.maximum(mask, shape_mask)

            # Ensure mask has some content
            if np.sum(mask) == 0:
                # Create a simple default pattern based on problem_id hash
                hash_val = hash(problem_id) % 1000
                size = 50 + (hash_val % 100)
                cv2.rectangle(mask, (center_x - size//2, center_y - size//2),
                            (center_x + size//2, center_y + size//2), 1, thickness=3)

            logging.debug(f"Synthetic mask for {problem_id}: shape={mask.shape}, nonzero={np.count_nonzero(mask)}")
            return mask

        except Exception as e:
            logging.error(f"Synthetic mask generation failed for {problem_id}: {e}")
            # Return minimal default mask
            mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.rectangle(mask, (200, 200), (300, 300), 1, thickness=3)
            return mask

# Legacy compatibility
HybridAugmentor = ActionMaskGenerator
