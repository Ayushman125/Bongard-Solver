import os
import glob
import shutil
import random
import json
import logging
from pathlib import Path
from copy import deepcopy
from collections import Counter
import csv
import multiprocessing
import time
import torch
import yaml
import optuna
import numpy as np
from PIL import Image, ImageDraw # For PIL shape generation
import cv2 # For image operations in PIL shape generation and procedural generation
from sklearn.model_selection import train_test_split # For stratified splitting

# Import CONFIG and logger from the new config_loader.py
from yolofinetuning.config_loader import CONFIG, logger, YOLO_CLASSES_LIST

# Import modules from the same 'yolofinefrom yolofinetuning import my_data_utils
import yolofinetuning.my_data_utils as my_data_utils # Use explicit import with alias
import yolofinetuning.pipeline_workers as pipeline_workers # Use explicit import with alias
import yolofinetuning.yolo_fine_tuning as yolo_fine_tuning # Use explicit import with alias

# Import DALI specific components needed for iterator
try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    import math # For math.ceil
except ImportError:
    DALIGenericIterator = None
    LastBatchPolicy = None
    math = None

# --- UTILITIES FOR TUNER ──────────────────────────────────────────────────────
def class_entropy(counts):
    """Calculates entropy for class distribution (higher is better for balance)."""
    total = sum(counts.values())
    if total == 0: return 0.0
    probs = [v / total for v in counts.values() if v > 0]
    return -sum(p * np.log2(p) for p in probs)

def evaluate_params(params, sample_imgs_paths, objectives):
    """
    Evaluates a set of hyperparameters based on class balance and difficulty.
    Returns a score, or -1e9 if the configuration is invalid.
    """
    all_labels_flat, all_diffs = [], []
    temp_config = deepcopy(CONFIG)
    temp_config.update(params)
    for p_img in sample_imgs_paths:
        anno_path = Path(CONFIG['raw_annotations_dir']) / (Path(p_img).stem + '.json')
        if not anno_path.exists():
            logger.warning(f"Annotation file not found for tuning sample {p_img}. Skipping.")
            continue
        try:
            with open(anno_path, 'r') as f:
                anno_data = json.load(f)
            labels = [l[0] for l in anno_data.get('yolo_labels', [])]
            diff = anno_data.get('difficulty_score', 0.0)
        except json.JSONDecodeError as e:
            logger.warning(f"Error reading annotation JSON for tuning sample {p_img}: {e}. Skipping.")
            continue
        if not labels:
            continue
        all_labels_flat.extend(labels)
        all_diffs.append(diff)

    if len(all_labels_flat) < objectives['min_labels_per_sample'] or \
       len(all_diffs) < objectives['min_diffs_per_sample']:
        logging.getLogger().debug(f"Rejected config (too few samples): {params}")
        return -1e9

    counts = Counter(all_labels_flat)
    balance_score = class_entropy(counts)
    # Use CONFIG['class_names'] instead of my_data_utils.YOLO_CLASSES_LIST
    balance_score = np.clip(balance_score, 0, np.log2(len(CONFIG['class_names'])))
    diff_score = np.std(all_diffs) if len(all_diffs) > 1 else 0.0
    diff_score = np.clip(diff_score, 0, 1.0)

    score = (objectives['balance_weight'] * balance_score +
             objectives['difficulty_weight'] * diff_score)
    return score

# --- HYPERPARAMETER TUNER (OPTUNA INTEGRATED) ────────────────────────────────
def tune_hyperparams(all_imgs_paths, param_space, objectives, n_trials=50, subset_size=200):
    """
    Performs hyperparameter tuning using Optuna with a stratified sampling strategy.
    """
    logger.info("Starting automatic hyperparameter tuning with Optuna (Data-centric study)...")
    image_class_ids = {}
    for img_path in all_imgs_paths:
        anno_path = Path(CONFIG['raw_annotations_dir']) / (Path(img_path).stem + '.json')
        if anno_path.exists():
            try:
                with open(anno_path, 'r') as f:
                    anno_data = json.load(f)
                classes_in_image = sorted(list(set([l[0] for l in anno_data.get('yolo_labels', [])])))
                if classes_in_image:
                    generated_image_class_ids[str(img_path)] = str(classes_in_image)
                    image_class_ids[str(img_path)] = str(classes_in_image)
                else:
                    image_class_ids[str(img_path)] = "no_objects"
            except json.JSONDecodeError:
                image_class_ids[str(img_path)] = "error"
        else:
            image_class_ids[str(img_path)] = "no_annotation"

    valid_imgs_for_strat = [p for p in all_imgs_paths if image_class_ids.get(str(p)) not in ["no_objects", "error", "no_annotation"]]
    labels_for_strat = [image_class_ids[str(p)] for p in valid_imgs_for_strat]

    if len(valid_imgs_for_strat) < subset_size:
        logger.warning(f"Not enough valid images ({len(valid_imgs_for_strat)}) for stratified tuning subset of size {subset_size}. Using all valid images.")
        sample_imgs = valid_imgs_for_strat
    else:
        _, sample_imgs = train_test_split(valid_imgs_for_strat,
                                          test_size=subset_size,
                                          stratify=labels_for_strat,
                                          random_state=CONFIG['seed'])

    logger.info(f"Created stratified tuning sample of size {len(sample_imgs)}.")
    if not sample_imgs:
        logger.warning("      ⚠️         No valid images found for hyperparameter tuning. Skipping Optuna and using default CONFIG.")
        return None

    def objective(trial):
        trial_params = {
            'elastic_p': trial.suggest_float('elastic_p', param_space['elastic_p'][0], param_space['elastic_p'][-1]),
            'phot_blur_p': trial.suggest_float('phot_blur_p', param_space['phot_blur_p'][0], param_space['phot_blur_p'][-1]),
            'jpeg_p': trial.suggest_float('jpeg_p', param_space['jpeg_p'][0], param_space['jpeg_p'][-1]),
            'occlude_p': trial.suggest_float('occlude_p', param_space['occlude_p'][0], param_space['occlude_p'][-1]),
            'augmix_p': trial.suggest_float('augmix_p', param_space['augmix_p'][0], param_space['augmix_p'][-1]),
            'rand_nops': trial.suggest_int('rand_nops', param_space['rand_nops'][0], param_space['rand_nops'][-1]),
            'rand_mag': trial.suggest_int('rand_mag', param_space['rand_mag'][0], param_space['rand_mag'][-1]),
            'mixup_alpha': trial.suggest_float('mixup_alpha', param_space['mixup_alpha'][0], param_space['mixup_alpha'][-1]),
            'fract_depth': trial.suggest_int('fract_depth', param_space['fract_depth'][0], param_space['fract_depth'][-1]),
            'fill_contour_p': trial.suggest_float('fill_contour_p', 0.0, 1.0),
            'num_clutter_patches': trial.suggest_int('num_clutter_patches', 1, 5),
            'clutter_max_factor': trial.suggest_float('clutter_max_factor', 0.05, 0.2),
            'yolo_learning_rate': trial.suggest_float('yolo_learning_rate', param_space['yolo_learning_rate'][0], param_space['yolo_learning_rate'][-1], log=True),
            'yolo_final_lr_factor': trial.suggest_float('yolo_final_lr_factor', param_space['yolo_final_lr_factor'][0], param_space['yolo_final_lr_factor'][-1]),
            'yolo_batch_size': trial.suggest_categorical('yolo_batch_size', param_space['yolo_batch_size']),
            # NEW DALI-specific ranges for Optuna
            'dali_rotation_p': trial.suggest_float('dali_rotation_p', 0.0, 1.0),
            'dali_translation_p': trial.suggest_float('dali_translation_p', 0.0, 1.0),
            'dali_shear_p': trial.suggest_float('dali_shear_p', 0.0, 1.0),
            'dali_hsv_p': trial.suggest_float('dali_hsv_p', 0.0, 1.0),
            'dali_gaussian_noise_p': trial.suggest_float('dali_gaussian_noise_p', 0.0, 1.0),
            'dali_salt_pepper_p': trial.suggest_float('dali_salt_pepper_p', 0.0, 1.0),
        }
        score = evaluate_params(trial_params, sample_imgs, objectives)
        if score == -1e9:
            with open("rejected_configs.log", "a") as f:
                f.write(json.dumps({"trial_id": trial.number, "config": trial_params, "reason": "invalid_score"}) + "\n")
            raise optuna.exceptions.TrialPruned(f"Invalid configuration: {trial_params}")
        return score

    study_name = "bongard_augmentation_tuning"
    storage_path = CONFIG['tuning_db_path']
    try:
        optuna.delete_study(study_name=study_name, storage=storage_path)
        logger.info(f"Existing Optuna study '{study_name}' deleted from {storage_path}.")
    except (KeyError, ValueError):
        pass

    # Enable Pruning
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, storage=storage_path, direction="maximize", pruner=pruner)
    logger.info(f"Created new Optuna study '{study_name}' at {storage_path}.")

    from tqdm import tqdm # Import tqdm for Optuna progress bar
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    try:
        best_params = study.best_trial.params
        best_score = study.best_trial.value
        logger.info(f"      ✅             Optuna tuning finished. Best score: {best_score:.4f} with params: {best_params}")
        return best_params
    except ValueError:
        logger.warning("      ⚠️         Optuna tuning failed to find any valid configurations. Using default CONFIG.")
        return None

# --- CLEAN OUTPUT ─────────────────────────────────────────────────────────────
def prepare_output_directories():
    """
    Ensures output directories exist. Does NOT delete existing data for persistence.
    """
    out = Path(CONFIG['output_root'])
    for sp in CONFIG['splits']:
        (out/'images'/sp).mkdir(parents=True, exist_ok=True)
        (out/'labels'/sp).mkdir(parents=True, exist_ok=True)
        (out/'annotations'/sp).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['raw_annotations_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['hard_examples_dir']).mkdir(parents=True, exist_ok=True)
    logger.info("Ensured output folders exist for data generation.")

# --- COLLECT IMAGES ───────────────────────────────────────────────────────────
def collect_images():
    """
    Collects paths to all raw Bongard images, using a cache file for speed.
    """
    cache_path = Path(CONFIG['raw_image_paths_cache'])
    if cache_path.exists():
        file_mod_time = os.path.getmtime(cache_path)
        current_time = time.time()
        freshness_threshold_seconds = CONFIG['cache_freshness_days'] * 24 * 60 * 60
        if (current_time - file_mod_time) < freshness_threshold_seconds:
            logger.info(f"Loading raw image paths from cache: {cache_path}")
            try:
                with open(cache_path, 'r') as f:
                    imgs = json.load(f)
                logger.info(f"Loaded {len(imgs)} raw image paths from cache.")
                return imgs
            except json.JSONDecodeError as e:
                logger.warning(f"Error reading image path cache: {e}. Recalculating paths.")
        else:
            logger.info(f"Image path cache is stale (older than {CONFIG['cache_freshness_days']} days). Recalculating paths.")
    else:
        logger.info("Image path cache not found. Collecting raw image paths...")

    start_time = time.time()
    # Adjust bongard_root to be relative to the project root
    bongard_root_abs = Path(os.getcwd()) / CONFIG['bongard_root']
    imgs = list(bongard_root_abs.rglob('*.png'))
    end_time = time.time()
    logger.info(f"Collected {len(imgs)} raw images from {bongard_root_abs} in {end_time - start_time:.2f} seconds.")

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump([str(p) for p in imgs], f, indent=4) # Save as strings
        logger.info(f"Saved raw image paths to cache: {cache_path}")
    except Exception as e:
            logger.error(f"Failed to save raw image paths to cache: {e}")
    return [str(p) for p in imgs] # Return as strings

# --- DCGAN STUB / StyleGAN2-ADA Placeholder ───────────────────────────────────
# Note: Integrating StyleGAN2-ADA properly requires more than just
# pip install and a few lines. It typically involves downloading pre-trained
# models, setting up specific configurations, and handling its own training/generation
# pipeline. For this comprehensive plan, we'll keep the DCGAN_G stub but add a comment
# for where StyleGAN2-ADA would be integrated.
class DCGAN_G(torch.nn.Module): # Use torch.nn directly as it's imported
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(ngf * 8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 4),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

def generate_gan_images(n_images=100, output_dir=None, img_size=(224, 224)):
    logger.info(f"Generating {n_images} synthetic images using DCGAN stub...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # --- StyleGAN2-ADA Placeholder ---
    # To integrate StyleGAN2-ADA, you would typically:
    # 1. pip install stylegan2-ada-pytorch
    # 2. Download a pre-trained model (e.g., from their GitHub releases)
    # 3. Load the generator:
    #    from stylegan2_ada_pytorch.generate import parse_args, create_network
    #    args = parse_args(['--network=path/to/your/pretrained_model.pkl', '--outdir=temp_out', '--seeds=0-99'])
    #    G = create_network(args.network) # This loads the generator
    #    G.to(device)
    #    # Then, generate images using G(z, c) where z is latent vector, c is class label (if conditional)
    #    # and save them. This is a more involved process than a simple DCGAN stub replacement.
    # For now, we proceed with the simple DCGAN_G.
    G = DCGAN_G(nc=3).to(device).eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_count = 0
    from tqdm import tqdm # Import tqdm here for local use
    with torch.no_grad():
        for i in tqdm(range(n_images), desc="Generating GAN images"):
            filename = output_path / f"gan_{i}.png"
            if filename.exists():
                generated_count += 1
                continue
            noise = torch.randn(1, 100, 1, 1, device=device)
            generated_image_tensor = G(noise).cpu().detach()
            img_np = ((generated_image_tensor[0].numpy() + 1) / 2 * 255).astype(np.uint8)
            img_np = np.transpose(img_np, (1, 2, 0))
            if img_np.shape[0] != img_size[0] or img_np.shape[1] != img_size[1]:
                img_np = cv2.resize(img_np, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(filename), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            generated_count += 1
    logger.info(f"Finished generating {generated_count} synthetic images to {output_path}.")

# --- PIL Batch-Generator for Basic Shapes (Enhanced) ──────────────────────────
def generate_pil_shapes_advanced(class_name, n=2000, size=(224,224), outdir='data/pil_shapes', fill_p=0.5, rotation_range=180):
    """
    Generates basic shapes (circle, square, triangle, line, dot, polygon) using PIL,
    with added support for overlaps, variable stroke widths, and boolean operations.
    Args:
        class_name (str): 'circle', 'square', 'triangle', 'line', 'dot', 'polygon'.
        n (int): Number of images to generate for this class.
        size (tuple): Image size (width, height).
        outdir (str): Base output directory.
        fill_p (float): Probability of filling the shape.
        rotation_range (int): Max rotation angle in degrees.
    """
    class_output_dir = Path(outdir) / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating {n} {class_name} shapes using PIL (advanced)...")
    from tqdm import tqdm # Import tqdm here for local use
    for i in tqdm(range(n), desc=f"Generating PIL {class_name} shapes"):
        filename = class_output_dir / f"{class_name}_{i}.png"
        if filename.exists():
            continue
        img = Image.new('RGB', size, 'white')
        draw = ImageDraw.Draw(img)
        outline_color = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        # Variable stroke width
        stroke_width = random.randint(1, 6)
        # Variable fill patterns (None for outline, outline_color for solid, or random color)
        fill_choice = random.choice([None, outline_color, (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))])
        fill_color = fill_choice if random.random() < fill_p else None

        min_dim = min(size)
        max_size_factor = random.uniform(0.3, 0.7)
        shape_size = int(min_dim * max_size_factor)
        x_center = random.randint(shape_size // 2, size[0] - shape_size // 2)
        y_center = random.randint(shape_size // 2, size[1] - shape_size // 2)
        x1 = x_center - shape_size // 2
        y1 = y_center - shape_size // 2
        x2 = x_center + shape_size // 2
        y2 = y_center + shape_size // 2

        current_shape_bbox = None

        if class_name == 'circle':
            draw.ellipse([x1, y1, x2, y2], outline=outline_color, fill=fill_color, width=stroke_width)
            current_shape_bbox = [x1, y1, x2, y2]
        elif class_name == 'square':
            draw.rectangle([x1, y1, x2, y2], outline=outline_color, fill=fill_color, width=stroke_width)
            current_shape_bbox = [x1, y1, x2, y2]
        elif class_name == 'triangle':
            p1 = (x_center, y1)
            p2 = (x1, y2)
            p3 = (x2, y2)
            draw.polygon([p1, p2, p3], outline=outline_color, fill=fill_color, width=stroke_width)
            current_shape_bbox = [min(p1[0],p2[0],p3[0]), min(p1[1],p2[1],p3[1]), max(p1[0],p2[0],p3[0]), max(p1[1],p2[1],p3[1])]
        elif class_name == 'line':
            line_x1 = random.randint(0, size[0])
            line_y1 = random.randint(0, size[1])
            line_x2 = random.randint(0, size[0])
            line_y2 = random.randint(0, size[1])
            draw.line([line_x1, line_y1, line_x2, line_y2], fill=outline_color, width=random.randint(1, 5))
            current_shape_bbox = [min(line_x1,line_x2), min(line_y1,line_y2), max(line_x1,line_x2), max(line_y1,line_y2)]
        elif class_name == 'dot':
            dot_radius = random.randint(2, 5)
            draw.ellipse([x_center - dot_radius, y_center - dot_radius,
                          x_center + dot_radius, y_center + dot_radius],
                          outline=outline_color, fill=fill_color or outline_color, width=1)
            current_shape_bbox = [x_center - dot_radius, y_center - dot_radius, x_center + dot_radius, y_center + dot_radius]
        elif class_name == 'polygon':
            num_vertices = random.randint(5, 8)
            points = []
            for _ in range(num_vertices):
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0.3 * shape_size, 0.5 * shape_size)
                px = x_center + radius * np.cos(angle)
                py = y_center + radius * np.sin(angle)
                points.append((px, py))
            draw.polygon(points, outline=outline_color, fill=fill_color, width=stroke_width)
            current_shape_bbox = [min([p[0] for p in points]), min([p[1] for p in points]), max([p[0] for p in points]), max([p[1] for p in points])]

        # Overlaps & Nesting
        if random.random() < 0.3 and current_shape_bbox:
            # Sample another random shape and composite it
            second_shape_class = random.choice(CONFIG['class_names']) # Use CONFIG['class_names']
            temp_img = Image.new('RGBA', size, (0,0,0,0)) # Transparent background
            temp_draw = ImageDraw.Draw(temp_img)

            # Random position and size for the second shape, potentially overlapping
            second_shape_size = int(min_dim * random.uniform(0.2, 0.6))
            x_center2 = random.randint(0, size[0])
            y_center2 = random.randint(0, size[1])
            x1_2 = x_center2 - second_shape_size // 2
            y1_2 = y_center2 - second_shape_size // 2
            x2_2 = x_center2 + second_shape_size // 2
            y2_2 = y_center2 + second_shape_size // 2

            outline_color2 = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
            fill_color2 = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200)) if random.random() < fill_p else None

            if second_shape_class == 'circle':
                temp_draw.ellipse([x1_2, y1_2, x2_2, y2_2], outline=outline_color2, fill=fill_color2, width=random.randint(1,4))
            elif second_shape_class == 'square':
                temp_draw.rectangle([x1_2, y1_2, x2_2, y2_2], outline=outline_color2, fill=fill_color2, width=random.randint(1,4))
            elif second_shape_class == 'triangle':
                p1_2 = (x_center2, y1_2)
                p2_2 = (x1_2, y2_2)
                p3_2 = (x2_2, y2_2)
                temp_draw.polygon([p1_2, p2_2, p3_2], outline=outline_color2, fill=fill_color2, width=random.randint(1,4))
            elif second_shape_class == 'line':
                line_x1_2 = random.randint(0, size[0])
                line_y1_2 = random.randint(0, size[1])
                line_x2_2 = random.randint(0, size[0])
                line_y2_2 = random.randint(0, size[1])
                temp_draw.line([line_x1_2, line_y1_2, line_x2_2, line_y2_2], fill=outline_color2, width=random.randint(1, 5))
            elif second_shape_class == 'dot':
                dot_radius2 = random.randint(2, 5)
                temp_draw.ellipse([x_center2 - dot_radius2, y_center2 - dot_radius2,
                                   x_center2 + dot_radius2, y_center2 + dot_radius2],
                                   outline=outline_color2, fill=fill_color2 or outline_color2, width=1)
            elif second_shape_class == 'polygon':
                num_vertices2 = random.randint(5, 8)
                points2 = []
                for _ in range(num_vertices2):
                    angle2 = random.uniform(0, 2 * np.pi)
                    radius2 = random.uniform(0.3 * second_shape_size, 0.5 * second_shape_size)
                    px2 = x_center2 + radius2 * np.cos(angle2)
                    py2 = y_center2 + radius2 * np.sin(angle2)
                    points2.append((px2, py2))
                temp_draw.polygon(points2, outline=outline_color2, fill=fill_color2, width=random.randint(1,4))

            # Composite with random alpha
            alpha = random.uniform(0.5, 1.0)
            img = Image.alpha_composite(img.convert('RGBA'), Image.blend(Image.new('RGBA', size, (0,0,0,0)), temp_img, alpha)).convert('RGB')
            draw = ImageDraw.Draw(img) # Re-initialize draw object for the new image

        # Boolean Operations (via shapely + PIL mask) - Example for two polygons
        # This is more complex and typically done on masks, then applied to PIL image.
        # Requires converting PIL shapes to Shapely polygons, performing operations,
        # then converting back to PIL masks and compositing.
        # For simplicity, this is a conceptual placeholder. A full implementation
        # would involve:
        # from shapely.geometry import Polygon
        # from PIL import Image, ImageDraw
        # if class_name == 'polygon' and random.random() < 0.1: # Example condition
        #     # Create two polygons as Shapely objects
        #     poly1_points = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)] # Example square
        #     poly2_points = [(x_center, y1), (x1, y2), (x2, y2)] # Example triangle
        #     poly1 = Polygon(poly1_points)
        #     poly2 = Polygon(poly2_points)
        #
        #     # Perform boolean operation (e.g., union, difference, intersection)
        #     union_poly = poly1.union(poly2)
        #
        #     # Create a mask from the resulting Shapely polygon
        #     mask_img = Image.new('L', size, 0) # L for grayscale mask
        #     mask_draw = ImageDraw.Draw(mask_img)
        #     if union_poly.geom_type == 'MultiPolygon':
        #         for p in union_poly.geoms:
        #             mask_draw.polygon(list(p.exterior.coords), fill=255)
        #     else:
        #         mask_draw.polygon(list(union_poly.exterior.coords), fill=255)
        #
        #     # Apply the mask to the image
        #     color_layer = Image.new('RGB', size, outline_color) # Or fill_color
        #     img.paste(color_layer, (0,0), mask_img)

        if rotation_range > 0:
            angle = random.uniform(-rotation_range, rotation_range)
            img = img.rotate(angle, expand=False, center=(x_center, y_center))

        # Domain Randomization for backgrounds and lighting
        # This assumes `my_data_utils.random_bg` is available and takes PIL Image as input
        # Convert PIL Image to OpenCV format (numpy array) for `random_bg`
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
        background_paths = list(Path(CONFIG['background_root']).rglob('*.[pj][pn]g'))
        if background_paths:
            img_np = my_data_utils.random_bg(img_np, background_paths) # Use aliased import
        # Convert back to PIL Image for saving
        img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        img.save(filename)
    logger.info(f"Finished generating {n} {class_name} shapes to {class_output_dir}.")

# --- MAIN PIPELINE ───────────────────────────────────────────────────────────
def run_pipeline():
    random.seed(CONFIG['seed']); np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['seed'])
        torch.cuda.manual_seed_all(CONFIG['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Adjust paths to be relative to the current working directory of main.py
    # This ensures that when main.py is run from the project root, paths are correct.
    project_root = Path(os.getcwd())
    CONFIG['output_root'] = str(project_root / CONFIG['output_root'])
    CONFIG['difficulty_csv_path'] = str(project_root / CONFIG['difficulty_csv_path'])
    CONFIG['model_save_dir'] = str(project_root / CONFIG['model_save_dir'])
    CONFIG['visualize_yolo_save_dir'] = str(project_root / CONFIG['visualize_yolo_save_dir'])
    CONFIG['raw_image_paths_cache'] = str(project_root / CONFIG['raw_image_paths_cache'])
    CONFIG['data_generated_flag_file'] = str(project_root / CONFIG['data_generated_flag_file'])
    CONFIG['raw_annotations_precomputed_flag_file'] = str(project_root / CONFIG['raw_annotations_precomputed_flag_file'])
    CONFIG['raw_annotations_dir'] = str(project_root / CONFIG['raw_annotations_dir'])
    CONFIG['temp_generated_data_dir'] = str(project_root / CONFIG['temp_generated_data_dir'])
    CONFIG['dali_file_list_path'] = str(project_root / CONFIG['dali_file_list_path'])
    CONFIG['gan_persistent_dir'] = str(project_root / CONFIG['gan_persistent_dir'])
    CONFIG['pil_persistent_dir'] = str(project_root / CONFIG['pil_persistent_dir'])
    CONFIG['dali_processed_stems_tracker'] = str(project_root / CONFIG['dali_processed_stems_tracker'])
    CONFIG['hard_examples_dir'] = str(project_root / CONFIG['hard_examples_dir'])
    CONFIG['tuning_db_path'] = str(project_root / CONFIG['tuning_db_path'])

    # NEW: Paths for active learning
    if 'active' in CONFIG['hard_mining'] and 'pool_path' in CONFIG['hard_mining']['active']:
        CONFIG['hard_mining']['active']['pool_path'] = str(project_root / CONFIG['hard_mining']['active']['pool_path'])

    # NEW: Paths for distillation and semi-supervised teacher weights
    if CONFIG['distillation']['enabled'] and CONFIG['distillation']['teacher']:
        CONFIG['distillation']['teacher'] = str(project_root / CONFIG['distillation']['teacher'])
    if CONFIG['semi_supervised']['enabled'] and CONFIG['semi_supervised']['teacher_weights']:
        CONFIG['semi_supervised']['teacher_weights'] = str(project_root / CONFIG['semi_supervised']['teacher_weights'])

    # NEW: Path for curriculum score map
    if CONFIG['curriculum']['enabled'] and CONFIG['curriculum']['score_map']:
        CONFIG['curriculum']['score_map'] = str(project_root / CONFIG['curriculum']['score_map'])

    # NEW: Paths for SimGAN/CycleGAN
    if CONFIG['simgan']['enabled'] and 'path' in CONFIG['simgan']: # Check for 'path' key
        CONFIG['simgan']['path'] = str(project_root / CONFIG['simgan']['path'])
    if CONFIG['cyclegan']['enabled'] and 'path' in CONFIG['cyclegan']: # Check for 'path' key
        CONFIG['cyclegan']['path'] = str(project_root / CONFIG['cyclegan']['path'])

    # Ensure background root is also absolute
    CONFIG['background_root'] = str(project_root / CONFIG['background_root'])

    prepare_output_directories()
    all_raw_imgs = collect_images()

    if not all_raw_imgs:
        logger.error(f"      ❌         No raw images found in {CONFIG['bongard_root']}. Please ensure the dataset exists and is correctly structured.")
        logger.error("Exiting pipeline as there is no data to process.")
        return

    # --- GAN Image Generation ---
    gan_persistent_dir = Path(CONFIG['gan_persistent_dir'])
    generate_gan_images(n_images=CONFIG['gan_generate_n'],
                        output_dir=gan_persistent_dir,
                        img_size=CONFIG['image_size'])

    # --- PIL Batch-Generator for Basic Shapes ---
    pil_persistent_dir = Path(CONFIG['pil_persistent_dir'])
    for class_name in CONFIG['class_names']: # Use CONFIG['class_names']
        # Only generate basic shapes that PIL can easily create
        if class_name in ['circle', 'square', 'triangle', 'line', 'dot', 'polygon']:
            generate_pil_shapes_advanced(class_name, n=CONFIG['pil_generate_n_per_class'],
                                         size=tuple(CONFIG['image_size']),
                                         outdir=str(pil_persistent_dir))

    # --- Advanced Procedural Generator ---
    # Import procedural functions
    try:
        from yolofinetuning import procedural # Use explicit import
        procedural_output_dir = Path(CONFIG['output_root']) / 'procedural_images'
        procedural_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Generating advanced procedural images...")
        all_images_for_precomputation_procedural = [] # Collect procedural images for annotation
        for i in range(CONFIG['gan_generate_n'] // 5): # Generate a fifth of GAN images as procedural
            img_type = random.choice(['cellular_automata', 'texture'])
            if img_type == 'cellular_automata':
                rule = random.choice([30, 90, 110])
                img_np = procedural.gen_cellular_automata(tuple(CONFIG['image_size']), rule=rule)
                filename = procedural_output_dir / f"ca_{i}_{rule}.png"
            else: # texture
                img_np = procedural.gen_texture(tuple(CONFIG['image_size']))
                filename = procedural_output_dir / f"texture_{i}.png"

            # Apply random background to procedural images
            background_paths = list(Path(CONFIG['background_root']).rglob('*.[pj][pn]g'))
            # Check if background_np is defined before using it
            background_np = None # Initialize to None or a default value
            if background_paths:
                # Assuming procedural.py functions return numpy arrays
                # and random_bg expects a numpy array and background paths.
                # If random_bg is in my_data_utils, call it from there.
                # If `img_np` is expected to be modified in-place, ensure it's mutable.
                img_np = my_data_utils.random_bg(img_np, background_paths) # Use aliased import

            if img_np is not None:
                cv2.imwrite(str(filename), img_np)
                # Add to all_images_for_precomputation for annotation
                all_images_for_precomputation_procedural.append(str(filename))
        logger.info("Finished generating advanced procedural images.")
    except ImportError:
        logger.warning("Could not import 'procedural.py'. Skipping advanced procedural generation.")
        all_images_for_precomputation_procedural = []
    except Exception as e:
        logger.error(f"Error during procedural image generation: {e}", exc_info=True)
        all_images_for_precomputation_procedural = []

    # Combine original raw images, generated GAN images, generated PIL images, and procedural images for pre-computation
    all_images_for_precomputation = all_raw_imgs + \
                                    [str(p) for p in gan_persistent_dir.rglob('*.png')] + \
                                    [str(p) for p in pil_persistent_dir.rglob('*.png')] + \
                                    all_images_for_precomputation_procedural
    random.shuffle(all_images_for_precomputation)

    # --- Pre-compute Raw Image Annotations ---
    pipeline_workers.precompute_raw_image_annotations(all_images_for_precomputation, CONFIG) # Use aliased import

    # 1) Automatic Hyperparameter Tuning for Augmentation Probabilities & Depths (Optuna)
    param_space = {
        'elastic_p':   [0.1, 0.7],
        'phot_blur_p': [0.1, 0.7],
        'jpeg_p':      [0.1, 0.7],
        'occlude_p':   [0.1, 0.7],
        'augmix_p':    [0.3, 0.9],
        'rand_nops':   [1, 3],
        'rand_mag':    [5, 10],
        'mixup_alpha': [0.2, 0.8],
        'fract_depth': [3, 6],
        'fill_contour_p': [0.0, 1.0],
        'num_clutter_patches': [1, 5],
        'clutter_max_factor': [0.05, 0.2],
        'yolo_learning_rate': [1e-4, 1e-2],
        'yolo_final_lr_factor': [0.01, 0.2],
        'yolo_batch_size': [2, 4, 8],
        # NEW DALI-specific ranges for Optuna
        'dali_rotation_p': [0.0, 1.0],
        'dali_translation_p': [0.0, 1.0],
        'dali_shear_p': [0.0, 1.0],
        'dali_hsv_p': [0.0, 1.0],
        'dali_gaussian_noise_p': [0.0, 1.0],
        'dali_salt_pepper_p': [0.0, 1.0],
    }
    objectives_for_tuner = {
        'target_per_class': len(CONFIG['class_names']) * 5, # Use CONFIG['class_names']
        'balance_weight': 1.0,
        'difficulty_weight': 0.5,
        'min_labels_per_sample': CONFIG['tuning_min_labels_per_sample'],
        'min_diffs_per_sample': CONFIG['tuning_min_diffs_per_sample']
    }

    tuned_params_path = Path(CONFIG['output_root']) / 'tuned_hyperparams.json'
    if tuned_params_path.exists():
        logger.info(f"Loading tuned hyperparameters from {tuned_params_path}")
        try:
            best_tuned_params = json.load(open(tuned_params_path, 'r'))
            CONFIG.update(best_tuned_params)
        except json.JSONDecodeError as e:
            logger.error(f"Error loading tuned hyperparameters from {tuned_params_path}: {e}. Re-running tuning.")
            best_tuned_params = tune_hyperparams(all_images_for_precomputation, param_space, objectives_for_tuner,
                                                 n_trials=CONFIG['tuning_n_trials'], subset_size=CONFIG['tuning_subset_size'])
            if best_tuned_params is None:
                logger.warning("Optuna tuning failed to find a valid configuration. Using default CONFIG.")
            else:
                CONFIG.update(best_tuned_params)
                with open(tuned_params_path, 'w') as f:
                    json.dump(best_tuned_params, f, indent=4)
                logger.info(f"Saved re-tuned hyperparameters to {tuned_params_path}")
    else:
        logger.info("No saved tuned hyperparameters found. Running tuning process.")
        best_tuned_params = tune_hyperparams(all_images_for_precomputation, param_space, objectives_for_tuner,
                                             n_trials=CONFIG['tuning_n_trials'], subset_size=CONFIG['tuning_subset_size'])
        if best_tuned_params is None:
            logger.warning("Optuna tuning failed to find a valid configuration. Using default CONFIG.")
        else:
            CONFIG.update(best_tuned_params)
            with open(tuned_params_path, 'w') as f:
                json.dump(best_tuned_params, f, indent=4)
            logger.info(f"Saved tuned hyperparameters to {tuned_params_path}")

    # --- YOLO-centric Optuna Study ---
    try:
        from yolofinetuning import optuna_yolo # Use explicit import
        logger.info("\n--- Starting YOLO-centric Hyperparameter Optimization with Optuna ---")
        yolo_tuned_params = optuna_yolo.run_yolo_tuning(CONFIG)
        if yolo_tuned_params:
            CONFIG.update(yolo_tuned_params)
            logger.info(f"Updated CONFIG with YOLO-centric tuned parameters: {yolo_tuned_params}")
        else:
            logger.warning("YOLO-centric Optuna tuning did not return valid parameters. Using existing CONFIG.")
    except ImportError:
        logger.warning("Could not import 'optuna_yolo.py'. Skipping YOLO-centric hyperparameter optimization.")
    except Exception as e:
        logger.error(f"Error during YOLO-centric Optuna tuning: {e}", exc_info=True)

    # --- Main Data Generation and Persistence (DALI, Splitting, Moving) ---
    data_generated_flag_file = Path(CONFIG['data_generated_flag_file'])
    temp_generated_data_dir = Path(CONFIG['temp_generated_data_dir'])
    dali_processed_stems_tracker = Path(CONFIG['dali_processed_stems_tracker'])

    if data_generated_flag_file.exists():
        logger.info(f"Data generation flag file found at {data_generated_flag_file}. Skipping DALI processing and data splitting/moving.")
        splits = {}
        all_difficulty_scores = []
        for split_name in CONFIG['splits']:
            split_img_dir = Path(CONFIG['output_root']) / 'images' / split_name
            current_split_images = list(split_img_dir.rglob('*.png'))
            splits[split_name] = current_split_images
            logger.info(f"Loaded {len(splits[split_name])} images for '{split_name}' split from existing data.")

        difficulty_csv_path = Path(CONFIG['difficulty_csv_path'])
        if difficulty_csv_path.exists():
            logger.info(f"Loading difficulty summary from {difficulty_csv_path}...")
            with open(difficulty_csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                all_difficulty_scores.extend(list(reader))
            logger.info("Loaded existing difficulty summary.")
        else:
            logger.warning("Existing data found, but difficulty summary CSV is missing. Difficulty scores will not be available for this run.")
    else: # Overall data generation needs to run (DALI, splitting, moving)
        logger.info(f"Data generation completion flag file not found at {data_generated_flag_file}. Starting DALI processing and data splitting/moving.")

        processed_stems = set()
        if dali_processed_stems_tracker.exists():
            try:
                with open(dali_processed_stems_tracker, 'r') as f:
                    for line in f:
                        processed_stems.add(line.strip())
                logger.info(f"Resuming DALI generation. Found {len(processed_stems)} previously processed stems in tracker file.")
            except Exception as e:
                logger.warning(f"Error loading DALI processed stems tracker file: {e}. Starting DALI generation from scratch for DALI stage.")
                processed_stems = set()
                if temp_generated_data_dir.exists() and len(os.listdir(temp_generated_data_dir)) > 0:
                    logger.info(f"Cleaning up non-empty temporary DALI output directory: {temp_generated_data_dir}")
                    shutil.rmtree(temp_generated_data_dir)
                    temp_generated_data_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info("No DALI processed stems tracker file found. Starting DALI generation from scratch for DALI stage.")
            if temp_generated_data_dir.exists() and len(os.listdir(temp_generated_data_dir)) > 0:
                logger.info(f"Cleaning up non-empty temporary DALI output directory: {temp_generated_data_dir}")
                shutil.rmtree(temp_generated_data_dir)
                temp_generated_data_dir.mkdir(parents=True, exist_ok=True)

        temp_generated_data_dir.mkdir(parents=True, exist_ok=True)

        images_to_process_dali = []
        for img_path_str in all_images_for_precomputation:
            img_stem = Path(img_path_str).stem
            if img_stem not in processed_stems:
                images_to_process_dali.append(img_path_str)
            else:
                logger.debug(f"Skipping DALI processing for {Path(img_path_str).name}: Already processed in previous DALI run.")
        random.shuffle(images_to_process_dali)

        if not my_data_utils.HAS_DALI: # Use aliased import
            logger.error("NVIDIA DALI is not found. Cannot proceed with GPU-accelerated data generation.")
            logger.error("Please install NVIDIA DALI to use this feature. Exiting.")
            return

        dali_file_list_path = Path(CONFIG['dali_file_list_path'])
        dali_file_list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dali_file_list_path, 'w') as f:
            for line in images_to_process_dali:
                f.write(f"{line}\n")

        total_images_to_process_dali = len(images_to_process_dali)
        total_dali_batches = math.ceil(total_images_to_process_dali / CONFIG['dali_batch_size']) if CONFIG['dali_batch_size'] > 0 else 0
        logger.info(f"Starting DALI-accelerated data generation for {total_images_to_process_dali} images across {total_dali_batches} batches.")
        logger.info(f"A progress bar will appear below (if your environment supports it).")
        logger.info(f"DALI pipeline will attempt to use GPU: {CONFIG['yolo_device'] == 'cuda' and torch.cuda.is_available()}")
        logger.info(f"Detected CUDA availability: {torch.cuda.is_available()}")
        logger.info(f"Configured DALI device_id: {0 if CONFIG['yolo_device'] == 'cuda' and torch.cuda.is_available() else -1}")
        device_id = 0 if CONFIG['yolo_device'] == 'cuda' and torch.cuda.is_available() else -1

        try:
            dali_pipeline = my_data_utils.BongardDaliPipeline( # Use aliased import
                file_root=dali_file_list_path.parent,
                file_list=dali_file_list_path,
                config=CONFIG,
                device_id=device_id,
                is_training=True
            )
            dali_pipeline.build()
        except Exception as e:
            logger.error(f"Error building DALI pipeline: {e}", exc_info=True)
            logger.error("DALI pipeline failed to build. Please check your DALI installation and configuration.")
            return

        try:
            from tqdm import tqdm # Import tqdm here for local use
            dali_iterator = DALIGenericIterator(
                dali_pipeline,
                ['images', 'yolo_labels', 'annotations_json', 'difficulty_score'],
                auto_reset=True,
                last_batch_policy=LastBatchPolicy.PARTIAL
            )

            processed_count_current_run = 0
            all_generated_temp_paths_for_split = []
            all_difficulty_scores = []

            with open(dali_processed_stems_tracker, 'a+') as stems_f:
                for i, data in enumerate(tqdm(dali_iterator, total=total_dali_batches, desc="DALI Data Generation Progress")):
                    if i % 10 == 0 or i == total_dali_batches - 1:
                        logger.info(f"Processing DALI batch {i+1}/{total_dali_batches}...")
                    try:
                        images_batch = data[0]['images']
                        yolo_labels_batch = data[0]['yolo_labels']
                        annotations_json_batch = data[0]['annotations_json']
                        difficulty_score_batch = data[0]['difficulty_score']

                        for j in range(images_batch.shape[0]):
                            try:
                                img_tensor = images_batch[j]
                                yolo_labels_np = yolo_labels_batch[j]
                                annotations_json_str = annotations_json_batch[j].item()
                                difficulty_score_val = difficulty_score_batch[j].item()

                                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                                variant_stem = f"dali_aug_{processed_count_current_run + len(processed_stems)}"
                                temp_images_split_dir = temp_generated_data_dir / 'images' / 'temp'
                                temp_labels_split_dir = temp_generated_data_dir / 'labels' / 'temp'
                                temp_annotations_split_dir = temp_generated_data_dir / 'annotations' / 'temp'

                                temp_images_split_dir.mkdir(parents=True, exist_ok=True)
                                temp_labels_split_dir.mkdir(parents=True, exist_ok=True)
                                temp_annotations_split_dir.mkdir(parents=True, exist_ok=True)

                                variant_img_path = temp_images_split_dir / f"{variant_stem}.png"
                                variant_label_path = temp_labels_split_dir / f"{variant_stem}.txt"
                                variant_anno_path = temp_annotations_split_dir / f"{variant_stem}.json"

                                cv2.imwrite(str(variant_img_path), img_np)
                                with open(variant_label_path, 'w') as f:
                                    for label_row in yolo_labels_np:
                                        f.write(f"{int(label_row[0])} {label_row[1]:.6f} {label_row[2]:.6f} {label_row[3]:.6f} {label_row[4]:.6f}\n")

                                with open(variant_anno_path, 'w') as f:
                                    anno_dict = json.loads(annotations_json_str)
                                    anno_dict['filename'] = str(variant_img_path.relative_to(temp_generated_data_dir))
                                    anno_dict['difficulty_score'] = float(difficulty_score_val)
                                    json.dump(anno_dict, f, indent=4)

                                all_difficulty_scores.append({
                                    'filename': str(variant_img_path.relative_to(temp_generated_data_dir)),
                                    'split': 'temp',
                                    'difficulty_score': float(difficulty_score_val)
                                })
                                all_generated_temp_paths_for_split.append(variant_img_path)
                                processed_count_current_run += 1
                                stems_f.write(f"{variant_stem}\n")
                                stems_f.flush()
                                logger.debug(f"     Processed image {processed_count_current_run} ({variant_stem}) in batch {i+1}.")
                            except Exception as e:
                                logger.error(f"Error processing image {j} in DALI batch {i}: {e}", exc_info=True)
                                continue
                    except Exception as e:
                        logger.error(f"Error processing DALI batch {i}: {e}", exc_info=True)
                        continue
            logger.info(f"Finished DALI-accelerated data generation. Processed {processed_count_current_run} new images (total: {processed_count_current_run + len(processed_stems)}).")
            dali_pipeline.empty()
            del dali_pipeline
            del dali_iterator
        except Exception as e:
            logger.error(f"An error occurred during DALI data iteration: {e}", exc_info=True)
            logger.error("DALI data generation failed. You can restart, and it will resume from the last successfully processed image.")
            return

        # --- Stratified Splits + Challenge Subset ---
        logger.info("Splitting generated data into train, val, test, and challenge_val sets...")
        generated_image_class_ids = {}
        for img_path in all_generated_temp_paths_for_split:
            anno_path = temp_generated_data_dir / 'annotations' / 'temp' / (img_path.stem + '.json')
            if anno_path.exists():
                try:
                    with open(anno_path, 'r') as f:
                        anno_data = json.load(f)
                    classes_in_image = sorted(list(set([l[0] for l in anno_data.get('yolo_labels', [])])))
                    if classes_in_image:
                        generated_image_class_ids[str(img_path)] = str(classes_in_image)
                    else:
                        generated_image_class_ids[str(img_path)] = "no_objects"
                except json.JSONDecodeError:
                    generated_image_class_ids[str(img_path)] = "error"
            else:
                generated_image_class_ids[str(img_path)] = "no_annotation"

        valid_generated_imgs = [p for p in all_generated_temp_paths_for_split if generated_image_class_ids.get(str(p)) not in ["no_objects", "error", "no_annotation"]]
        labels_for_strat_gen = [generated_image_class_ids[str(p)] for p in valid_generated_imgs]

        if len(valid_generated_imgs) < 100:
            logger.warning("Not enough valid generated images for robust stratification. Performing simple random split.")
            t, rem = train_test_split(all_generated_temp_paths_for_split, train_size=CONFIG['splits']['train'],
                                      random_state=CONFIG['seed'])
            v, te = train_test_split(rem, test_size=CONFIG['splits']['test'] /
                                     (CONFIG['splits']['val'] + CONFIG['splits']['test']),
                                     random_state=CONFIG['seed'])
            splits = {'train': t, 'val': v, 'test': te, 'challenge_val': []}
        else:
            # First, split into train and (val+test+challenge_val)
            train_size_ratio = CONFIG['splits']['train']
            val_test_challenge_ratio = CONFIG['splits']['val'] + CONFIG['splits']['test'] + CONFIG['splits']['challenge_val']
            t_paths, val_test_challenge_paths = train_test_split(valid_generated_imgs,
                                                                 train_size=train_size_ratio,
                                                                 stratify=labels_for_strat_gen,
                                                                 random_state=CONFIG['seed'])

            # Now split val_test_challenge_paths into val, test, and challenge_val
            labels_val_test_challenge = [generated_image_class_ids[str(p)] for p in val_test_challenge_paths]
            total_remaining = len(val_test_challenge_paths)

            # Calculate sizes for val, test, challenge from the remaining paths
            challenge_size = int(total_remaining * (CONFIG['splits']['challenge_val'] / val_test_challenge_ratio))
            test_size = int(total_remaining * (CONFIG['splits']['test'] / val_test_challenge_ratio))
            val_size = total_remaining - challenge_size - test_size

            # Ensure non-zero sizes for stratification and handle potential rounding errors
            challenge_size = max(0, challenge_size)
            test_size = max(0, test_size)
            val_size = max(0, val_size)

            # Adjust sizes to ensure sum matches total_remaining
            current_sum = challenge_size + test_size + val_size
            if current_sum != total_remaining:
                diff = total_remaining - current_sum
                if diff > 0: # Add remainder to the largest group
                    if val_size >= test_size and val_size >= challenge_size: val_size += diff
                    elif test_size >= val_size and test_size >= challenge_size: test_size += diff
                    else: challenge_size += diff
                elif diff < 0: # Subtract from the largest group
                    if val_size >= test_size and val_size >= challenge_size: val_size = max(0, val_size + diff)
                    elif test_size >= val_size and test_size >= challenge_size: test_size = max(0, test_size + diff)
                    else: challenge_size = max(0, challenge_size + diff)

            # Perform splits sequentially
            if total_remaining > 0:
                # Split off challenge_val first
                val_test_rem_paths, challenge_val_paths = train_test_split(val_test_challenge_paths,
                                                                            test_size=challenge_size,
                                                                            stratify=[generated_image_class_ids[str(p)] for p in val_test_challenge_paths],
                                                                            random_state=CONFIG['seed'])
                # Then split remaining into val and test
                labels_val_test_rem = [generated_image_class_ids[str(p)] for p in val_test_rem_paths]
                val_paths, test_paths = train_test_split(val_test_rem_paths,
                                                          test_size=test_size,
                                                          stratify=labels_val_test_rem,
                                                          random_state=CONFIG['seed'])
            else:
                val_paths, test_paths, challenge_val_paths = [], [], []

            splits = {'train': t_paths, 'val': val_paths, 'test': test_paths, 'challenge_val': challenge_val_paths}

        logger.info(f"Generated dataset split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}, Challenge_Val={len(splits['challenge_val'])}")
        logger.info("Moving generated data to final split directories...")

        temp_stem_to_final_split = {}
        for split_name, paths_list in splits.items():
            for p in paths_list:
                temp_stem_to_final_split[p.stem] = split_name

        for split_name, image_paths_list in splits.items():
            target_img_dir = Path(CONFIG['output_root']) / 'images' / split_name
            target_label_dir = Path(CONFIG['output_root']) / 'labels' / split_name
            target_anno_dir = Path(CONFIG['output_root']) / 'annotations' / split_name

            target_img_dir.mkdir(parents=True, exist_ok=True)
            target_label_dir.mkdir(parents=True, exist_ok=True)
            target_anno_dir.mkdir(parents=True, exist_ok=True)

            from tqdm import tqdm # Import tqdm here for local use
            for img_path_temp in tqdm(image_paths_list, desc=f"Moving {split_name} data"):
                stem = img_path_temp.stem
                temp_label_path = temp_generated_data_dir / 'labels' / 'temp' / f"{stem}.txt"
                temp_anno_path = temp_generated_data_dir / 'annotations' / 'temp' / f"{stem}.json"

                final_img_path = target_img_dir / f"{stem}.png"
                final_label_path = target_label_dir / f"{stem}.txt"
                final_anno_path = target_anno_dir / f"{stem}.json"

                try:
                    if img_path_temp.exists():
                        shutil.move(img_path_temp, final_img_path)
                    if temp_label_path.exists():
                        shutil.move(temp_label_path, final_label_path)
                    if temp_anno_path.exists():
                        shutil.move(temp_anno_path, final_anno_path)
                except Exception as e:
                    logger.error(f"Error moving file {img_path_temp.name} to {split_name} split: {e}", exc_info=True)
                    continue

        logger.info("Finished moving data to final split directories.")

        for entry in all_difficulty_scores:
            temp_filename_path_relative = Path(entry['filename'])
            final_split = temp_stem_to_final_split.get(temp_filename_path_relative.stem)
            if final_split:
                entry['split'] = final_split
                entry['filename'] = str(Path('images') / final_split / temp_filename_path_relative.name)
            else:
                logger.warning(f"Could not determine final split for temp file: {entry['filename']}. Skipping path update.")

        logger.info(f"Cleaning up temporary generated data directory: {temp_generated_data_dir}")
        if temp_generated_data_dir.exists():
            shutil.rmtree(temp_generated_data_dir)

        data_generated_flag_file.parent.mkdir(parents=True, exist_ok=True)
        data_generated_flag_file.touch()
        logger.info(f"Created data generation completion flag file: {data_generated_flag_file}")

    # --- Curriculum Difficulty Scoring & CSV (always save/update if data was generated or loaded) ---
    logger.info(f"Saving difficulty summary to {CONFIG['difficulty_csv_path']}...")
    with open(CONFIG['difficulty_csv_path'], 'w', newline='') as csvfile:
        fieldnames = ['filename', 'split', 'difficulty_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_difficulty_scores)
    logger.info("Difficulty summary CSV saved.")

    # 6) data.yaml
    data_yaml_path = Path(CONFIG['output_root'])/'data.yaml'
    data_yaml_content = {
        'path': str(Path(CONFIG['output_root']).resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CONFIG['class_names']), # Use CONFIG['class_names']
        'names': CONFIG['class_names'] # Use CONFIG['class_names']
    }
    # Add challenge_val split to data.yaml if it exists and has images
    challenge_val_img_dir_check = Path(CONFIG['output_root']) / 'images' / 'challenge_val'
    if challenge_val_img_dir_check.exists() and len(list(challenge_val_img_dir_check.rglob('*.png'))) > 0:
        data_yaml_content['challenge_val'] = 'images/challenge_val'
    yaml.dump(data_yaml_content, open(data_yaml_path,'w'), sort_keys=False)
    logger.info(f"Generated data.yaml at: {data_yaml_path}")

    # --- PHASE 2: YOLO MODEL FINE-TUNING ---
    trained_model = yolo_fine_tuning.fine_tune_yolo_model(CONFIG) # Use aliased import

    # --- Hard Example Mining and Retraining ---
    if trained_model:
        logger.info("\n--- Hard Example Mining and Retraining Phase ---")
        # Mine hard examples from the validation set
        hard_examples_val = yolo_fine_tuning.mine_hard_examples( # Use aliased import
            model_path=str(trained_model.ckpt_path), # Use the path to the best model
            data_root=CONFIG['output_root'],
            split='val', # Mine from validation set
            output_dir=CONFIG['hard_examples_dir'],
            conf_thresh=CONFIG['hard_mining_conf_thresh']
        )

        # Filter by difficulty score (D.2)
        if hard_examples_val:
            logger.info(f"Filtering {len(hard_examples_val)} hard examples by difficulty score...")
            scores = []
            for p in hard_examples_val:
                anno_path = Path(CONFIG['raw_annotations_dir'])/f"{Path(p).stem}.json"
                if anno_path.exists():
                    try:
                        anno = json.load(open(anno_path, 'r'))
                        scores.append((p, anno.get('difficulty_score', 0.0)))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error reading annotation for difficulty score for {p}: {e}. Skipping.")
                        scores.append((p, 0.0)) # Assign a default low score
                else:
                    logger.warning(f"Annotation file not found for {p}. Cannot get difficulty score. Assigning 0.0.")
                    scores.append((p, 0.0))

            # Sort & pick top X%
            scores.sort(key=lambda x: x[1], reverse=True)
            n_keep = int(len(scores) * CONFIG.get('hard_mining_topk_frac', 0.3))
            hard_final = [p for p,_ in scores[:n_keep]]
            logger.info(f"Selected {len(hard_final)} top hard examples based on difficulty.")
        else:
            hard_final = []
            logger.info("No hard examples found after initial mining.")

        # Retrain on hard examples
        if hard_final:
            yolo_fine_tuning.retrain_on_hard_examples(trained_model, hard_final, CONFIG) # Use aliased import
            logger.info("Hard example retraining completed. Final evaluation on validation set after retraining.")

            # Re-evaluate after hard example retraining
            final_model_path_after_retrain = Path(CONFIG['model_save_dir'] + '_hard_retrain') / 'weights' / 'best.pt'
            if final_model_path_after_retrain.exists():
                from ultralytics import YOLO # Import YOLO here for evaluation
                retrained_model = YOLO(str(final_model_path_after_retrain))
                retrained_model.to(CONFIG['yolo_device'])
                logger.info(f"Loaded best model after hard example retraining: {final_model_path_after_retrain}")
                metrics_after_retrain = retrained_model.val(
                    data=str(data_yaml_path),
                    imgsz=CONFIG['yolo_img_size'][-1],
                    batch=CONFIG['yolo_batch_size'],
                    project=Path(CONFIG['model_save_dir']).parent,
                    name=f"{Path(CONFIG['model_save_dir']).name}_val_after_hard_retrain",
                    split='val',
                    seed=CONFIG['seed'],
                    workers=max(1, os.cpu_count() - 1)
                )
                logger.info(f"Validation Metrics After Hard Retraining: {metrics_after_retrain.results_dict}")
                logger.info(f"mAP50-95 After Hard Retraining: {metrics_after_retrain.box.map}")
                logger.info(f"mAP50 After Hard Retraining: {metrics_after_retrain.box.map50}")
            else:
                logger.warning("Model after hard example retraining not found for final evaluation.")
        else:
            logger.info("No hard examples found for retraining after difficulty filtering.")
    else:
        logger.error("YOLO model was not successfully fine-tuned. Skipping hard example mining and retraining.")

    logger.info("\nFull YOLO Pipeline execution completed successfully!")

def main():
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing start method (might already be set): {e}")
    run_pipeline()

if __name__ == "__main__":
    main()
