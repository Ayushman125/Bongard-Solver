# scripts/prepare_yolo_dataset.py
import logging  # Moved to the very top
# Configure logging for better output control
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)  # Logger defined here
# ======== ensure these exist before any code runs =======
# These flags will now be part of _worker_globals and initialized per process
# USE_PROGRAM = False
# USE_SAM     = False
# HAS_EMBED_MODEL = False

import os

# Secure OpenAI API Key loading
def get_openai_api_key():
    """
    Securely retrieves the OpenAI API key from environment variables.
    Never logs or exposes the key in code or output.
    """
    key = os.environ.get("OPENAI_API_KEY", None)
    if key is None or not key.strip():
        logger.warning("OpenAI API key not found in environment. LLM features will be disabled.")
        return None
    return key

# Usage: pass get_openai_api_key() to OpenAI client initialization, never expose or print the key

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import random
import shutil
import argparse
import cv2
import torch
from tqdm import tqdm
import json  # New import for JSON export
import numpy as np  # New import for mask/relation operations
from pathlib import Path  # New import for Path objects
from shapely.geometry import Polygon  # New import for spatial relations
import networkx as nx  # New import for spatial relations graph
import skimage.morphology as morph  # For skeletonization, convex hull
import skimage.measure as meas  # For connected components if needed

# New imports for DINOv2 and Albumentations
from transformers import AutoImageProcessor, AutoModel # Changed AutoFeatureExtractor to AutoImageProcessor
import albumentations as A

# Import CopyPaste from your local albumentations_augmix.py
try:
    from albumentations_augmix import CopyPaste
    logger.info("Successfully imported CopyPaste from albumentations_augmix.py.")
except ImportError:
    logger.error("Could not import CopyPaste from albumentations_augmix.py. "
                 "Please ensure the file is in the Python path and contains the CopyPaste class.")
    raise ImportError("CopyPaste class is required but could not be imported from albumentations_augmix.py.")

# New imports for optimizations
import cupy as cp
from joblib import Memory
import pyfftw
from numba import njit # New import for Numba JIT compilation
from torch.cuda.amp import autocast # New import for mixed precision

# giotto-tda for topological features
# HAS_GTDA flag will be part of _worker_globals
try:
    from gtda.homology import CubicalPersistence
    # HAS_GTDA = True # Moved to _worker_globals
    logger.info("giotto-tda found. Topological feature extraction enabled.")
except ImportError:
    # HAS_GTDA = False # Moved to _worker_globals
    logger.warning("giotto-tda not found. Topological feature extraction will be skipped.")

# Optional: OpenAI for LLM-based captioning
# HAS_OPENAI flag will be part of _worker_globals
try:
    from openai import OpenAI
    # client = OpenAI() # Moved to _worker_globals
    # HAS_OPENAI = True # Moved to _worker_globals
    logger.info("OpenAI client initialized for LLM captioning.")
except ImportError:
    # HAS_OPENAI = False # Moved to _worker_globals
    logger.warning("OpenAI library not found. LLM captioning will be skipped.")
except Exception as e:
    # HAS_OPENAI = False # Moved to _worker_globals
    logger.warning(f"Could not initialize OpenAI client for captioning: {e}. LLM captioning will be skipped.")

# Optional: Segment Anything Model (SAM) for confidence maps and automatic mask generation
# predictor = None # Moved to _worker_globals
# mask_generator = None # Moved to _worker_globals
# USE_SAM flag will be part of _worker_globals
try:
    # Import SAM loading and utility functions from your local sam.py
    from sam import load_sam_model, get_mask_generator, save_mask_png, sam_masks_to_yolo, generate_relation_graph, get_symbolic_labels, overlay_symbolic_debugger, generate_reasoning_chain
    logger.info("Attempting to load SAM model and utilities from sam.py...")
    
    # SAM checkpoint path and model type (constants, can remain global or be passed)
    SAM_CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"
    SAM_MODEL_TYPE = "vit_h"
    
    # Ensure the 'weights' directory exists
    os.makedirs("weights", exist_ok=True)

    # Models will be loaded in _worker_init
except ImportError as ie:
    logger.warning(f"Could not import SAM components from sam.py: {ie}. Please ensure sam.py exists and is in the Python path. SAM functionalities will be skipped.")
except Exception as e:
    logger.warning(f"Error initializing SAM from sam.py: {e}. SAM functionalities will be skipped.")


# Optional: Pre-trained embedding model (DINOv2)
# embed_model = None # Moved to _worker_globals
# HAS_EMBED_MODEL flag will be part of _worker_globals
# device = "cuda" if torch.cuda.is_available() else "cpu" # Moved to _worker_globals

try:
    # DINOv2 Integration
    logger.info("Attempting to load DINOv2 for shape embeddings...")
    # processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base") # Moved to _worker_globals
    # dinov2 = AutoModel.from_pretrained("facebook/dinov2-base").to(device) # Moved to _worker_globals
    # dinov2.eval() # Moved to _worker_globals
    
    # embed_model = dinov2 # Moved to _worker_globals
    # HAS_EMBED_MODEL = True # Moved to _worker_globals
    logger.info(f"DINOv2 embedding model will be initialized per worker.")
except ImportError:
    logger.warning("HuggingFace Transformers library not found. DINOv2 embedding generation will be skipped.")
except Exception as e:
    logger.warning(f"Error setting up DINOv2 import: {e}. Shape embedding generation will be skipped.")

# Import CONFIG from config_loader.py
try:
    from yolofinetuning.config_loader import CONFIG, YOLO_CLASSES_LIST
    logger.info("CONFIG and YOLO_CLASSES_LIST imported from config_loader.py.")
except ImportError:
    logger.error("Could not import CONFIG or YOLO_CLASSES_LIST from yolofinetuning.config_loader. "
                 "Please ensure the file exists and is in the Python path.")
     # Fallback for standalone execution/testing if config_loader is not set up as a package
    YOLO_CLASSES_LIST = ['circle', 'square', 'triangle', 'line', 'dot', 'polygon',
                         'pentagon', 'hexagon', 'star', 'cross', 'plus', 'ellipse', 'unknown_shape']
    CONFIG = {
        'seed': 42,
        'data_root': './data',   # Default data root for output
        'yolo_img_size': [640, 640],   # Default image size
        'label_directories': {  # Minimal default for fallback
            "boxes": "labels", "masks": "masks", "polygons": "polygons",
            "programs": "programs", "relations": "relations", "topo": "topo",
            "descriptors": "stats", "captions": "captions",
            "oriented": "oriented_boxes", "stroke_meta": "strokes_meta",
            "keypoints": "keypoints", "sam_conf": "sam_confidence",
            "pyramid": "mask_pyramid", "fourier": "fourier_desc",
            "embed": "embeddings", "rule": "rule_explanations",
            "aug_meta": "aug_metadata", "neg_flags": "negative_flags",
            "sam_masks": "sam_masks", # New directory for SAM generated masks
            "sam_yolo": "sam_yolo_labels", # New directory for YOLO labels from SAM
            "sam_relations": "sam_relations", # New directory for relations from SAM masks
            "symbolic": "symbolic_labels", # New directory for symbolic labels
            "visual_debug": "visual_debug_overlays", # New directory for visual debugger output
            "reasoning_chains": "reasoning_chains", # New directory for reasoning chains
            "complexity_scores": "complexity_scores" # New directory for complexity scores
        }
    }
# 2.1 Programmatic labels via Bongard library
# USE_PROGRAM flag will be part of _worker_globals
try:
    from bongard import BongardProblem
    # USE_PROGRAM = True # Moved to _worker_globals
    logger.info("Bongard library found. Programmatic labeling enabled (strict mode).")
except ImportError:
    logger.error("Bongard library not found. Programmatic labeling is REQUIRED for strict mode.")
    # USE_PROGRAM remains False from its initial definition, which will trigger the RuntimeError later.

# Global dictionary for class IDs - now derived from YOLO_CLASSES_LIST from CONFIG
CLASS_ID = {name: idx for idx, name in enumerate(YOLO_CLASSES_LIST)}
# Determine the number of classes based on the mapped IDs
NUM_CLASSES = len(set(CLASS_ID.values()))
CLASS_NAMES = sorted(CLASS_ID.keys(), key=lambda k: CLASS_ID[k])    # For data.yaml

# Define augmentation transforms globally (Albumentations) - will be accessed from _worker_globals
# train_augs = A.Compose([...]) # Moved to _worker_globals


# Global dictionary to store objects initialized per worker process
_worker_globals = {}

def _worker_init(worker_id):
    """
    Initializes heavy models and other global resources for each worker process.
    This function is passed to multiprocessing.Pool(initializer=_worker_init).
    """
    global _worker_globals
    logger.info(f"Worker {worker_id} initializing global resources.")

    # Configuration and Class IDs
    _worker_globals['CONFIG'] = CONFIG
    _worker_globals['CLASS_ID'] = CLASS_ID
    _worker_globals['NUM_CLASSES'] = NUM_CLASSES
    _worker_globals['CLASS_NAMES'] = CLASS_NAMES

    # Device for PyTorch models
    _worker_globals['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Bongard Problem (needed for programmatic labeling)
    _worker_globals['USE_PROGRAM'] = False
    try:
        from bongard import BongardProblem
        _worker_globals['BongardProblem'] = BongardProblem
        _worker_globals['USE_PROGRAM'] = True
    except ImportError:
        logger.warning(f"Worker {worker_id}: Bongard library not found. Programmatic labeling disabled.")
    
    # SAM Models
    _worker_globals['USE_SAM'] = False
    _worker_globals['predictor'] = None
    _worker_globals['mask_generator'] = None
    try:
        from sam import load_sam_model, get_mask_generator, save_mask_png, sam_masks_to_yolo, generate_relation_graph, get_symbolic_labels, overlay_symbolic_debugger, generate_reasoning_chain
        _worker_globals['save_mask_png'] = save_mask_png
        _worker_globals['sam_masks_to_yolo'] = sam_masks_to_yolo
        _worker_globals['generate_relation_graph'] = generate_relation_graph
        _worker_globals['get_symbolic_labels'] = get_symbolic_labels
        _worker_globals['overlay_symbolic_debugger'] = overlay_symbolic_debugger
        _worker_globals['generate_reasoning_chain'] = generate_reasoning_chain

        _worker_globals['predictor'] = load_sam_model(
            checkpoint_path=SAM_CHECKPOINT_PATH,
            model_type=SAM_MODEL_TYPE,
            device=_worker_globals['device']
        )
        _worker_globals['mask_generator'] = get_mask_generator(
            checkpoint_path=SAM_CHECKPOINT_PATH,
            model_type=SAM_MODEL_TYPE,
            device=_worker_globals['device']
        )
        if _worker_globals['predictor'] and _worker_globals['mask_generator']:
            _worker_globals['USE_SAM'] = True
            logger.info(f"Worker {worker_id}: SAM models loaded.")
        else:
            logger.warning(f"Worker {worker_id}: Failed to load SAM models. SAM functionalities disabled.")
    except Exception as e:
        logger.warning(f"Worker {worker_id}: Error loading SAM components: {e}. SAM functionalities disabled.")

    # DINOv2 Embedding Model
    _worker_globals['HAS_EMBED_MODEL'] = False
    _worker_globals['processor'] = None
    _worker_globals['embed_model'] = None
    try:
        from transformers import AutoImageProcessor, AutoModel
        _worker_globals['processor'] = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True) # use_fast=True
        _worker_globals['embed_model'] = AutoModel.from_pretrained("facebook/dinov2-base").to(_worker_globals['device'])
        _worker_globals['embed_model'].eval()
        _worker_globals['HAS_EMBED_MODEL'] = True
        logger.info(f"Worker {worker_id}: DINOv2 model loaded.")
    except Exception as e:
        logger.warning(f"Worker {worker_id}: Error loading DINOv2 model: {e}. DINOv2 embeddings disabled.")

    # OpenAI Client
    _worker_globals['HAS_OPENAI'] = False
    _worker_globals['client'] = None
    try:
        from openai import OpenAI
        _worker_globals['client'] = OpenAI()
        _worker_globals['HAS_OPENAI'] = True
        logger.info(f"Worker {worker_id}: OpenAI client initialized.")
    except Exception as e:
        logger.warning(f"Worker {worker_id}: Error initializing OpenAI client: {e}. LLM functionalities disabled.")

    # GTDA Caching and CuPy
    # Ensure cache directory exists and is unique per worker for multiprocessing safety
    cache_dir_worker = os.path.join(os.getcwd(), f".cache_gtda_worker_{worker_id}")
    os.makedirs(cache_dir_worker, exist_ok=True)
    _worker_globals['memory'] = Memory(location=cache_dir_worker, verbose=0)
    
    _worker_globals['HAS_CUPY'] = False
    try:
        import cupy as cp
        _worker_globals['cp'] = cp # Store cupy module
        _worker_globals['HAS_CUPY'] = True
        logger.info(f"Worker {worker_id}: CuPy found.")
    except ImportError:
        logger.warning(f"Worker {worker_id}: CuPy not found.")
    
    _worker_globals['HAS_GTDA'] = False
    try:
        from gtda.homology import CubicalPersistence
        _worker_globals['CubicalPersistence'] = CubicalPersistence
        _worker_globals['HAS_GTDA'] = True
        logger.info(f"Worker {worker_id}: giotto-tda found.")
    except ImportError:
        logger.warning(f"Worker {worker_id}: giotto-tda not found.")

    # pyFFTW
    _worker_globals['L_max'] = 2048
    _worker_globals['_in_fft'] = None
    _worker_globals['_out_fft'] = None
    _worker_globals['fft_obj'] = None
    try:
        _worker_globals['_in_fft'] = pyfftw.empty_aligned(_worker_globals['L_max'], dtype="complex128")
        _worker_globals['_out_fft'] = pyfftw.empty_aligned(_worker_globals['L_max'], dtype="complex128")
        _worker_globals['fft_obj'] = pyfftw.FFTW(_worker_globals['_in_fft'], _worker_globals['_out_fft'], axes=(0,), threads=1) # Use 1 thread per FFT for worker
        logger.info(f"Worker {worker_id}: pyFFTW initialized.")
    except Exception as e:
        logger.warning(f"Worker {worker_id}: Error initializing pyFFTW: {e}. Fourier descriptors will use NumPy's FFT.")

    # Albumentations Augmentations (can be initialized once)
    _worker_globals['train_augs'] = A.Compose([
        A.Mosaic(p=0.5),
        CopyPaste(mask_format='binary', p=0.5), 
        A.RandomBrightnessContrast(p=0.5),
        A.CoarseDropout(
            max_holes=8, max_height=16, max_width=16, min_holes=1, fill_value=0, p=0.5
        ),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.7),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# --- Optimizations Global Setup --- (These functions now access _worker_globals)

# Removed @_worker_globals['memory'].cache decorator here
def compute_topo_features_internal(gray_img_array: np.ndarray):
    """
    Internal function to extract topological features (Betti numbers, persistence) using CubicalPersistence,
    with optional CuPy transfer for initial data handling.
    This function is called by compute_topo_cached which applies joblib.Memory caching.
    - gray_img_array: Grayscale image (H, W)
    Returns a dictionary of topological features.
    """
    if not _worker_globals['HAS_GTDA']:
        logger.warning("giotto-tda not available. Skipping topological feature extraction.")
        return {}

    X = None
    if _worker_globals['HAS_CUPY']:
        try:
            # Move data to GPU for initial processing if CuPy is available
            X_gpu = _worker_globals['cp'].asarray(gray_img_array[..., None].astype(np.float32) / 255.0)
            # Transfer back to NumPy once preprocessing is done, as GTDA is CPU-only
            X = _worker_globals['cp'].asnumpy(X_gpu)
            _worker_globals['cp'].get_default_memory_pool().free_all_blocks() # Free GPU memory
            _worker_globals['cp'].get_default_pinned_memory_pool().free_all_blocks() # Free pinned memory
        except Exception as e:
            logger.warning(f"CuPy GPU transfer failed: {e}. Falling back to CPU for topological features.")
            X = gray_img_array[...,None].astype(np.float32) / 255.0
    else:
        X = gray_img_array[...,None].astype(np.float32) / 255.0
    
    try:
        cp_instance = _worker_globals['CubicalPersistence'](homology_dimensions=[0,1], n_jobs=1)
        diagrams = cp_instance.fit_transform([X])
        
        if len(diagrams) > 0 and diagrams[0].size > 0:
            diagram = diagrams[0]
            births = diagram[:,1]
            deaths = diagram[:,2]
            
            betti0 = np.sum(np.isfinite(deaths[diagram[:,0] == 0]))
            sum_persistence = np.sum(deaths - births)
            
            return {
                "betti0": int(betti0),
                "sum_persistence": float(sum_persistence)
            }
        else:
            logger.warning("No persistence diagrams extracted. Returning empty topological features.")
            return {}
    except Exception as e:
        logger.error(f"Error extracting topological features: {e}. Returning empty features.")
        return {}

# Define the cached version of the function
def compute_topo_cached(gray_img_array: np.ndarray):
    """
    Extracts topological features with caching.
    This function applies the caching decorator from _worker_globals['memory'].
    """
    # This ensures 'memory' is available after _worker_init has run
    return _worker_globals['memory'].cache(compute_topo_features_internal)(gray_img_array)


# Modified compute_fourier_descriptors to use pyFFTW
def compute_fourier_descriptors(points: list[tuple[int,int]], n_coeffs: int = 16):
    """
    Computes first n_coeffs complex Fourier coefficients from contour points,
    using pyFFTW if available, otherwise NumPy.
    Returns list of complex numbers (as real/imag tuples).
    """
    if not points or len(points) < 2:  # Need at least 2 points for a contour
        return []
    
    pts = np.array(points, dtype=np.complex128)
    N = len(pts)

    if _worker_globals['fft_obj']: # Use pyFFTW if initialized
        if N > _worker_globals['L_max']:
            logger.warning(f"Contour length {N} exceeds L_max ({_worker_globals['L_max']}) for pyFFTW. Falling back to NumPy FFT for this contour.")
            coeffs = np.fft.fft(pts) / N
        else:
            # Copy into aligned buffer and execute
            _worker_globals['_in_fft'][:N] = pts
            _worker_globals['_in_fft'][N:] = 0 # Pad with zeros if N < _worker_globals['L_max']
            out = _worker_globals['fft_obj']()
            coeffs = out[:n_coeffs] / N # Normalize by original N
    else: # Fallback to NumPy FFT
        coeffs = np.fft.fft(pts) / N
    
     # Return first n_coeffs (excluding DC component if desired, but user asks for first n_coeffs)
     # Ensure n_coeffs does not exceed available coefficients
    n_coeffs = min(n_coeffs, len(coeffs))
    
    return [(float(c.real), float(c.imag)) for c in coeffs[:n_coeffs]]

# Internal helper for single patch embedding (now called by batched function)
def _embed_single_shape_patch(patch: np.ndarray) -> torch.Tensor:
    """
    Extracts a feature embedding from a DINOv2 model for a single shape crop.
    Returns a PyTorch tensor.
    """
    # Ensure patch is RGB and has 3 channels for DINOv2 input
    if patch.ndim == 2:  # Grayscale
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
    elif patch.shape[2] == 4:  # RGBA
        patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)
    
    # Resize to DINOv2’s expected input size (e.g., 518×518)
    patch_resized = cv2.resize(patch, (518, 518))
    
    # Preprocess via HuggingFace processor (global `processor` is used)
    inputs = _worker_globals['processor'](images=patch_resized, return_tensors="pt").to(_worker_globals['device']) 
    
    # Forward with mixed precision
    with autocast():
        with torch.no_grad():
            out = _worker_globals['embed_model'](**inputs).last_hidden_state  # (1, N_patches, dim)
            feat = out.mean(dim=1)                   # global average pool (1, dim)
    
    return feat.squeeze() # Return as 1D tensor

# 2. Batched DINOv2 Patch Embeddings
def embed_patches_batched(patches: list[np.ndarray]) -> list:
    """
    Extracts feature embeddings from a DINOv2 model for a batch of shape crops.
    Returns a list of feature vectors.
    """
    if _worker_globals['embed_model'] is None or not _worker_globals['HAS_EMBED_MODEL'] or not patches:
        logger.warning("Embedding model not initialized/enabled or no patches provided. Skipping batched shape embedding.")
        return [[] for _ in patches] # Return empty list for each input patch if no model or no patches

    # Preprocess all patches in a list
    # Ensure patches are RGB for the processor
    processed_patches = []
    for patch in patches:
        if patch.ndim == 2:
            processed_patches.append(cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB))
        elif patch.shape[2] == 4:
            processed_patches.append(cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB))
        else:
            processed_patches.append(patch) # Already RGB

    # Resize all patches to DINOv2’s expected input size (e.g., 518×518)
    resized_patches = [cv2.resize(p, (518, 518)) for p in processed_patches]

    # Preprocess via HuggingFace processor
    inputs = _worker_globals['processor'](images=resized_patches, return_tensors="pt").to(_worker_globals['device'])
    
    # Forward in one batch with mixed precision
    with autocast():
        with torch.no_grad():
            out = _worker_globals['embed_model'](**inputs).last_hidden_state  # (B, N_patches, dim)
    
    # Pool and return as list of B vectors
    feats = out.mean(dim=1).cpu().tolist() # list of B vectors
    return feats

# 1.8 Rule explanation via LLM
def generate_rule_explanation(pos_image_paths: list, neg_image_paths: list):
    """
    Prompts an LLM to describe the distinguishing rule based on image paths.
    """
    if not _worker_globals['HAS_OPENAI'] or _worker_globals['client'] is None:
        logger.warning("OpenAI client not available. Skipping rule explanation generation.")
        return ""
     # Use only a few examples to keep prompt short and within token limits
    pos_examples = [Path(p).name for p in pos_image_paths[:3]]
    neg_examples = [Path(p).name for p in neg_image_paths[:3]]
    prompt = (
        f"Given positive examples (images): {', '.join(pos_examples)} and negative examples (images): {', '.join(neg_examples)}, "
        "write a concise English rule that separates them. Focus on visual properties and relations of shapes. "
        "Example: 'The positive examples contain a red circle, while negative examples do not.'"
    )
    
    try:
        resp = _worker_globals['client'].chat.completions.create(
            model="gpt-4o-mini",  # Use a text-based model for rule explanation
            messages=[{"role":"user","content":prompt}]
        )
        rule_text = resp.choices[0].message.content
        return rule_text
    except Exception as e:
        logger.error(f"Error generating LLM rule explanation: {e}. Returning empty string.")
        return ""

# --- Utility Functions for New Label Types ---
def stroke_mask(img_shape: tuple, strokes: list) -> np.ndarray:
    """
    Render a binary mask of shape (H,W) from programmatic stroke polygons.
    - img_shape: tuple (H, W, C) or (H, W)
    - strokes:   list of stroke objects with `.points` attribute as List[(x,y)]
    Returns: a NumPy array (H, W) with pixel values 0 or 1.
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for s in strokes:
         # Ensure points are integers and in the correct shape for cv2.fillPoly
        pts = np.array(s.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], color=1)
    return mask

def mask_to_polygon(mask: np.ndarray) -> list:
    """
    Converts a binary mask to a list of (x,y) points representing the largest contour.
    Returns an empty list if no contours are found.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
         # Find the largest contour by area
        cnt = max(cnts, key=cv2.contourArea)
        return cnt.reshape(-1, 2).tolist()
    return []

def export_action_program(prog, out_path: Path):
    """
    Write the LOGO action sequence to JSON.
    - prog:     a Bongard “program” object with .actions or .get_action_string_list()
    - out_path: Path where to save .json
    """
    seq = []
    try:
        seq = prog.get_action_string_list()
    except AttributeError:
         # Fallback if get_action_string_list() is not available
        if hasattr(prog, 'actions'):
            seq = prog.actions
        else:
            logger.warning(f"Program object for {out_path.stem} has no 'actions' or 'get_action_string_list()'. Exporting empty list.")
    try:
        with open(out_path, "w") as f:
            json.dump(seq, f, indent=2)
    except IOError as e:
        logger.error(f"Could not write action program to {out_path}: {e}")

@njit # Numba JIT for faster overlap check
def _fast_mask_overlap_sum(mask1: np.ndarray, mask2: np.ndarray) -> int:
    """
    Computes the sum of overlapping pixels between two boolean masks using Numba.
    """
    return np.sum(np.logical_and(mask1, mask2))

def compute_relations(polygons: list[list[tuple]], masks: list[np.ndarray]) -> dict:
    """
    Compute pairwise relations among a list of polygons (each a list of (x,y) points).
    Returns a NetworkX graph converted to node-link data format.
    Relations include: touches, contains, left_of, above, overlaps.
    
    Args:
        polygons (list[list[tuple]]): List of polygon points.
        masks (list[np.ndarray]): List of corresponding binary masks.
    """
    G = nx.DiGraph()
    shapely_polygons = []
    valid_indices = []
     # Convert to shapely Polygons and filter out invalid ones
    for i, poly_pts in enumerate(polygons):
        if len(poly_pts) >= 3:  # A polygon needs at least 3 points
            try:
                shapely_polygons.append(Polygon(poly_pts))
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Could not create shapely Polygon from points for index {i}: {e}. Skipping.")
                shapely_polygons.append(None)  # Placeholder for invalid polygon
        else:
            shapely_polygons.append(None)  # Placeholder for invalid polygon
    
    for i_idx, p1_shapely in enumerate(shapely_polygons):
        if p1_shapely is None: continue
        
        original_i = valid_indices[i_idx]
        G.add_node(original_i)  # Add node using original index
        c1x, c1y = p1_shapely.centroid.x, p1_shapely.centroid.y
        
        for j_idx, p2_shapely in enumerate(shapely_polygons):
            if i_idx == j_idx or p2_shapely is None: continue
            original_j = valid_indices[j_idx]
            c2x, c2y = p2_shapely.centroid.x, p2_shapely.centroid.y
            
            # Use Numba-accelerated overlap check for masks
            mask1_bool = masks[original_i] > 0
            mask2_bool = masks[original_j] > 0
            
            if _fast_mask_overlap_sum(mask1_bool, mask2_bool) > 0:
                G.add_edge(original_i, original_j, relation="overlaps")

             # Touches
            if p1_shapely.touches(p2_shapely):
                G.add_edge(original_i, original_j, relation="touches")
             # Contains
            if p1_shapely.contains(p2_shapely):
                G.add_edge(original_i, original_j, relation="contains")
             # Left-of
            if c1x < c2x:
                G.add_edge(original_i, original_j, relation="left_of")
             # Above
            if c1y < c2y:
                G.add_edge(original_i, original_j, relation="above")
    return nx.readwrite.json_graph.node_link_data(G)

# Modified extract_topo_features to use caching and CuPy
def extract_topo_features(gray_img_array: np.ndarray):
    """
    Extracts topological features (Betti numbers, persistence) using CubicalPersistence,
    with caching and optional CuPy transfer for initial data handling.
    - gray_img_array: Grayscale image (H, W)
    Returns a dictionary of topological features.
    """
    return compute_topo_cached(gray_img_array)


def extract_shape_descriptors(mask: np.ndarray):
    """
    Extracts shape descriptors (Hu Moments) from a binary mask.
    Returns a dictionary with Hu Moments.
    """
     # Find largest contour in the mask
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M['m00'] != 0:  # Avoid division by zero for empty contours
            hu = cv2.HuMoments(M).flatten().tolist()
            return {"hu_moments": hu}
    return {"hu_moments": []}  # Return empty list if no valid contour

# 1.1 Oriented (rotated) bounding box
def oriented_bbox(points: list[tuple[int,int]]):
    """
    Computes the minimum-area rotated rectangle for a contour polygon.
    Returns (cx, cy, w, h, angle_deg).
    """
    if not points:
        return (0, 0, 0, 0, 0)  # Return default for empty points
    pts = np.array(points, dtype=np.int32).reshape(-1,2)
    rect = cv2.minAreaRect(pts)
    (cx,cy),(w,h),ang = rect
    return (cx, cy, w, h, ang)

# 1.2 Per-stroke temporal metadata
def extract_stroke_metadata(strokes: list):
    """
    Extract order, pen-width, color for each stroke.
    Returns list of dicts: {id, order, pen_width, pen_color}.
    """
    meta = []
    for idx, s in enumerate(strokes):
        meta.append({
            "id": idx,
            "order": getattr(s, "order", idx),
            "pen_width": getattr(s, "width", None),
            "pen_color": getattr(s, "color", None)
        })
    return meta

# 1.3 Skeleton & junction keypoints
def extract_skeleton_keypoints(mask: np.ndarray):
    """
    Skeletonize mask & find end/intersection keypoints.
    Returns keypoints list of (x,y).
    """
    
    if mask.sum() == 0:  # Handle empty mask
        return []
    ske = morph.skeletonize(mask>0)
    coords = np.column_stack(np.where(ske))
    
    keypoints = []
     # Iterate through skeleton pixels to find endpoints and junctions
     # A pixel (y,x) is an endpoint if it has exactly one neighbor in the 3x3 window
     # A pixel (y,x) is a junction if it has more than two neighbors in the 3x3 window
    for y,x in coords:
        neighbors = 0
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < ske.shape[0] and 0 <= nx < ske.shape[1] and ske[ny, nx]:
                    neighbors += 1
        
        if neighbors == 1 or neighbors > 2:  # Endpoint or junction
            keypoints.append((int(x), int(y)))  # Store as (x,y)
    
    return keypoints

# 1.4 Mask confidence map via SAM
def generate_sam_confidence_map(image: np.ndarray, mask: np.ndarray):
    """
    Uses SAM predictor to return per-pixel confidence for the given mask.
    Assumes predictor.set_image(image) called already.
    """
    predictor = _worker_globals['predictor']
    USE_SAM = _worker_globals['USE_SAM']

    if predictor is None or not USE_SAM:
        logger.warning("SAM predictor not initialized or enabled. Skipping SAM confidence map generation.")
        return np.zeros_like(mask, dtype=np.float32), 0.0  # Return empty map and 0 confidence
     # Find bounding box of the mask
    y_coords, x_coords = np.where(mask > 0)
    if y_coords.size == 0:  # Empty mask
        return np.zeros_like(mask, dtype=np.float32), 0.0
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    box = np.array([x_min, y_min, x_max, y_max])
     # Predict masks and scores with mixed precision
    with autocast():
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=True  # Get multiple masks to find best overlap
        )
    
    if masks is None or len(masks) == 0:
        logger.warning("SAM did not generate any masks. Returning empty confidence map.")
        return np.zeros_like(mask, dtype=np.float32), 0.0
     # Choose mask with highest IoU overlap with the ground truth mask
     # Ensure both masks are boolean for IoU calculation
    gt_mask_bool = mask > 0
    iou_scores = []
    for m in masks:
        m_bool = m > 0
        intersection = np.logical_and(gt_mask_bool, m_bool).sum()
        union = np.logical_or(gt_mask_bool, m_bool).sum()
        iou_scores.append(intersection / union if union > 0 else 0)
    
    best_mask_idx = int(np.argmax(iou_scores))
    best_mask = masks[best_mask_idx]
    best_score = float(scores[best_mask_idx]) # Convert to float for JSON serialization
     # Return confidence map: score multiplied by the best predicted mask
    confidence_map = (best_mask.astype(np.float32) * best_score)
    return confidence_map, best_score

# 1.5 Multi ‐ resolution mask pyramid
def build_mask_pyramid(mask: np.ndarray, levels: int = 3):
    """
    Builds downsampled versions of mask for scale ‐ invariant reasoning.
    Returns list of masks [full_res, half, quarter, ...].
    """
    pyramid = [mask]
    for i in range(1, levels):
         # Ensure mask is uint8 for cv2.resize
        resized_mask = cv2.resize(
            mask.astype(np.uint8),
            (mask.shape[1]//(2**i), mask.shape[0]//(2**i)),
            interpolation=cv2.INTER_NEAREST
        )
        pyramid.append(resized_mask)
    return pyramid

# 1.6 DALI Sample-Mix Augmentations (Mosaic, MixUp, CutMix)
def dali_sample_mix(image_batch, mask_batch, op_type="mosaic", p=0.5):
    """
    Placeholder for DALI-based sample-mix augmentations (Mosaic, MixUp, CutMix).
    Integrate with NVIDIA DALI pipeline for GPU-accelerated batch ops.
    """
    # TODO: Implement DALI pipeline for sample-mix ops
    # This function should be called in the main data loader loop for GPU augmentations
    # Example: Use nvidia.dali.fn.mixup, mosaic, cutmix, etc. with probability p
    # Return augmented batch and metadata
    return image_batch, mask_batch, {"op_type": op_type, "prob": p, "status": "not_implemented"}

# 1.7 Streaming Metrics Integration (W&B, Prometheus)
def log_streaming_metrics(metrics_dict, step=None):
    """
    Hook for streaming metrics to Weights & Biases and Prometheus.
    """
    # TODO: Integrate with wandb.log and prometheus_client
    # Example: wandb.log(metrics_dict, step=step)
    # Example: prometheus_client.Gauge(...).set(metrics_dict[...])
    pass

# 1.8 Batch Review UI & Authentication Hooks
def batch_review_ui_hook(batch_data, user_id=None):
    """
    Placeholder for batch review endpoint integration with Flask UI.
    """
    # TODO: Connect to Flask UI batch review endpoint (ui/validate_iou.py)
    # Example: POST batch_data to /api/batch_review with authentication
    # Use Flask-Login for user_id authentication
    return {"status": "not_implemented", "user_id": user_id}

def authenticate_user(username, password):
    """
    Placeholder for authentication logic (Flask-Login, Flask-Bcrypt).
    """
    # TODO: Integrate with Flask-Login and Flask-Bcrypt
    # Example: Check hashed password, return user session
    return {"authenticated": False, "username": username}

# 1.11 Automated Background Harvesting Hook
def harvest_backgrounds(target_dir, schedule="daily"):
    """
    Placeholder for automated background harvesting (Flickrapi, APScheduler).
    """
    # TODO: Implement background crawler using Flickrapi and schedule with APScheduler
    # Example: backgrounds downloaded to target_dir on schedule
    return {"status": "not_implemented", "target_dir": target_dir, "schedule": schedule}

# 1.9 Domain ‐ randomized augmentation metadata
def random_augment(image: np.ndarray):
    """
    Applies random jitter to line width, adds noise.
    Returns augmented image and metadata dict.
    """
    aug = image.copy()
    
     # Add Gaussian noise
    noise_std = random.uniform(0, 15)  # Random noise level
    noise = (np.random.randn(*aug.shape) * noise_std).clip(-255, 255).astype(np.int16)
    aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)
     # Simulate line thickness jitter (conceptually, by applying morphological operations)
     # This is a simplified conceptual implementation. A real one would modify strokes before rendering.
    kernel_size = random.choice([1, 3, 5])
    if random.random() < 0.5:  # Randomly apply dilation or erosion
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        aug = cv2.dilate(aug, kernel, iterations=1)
        op_type = "dilate"
    else:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        aug = cv2.erode(aug, kernel, iterations=1)
        op_type = "erode"
    return aug, {"noise_std": noise_std, "morph_op": op_type, "morph_kernel_size": kernel_size}

# 1.10 Negative ‐ flag for low ‐ IoU shapes
def is_ambiguous(mask: np.ndarray, hull_mask: np.ndarray):
    """
    Flags if mask vs convex hull IoU < threshold (e.g. 0.8).
    Takes boolean masks as input.
    """
     # Ensure inputs are boolean
    mask_bool = mask > 0
    hull_mask_bool = hull_mask > 0
    intersection = np.logical_and(mask_bool, hull_mask_bool).sum()
    union = np.logical_or(mask_bool, hull_mask_bool).sum()
    
    if union == 0:  # Avoid division by zero if both masks are empty
        return False  # Not ambiguous if nothing is present
    iou = intersection / union
    return iou < 0.8  # Threshold can be configured

# --- Albumentations Helper Functions ---
def make_sample_dict(image_np, boxes, masks, class_ids):
    """
    - image_np: H×W×3 uint8
    - boxes:  [[x_center,y_center,w,h], ...] in YOLO format (values 0–1 or pixels)
    - masks: list of H×W binary masks matching boxes order
    - class_ids: [cid, cid, ...]
    """
    return {
        'image': image_np,
        'bboxes': boxes,
        'masks': masks,
        'class_labels': class_ids,
    }

def augment_and_unpack(sample):
    """
    Applies one of Mosaic / MixUp / CopyPaste, then unpacks back into image and labels.
    """
    aug = _worker_globals['train_augs']( # Access from worker globals
        image=sample['image'],
        bboxes=sample['bboxes'],
        masks=sample['masks'],
        class_labels=sample['class_labels']
    )
    image_aug = aug['image']
    boxes_aug = aug['bboxes']
    masks_aug = aug['masks']
    cids_aug  = aug['class_labels']
    
    # Convert YOLO boxes back to pixel coords: [x_center,y_center,w,h] (normalized) to [x1,y1,x2,y2] (pixel)
    h, w = image_aug.shape[:2]
    boxes_px = []
    for xc,yc,wb,hb in boxes_aug:
        x1 = int((xc - wb/2) * w)
        y1 = int((yc - hb/2) * h)
        x2 = int((xc + wb/2) * w)
        y2 = int((yc + hb/2) * h)
        boxes_px.append([x1, y1, x2, y2])
    
    # Polygons from masks
    polys_aug = [mask_to_polygon(m) for m in masks_aug]
    
    return image_aug, boxes_px, masks_aug, polys_aug, cids_aug

# Custom Static MixUp Logic
def apply_static_mixup(sample1: dict, sample2: dict, alpha: float = 1.0):
    """
    Applies MixUp to two image samples (numpy arrays) and concatenates their labels.
    Returns a single mixed sample dictionary.
    
    Args:
        sample1 (dict): Dictionary for the first image, with keys 'image', 'bboxes', 'masks', 'class_labels'.
        sample2 (dict): Dictionary for the second image, with keys 'image', 'bboxes', 'masks', 'class_labels'.
        alpha (float): Beta distribution parameter for lambda.
    
    Returns:
        dict: A new sample dictionary with mixed image and concatenated labels.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    img1 = sample1['image'].astype(np.float32)
    img2 = sample2['image'].astype(np.float32)

    # Ensure images are the same size for blending
    # If not, resize one to match the other or both to a common size
    if img1.shape != img2.shape:
        # For simplicity, resize img2 to match img1. You might want a more sophisticated strategy.
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)
        logger.warning(f"Resized second image for MixUp from {sample2['image'].shape} to {img1.shape}. "
                       "Ensure consistent image sizes for better MixUp results.")

    # Blend images
    mixed_image = (lam * img1 + (1 - lam) * img2).astype(np.uint8)

    # Concatenate labels
    mixed_bboxes = sample1['bboxes'] + sample2['bboxes']
    mixed_masks = sample1['masks'] + sample2['masks']
    mixed_class_labels = sample1['class_labels'] + sample2['class_labels']

    # Recalculate polygons from mixed masks (or from combined original polygons if preferred)
    # For simplicity, we'll re-derive from the new masks.
    # Note: If masks overlap significantly after concatenation, this might not be ideal.
    # A more robust approach for MixUp with segmentation is complex and often involves
    # handling overlapping masks carefully. Here, we're just concatenating.
    
    # Combine masks into a single mask for polygon derivation
    combined_mask_for_polys = np.zeros(mixed_image.shape[:2], dtype=np.uint8)
    for m in mixed_masks:
        # Ensure mask 'm' is the same size as mixed_image, resize if necessary
        if m.shape != mixed_image.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (mixed_image.shape[1], mixed_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        combined_mask_for_polys = np.bitwise_or(combined_mask_for_polys, m)

    mixed_polygons = mask_to_polygon(combined_mask_for_polys) # Derive one large polygon from combined mask

    return {
        'image': mixed_image,
        'bboxes': mixed_bboxes,
        'masks': mixed_masks,
        'class_labels': mixed_class_labels,
        'mixup_lambda': lam # Store lambda as metadata
    }, mixed_polygons # Return mixed_polygons separately for direct use


# --- Core Dataset Preparation Functions ---
def find_problem_dirs(src_root: str) -> list[tuple[str, float]]:
    """
    Recursively finds all subdirectories under src_root that contain both '0/' and '1/'
    subdirectories, indicating a Bongard problem folder.
    Returns a list of (problem_path, heuristic_complexity_score) tuples.
    """
    problems_with_scores = []
    logger.info(f"Searching for problem directories under: {src_root}")
    for root, dirs, _ in os.walk(src_root):
        if "0" in dirs and "1" in dirs:
            problem_path = root
            # Heuristic complexity score: number of images in the problem
            num_pos_images = len(glob.glob(os.path.join(problem_path, "1", "*.png")))
            num_neg_images = len(glob.glob(os.path.join(problem_path, "0", "*.png")))
            heuristic_score = num_pos_images + num_neg_images
            problems_with_scores.append((problem_path, heuristic_score))
    
    # Sort problems by complexity score (descending)
    problems_with_scores.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Found {len(problems_with_scores)} problem directories, sorted by heuristic complexity.")
    return problems_with_scores

def split_problems(problems_with_scores: list[tuple[str, float]], frac: float, seed: int = 42) -> tuple[list[str], list[str]]:
    """
    Splits a list of (problem_path, score) tuples into training and validation sets,
    preserving the order (which is already sorted by complexity).
    """
    if not (0.0 <= frac <= 1.0):
        raise ValueError("frac must be between 0.0 and 1.0")
    
    # No need to shuffle if already sorted by complexity for curriculum learning
    n_train = int(len(problems_with_scores) * frac)
    
    train_ps = [p[0] for p in problems_with_scores[:n_train]]
    val_ps = [p[0] for p in problems_with_scores[n_train:]]
    
    logger.info(f"Splitting {len(problems_with_scores)} problems: {len(train_ps)} for training, {len(val_ps)} for validation.")
    return train_ps, val_ps

def write_yolo_boxes(boxes: list[list[float]], img_shape: tuple[int, int], out_txt: str) -> None:
    """
    Writes bounding box coordinates to a YOLO-format .txt file.
    """
    h, w = img_shape
    lines = []
    for x1, y1, x2, y2, cid in boxes:
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{int(cid)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    try:
        with open(out_txt, "w") as f:
            f.write("\n".join(lines))
    except IOError as e:
        logger.error(f"Could not write label file {out_txt}: {e}")

def label_with_bongard(img_path: str, prog) -> list[list[float]]:
    """
    Extracts exact stroke bounding boxes from a BongardProblem object.
    """
    out_boxes = []
    CLASS_ID = _worker_globals['CLASS_ID'] # Access from worker globals
    for stroke in prog.strokes:
        x1, y1, x2, y2 = stroke.bounding_box()
        shape_type = stroke.shape_type
        cid = CLASS_ID.get(shape_type, CLASS_ID['unknown_shape'])
        if cid == CLASS_ID['unknown_shape']:
            logger.warning(f"Unknown shape type '{shape_type}' encountered in Bongard problem for {img_path}. "
                            f"Assigning to 'unknown_shape' class ID {CLASS_ID['unknown_shape']}.")
        out_boxes.append([x1, y1, x2, y2, cid])
    return out_boxes

def _process_single_subdir(args_tuple) -> None:
    """
    Processes all PNG images in a source directory for a single worker.
    This function is designed to be called by multiprocessing.Pool.
    
    Args:
        args_tuple (tuple): A tuple containing:
            src_dir (str): The source directory containing PNG images (e.g., 'problem_path/0' or 'problem_path/1').
            dst_img_dir_orig (str): The destination directory for copied original images.
            dst_lbl_dir_orig (str): The destination directory for original YOLO label files.
            label_base_paths_orig (dict): Dictionary of base Path objects for different original label types.
            dst_img_dir_aug (str): The destination directory for copied augmented images.
            dst_lbl_dir_aug (str): The destination directory for augmented YOLO label files.
            label_base_paths_aug (dict): Dictionary of base Path objects for different augmented label types.
            current_split (str): The current split ('train' or 'val').
    """
    # Access global resources initialized by _worker_init
    CONFIG = _worker_globals['CONFIG']
    CLASS_ID = _worker_globals['CLASS_ID']
    USE_PROGRAM = _worker_globals['USE_PROGRAM']
    BongardProblem = _worker_globals.get('BongardProblem')
    USE_SAM = _worker_globals['USE_SAM']
    predictor = _worker_globals['predictor']
    mask_generator = _worker_globals['mask_generator']
    HAS_EMBED_MODEL = _worker_globals['HAS_EMBED_MODEL']
    embed_model = _worker_globals['embed_model']
    HAS_OPENAI = _worker_globals['HAS_OPENAI']
    client = _worker_globals['client']
    save_mask_png = _worker_globals['save_mask_png']
    sam_masks_to_yolo = _worker_globals['sam_masks_to_yolo']
    generate_relation_graph_sam = _worker_globals['generate_relation_graph']
    get_symbolic_labels = _worker_globals['get_symbolic_labels']
    overlay_symbolic_debugger = _worker_globals['overlay_symbolic_debugger']
    generate_reasoning_chain = _worker_globals['generate_reasoning_chain']
    train_augs = _worker_globals['train_augs']

    src_dir, dst_img_dir_orig, dst_lbl_dir_orig, label_base_paths_orig, \
    dst_img_dir_aug, dst_lbl_dir_aug, label_base_paths_aug, current_split = args_tuple

    os.makedirs(dst_img_dir_orig, exist_ok=True)
    os.makedirs(dst_lbl_dir_orig, exist_ok=True)
    os.makedirs(dst_img_dir_aug, exist_ok=True) # Create augmented image dir
    os.makedirs(dst_lbl_dir_aug, exist_ok=True) # Create augmented label dir

    # Only proceed if programmatic labeling is enabled
    if not USE_PROGRAM or BongardProblem is None:
        logger.error(f"Worker processing {src_dir}: Programmatic labeling is not enabled. Skipping.")
        return

    prob_folder = Path(src_dir).parents[1]
    pos_dir = prob_folder / "1"
    neg_dir = prob_folder / "0"
    pos_images_glob = sorted(glob.glob(str(pos_dir / "*.png")))
    neg_images_glob = sorted(glob.glob(str(neg_dir / "*.png")))
    
    # This BongardProblem instance is lightweight and can be created per problem folder
    prob = BongardProblem(pos_images_glob, neg_images_glob)
    pos_basenames = [os.path.basename(p) for p in pos_images_glob]
    pos_rules = prob.get_positive_rules() or []
    neg_basenames = [os.path.basename(p) for p in neg_images_glob]
    neg_rules = prob.get_negative_rules() or []
    pos_map = {bn: rule for bn, rule in zip(pos_basenames, pos_rules)}
    neg_map = {bn: rule for bn, rule in zip(neg_basenames, neg_rules)}
    is_pos = Path(src_dir).name == "1"
    prog_map = pos_map if is_pos else neg_map

    # Get all image paths in the current source directory for potential MixUp partners
    all_img_paths_in_src_dir = glob.glob(os.path.join(src_dir, "*.png"))

    for img_path_str in glob.glob(os.path.join(src_dir, "*.png")):
        img_path = Path(img_path_str)
        stem = img_path.stem

        # --- Process ORIGINAL data ---
        dst_image_orig = Path(dst_img_dir_orig) / f"{stem}.png"
        dst_label_orig = Path(dst_lbl_dir_orig) / f"{stem}.txt"

        try:
            shutil.copy(img_path, dst_image_orig)
        except IOError as e:
            logger.error(f"Could not copy original image {img_path} to {dst_image_orig}: {e}")
            continue

        img_np_orig = cv2.imread(str(img_path))
        if img_np_orig is None:
            logger.error(f"Could not read original image {img_path}. Skipping this image.")
            continue
        img_shape_orig = img_np_orig.shape
        gray_img_np_orig = cv2.cvtColor(img_np_orig, cv2.COLOR_BGR2GRAY)
        img_rgb_orig = cv2.cvtColor(img_np_orig, cv2.COLOR_BGR2RGB) # For SAM input

        boxes_orig = []
        combined_mask_orig = np.zeros(img_shape_orig[:2], dtype=np.uint8)
        combined_polygon_pts_orig = []
        matched_program_orig = None
        current_image_rule_text = "" # Initialize empty rule text
        image_complexity_score = 0.0 # Initialize image complexity score

        try:
            base = img_path.name
            p = prog_map.get(base, None)
            if p is None:
                logger.warning(f"{base}: No matching programmatic program found. Falling back to contour-based box and mask generation for original image.")
                _, binary_img = cv2.threshold(gray_img_np_orig, 1, 255, cv2.THRESH_BINARY)
                cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cid_for_unknown = CLASS_ID['unknown_shape']
                for c in cnts:
                    x, y, w, h = cv2.boundingRect(c)
                    boxes_orig.append([x, y, x + w, y + h, cid_for_unknown])
                combined_mask_orig = np.zeros(img_shape_orig[:2], dtype=np.uint8)
                cv2.drawContours(combined_mask_orig, cnts, -1, 1, cv2.FILLED)
                combined_polygon_pts_orig = mask_to_polygon(combined_mask_orig)
            else:
                matched_program_orig = p
                boxes_orig = label_with_bongard(str(img_path), p)
                combined_mask_orig = stroke_mask(img_shape_orig, matched_program_orig.strokes)
                combined_polygon_pts_orig = mask_to_polygon(combined_mask_orig)
                # Extract the rule text for the current image
                if is_pos:
                    idx = pos_basenames.index(base)
                    current_image_rule_text = pos_rules[idx] if idx < len(pos_rules) else ""
                else:
                    idx = neg_basenames.index(base)
                    current_image_rule_text = neg_rules[idx] if idx < len(neg_rules) else ""

            if not boxes_orig:
                logger.warning(f"No boxes found for original image {img_path}. This image will have an empty label file.")
        except Exception as e:
            logger.error(f"Error during original labeling for {img_path}: {e}. Skipping additional label generation for this image.")
            boxes_orig = []
            matched_program_orig = None
            current_image_rule_text = ""

        # Write original YOLO boxes
        write_yolo_boxes(boxes_orig, img_shape_orig[:2], str(dst_label_orig))

        # Initialize SAM confidence and mask entropy for complexity score
        sam_confidence_score = 0.0
        mask_entropy = 0.0

        # Generate & Save Additional ORIGINAL Labels
        if combined_mask_orig.sum() > 0:
            # Mask
            mask_path_orig = label_base_paths_orig["masks"] / current_split / f"{stem}.png"
            try:
                _worker_globals['save_mask_png'](combined_mask_orig, str(mask_path_orig), colorize=True)
            except Exception as e:
                logger.error(f"Error writing original mask for {img_path}: {e}")
            # Polygons
            poly_path_orig = label_base_paths_orig["polygons"] / current_split / f"{stem}.json"
            try:
                with open(poly_path_orig, "w") as f: json.dump(combined_polygon_pts_orig, f)
            except IOError as e:
                logger.error(f"Could not write original polygon data to {poly_path_orig}: {e}")
            # Topological Summary (using optimized function)
            topo_features_orig = compute_topo_cached(gray_img_np_orig) # Call the cached version
            topo_path_orig = label_base_paths_orig["topo"] / current_split / f"{stem}.json"
            try:
                with open(topo_path_orig, "w") as f: json.dump(topo_features_orig, f, indent=2)
            except IOError as e:
                logger.error(f"Could not write original topological features to {topo_path_orig}: {e}")
            # Shape Descriptors
            desc_features_orig = extract_shape_descriptors(combined_mask_orig)
            desc_path_orig = label_base_paths_orig["descriptors"] / current_split / f"{stem}.json"
            try:
                with open(desc_path_orig, "w") as f: json.dump(desc_features_orig, f, indent=2)
            except IOError as e:
                logger.error(f"Could not write original shape descriptors to {desc_path_orig}: {e}")
            # LLM Caption
            if HAS_OPENAI and client:
                caption_text_orig = generate_rule_explanation([str(img_path)], []) # Using generate_rule_explanation for single image captioning
                cap_path_orig = label_base_paths_orig["captions"] / current_split / f"{stem}.txt"
                try:
                    with open(cap_path_orig, "w") as f: f.write(caption_text_orig)
                except IOError as e:
                    logger.error(f"Could not write original caption to {cap_path_orig}: {e}")
            # Oriented Bounding Box
            obox_orig = oriented_bbox(combined_polygon_pts_orig)
            obox_path_orig = label_base_paths_orig["oriented"] / current_split / f"{stem}.json" 
            try:
                with open(obox_path_orig, "w") as f: json.dump({"obox": obox_orig}, f, indent=2)
            except IOError as e:
                logger.error(f"Could not write original oriented bounding box to {obox_path_orig}: {e}")
            # Skeleton Keypoints
            kpts_orig = extract_skeleton_keypoints(combined_mask_orig)
            kpt_path_orig = label_base_paths_orig["keypoints"] / current_split / f"{stem}.json"
            try:
                with open(kpt_path_orig, "w") as f: json.dump(kpts_orig, f, indent=2)
            except IOError as e:
                logger.error(f"Could not write original keypoints to {kpt_path_orig}: {e}")
            # SAM confidence map (using SamPredictor)
            if USE_SAM and predictor:
                try:
                    predictor.set_image(img_rgb_orig) # Set image for predictor
                    conf_map_orig, sam_confidence_score = generate_sam_confidence_map(img_rgb_orig, combined_mask_orig)
                    conf_path_orig = label_base_paths_orig["sam_conf"] / current_split / f"{stem}.npy"
                    np.save(str(conf_path_orig), conf_map_orig)
                except Exception as e:
                    logger.error(f"Error generating/saving original SAM confidence map for {img_path}: {e}")
            # Mask Pyramid
            pyr_orig = build_mask_pyramid(combined_mask_orig, levels=4)
            for lvl, m in enumerate(pyr_orig):
                pyr_path_orig = label_base_paths_orig["pyramid"] / current_split / f"{stem}_lvl{lvl}.png"
                try:
                    cv2.imwrite(str(pyr_path_orig), (m*255).astype(np.uint8))
                except Exception as e:
                    logger.error(f"Error writing original mask pyramid level {lvl} to {pyr_path_orig}: {e}")
            # Fourier Descriptors (using optimized function)
            fd_orig = compute_fourier_descriptors(combined_polygon_pts_orig, n_coeffs=32)
            fd_path_orig = label_base_paths_orig["fourier"] / current_split / f"{stem}.json"
            try:
                with open(fd_path_orig,"w") as f: json.dump(fd_orig, f, indent=2)
            except IOError as e:
                logger.error(f"Could not write original fourier descriptors to {fd_path_orig}: {e}")
            
            # Shape Embeddings (batched processing)
            patches_to_embed_orig = []
            valid_box_indices_orig = []
            for box_idx, (x1, y1, x2, y2, cid) in enumerate(boxes_orig): # boxes_orig is [x1,y1,x2,y2,cid]
                x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(img_shape_orig[1], x2)), int(min(img_shape_orig[0], y2))
                if x2 > x1 and y2 > y1:
                    patch = img_np_orig[y1:y2, x1:x2]
                    patches_to_embed_orig.append(patch)
                    valid_box_indices_orig.append(box_idx)
                else:
                    logger.warning(f"Skipping original embedding for {img_path} shape {box_idx}: Invalid patch coordinates ({x1},{y1},{x2},{y2}).")

            if HAS_EMBED_MODEL and embed_model and patches_to_embed_orig:
                try:
                    embeddings_batch_orig = embed_patches_batched(patches_to_embed_orig)
                    for i, emb in enumerate(embeddings_batch_orig):
                        original_box_idx = valid_box_indices_orig[i]
                        emb_path_orig = label_base_paths_orig["embed"] / current_split / f"{stem}_shape{original_box_idx}.json"
                        with open(emb_path_orig,"w") as f: json.dump(emb, f, indent=2)
                except Exception as e:
                    logger.error(f"Error generating/saving batched original embeddings for {img_path}: {e}")
            elif HAS_EMBED_MODEL and not patches_to_embed_orig:
                logger.info(f"No valid patches found for original embeddings in {img_path}.")
            else:
                logger.warning(f"Skipping original shape embeddings for {img_path}: Embedding model not enabled/configured.")

            # Domain-randomized Augmentation Metadata (for original image)
            _, aug_meta_orig = random_augment(img_np_orig)
            
            # Calculate mask entropy
            mask_pixels = combined_mask_orig.flatten()
            if mask_pixels.size > 0:
                counts = np.bincount(mask_pixels)
                probabilities = counts / mask_pixels.size
                mask_entropy = -np.sum(p * np.log2(p) for p in probabilities if p > 0)
            
            # Add complexity score to metadata
            num_relations = 0
            if matched_program_orig and hasattr(matched_program_orig, 'relations'):
                # This part assumes 'relations' attribute exists and is a list of relations
                # You might need to adjust based on the actual structure of your BongardProblem object
                try:
                    individual_polygons_for_relations = []
                    individual_masks_for_relations = []
                    for stroke in matched_program_orig.strokes:
                        individual_stroke_mask = stroke_mask(img_shape_orig, [stroke])
                        individual_poly = mask_to_polygon(individual_stroke_mask)
                        if individual_poly:
                            individual_polygons_for_relations.append(individual_poly)
                            individual_masks_for_relations.append(individual_stroke_mask)
                    # Compute relations to get the count
                    relations_graph = compute_relations(individual_polygons_for_relations, individual_masks_for_relations)
                    num_relations = len(relations_graph.get('links', []))
                except Exception as e:
                    logger.warning(f"Could not compute relations for complexity score: {e}")
                    num_relations = 0


            image_complexity_score = (
                0.4 * len(boxes_orig) + # Number of primitives (approximated by boxes)
                0.3 * num_relations + # Number of relations
                0.2 * mask_entropy +
                0.1 * (1 - sam_confidence_score)
            )
            aug_meta_orig["image_complexity_score"] = float(image_complexity_score) # Ensure float for JSON
            aug_meta_orig["sam_confidence_score"] = float(sam_confidence_score)
            aug_meta_orig["mask_entropy"] = float(mask_entropy)
            aug_meta_orig["num_relations_for_complexity"] = int(num_relations)


            aug_meta_path_orig = label_base_paths_orig["aug_meta"] / current_split / f"{stem}.json"
            try:
                with open(aug_meta_path_orig,"w") as f: json.dump(aug_meta_orig, f, indent=2)
            except IOError as e:
                logger.error(f"Could not write original augmentation metadata to {aug_meta_path_orig}: {e}")
            
            # Save complexity score separately for easier access
            complexity_score_path_orig = label_base_paths_orig["complexity_scores"] / current_split / f"{stem}.json"
            try:
                with open(complexity_score_path_orig, "w") as f:
                    json.dump({"complexity_score": float(image_complexity_score)}, f, indent=2)
            except IOError as e:
                logger.error(f"Could not write complexity score to {complexity_score_path_orig}: {e}")


            # Negative-flag
            hull_orig = morph.convex_hull_image(combined_mask_orig > 0).astype(np.uint8)
            flag_orig = is_ambiguous(combined_mask_orig, hull_orig)
            neg_path_orig = label_base_paths_orig["neg_flags"] / current_split / f"{stem}.json"
            try:
                with open(neg_path_orig,"w") as f: json.dump({"ambiguous": bool(flag_orig)}, f, indent=2)
            except IOError as e:
                logger.error(f"Could not write original negative flag to {neg_path_orig}: {e}")
            # Programmatic-specific labels for original (if matched_program_orig exists)
            if matched_program_orig:
                prog_path_orig = label_base_paths_orig["programs"] / current_split / f"{stem}.json"
                try:
                    export_action_program(matched_program_orig, prog_path_orig)
                except Exception as e:
                    logger.error(f"Error exporting original action program to {prog_path_orig}: {e}")
                individual_polygons_orig = []
                individual_masks_orig = [] # Store individual masks for compute_relations
                for stroke in matched_program_orig.strokes:
                    individual_stroke_mask_orig = stroke_mask(img_shape_orig, [stroke])
                    individual_poly_orig = mask_to_polygon(individual_stroke_mask_orig)
                    if individual_poly_orig:
                        individual_polygons_orig.append(individual_poly_orig)
                        individual_masks_orig.append(individual_stroke_mask_orig)
                relations_graph_orig = compute_relations(individual_polygons_orig, individual_masks_orig) # Pass individual masks
                rel_path_orig = label_base_paths_orig["relations"] / current_split / f"{stem}.json"
                try:
                    with open(rel_path_orig, "w") as f: json.dump(relations_graph_orig, f, indent=2)
                except IOError as e:
                    logger.error(f"Could not write original relations to {rel_path_orig}: {e}")
                meta_orig = extract_stroke_metadata(matched_program_orig.strokes)
                meta_path_orig = label_base_paths_orig["stroke_meta"] / current_split / f"{stem}.json"
                try:
                    with open(meta_path_orig, "w") as f: json.dump(meta_orig, f, indent=2)
                except IOError as e:
                    logger.error(f"Could not write original stroke metadata to {meta_path_orig}: {e}")
        else:
            logger.info(f"Skipping mask/polygon-dependent original labels for {img_path} (no valid mask/polygon generated).")

        # --- Generate and Save SAM-based Labels (Original Image) ---
        if USE_SAM and mask_generator:
            try:
                # Generate masks automatically using SAM
                with autocast(): # Apply mixed precision
                    sam_masks_raw = mask_generator.generate(img_rgb_orig)
                logger.info(f"SAM generated {len(sam_masks_raw)} masks for {img_path.name}.")

                # Extract boolean masks for functions that expect them
                sam_masks_boolean = [m["segmentation"] for m in sam_masks_raw]

                # Save individual SAM masks
                sam_masks_dir = label_base_paths_orig["sam_masks"] / current_split
                sam_masks_dir.mkdir(parents=True, exist_ok=True)
                for i, mask_dict in enumerate(sam_masks_raw):
                    mask_save_path = sam_masks_dir / f"{stem}_sam_mask_{i}.png"
                    _worker_globals['save_mask_png'](mask_dict["segmentation"], str(mask_save_path), colorize=True)
                
                # Convert SAM masks to YOLO format and save
                sam_yolo_labels = _worker_globals['sam_masks_to_yolo'](sam_masks_raw, img_shape_orig[:2], class_id=CLASS_ID.get('unknown_shape', 0))
                sam_yolo_path = label_base_paths_orig["sam_yolo"] / current_split / f"{stem}_sam.txt"
                sam_yolo_path.parent.mkdir(parents=True, exist_ok=True)
                with open(sam_yolo_path, "w") as f:
                    f.write("\n".join(sam_yolo_labels))
                logger.info(f"Saved SAM-generated YOLO labels to {sam_yolo_path}")

                # Generate and save relational graph from SAM masks
                sam_relations_graph = generate_relation_graph_sam(sam_masks_raw) # Use the imported function
                sam_relations_path = label_base_paths_orig["sam_relations"] / current_split / f"{stem}_sam_relations.json"
                sam_relations_path.parent.mkdir(parents=True, exist_ok=True)
                with open(sam_relations_path, "w") as f:
                    json.dump(sam_relations_graph, f, indent=2)
                logger.info(f"Saved SAM-generated relational graph to {sam_relations_path}")

                # --- Symbolic Label Fusion and Debugging ---
                if USE_PROGRAM and current_image_rule_text: # Ensure Bongard program is available
                    try:
                        symbolic_annotations = get_symbolic_labels(sam_masks_raw, img_rgb_orig, current_image_rule_text)
                        logger.info(f"Generated {len(symbolic_annotations)} symbolic annotations for {img_path.name}.")

                        # Save symbolic JSON per object
                        symbolic_dir = label_base_paths_orig["symbolic"] / current_split
                        symbolic_dir.mkdir(parents=True, exist_ok=True)
                        for i, symbolic_obj in enumerate(symbolic_annotations):
                            symbolic_path = symbolic_dir / f"{stem}_obj{i}.json"
                            with open(symbolic_path, "w") as f:
                                json.dump(symbolic_obj, f, indent=2)
                        logger.info(f"Saved symbolic annotations for {img_path.name}.")

                        # Generate and save visual debugger overlay
                        if len(sam_masks_boolean) == len(symbolic_annotations):
                            debug_image = overlay_symbolic_debugger(img_np_orig, sam_masks_boolean, symbolic_annotations)
                            debug_dir = label_base_paths_orig["visual_debug"] / current_split
                            debug_dir.mkdir(parents=True, exist_ok=True)
                            debug_path = debug_dir / f"{stem}_debug.png"
                            cv2.imwrite(str(debug_path), debug_image)
                            logger.info(f"Saved visual debugger overlay to {debug_path}.")
                        else:
                            logger.warning(f"Skipping visual debugger for {img_path.name}: Mismatch between SAM masks ({len(sam_masks_boolean)}) and symbolic annotations ({len(symbolic_annotations)}).")

                        # Generate and save reasoning chain
                        reasoning_chain_text = generate_reasoning_chain(symbolic_annotations)
                        reasoning_dir = label_base_paths_orig["reasoning_chains"] / current_split
                        reasoning_dir.mkdir(parents=True, exist_ok=True)
                        reasoning_path = reasoning_dir / f"{stem}_reasoning.txt"
                        with open(reasoning_path, "w") as f:
                            f.write(reasoning_chain_text)
                        logger.info(f"Saved reasoning chain to {reasoning_path}.")

                    except Exception as e:
                        logger.error(f"Error during symbolic label fusion or debugging for {img_path}: {e}")
                else:
                    logger.info(f"Skipping symbolic label fusion for {img_path.name}: Bongard programmatic labels or rule text not available.")

            except Exception as e:
                logger.error(f"Error during SAM-based label generation for {img_path}: {e}")
        elif USE_SAM and not mask_generator:
            logger.warning(f"SAM AutomaticMaskGenerator not available for {img_path}. Skipping SAM-based label generation.")
        else:
            logger.info(f"Skipping SAM-based label generation for {img_path}: SAM not enabled/configured.")


        # --- Process AUGMENTED data (Albumentations & Custom MixUp) ---
        # Only augment if original image and labels are valid enough to start with
        if boxes_orig and combined_mask_orig.sum() > 0:
            # --- Albumentations Augmentations ---
            dst_image_alb_aug = Path(dst_img_dir_aug) / f"{stem}_alb.png" # Differentiate augmented files
            dst_label_alb_aug = Path(dst_lbl_dir_aug) / f"{stem}_alb.txt"

            # Prepare sample for Albumentations augmentation
            h_orig, w_orig = img_np_orig.shape[:2]
            yolo_boxes_orig_norm = []
            for x1,y1,x2,y2,cid in boxes_orig:
                x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w_orig, x2)), int(min(h_orig, y2))
                if x2 > x1 and y2 > y1:
                    xc = ((x1 + x2) / 2) / w_orig
                    yc = ((y1 + y2) / 2) / h_orig
                    bw = (x2 - x1) / w_orig
                    bh = (y2 - y1) / h_orig
                    yolo_boxes_orig_norm.append([xc, yc, bw, bh])
                else:
                    logger.warning(f"Invalid original box for Albumentations input: [{x1},{y1},{x2},{y2}] for {img_path}. Skipping this box.")

            orig_masks_for_aug = []
            orig_cids_for_aug = []
            for (x1,y1,x2,y2,cid) in boxes_orig:
                x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(img_np_orig.shape[1], x2)), int(min(img_np_orig.shape[0], y2))
                if x2 > x1 and y2 > y1:
                    m = np.zeros(img_np_orig.shape[:2], dtype=np.uint8)
                    m[y1:y2, x1:x2] = 1
                    orig_masks_for_aug.append(m)
                    orig_cids_for_aug.append(cid)
                else:
                    logger.warning(f"Invalid original mask for Albumentations input: [{x1},{y1},{x2},{y2}] for {img_path}. Skipping this mask.")

            sample_for_alb_aug = make_sample_dict(img_np_orig, yolo_boxes_orig_norm, orig_masks_for_aug, orig_cids_for_aug)

            try:
                image_alb_aug, boxes_px_alb_aug, masks_alb_aug, polys_alb_aug, cids_alb_aug = augment_and_unpack(sample_for_alb_aug)

                # Save augmented image (Albumentations)
                try:
                    cv2.imwrite(str(dst_image_alb_aug), image_alb_aug)
                except Exception as e:
                    logger.error(f"Error writing Albumentations augmented image {dst_image_alb_aug}: {e}")

                # Convert augmented pixel boxes back to [x1,y1,x2,y2,cid] format for writing
                boxes_alb_aug_final = [ [*box_px, cid] for box_px, cid in zip(boxes_px_alb_aug, cids_alb_aug) ]

                # Write augmented YOLO boxes (Albumentations)
                write_yolo_boxes(boxes_alb_aug_final, image_alb_aug.shape[:2], str(dst_label_alb_aug))

                # Generate & Save Additional AUGMENTED Labels (Albumentations)
                combined_mask_alb_aug = np.bitwise_or.reduce(masks_alb_aug).astype(np.uint8) if masks_alb_aug else np.zeros(image_alb_aug.shape[:2], dtype=np.uint8)
                combined_polygon_pts_alb_aug = polys_alb_aug

                if combined_mask_alb_aug.sum() > 0:
                    mask_path_alb_aug = label_base_paths_aug["masks"] / current_split / f"{stem}_alb.png"
                    cv2.imwrite(str(mask_path_alb_aug), (combined_mask_alb_aug * 255).astype(np.uint8))
                    poly_path_alb_aug = label_base_paths_aug["polygons"] / current_split / f"{stem}_alb.json"
                    with open(poly_path_alb_aug, "w") as f: json.dump(combined_polygon_pts_alb_aug, f)
                    topo_features_alb_aug = compute_topo_cached(cv2.cvtColor(image_alb_aug, cv2.COLOR_BGR2GRAY)) # Call the cached version
                    topo_path_alb_aug = label_base_paths_aug["topo"] / current_split / f"{stem}_alb.json"
                    with open(topo_path_alb_aug, "w") as f: json.dump(topo_features_alb_aug, f, indent=2)
                    desc_features_alb_aug = extract_shape_descriptors(combined_mask_alb_aug)
                    desc_path_alb_aug = label_base_paths_aug["descriptors"] / current_split / f"{stem}_alb.json"
                    with open(desc_path_alb_aug, "w") as f: json.dump(desc_features_alb_aug, f, indent=2)
                    obox_alb_aug = oriented_bbox(combined_polygon_pts_alb_aug)
                    obox_path_alb_aug = label_base_paths_aug["oriented"] / current_split / f"{stem}_alb.json" 
                    with open(obox_path_alb_aug, "w") as f: json.dump({"obox": obox_alb_aug}, f, indent=2)
                    kpts_alb_aug = extract_skeleton_keypoints(combined_mask_alb_aug)
                    kpt_path_alb_aug = label_base_paths_aug["keypoints"] / current_split / f"{stem}_alb.json"
                    with open(kpt_path_alb_aug, "w") as f: json.dump(kpts_alb_aug, f, indent=2)
                    pyr_alb_aug = build_mask_pyramid(combined_mask_alb_aug, levels=4)
                    for lvl, m in enumerate(pyr_alb_aug):
                        pyr_path_alb_aug = label_base_paths_aug["pyramid"] / current_split / f"{stem}_alb_lvl{lvl}.png"
                        cv2.imwrite(str(pyr_path_alb_aug), (m*255).astype(np.uint8))
                    fd_alb_aug = compute_fourier_descriptors(combined_polygon_pts_alb_aug, n_coeffs=32)
                    fd_path_alb_aug = label_base_paths_aug["fourier"] / current_split / f"{stem}_alb.json"
                    with open(fd_path_alb_aug,"w") as f: json.dump(fd_alb_aug, f, indent=2)
                    
                    patches_to_embed_alb_aug = []
                    valid_box_indices_alb_aug = []
                    for box_idx, (x1, y1, x2, y2, cid) in enumerate(boxes_alb_aug_final):
                        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(image_alb_aug.shape[1], x2)), int(min(image_alb_aug.shape[0], y2))
                        if x2 > x1 and y2 > y1:
                            patch_alb_aug = image_alb_aug[y1:y2, x1:x2]
                            patches_to_embed_alb_aug.append(patch_alb_aug)
                            valid_box_indices_alb_aug.append(box_idx)
                    if HAS_EMBED_MODEL and embed_model and patches_to_embed_alb_aug:
                        embeddings_batch_alb_aug = embed_patches_batched(patches_to_embed_alb_aug)
                        for i, emb in enumerate(embeddings_batch_alb_aug):
                            original_box_idx = valid_box_indices_alb_aug[i]
                            emb_path_alb_aug = label_base_paths_aug["embed"] / current_split / f"{stem}_alb_shape{original_box_idx}.json"
                            with open(emb_path_alb_aug,"w") as f: json.dump(emb, f, indent=2)
                    _, aug_meta_alb_aug = random_augment(image_alb_aug) # Apply random augment to the already alb augmented image
                    aug_meta_path_alb_aug = label_base_paths_aug["aug_meta"] / current_split / f"{stem}_alb.json"
                    with open(aug_meta_path_alb_aug,"w") as f: json.dump(aug_meta_alb_aug, f, indent=2)
                    hull_alb_aug = morph.convex_hull_image(combined_mask_alb_aug > 0).astype(np.uint8)
                    flag_alb_aug = is_ambiguous(combined_mask_alb_aug, hull_alb_aug)
                    neg_path_alb_aug = label_base_paths_aug["neg_flags"] / current_split / f"{stem}_alb.json"
                    with open(neg_path_alb_aug,"w") as f: json.dump({"ambiguous": bool(flag_alb_aug)}, f, indent=2)
                else:
                    logger.info(f"Skipping mask/polygon-dependent Albumentations augmented labels for {img_path} (no valid mask/polygon generated).")

            except Exception as e:
                logger.error(f"Error during Albumentations augmentation or saving augmented data for {img_path}: {e}")
            
            # --- Custom Static MixUp Augmentation ---
            # Randomly select a second image from the same source directory for MixUp
            # Ensure it's not the current image and is a valid image path
            mixup_partner_paths = [p for p in all_img_paths_in_src_dir if Path(p) != img_path]
            
            if mixup_partner_paths:
                mixup_partner_path = Path(random.choice(mixup_partner_paths))
                img_np_partner = cv2.imread(str(mixup_partner_path))
                if img_np_partner is None:
                    logger.warning(f"Could not read MixUp partner image {mixup_partner_path}. Skipping MixUp for {img_path}.")
                else:
                    # Get labels for the MixUp partner (simplified: assume programmatic labels for partner)
                    partner_stem = mixup_partner_path.stem
                    p_partner = prog_map.get(mixup_partner_path.name, None)
                    if p_partner is None:
                        logger.warning(f"No programmatic program for MixUp partner {mixup_partner_path.name}. Skipping MixUp for {img_path}.")
                    else:
                        boxes_partner = label_with_bongard(str(mixup_partner_path), p_partner)
                        combined_mask_partner = stroke_mask(img_np_partner.shape, p_partner.strokes)
                        
                        # Convert partner's pixel boxes to YOLO normalized format for make_sample_dict
                        h_partner, w_partner = img_np_partner.shape[:2]
                        yolo_boxes_partner_norm = []
                        for x1,y1,x2,y2,cid in boxes_partner:
                            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w_partner, x2)), int(min(h_partner, y2))
                            if x2 > x1 and y2 > y1:
                                xc = ((x1 + x2) / 2) / w_partner
                                yc = ((y1 + y2) / 2) / h_partner
                                bw = (x2 - x1) / w_partner
                                bh = (y2 - y1) / h_partner
                                yolo_boxes_partner_norm.append([xc, yc, bw, bh])

                        orig_masks_for_mixup_partner = []
                        orig_cids_for_mixup_partner = []
                        for (x1,y1,x2,y2,cid) in boxes_partner:
                            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(img_np_partner.shape[1], x2)), int(min(img_np_partner.shape[0], y2))
                            if x2 > x1 and y2 > y1:
                                m = np.zeros(img_np_partner.shape[:2], dtype=np.uint8)
                                m[y1:y2, x1:x2] = 1
                                orig_masks_for_mixup_partner.append(m)
                                orig_cids_for_mixup_partner.append(cid)

                        sample1_for_mixup = make_sample_dict(img_np_orig, yolo_boxes_orig_norm, orig_masks_for_aug, orig_cids_for_aug)
                        sample2_for_mixup = make_sample_dict(img_np_partner, yolo_boxes_partner_norm, orig_masks_for_mixup_partner, orig_cids_for_mixup_partner)

                        try:
                            mixed_sample_dict, mixed_polygons_from_masks = apply_static_mixup(sample1_for_mixup, sample2_for_mixup, alpha=1.0)
                            image_mixup = mixed_sample_dict['image']
                            boxes_mixup_norm = mixed_sample_dict['bboxes'] # These are YOLO normalized
                            masks_mixup = mixed_sample_dict['masks']
                            cids_mixup = mixed_sample_dict['class_labels']
                            mixup_lambda = mixed_sample_dict['mixup_lambda']

                            # Convert YOLO normalized boxes back to pixel for saving
                            h_mixup, w_mixup = image_mixup.shape[:2]
                            boxes_px_mixup = []
                            for xc,yc,wb,hb in boxes_mixup_norm:
                                x1 = int((xc - wb/2) * w_mixup)
                                y1 = int((yc - hb/2) * h_mixup)
                                x2 = int((xc + wb/2) * w_mixup)
                                y2 = int((yc + hb/2) * h_mixup)
                                boxes_px_mixup.append([x1, y1, x2, y2])
                            
                            boxes_mixup_final = [ [*box_px, cid] for box_px, cid in zip(boxes_px_mixup, cids_mixup) ]

                            # Save MixUp augmented image and labels
                            dst_image_mixup = Path(dst_img_dir_aug) / f"{stem}_mixup.png"
                            dst_label_mixup = Path(dst_lbl_dir_aug) / f"{stem}_mixup.txt"
                            cv2.imwrite(str(dst_image_mixup), image_mixup)
                            write_yolo_boxes(boxes_mixup_final, image_mixup.shape[:2], str(dst_label_mixup))

                            # Save other MixUp augmented labels
                            combined_mask_mixup = np.bitwise_or.reduce(masks_mixup).astype(np.uint8) if masks_mixup else np.zeros(image_mixup.shape[:2], dtype=np.uint8)
                            
                            if combined_mask_mixup.sum() > 0:
                                mask_path_mixup = label_base_paths_aug["masks"] / current_split / f"{stem}_mixup.png"
                                cv2.imwrite(str(mask_path_mixup), (combined_mask_mixup * 255).astype(np.uint8))
                                poly_path_mixup = label_base_paths_aug["polygons"] / current_split / f"{stem}_mixup.json"
                                with open(poly_path_mixup, "w") as f: json.dump(mixed_polygons_from_masks, f) # Use the derived polygons
                                topo_features_mixup = compute_topo_cached(cv2.cvtColor(image_mixup, cv2.COLOR_BGR2GRAY)) # Call the cached version
                                topo_path_mixup = label_base_paths_aug["topo"] / current_split / f"{stem}_mixup.json"
                                with open(topo_path_mixup, "w") as f: json.dump(topo_features_mixup, f, indent=2)
                                desc_features_mixup = extract_shape_descriptors(combined_mask_mixup)
                                desc_path_mixup = label_base_paths_aug["descriptors"] / current_split / f"{stem}_mixup.json"
                                with open(desc_path_mixup, "w") as f: json.dump(desc_features_mixup, f, indent=2)
                                obox_mixup = oriented_bbox(mixed_polygons_from_masks)
                                obox_path_mixup = label_base_paths_aug["oriented"] / current_split / f"{stem}_mixup.json" 
                                with open(obox_path_mixup, "w") as f: json.dump({"obox": obox_mixup}, f, indent=2)
                                kpts_mixup = extract_skeleton_keypoints(combined_mask_mixup)
                                kpt_path_mixup = label_base_paths_aug["keypoints"] / current_split / f"{stem}_mixup.json"
                                with open(kpt_path_mixup, "w") as f: json.dump(kpts_mixup, f, indent=2)
                                pyr_mixup = build_mask_pyramid(combined_mask_mixup, levels=4)
                                for lvl, m in enumerate(pyr_mixup):
                                    pyr_path_mixup = label_base_paths_aug["pyramid"] / current_split / f"{stem}_mixup_lvl{lvl}.png"
                                    cv2.imwrite(str(pyr_path_mixup), (m*255).astype(np.uint8))
                                fd_mixup = compute_fourier_descriptors(mixed_polygons_from_masks, n_coeffs=32)
                                fd_path_mixup = label_base_paths_aug["fourier"] / current_split / f"{stem}_mixup.json"
                                with open(fd_path_mixup,"w") as f: json.dump(fd_mixup, f, indent=2)

                                patches_to_embed_mixup = []
                                valid_box_indices_mixup = []
                                for box_idx, (x1, y1, x2, y2, cid) in enumerate(boxes_mixup_final):
                                    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(image_mixup.shape[1], x2)), int(min(image_mixup.shape[0], y2))
                                    if x2 > x1 and y2 > y1:
                                        patch_mixup = image_mixup[y1:y2, x1:x2]
                                        patches_to_embed_mixup.append(patch_mixup)
                                        valid_box_indices_mixup.append(box_idx)
                                if HAS_EMBED_MODEL and embed_model and patches_to_embed_mixup:
                                    embeddings_batch_mixup = embed_patches_batched(patches_to_embed_mixup)
                                    for i, emb in enumerate(embeddings_batch_mixup):
                                        original_box_idx = valid_box_indices_mixup[i]
                                        emb_path_mixup = label_base_paths_aug["embed"] / current_split / f"{stem}_mixup_shape{original_box_idx}.json"
                                        with open(emb_path_mixup,"w") as f: json.dump(emb, f, indent=2)
                                
                                # Store MixUp specific metadata
                                aug_meta_mixup = {"mixup_lambda": float(mixup_lambda)}
                                aug_meta_path_mixup = label_base_paths_aug["aug_meta"] / current_split / f"{stem}_mixup.json"
                                with open(aug_meta_path_mixup,"w") as f: json.dump(aug_meta_mixup, f, indent=2)
                                
                                hull_mixup = morph.convex_hull_image(combined_mask_mixup > 0).astype(np.uint8)
                                flag_mixup = is_ambiguous(combined_mask_mixup, hull_mixup)
                                neg_path_mixup = label_base_paths_aug["neg_flags"] / current_split / f"{stem}_mixup.json"
                                with open(neg_path_mixup,"w") as f: json.dump({"ambiguous": bool(flag_mixup)}, f, indent=2)
                            else:
                                logger.info(f"Skipping mask/polygon-dependent MixUp augmented labels for {img_path} (no valid mask/polygon generated).")
                        except Exception as e:
                            logger.error(f"Error during MixUp augmentation or saving MixUp data for {img_path}: {e}")
            else:
                logger.info(f"No suitable MixUp partner found for {img_path}. Skipping MixUp for this image.")
        else:
            logger.info(f"Skipping all augmentations for {img_path} due to invalid original labels (no boxes or mask).")


def prepare_dataset(src_root: str, out_root: str, frac: float, shard_id: int = 0, num_shards: int = 1) -> None:
    """
    Orchestrates the entire dataset preparation process:
      1. Discovers all Bongard problem folders and assigns complexity scores.
      2. Splits these problems into training and validation sets.
      3. Creates necessary output directories for all label types (original and augmented).
      4. Iterates through each problem's '0/' and '1/' subdirectories in parallel,
         auto-labels images, and copies them to the final YOLO dataset structure.
         Also generates masks, action programs, spatial relations, topological features,
         shape descriptors, and optional LLM captions for both original and augmented data.
    Args:
        src_root (str): The root directory of your ShapeBongard_V2 data
                        (e.g., the parent of 'hd/images', 'bd/images', 'ff/images').
        out_root (str): The desired output root directory for the prepared dataset.
                        This is where 'images/', 'labels/', and all new label type
                        subdirectories will be created for both original and augmented data.
        frac (float): The fraction of problems to use for the training set.
        shard_id (int): The ID of the current shard (0-indexed).
        num_shards (int): The total number of shards.
    """
    # Check if programmatic labeling is enabled before starting the process
    # This check is now done per worker in _worker_init, but a top-level check is still good
    try:
        from bongard import BongardProblem # Attempt import at top level
    except ImportError:
        logger.error("Bongard library not found. Programmatic labeling is REQUIRED for strict mode. Aborting.")
        return

    problems_with_scores = find_problem_dirs(src_root)
    if not problems_with_scores:
        logger.error(f"No problem directories found under {src_root}. Please check the --src path.")
        return
    
    # 6. Distributed Execution: Shard the problems list (already sorted by complexity)
    if num_shards > 1:
        if not (0 <= shard_id < num_shards):
            raise ValueError(f"shard_id ({shard_id}) must be between 0 and num_shards-1 ({num_shards-1}).")
        
        # Split into shards based on the pre-sorted list
        # Using numpy.array_split for even distribution
        all_problem_paths_sorted = [p[0] for p in problems_with_scores]
        shards = np.array_split(all_problem_paths_sorted, num_shards)
        problems_for_this_shard = shards[shard_id].tolist() # Convert back to list
        logger.info(f"Processing shard {shard_id + 1}/{num_shards} with {len(problems_for_this_shard)} problems.")
    else:
        problems_for_this_shard = [p[0] for p in problems_with_scores] # Take all paths
        logger.info("Processing all problems in a single shard.")

     # Use CONFIG['seed'] for reproducibility if available, otherwise default
    train_ps, val_ps = split_problems([(p, 0) for p in problems_for_this_shard], frac, seed=CONFIG.get('seed', 42)) # Pass dummy scores for split_problems

     # Define the base paths for images and labels within the output root
    out_images_dir = os.path.join(out_root, 'images')
    out_labels_dir = os.path.join(out_root, 'labels')

    # Define augmented paths
    out_images_aug_dir = os.path.join(out_root, 'images_aug')
    out_labels_aug_dir = os.path.join(out_root, 'labels_aug')

     # Directories for new label types - created once at startup
    LABEL_DIRS_MAP = CONFIG['label_directories'] 
    label_base_paths_original = {}
    label_base_paths_augmented = {}

    for name, folder in LABEL_DIRS_MAP.items():
        # Original paths
        base_path_orig = Path(out_root) / folder
        label_base_paths_original[name] = base_path_orig
        (base_path_orig / "train").mkdir(parents=True, exist_ok=True)
        (base_path_orig / "val").mkdir(parents=True, exist_ok=True)

        # Augmented paths
        base_path_aug = Path(out_root) / f"{folder}_aug" # e.g., masks_aug, polygons_aug
        label_base_paths_augmented[name] = base_path_aug
        (base_path_aug / "train").mkdir(parents=True, exist_ok=True)
        (base_path_aug / "val").mkdir(parents=True, exist_ok=True)

    logger.info(f"Created directories for all original and augmented label types under {out_root}.")
    
     # Set to track processed problem folders for rule explanations (only for original data)
    processed_problem_folders = set()
    
    # Prepare arguments for multiprocessing pool
    all_subdir_args = []
    for phase, plist in [("train", train_ps), ("val", val_ps)]:
        dst_imgs_phase_orig = os.path.join(out_images_dir, phase)
        dst_lbls_phase_orig = os.path.join(out_labels_dir, phase)
        dst_imgs_phase_aug = os.path.join(out_images_aug_dir, phase)
        dst_lbls_phase_aug = os.path.join(out_labels_aug_dir, phase)

        for p_folder in plist:
            # Generate rule explanation once per problem folder (only for original)
            # This part runs on the main process before pool starts
            if HAS_OPENAI and client and p_folder not in processed_problem_folders:
                try:
                    pos_images_for_rule = sorted(glob.glob(str(Path(p_folder) / "1" / "*.png")))
                    neg_images_for_rule = sorted(glob.glob(str(Path(p_folder) / "0" / "*.png")))
                    
                    # Call the function directly, it will use _worker_globals['client'] if available
                    rule_txt = generate_rule_explanation(pos_images_for_rule, neg_images_for_rule) 
                    rule_path = label_base_paths_original["rule"] / phase / f"{Path(p_folder).name}.txt"
                    rule_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(rule_path,"w") as f:
                        f.write(rule_txt)
                    processed_problem_folders.add(p_folder)
                except Exception as e:
                    logger.error(f"Error generating rule explanation for problem {Path(p_folder).name}: {e}")
            
            for sub in ["0","1"]:
                src_imgs_subdir = os.path.join(p_folder, sub)
                if not os.path.isdir(src_imgs_subdir):
                    logger.warning(f"Directory not found: {src_imgs_subdir}. Skipping.")
                    continue
                all_subdir_args.append((
                    src_imgs_subdir,
                    dst_imgs_phase_orig,
                    dst_lbls_phase_orig,
                    label_base_paths_original,
                    dst_imgs_phase_aug,
                    dst_lbls_phase_aug,
                    label_base_paths_augmented,
                    phase
                ))
    
    # Use multiprocessing Pool for parallel processing
    num_processes = os.cpu_count() # Use all available CPU cores
    logger.info(f"Starting parallel processing with {num_processes} workers.")
    with Pool(processes=num_processes, initializer=_worker_init, initargs=(0,)) as pool: # Pass dummy worker_id for initializer
        # Use imap_unordered for progress bar and flexible order
        for _ in tqdm(pool.imap_unordered(_process_single_subdir, all_subdir_args), total=len(all_subdir_args), desc="Processing image subdirectories"):
            pass # Progress bar will update here

if __name__ == "__main__":
     # Inform user about required dependencies
    print("Please ensure you have the following Python packages installed:")
    print("   - opencv-python-headless (pip install opencv-python-headless)")
    print("   - tqdm (pip install tqdm)")
    print("   - bongard (pip install bongard) - REQUIRED for programmatic labels in strict mode.")
    print("   - shapely (pip install shapely) - For polygon operations.")
    print("   - networkx (pip install networkx) - For graph relations.")
    print("   - scikit-image (pip install scikit-image) - For skeletonization and convex hull.")
    print("   - giotto-tda (pip install giotto-tda) - For topological features (optional).")
    print("   - cupy (pip install cupy-cudaXX where XX is your CUDA version, e.g., 11x) - For GPU-accelerated GTDA data transfer (optional).")
    print("   - pyfftw (pip install pyfftw) - For accelerated Fourier descriptors (optional).")
    print("   - openai (pip install openai) - For LLM captioning & rule explanations (optional, requires API key).")
    print("   - segment-anything (pip install git+https://github.com/facebookresearch/segment-anything.git) - For SAM confidence maps (optional, requires model checkpoint).")
    print("   - torch torchvision transformers timm (pip install torch torchvision transformers timm) - For DINOv2 embeddings.")
    print("   - albumentations (pip install albumentations opencv-python) - For advanced data augmentation.")
    print("   - Make sure 'albumentations_augmix.py' is in your Python path and contains 'CopyPaste' class.")
    print("   - numba (pip install numba) - For JIT compilation of numeric operations.")
    print("\n")
    parser = argparse.ArgumentParser(
        description='Prepare Bongard-LOGO dataset for YOLOv8 fine-tuning.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Show default values in help
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Root directory of your ShapeBongard_V2 data (e.g., path to the folder containing hd/images, bd/images, ff/images)."
    )
    parser.add_argument(
        "--out",
        required=True,
        help='Output root directory where the prepared dataset will be stored (e.g., "data/"). '
             'It will create subdirectories for images, labels, masks, programs, relations, topo, stats, and captions under it, for both original and augmented data.'
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Problem-wise train split fraction."
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Shard ID for distributed processing (0-indexed). Used with --num_shards."
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards for distributed processing. Used with --shard."
    )
    args = parser.parse_args()
    logger.info("Starting dataset preparation in STRICT PROGRAMMATIC LABELING mode...")
    prepare_dataset(args.src, args.out, args.train_frac, args.shard, args.num_shards)
    logger.info(f"\n ✔  Dataset ready under {args.out}")
    logger.info("Remember to create your data.yaml file for YOLOv8 training:")
    logger.info(f"   train: {os.path.join(args.out, 'images', 'train')}")
    logger.info(f"   val:   {os.path.join(args.out, 'images', 'val')}")
    logger.info(f"   train_aug: {os.path.join(args.out, 'images_aug', 'train')}") # New line for augmented images
    logger.info(f"   val_aug:   {os.path.join(args.out, 'images_aug', 'val')}")   # New line for augmented images
    logger.info(f"   nc:    {NUM_CLASSES}")
    logger.info(f"   names: {CLASS_NAMES}")
    logger.info("\nAdditional labels generated:")
    logger.info(f"   Original Boxes: {os.path.join(args.out, 'labels')}")
    logger.info(f"   Augmented Boxes (Albumentations): {os.path.join(args.out, 'labels_aug')} (files with _alb.txt)")
    logger.info(f"   Augmented Boxes (MixUp): {os.path.join(args.out, 'labels_aug')} (files with _mixup.txt)")
    logger.info(f"   Original Masks: {os.path.join(args.out, 'masks')}")
    logger.info(f"   Augmented Masks (Albumentations): {os.path.join(args.out, 'masks_aug')} (files with _alb.png)")
    logger.info(f"   Augmented Masks (MixUp): {os.path.join(args.out, 'masks_aug')} (files with _mixup.png)")
    logger.info(f"   Original Polygons: {os.path.join(args.out, 'polygons')}")
    logger.info(f"   Augmented Polygons (Albumentations): {os.path.join(args.out, 'polygons_aug')} (files with _alb.json)")
    logger.info(f"   Augmented Polygons (MixUp): {os.path.join(args.out, 'polygons_aug')} (files with _mixup.json)")
    logger.info(f"   Programs: {os.path.join(args.out, 'programs')} (Original only)")
    logger.info(f"   Relations: {os.path.join(args.out, 'relations')} (Original only)")
    logger.info(f"   Topological Features (Original): {os.path.join(args.out, 'topo')}")
    logger.info(f"   Topological Features (Augmented Albumentations): {os.path.join(args.out, 'topo_aug')} (files with _alb.json)")
    logger.info(f"   Topological Features (Augmented MixUp): {os.path.join(args.out, 'topo_aug')} (files with _mixup.json)")
    logger.info(f"   Shape Descriptors (Original): {os.path.join(args.out, 'stats')}")
    logger.info(f"   Shape Descriptors (Augmented Albumentations): {os.path.join(args.out, 'stats_aug')} (files with _alb.json)")
    logger.info(f"   Shape Descriptors (Augmented MixUp): {os.path.join(args.out, 'stats_aug')} (files with _mixup.json)")
    logger.info(f"   Oriented Boxes (Original): {os.path.join(args.out, 'oriented_boxes')}")
    logger.info(f"   Oriented Boxes (Augmented Albumentations): {os.path.join(args.out, 'oriented_boxes_aug')} (files with _alb.json)")
    logger.info(f"   Oriented Boxes (Augmented MixUp): {os.path.join(args.out, 'oriented_boxes_aug')} (files with _mixup.json)")
    logger.info(f"   Stroke Metadata: {os.path.join(args.out, 'strokes_meta')} (Original only)")
    logger.info(f"   Keypoints (Original): {os.path.join(args.out, 'keypoints')}")
    logger.info(f"   Keypoints (Augmented Albumentations): {os.path.join(args.out, 'keypoints_aug')} (files with _alb.json)")
    logger.info(f"   Keypoints (Augmented MixUp): {os.path.join(args.out, 'keypoints_aug')} (files with _mixup.json)")
    logger.info(f"   Mask Pyramid (Original): {os.path.join(args.out, 'mask_pyramid')}")
    logger.info(f"   Mask Pyramid (Augmented Albumentations): {os.path.join(args.out, 'mask_pyramid_aug')} (files with _alb_lvlX.png)")
    logger.info(f"   Mask Pyramid (Augmented MixUp): {os.path.join(args.out, 'mask_pyramid_aug')} (files with _mixup_lvlX.png)")
    logger.info(f"   Fourier Descriptors (Original): {os.path.join(args.out, 'fourier_desc')}")
    logger.info(f"   Fourier Descriptors (Augmented Albumentations): {os.path.join(args.out, 'fourier_desc_aug')} (files with _alb.json)")
    logger.info(f"   Fourier Descriptors (Augmented MixUp): {os.path.join(args.out, 'fourier_desc_aug')} (files with _mixup.json)")
    logger.info(f"   Augmentation Metadata (Original): {os.path.join(args.out, 'aug_metadata')}") # Metadata *about* random augmentations applied to original
    logger.info(f"   Augmentation Metadata (Albumentations): {os.path.join(args.out, 'aug_metadata_aug')} (files with _alb.json)")
    logger.info(f"   Augmentation Metadata (MixUp): {os.path.join(args.out, 'aug_metadata_aug')} (files with _mixup.json, includes lambda)")
    logger.info(f"   Negative Flags (Original): {os.path.join(args.out, 'negative_flags')}")
    logger.info(f"   Negative Flags (Albumentations): {os.path.join(args.out, 'negative_flags_aug')} (files with _alb.json)")
    logger.info(f"   Negative Flags (MixUp): {os.path.join(args.out, 'negative_flags_aug')} (files with _mixup.json)")
    logger.info(f"   Image Complexity Scores: {os.path.join(args.out, 'complexity_scores')}") # New line for complexity scores
    if _worker_globals.get('HAS_OPENAI', False):
        logger.info(f"   Captions: {os.path.join(args.out, 'captions')} (Original only)")
        logger.info(f"   Rule Explanations: {os.path.join(args.out, 'rule_explanations')} (Original only)")
    else:
        logger.info("   Captions: Skipped (OpenAI not enabled/configured).")
        logger.info("   Rule Explanations: Skipped (OpenAI not enabled/configured).")
    
    if _worker_globals.get('USE_SAM', False):
        logger.info(f"   SAM Confidence Maps (Original): {os.path.join(args.out, 'sam_confidence')}")
        logger.info(f"   SAM Generated Masks: {os.path.join(args.out, 'sam_masks')}") # New line for SAM masks
        logger.info(f"   SAM Generated YOLO Labels: {os.path.join(args.out, 'sam_yolo_labels')}") # New line for SAM YOLO labels
        logger.info(f"   SAM Generated Relations: {os.path.join(args.out, 'sam_relations')}") # New line for SAM relations
        logger.info(f"   Symbolic Labels (from SAM + Bongard): {os.path.join(args.out, 'symbolic_labels')}") # New line for symbolic labels
        logger.info(f"   Visual Debug Overlays (from SAM + Bongard): {os.path.join(args.out, 'visual_debug_overlays')}") # New line for visual debug
        logger.info(f"   Reasoning Chains (from SAM + Bongard): {os.path.join(args.out, 'reasoning_chains')}") # New line for reasoning chains
    else:
        logger.info("   SAM Confidence Maps: Skipped (SAM not enabled/configured).")
        logger.info("   SAM Generated Masks: Skipped (SAM not enabled/configured).")
        logger.info("   SAM Generated YOLO Labels: Skipped (SAM not enabled/configured).")
        logger.info("   SAM Generated Relations: Skipped (SAM not enabled/configured).")
        logger.info("   Symbolic Labels: Skipped (SAM or Bongard symbolic fusion not enabled/configured).")
        logger.info("   Visual Debug Overlays: Skipped (SAM or Bongard symbolic fusion not enabled/configured).")
        logger.info("   Reasoning Chains: Skipped (SAM or Bongard symbolic fusion not enabled/configured).")
    if _worker_globals.get('HAS_EMBED_MODEL', False):
        logger.info(f"   Shape Embeddings (Original): {os.path.join(args.out, 'embeddings')}")
        logger.info(f"   Shape Embeddings (Augmented Albumentations): {os.path.join(args.out, 'embeddings_aug')} (files with _alb_shapeX.json)")
        logger.info(f"   Shape Embeddings (Augmented MixUp): {os.path.join(args.out, 'embeddings_aug')} (files with _mixup_shapeX.json)")
    else:
        logger.info("   Shape Embeddings: Skipped (Embedding model not enabled/configured).")
    logger.info("\nThis script now strictly relies on programmatic Bongard labels for maximum accuracy and rich multi-modal data.")
