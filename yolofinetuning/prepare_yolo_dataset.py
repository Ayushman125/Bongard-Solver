# scripts/prepare_yolo_dataset.py

import os
import glob
import random
import shutil
import argparse
import cv2
import torch
import logging
from tqdm import tqdm

# Configure logging for better output control
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 2.1 Optional: Programmatic labels via Bongard library
try:
    from bongard import BongardProblem
    USE_PROGRAM = True
    logger.info("Bongard library found. Programmatic labeling enabled (strict mode).")
except ImportError:
    USE_PROGRAM = False
    logger.error("Bongard library not found. Programmatic labeling is REQUIRED for strict mode. Exiting.")
    # In a real script, you might sys.exit(1) here if running strictly
    # For this interactive environment, we'll just set USE_PROGRAM to False and let the runtime error handle it later.

# SAM and Contour fallbacks are explicitly removed as per user request for strict programmatic labeling.
# The following blocks are commented out or removed to reflect this.
# 2.2 Optional: Zero-shot segmentation fallback with SAM
# try:
#     from segment_anything import sam_model_registry, SamPredictor
#     SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
#     if os.path.exists(SAM_CHECKPOINT_PATH):
#         SAM = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
#         SAM.to("cuda" if torch.cuda.is_available() else "cpu")
#         predictor = SamPredictor(SAM)
#         USE_SAM = True
#         logger.info(f"SAM model loaded from {SAM_CHECKPOINT_PATH}. Zero-shot labeling enabled.")
#     else:
#         USE_SAM = False
#         logger.warning(f"SAM checkpoint '{SAM_CHECKPOINT_PATH}' not found. Zero-shot labeling will be skipped.")
# except ImportError:
#     USE_SAM = False
#     logger.warning("Segment Anything Model (SAM) library not found. Zero-shot labeling will be skipped.")
# except Exception as e:
#     USE_SAM = False
#     logger.error(f"Error loading SAM model: {e}. Zero-shot labeling will be skipped.")
USE_SAM = False # Explicitly disable SAM

# Global dictionary for class IDs (extended for multi-class as per recommendation)
# Add more shape types as needed from your Bongard-LOGO dataset
CLASS_ID = {
    'circle': 0,
    'square': 1,
    'triangle': 2,
    'line': 3,
    'pentagon': 4,
    'hexagon': 5,
    'star': 6,
    'cross': 7,
    'plus': 8,
    'ellipse': 9,
    # Add any other shape types present in your Bongard dataset
    'unknown_shape': 99 # Fallback for shapes not explicitly mapped
}
# Determine the number of classes based on the mapped IDs
NUM_CLASSES = len(set(CLASS_ID.values()))
CLASS_NAMES = sorted(CLASS_ID.keys(), key=lambda k: CLASS_ID[k]) # For data.yaml

def find_problem_dirs(src_root: str) -> list[str]:
    """
    Recursively finds all subdirectories under src_root that contain both '0/' and '1/'
    subdirectories, indicating a Bongard problem folder.

    Args:
        src_root (str): The root directory to start searching from (e.g., 'hd/images').

    Returns:
        list[str]: A list of absolute paths to problem directories.
    """
    problems = []
    logger.info(f"Searching for problem directories under: {src_root}")
    for root, dirs, _ in os.walk(src_root):
        # Check if both '0' and '1' are direct subdirectories
        if "0" in dirs and "1" in dirs:
            problems.append(root)
    logger.info(f"Found {len(problems)} problem directories.")
    return problems

def split_problems(problems: list[str], frac: float, seed: int = 42) -> tuple[list[str], list[str]]:
    """
    Splits a list of problem directories into training and validation sets based on a
    specified fraction. The split is deterministic if the seed is fixed.

    Args:
        problems (list[str]): A list of problem directory paths.
        frac (float): The fraction of problems to allocate to the training set (e.g., 0.8 for 80%).
        seed (int): The random seed for reproducibility.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists: (train_problems, val_problems).
    """
    if not (0.0 <= frac <= 1.0):
        raise ValueError("frac must be between 0.0 and 1.0")

    random.seed(seed)
    shuffled_problems = problems[:] # Create a copy to shuffle
    random.shuffle(shuffled_problems)

    n_train = int(len(shuffled_problems) * frac)
    train_ps = shuffled_problems[:n_train]
    val_ps = shuffled_problems[n_train:]

    logger.info(f"Splitting {len(problems)} problems: {len(train_ps)} for training, {len(val_ps)} for validation.")
    return train_ps, val_ps

def write_yolo_boxes(boxes: list[list[float]], img_shape: tuple[int, int], out_txt: str) -> None:
    """
    Writes bounding box coordinates to a YOLO-format .txt file.

    Args:
        boxes (list[list[float]]): A list of bounding boxes, where each box is
                                   [x1, y1, x2, y2, class_id] in pixel coordinates.
        img_shape (tuple[int, int]): A tuple (height, width) of the original image.
        out_txt (str): The path to the output .txt file.
    """
    h, w = img_shape
    lines = []
    for x1, y1, x2, y2, cid in boxes:
        # Calculate normalized center coordinates and dimensions
        # YOLO format: <class_id> <center_x> <center_y> <width> <height> (all normalized to [0, 1])
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

    Args:
        img_path (str): Path to the image (used for context, though not directly in this function).
        prog: A BongardProblem program object containing stroke information.

    Returns:
        list[list[float]]: A list of bounding boxes, where each box is
                           [x1, y1, x2, y2, class_id] in pixel coordinates.
    """
    out_boxes = []
    for stroke in prog.strokes:
        x1, y1, x2, y2 = stroke.bounding_box()
        # Use stroke.shape_type for multi-class mapping
        shape_type = stroke.shape_type
        cid = CLASS_ID.get(shape_type, CLASS_ID['unknown_shape']) # Get ID, fallback to 'unknown_shape'
        if cid == CLASS_ID['unknown_shape']:
            logger.warning(f"Unknown shape type '{shape_type}' encountered in Bongard problem for {img_path}. "
                           f"Assigning to 'unknown_shape' class ID {CLASS_ID['unknown_shape']}.")
        out_boxes.append([x1, y1, x2, y2, cid])
    return out_boxes

# SAM and Contour labeling functions are removed as per user request for strict programmatic labeling.
# def label_with_sam(img_path: str) -> list[list[float]]:
#     # ... (function body removed) ...
#     pass

# def label_with_contours(img_path: str) -> list[list[float]]:
#     # ... (function body removed) ...
#     pass

def auto_label_and_copy(src_dir: str, dst_img_dir: str, dst_lbl_dir: str) -> None:
    """
    Processes all PNG images in a source directory:
      1. Copies each image to the destination image directory.
      2. Labels via programmatic (Bongard) method ONLY. Raises RuntimeError if it fails.
      3. Writes a YOLO-format .txt label file for each image in the
         destination label directory.

    Args:
        src_dir (str): The source directory containing PNG images.
        dst_img_dir (str): The destination directory for copied images.
        dst_lbl_dir (str): The destination directory for YOLO label files.
    """
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    for img_path in glob.glob(os.path.join(src_dir, "*.png")):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        dst_image = os.path.join(dst_img_dir, f"{stem}.png")
        dst_label = os.path.join(dst_lbl_dir, f"{stem}.txt")

        # 1) Copy image
        try:
            shutil.copy(img_path, dst_image)
        except IOError as e:
            logger.error(f"Could not copy image {img_path} to {dst_image}: {e}")
            continue

        img_read_for_shape = cv2.imread(img_path)
        if img_read_for_shape is None:
            raise RuntimeError(f"Could not read image {img_path} to determine shape. Cannot proceed.")
        img_shape = img_read_for_shape.shape[:2] # (height, width)

        # Rely EXCLUSIVELY on programmatic labeling
        if USE_PROGRAM:
            try:
                # Assuming problem folder is two levels up from img_path (e.g., prob_folder/0/image.png)
                prob_folder = os.path.dirname(os.path.dirname(src_dir))
                prob = BongardProblem(prob_folder)
                progs = prob.positive_programs + prob.negative_programs
                boxes = []
                found_program = False
                for p in progs:
                    # Match program to current image based on filename
                    if os.path.basename(p.image_path) == os.path.basename(img_path):
                        boxes = label_with_bongard(img_path, p)
                        found_program = True
                        break
                if not found_program:
                    raise RuntimeError(f"No matching Bongard program found for {img_path}. "
                                       "Strict mode requires programmatic labels for all images.")
                if not boxes:
                    logger.warning(f"Programmatic labeling found no boxes for {img_path}. "
                                   "This image will have an empty label file.")
            except Exception as e:
                # Raise a RuntimeError to halt if programmatic labeling fails in strict mode
                raise RuntimeError(f"Error during programmatic labeling for {img_path}: {e}. "
                                   "Strict mode requires perfect programmatic labels.")
        else:
            # This case should ideally be caught earlier if USE_PROGRAM is False at script start
            raise RuntimeError("Programmatic labeling is disabled or Bongard library not found. "
                               "Cannot proceed with dataset preparation in strict mode.")

        # Write the collected YOLO boxes
        write_yolo_boxes(boxes, img_shape, dst_label)


def prepare_dataset(src_root: str, out_root: str, frac: float) -> None:
    """
    Orchestrates the entire dataset preparation process:
      1. Discovers all Bongard problem folders.
      2. Splits these problems into training and validation sets.
      3. Iterates through each problem's '0/' and '1/' subdirectories,
         auto-labels images, and copies them to the final YOLO dataset structure.

    Args:
        src_root (str): The root directory of your ShapeBongard_V2 data
                        (e.g., the parent of 'hd/images', 'bd/images', 'ff/images').
        out_root (str): The desired output root directory for the prepared dataset.
                        This is where 'images/' and 'labels/' will be created.
        frac (float): The fraction of problems to use for the training set.
    """
    # Check if programmatic labeling is enabled before starting the process
    if not USE_PROGRAM:
        logger.error("Programmatic labeling is not enabled. Cannot proceed in strict mode.")
        return # Exit early if Bongard library is not available

    problems = find_problem_dirs(src_root)
    if not problems:
        logger.error(f"No problem directories found under {src_root}. Please check the --src path.")
        return

    train_ps, val_ps = split_problems(problems, frac)

    # Define the base paths for images and labels within the output root
    out_images_dir = os.path.join(out_root, 'images')
    out_labels_dir = os.path.join(out_root, 'labels')

    for phase, plist in [("train", train_ps), ("val", val_ps)]:
        logger.info(f"\n→ Preparing {phase.upper()} set ({len(plist)} problems)")
        # Create phase-specific directories
        dst_imgs_phase = os.path.join(out_images_dir, phase)
        dst_lbls_phase = os.path.join(out_labels_dir, phase)

        # Use tqdm for a progress bar over problems
        for p in tqdm(plist, desc=f"Processing {phase} problems"):
            for sub in ["0","1"]: # Iterate through positive (1) and negative (0) examples
                src_imgs_subdir = os.path.join(p, sub)
                if not os.path.isdir(src_imgs_subdir):
                    logger.warning(f"Directory not found: {src_imgs_subdir}. Skipping.")
                    continue
                try:
                    auto_label_and_copy(src_imgs_subdir, dst_imgs_phase, dst_lbls_phase)
                except RuntimeError as e:
                    logger.critical(f"Fatal error during labeling: {e}. Aborting dataset preparation.")
                    return # Stop execution on critical error

if __name__ == "__main__":
    # Inform user about required dependencies
    print("Please ensure you have the following Python packages installed:")
    print("  - opencv-python-headless (pip install opencv-python-headless)")
    print("  - tqdm (pip install tqdm)")
    print("  - bongard (pip install bongard) - REQUIRED for programmatic labels in strict mode.")
    print("\n")

    parser = argparse.ArgumentParser(
        description='Prepare Bongard-LOGO dataset for YOLOv8 fine-tuning.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
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
             'It will create "images/" and "labels/" subdirectories under it.'
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Problem-wise train split fraction."
    )
    args = parser.parse_args()

    logger.info("Starting dataset preparation in STRICT PROGRAMMATIC LABELING mode...")
    prepare_dataset(args.src, args.out, args.train_frac)
    logger.info(f"\n✔ Dataset ready under {args.out}")
    logger.info("Remember to create your data.yaml file for YOLOv8 training:")
    logger.info(f"  train: {os.path.join(args.out, 'images', 'train')}")
    logger.info(f"  val:   {os.path.join(args.out, 'images', 'val')}")
    logger.info(f"  nc:    {NUM_CLASSES}")
    logger.info(f"  names: {CLASS_NAMES}")
    logger.info("\nThis script now strictly relies on programmatic Bongard labels for maximum accuracy.")
