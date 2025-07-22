
# src/preprocess/crop_extractor.py

import numpy as np
import cv2 # Import OpenCV
from PIL import Image
from typing import List, Dict, Any, Tuple

class ClassicalCVCropper:
    """
    Uses classical computer vision techniques (e.g., contour detection)
    to segment objects and extract crops from an image.
    This replaces the Mask R-CNN based approach.
    """
    def __init__(self, conf_thresh: float = 0.1):
        """
        Initializes the ClassicalCVCropper.
        Args:
            conf_thresh (float): A threshold (e.g., for contour area or solidity)
                                 to filter detected objects.
        """
        self.conf_thresh = conf_thresh
        self.object_id_counter = 0 # To assign unique IDs to detected objects

    def segment_shapes(self, image: np.ndarray, debug_dir: str = None) -> List[Dict[str, Any]]:
        """
        Robust segmentation: adaptive thresholding, morphological cleaning, contour filtering, per-shape mask extraction.
        Optionally saves intermediate masks for debugging if debug_dir is provided.
        """
        import os
        from skimage.filters import threshold_otsu, threshold_local
        from skimage.morphology import closing, opening, remove_small_objects, disk
        img_area = image.shape[0] * image.shape[1]
        # 1. Convert to grayscale
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError("Unsupported image format. Expected 2D grayscale or 3D BGR/RGB.")

        # 2. Adaptive binarization (global Otsu, fallback to local)
        try:
            thresh = threshold_otsu(gray)
            mask = gray < thresh
            frac = mask.mean()
            if frac < 0.01 or frac > 0.5:
                block_size = 51
                local = threshold_local(gray, block_size, offset=10)
                mask = gray < local
        except Exception:
            # fallback to OpenCV Otsu
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mask = mask > 0

        # 3. Morphological cleaning
        mask = closing(mask, disk(2))
        mask = opening(mask, disk(1))
        mask = remove_small_objects(mask, min_size=50)

        # 4. Contour-based background filtering
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shape_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 0.5 * img_area:
                shape_contours.append(cnt)

        # 5. Per-shape mask extraction
        detected_objects = []
        shape_masks = []
        for contour in shape_contours:
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, x + w, y + h]
            mask_shape = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask_shape, [contour], -1, 255, cv2.FILLED)
            mask_binary = (mask_shape[y:y+h, x:x+w] > 0)
            patch = image[y:y+h, x:x+w]
            detected_objects.append({
                'id': self.object_id_counter,
                'patch': patch,
                'mask': mask_binary,
                'bbox': bbox,
                'contour': contour
            })
            shape_masks.append(mask_binary)
            self.object_id_counter += 1

        # 6. Sanity-check: shapes should cover <20% of image
        total_shape_area = sum(m.sum() for m in shape_masks)
        assert 0.005 < total_shape_area / img_area < 0.2, f"Shapes cover unexpected fraction of image: {total_shape_area/img_area:.3f}"

        # 7. Optionally save masks for debugging
        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "input_gray.png"), gray)
            cv2.imwrite(os.path.join(debug_dir, "mask_cleaned.png"), (mask*255).astype(np.uint8))
            for i, m in enumerate(shape_masks):
                cv2.imwrite(os.path.join(debug_dir, f"shape_mask_{i}.png"), (m*255).astype(np.uint8))


        return detected_objects

# --- Module-level function for direct import ---
def segment_shapes(image: np.ndarray, debug_dir: str = None) -> List[Dict[str, Any]]:
    """
    Module-level wrapper for robust segmentation. Instantiates ClassicalCVCropper and calls segment_shapes.
    """
    cropper = ClassicalCVCropper()
    return cropper.segment_shapes(image, debug_dir=debug_dir)

