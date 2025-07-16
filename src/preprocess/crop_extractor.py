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

    def segment_shapes(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segments shapes in the input image using OpenCV contour detection.
        Args:
            image (np.ndarray): Input image (H, W, C) in RGB or BGR.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing:
                                  - 'id': Unique ID for the object
                                  - 'patch': Cropped image patch (np.ndarray)
                                  - 'mask': Binary mask of the object (np.ndarray)
                                  - 'bbox': Bounding box [x1, y1, x2, y2]
                                  - 'contour': The raw OpenCV contour (np.ndarray)
        """
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2: # Already grayscale
            gray = image
        else:
            raise ValueError("Unsupported image format. Expected 2D grayscale or 3D BGR/RGB.")

        # Apply a binary threshold
        # You might need to adjust the thresholding method (e.g., adaptive thresholding)
        # depending on your image characteristics.
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_objects = []
        for contour in contours:
            # Filter small contours (noise) or based on confidence threshold
            area = cv2.contourArea(contour)
            if area < self.conf_thresh * (image.shape[0] * image.shape[1]): # Filter by relative area
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, x + w, y + h]

            # Create a mask for the current object
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
            mask_binary = (mask[y:y+h, x:x+w] > 0) # Crop and binarize mask

            # Extract the patch (crop)
            # Ensure the patch is in the same format as the original image (e.g., RGB)
            patch = image[y:y+h, x:x+w]

            detected_objects.append({
                'id': self.object_id_counter,
                'patch': patch,
                'mask': mask_binary,
                'bbox': bbox,
                'contour': contour # Keep the raw contour for shape inference
            })
            self.object_id_counter += 1
        
        return detected_objects

