# Folder: bongard_solver/
# File: crop_extraction.py
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2
import logging
from typing import List, Tuple, Dict, Any, Optional

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MaskRCNNCropper:
    """
    A class to perform object segmentation and cropping using a pre-trained Mask R-CNN model.
    """
    def __init__(self, conf_thresh: float = 0.7, device: Optional[str] = None):
        """
        Initializes the MaskRCNNCropper with a pre-trained Mask R-CNN model.

        Args:
            conf_thresh (float): Confidence threshold for object detection. Detections
                                 with scores below this threshold will be ignored.
            device (Optional[str]): The device to run the model on ('cuda' or 'cpu').
                                    If None, it will automatically detect if CUDA is available.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Initializing MaskRCNNCropper on device: {self.device}")
        try:
            # Load a pre-trained Mask R-CNN model with a ResNet-50 FPN backbone
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            self.model.eval().to(self.device) # Set model to evaluation mode and move to device
            self.conf_thresh = conf_thresh
            logger.info(f"Mask R-CNN model loaded successfully with confidence threshold: {conf_thresh}")
        except Exception as e:
            logger.error(f"Failed to load Mask R-CNN model: {e}")
            raise RuntimeError(f"Could not initialize MaskRCNNCropper. Error: {e}")

    def segment_shapes(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """
        Segments shapes in an input image using Mask R-CNN and returns cropped patches
        along with their corresponding masks and metadata.

        Args:
            image (np.ndarray): The input image as a NumPy array (H, W, C) in RGB format.
                                Assumed to be in the range [0, 255].

        Returns:
            List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]: A list of tuples, where each tuple contains:
                - patch (np.ndarray): The cropped image patch containing the segmented object.
                - mask (np.ndarray): The binary mask (0 or 1) of the segmented object, cropped to the patch size.
                - metadata (Dict[str, Any]): A dictionary containing additional information
                                            like bounding box coordinates (x1, y1, x2, y2)
                                            and confidence score.
        """
        if image is None or image.size == 0:
            logger.warning("Received an empty or invalid image for segmentation.")
            return []

        # Convert image to a PyTorch tensor and normalize to [0, 1]
        # F.to_tensor automatically converts HWC to CHW and normalizes to [0, 1]
        img_tensor = F.to_tensor(image).to(self.device)

        patches_with_masks = []
        try:
            with torch.no_grad(): # Disable gradient calculation for inference
                # The model expects a list of tensors, even for a single image
                outputs = self.model([img_tensor])[0]

            # Process each detected instance
            for i in range(len(outputs['masks'])):
                score = outputs['scores'][i].item()
                if score < self.conf_thresh:
                    continue # Skip detections below the confidence threshold

                # Extract mask, convert to NumPy, and threshold to binary
                # The mask is typically (1, H, W) or (N, H, W) where N is num_classes
                # We take the first channel [0] if it's (1, H, W)
                mask = outputs['masks'][i, 0].cpu().numpy()
                binary_mask = (mask > 0.5).astype(np.uint8) # Convert to binary mask (0 or 1)

                # Extract bounding box coordinates
                box = outputs['boxes'][i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box

                # Ensure bounding box coordinates are within image dimensions
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid bounding box [{x1},{y1},{x2},{y2}] for detection {i}. Skipping.")
                    continue

                # Apply the binary mask to the original image to get the masked patch
                # Create a copy to avoid modifying the original image
                masked_img = image.copy()
                # Zero out pixels outside the detected object region
                # Need to resize binary_mask to original image size before applying
                mask_resized = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                masked_img[mask_resized == 0] = 0 # Set background pixels to black

                # Crop the masked image to the bounding box
                patch = masked_img[y1:y2, x1:x2]

                # Crop the mask itself to the bounding box for consistency
                cropped_mask = binary_mask[y1:y2, x1:x2]

                metadata = {
                    'bbox': [x1, y1, x2, y2],
                    'score': score
                }
                patches_with_masks.append((patch, cropped_mask, metadata))

            logger.info(f"Segmented {len(patches_with_masks)} shapes from the image.")
            return patches_with_masks

        except Exception as e:
            logger.error(f"Error during shape segmentation with Mask R-CNN: {e}")
            return []

# Example Usage (for testing purposes, not part of the main pipeline)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create a dummy image (e.g., a white circle on a black background)
    dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
    # Draw a white circle
    cv2.circle(dummy_image, (128, 128), 50, (255, 255, 255), -1)
    # Draw a red square
    cv2.rectangle(dummy_image, (30, 30), (80, 80), (255, 0, 0), -1)

    cropper = MaskRCNNCropper(conf_thresh=0.7)
    segmented_results = cropper.segment_shapes(dummy_image)

    if segmented_results:
        print(f"Found {len(segmented_results)} objects.")
        for i, (patch, mask, metadata) in enumerate(segmented_results):
            print(f"Object {i+1}: BBox={metadata['bbox']}, Score={metadata['score']:.2f}")
            # Save or display the patch and mask for verification
            cv2.imwrite(f"object_patch_{i+1}.png", patch)
            cv2.imwrite(f"object_mask_{i+1}.png", mask * 255) # Save mask as grayscale image
            print(f"Saved object_patch_{i+1}.png and object_mask_{i+1}.png")
    else:
        print("No objects found or an error occurred.")

