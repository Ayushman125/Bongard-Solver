# albumentations_augmix.py
import albumentations as A
import numpy as np
import cv2
import random
import logging

logger = logging.getLogger(__name__)

class CopyPaste(A.BasicTransform):
    """
    Implements the Copy-Paste augmentation technique.
    
    Args:
        blend (bool): If True, blends the pasted objects with the background using Gaussian blur.
                      If False, directly overlays the pasted objects.
        sigma (int): Standard deviation for Gaussian blur when blending.
        pct_objects_paste (float): Percentage of objects from the pasted image to actually paste.
                                   (Note: Current implementation pastes all objects from the selected paste_masks/bboxes.
                                   This parameter is kept for API consistency but not fully utilized in this basic version).
        mask_format (str): Format of the masks. Currently supports 'binary'.
        always_apply (bool): Whether to always apply the transform.
        p (float): Probability of applying the transform.
    """
    def __init__(self, blend=True, sigma=1, pct_objects_paste=0.5, mask_format='binary', always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.blend = blend
        self.sigma = sigma
        self.pct_objects_paste = pct_objects_paste
        self.mask_format = mask_format
        # Define available keys for Albumentations to understand what inputs this transform expects
        # 'image', 'mask', 'bboxes' are standard for the main image
        # 'paste_image', 'paste_masks', 'paste_bboxes' are the additional inputs this transform needs
        self.available_keys = {"image", "mask", "bboxes", "paste_image", "paste_masks", "paste_bboxes", "class_labels", "paste_class_labels"}

    def apply(self, image, paste_image=None, paste_masks=None, **params):
        """
        Applies the copy-paste transformation to the image.
        """
        if paste_image is None or paste_masks is None or len(paste_masks) == 0:
            return image # No paste image or masks provided, return original image

        # Create a combined mask for all objects to be pasted
        # Ensure paste_masks are numpy arrays and sum them up
        combined_paste_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for pm in paste_masks:
            # Ensure the paste mask is the same size as the target image
            if pm.shape != image.shape[:2]:
                pm = cv2.resize(pm.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            combined_paste_mask = np.maximum(combined_paste_mask, pm)

        # Ensure paste_image is the same size as the target image
        if paste_image.shape != image.shape:
            paste_image = cv2.resize(paste_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Blend pasted image
        if self.blend:
            # Apply Gaussian blur to the combined mask for smooth blending
            blurred_mask = cv2.GaussianBlur(combined_paste_mask, (0, 0), self.sigma)
            # Normalize blurred mask to be between 0 and 1
            blurred_mask_normalized = blurred_mask[..., None] / 255.0
            
            # Blend the images using the normalized blurred mask
            image = (1 - blurred_mask_normalized) * image.astype(np.float32) + blurred_mask_normalized * paste_image.astype(np.float32)
            image = image.astype(np.uint8)
        else:
            # Directly overlay without blending
            image = np.where(combined_paste_mask[..., None] > 0, paste_image, image)

        return image

    def apply_to_mask(self, mask, paste_masks=None, **params):
        """
        Applies the copy-paste transformation to the mask.
        Combines the original mask with the pasted masks.
        """
        if paste_masks is None or len(paste_masks) == 0:
            return mask # No paste masks provided, return original mask

        combined_paste_mask = np.zeros(mask.shape, dtype=mask.dtype)
        for pm in paste_masks:
            if pm.shape != mask.shape:
                pm = cv2.resize(pm.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            combined_paste_mask = np.maximum(combined_paste_mask, pm)
            
        return np.maximum(mask, combined_paste_mask)

    def apply_to_bbox(self, bbox, paste_bboxes=None, **params):
        """
        Applies the copy-paste transformation to a single bounding box.
        This method is called for each bbox in the original image.
        For CopyPaste, we typically concatenate the original bboxes with the pasted bboxes.
        """
        # This method is designed to apply to a single bbox.
        # The concatenation logic should happen in the `recompose_data` or a custom `__call__` method
        # if this transform is used directly with `Compose`.
        # For now, we return the original bbox as this method is not directly used for concatenation.
        return bbox

    def get_transform_init_args_names(self):
        """
        Returns the names of the arguments that are passed to the __init__ method.
        """
        return ("blend", "sigma", "pct_objects_paste", "mask_format")

    def get_params_dependent_on_data(self, params):
        """
        This method is crucial for CopyPaste as it needs to select objects to paste.
        In a full implementation, this would involve loading random images and their annotations.
        For this basic setup, we assume `paste_image`, `paste_masks`, `paste_bboxes`, `paste_class_labels`
        are already provided in the input `params` dictionary by the dataset loader.
        """
        # This is where you would typically load a random image and its annotations
        # For now, we expect them to be passed in the input dictionary
        return {} # No new parameters generated here, assuming inputs are pre-loaded

    @property
    def targets_as_params(self):
        """
        Specifies which targets should be passed as parameters to other apply methods.
        This is how Albumentations knows to pass 'paste_image', 'paste_masks', 'paste_bboxes'
        to `apply`, `apply_to_mask`, etc.
        """
        return ["paste_image", "paste_masks", "paste_bboxes", "paste_class_labels"]

    def __call__(self, force_apply=False, **kwargs):
        """
        The main call method for the transform.
        This overrides the default BasicTransform.__call__ to handle multiple inputs.
        """
        if self.p < random.random() and not force_apply:
            return kwargs # Return original data if not applying

        image = kwargs["image"]
        masks = kwargs.get("masks", [])
        bboxes = kwargs.get("bboxes", [])
        class_labels = kwargs.get("class_labels", [])

        paste_image = kwargs.get("paste_image")
        paste_masks = kwargs.get("paste_masks", [])
        paste_bboxes = kwargs.get("paste_bboxes", [])
        paste_class_labels = kwargs.get("paste_class_labels", [])

        # Apply the image transformation
        transformed_image = self.apply(image=image, paste_image=paste_image, paste_masks=paste_masks)

        # Apply mask transformations
        # Each original mask is passed through apply_to_mask, but for copy-paste,
        # we generally want to combine all masks.
        # The apply_to_mask method above now handles combining all paste_masks.
        transformed_masks = [self.apply_to_mask(m, paste_masks=paste_masks) for m in masks]
        transformed_masks.extend(paste_masks) # Add the pasted masks to the list

        # Apply bbox transformations
        # For bboxes, we simply concatenate the original and pasted bboxes
        transformed_bboxes = bboxes + paste_bboxes
        transformed_class_labels = class_labels + paste_class_labels

        # Reconstruct the output dictionary
        return {
            "image": transformed_image,
            "masks": transformed_masks,
            "bboxes": transformed_bboxes,
            "class_labels": transformed_class_labels,
        }

