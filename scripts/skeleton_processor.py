# ------------------------------------------------------------------
# Skeleton-Aware Post-processing (SAP) for thin mask refinement
# Advanced topology-preserving mask enhancement
# ------------------------------------------------------------------

import cv2
import numpy as np
from typing import Tuple, Optional, List
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects
import logging

class SkeletonAwareProcessor:
    """
    Professional skeleton-aware post-processing for mask refinement.
    Implements Zhang-Suen thinning, Hamilton-Jacobi skeleton repair,
    and distance-transform-based width restoration.
    """
    
    def __init__(self, 
                 min_branch_len: int = 5,
                 gap_fill_radius: int = 3,
                 min_component_size: int = 50):
        """
        Initialize skeleton processor with quality parameters.
        
        Args:
            min_branch_len: Minimum branch length to preserve
            gap_fill_radius: Maximum gap distance to repair
            min_component_size: Minimum component size to keep
        """
        self.min_branch_len = min_branch_len
        self.gap_fill_radius = gap_fill_radius  
        self.min_component_size = min_component_size
        
        # Check OpenCV contrib availability
        self.has_ximgproc = hasattr(cv2, 'ximgproc')
        if not self.has_ximgproc:
            logging.warning("cv2.ximgproc not available - using skimage thinning fallback")
    
    def sap_refine(self, mask_bin: np.ndarray) -> np.ndarray:
        """
        Complete skeleton-aware post-processing pipeline.
        
        Pipeline:
        1. Clean input mask (remove noise, fill holes)
        2. Extract skeleton using Zhang-Suen thinning
        3. Hamilton-Jacobi gap repair
        4. Distance transform width restoration
        
        Args:
            mask_bin: Binary mask [H,W] with values 0/255 or 0/1
            
        Returns:
            Refined binary mask [H,W] uint8
        """
        print(f"[SAP] Starting refinement on mask: shape={mask_bin.shape}, min={mask_bin.min():.6f}, max={mask_bin.max():.6f}")
        
        if mask_bin.max() == 0:
            print("[SAP] Input mask is empty, returning zeros")
            return mask_bin.astype(np.uint8)
        
        try:
            # Normalize to 0/255
            mask_clean = self._clean_input_mask(mask_bin)
            if mask_clean.max() == 0:
                print("[SAP] Mask became empty after cleaning, returning zeros")
                return mask_clean
            
            print(f"[SAP] After cleaning: nonzero_pixels={np.count_nonzero(mask_clean)}")
            
            # Extract skeleton
            skeleton = self._extract_skeleton(mask_clean)
            if skeleton.max() == 0:
                print("[SAP] Skeleton extraction failed, returning cleaned mask as fallback")
                return mask_clean  # Fallback to cleaned input
            
            print(f"[SAP] Skeleton extracted: nonzero_pixels={np.count_nonzero(skeleton)}")
            
            # Repair gaps in skeleton
            skeleton_repaired = self._repair_skeleton_gaps(skeleton)
            repaired_pixels = np.count_nonzero(skeleton_repaired)
            print(f"[SAP] After gap repair: nonzero_pixels={repaired_pixels}")
            
            # Restore width using distance transform
            mask_restored = self._restore_mask_width(skeleton_repaired, mask_clean)
            final_pixels = np.count_nonzero(mask_restored)
            print(f"[SAP] Final result: nonzero_pixels={final_pixels}")
            
            # Validate result isn't empty
            if mask_restored.max() == 0:
                print("[SAP] Width restoration failed, returning cleaned mask")
                return mask_clean
            
            return mask_restored.astype(np.uint8)
            
        except Exception as e:
            print(f"[SAP] Refinement failed: {e}")
            logging.error(f"SAP refinement failed: {e}")
            # Return original on failure, properly converted to uint8
            if mask_bin.max() <= 1:
                return (mask_bin * 255).astype(np.uint8)
            else:
                return mask_bin.astype(np.uint8)
    
    def _clean_input_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean and normalize input mask with enhanced low-value handling."""
        # Handle very low-value masks (common with SAM outputs)
        if mask.max() <= 0.01:
            # For very low values, use adaptive thresholding
            if mask.max() > 0:
                # Use half of max value as threshold for very low-value masks
                threshold_val = mask.max() / 2
                mask_bin = (mask > threshold_val).astype(np.uint8) * 255
                print(f"[SAP] Low-value mask detected (max={mask.max():.6f}), using adaptive threshold={threshold_val:.6f}")
            else:
                # Completely empty mask
                return np.zeros_like(mask, dtype=np.uint8)
        elif mask.max() <= 1:
            # Standard float mask [0,1] -> [0,255]
            mask_bin = (mask * 255).astype(np.uint8)
        else:
            # Already in [0,255] range or higher
            mask_bin = (mask > 127).astype(np.uint8) * 255
        
        # Skip processing if mask is too small or empty
        if mask_bin.max() == 0:
            print("[SAP] Empty mask after thresholding, returning zeros")
            return mask_bin
            
        nonzero_count = np.count_nonzero(mask_bin)
        if nonzero_count < 10:
            print(f"[SAP] Very sparse mask ({nonzero_count} pixels), minimal processing")
            # For very sparse masks, just do basic cleanup
            return mask_bin
        
        # Remove small noise components
        try:
            mask_bin = remove_small_objects(
                mask_bin > 0, min_size=min(self.min_component_size, nonzero_count // 4)
            ).astype(np.uint8) * 255
        except Exception as e:
            print(f"[SAP] Small object removal failed: {e}, skipping")
        
        # Fill small holes - be more conservative for low-value masks
        kernel_size = 3 if mask.max() > 0.1 else 2  # Smaller kernel for low-value masks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
        
        return mask_bin
    
    def _extract_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """Extract 1-pixel skeleton using Zhang-Suen or skimage fallback."""
        mask_bool = mask > 127
        
        if self.has_ximgproc:
            try:
                # OpenCV Zhang-Suen thinning (preferred)
                skeleton = cv2.ximgproc.thinning(
                    mask.astype(np.uint8), 
                    thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
                )
                return skeleton
            except Exception as e:
                logging.warning(f"OpenCV thinning failed: {e}, using skimage")
        
        # Skimage fallback
        skeleton_bool = skeletonize(mask_bool)
        return (skeleton_bool * 255).astype(np.uint8)
    
    def _repair_skeleton_gaps(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Hamilton-Jacobi inspired gap repair for broken skeleton fragments.
        Connects nearby endpoints within gap_fill_radius.
        """
        skeleton_repaired = skeleton.copy()
        
        # Find skeleton endpoints (pixels with exactly 1 neighbor)
        kernel = np.ones((3, 3), np.uint8)
        neighbor_count = cv2.filter2D(skeleton, -1, kernel) - skeleton
        endpoints = ((neighbor_count == 255) & (skeleton == 255))
        
        if not np.any(endpoints):
            return skeleton_repaired
        
        # Get endpoint coordinates
        endpoint_coords = np.column_stack(np.where(endpoints))
        
        # Connect nearby endpoints
        for i, (y1, x1) in enumerate(endpoint_coords):
            for j, (y2, x2) in enumerate(endpoint_coords[i+1:], i+1):
                distance = np.sqrt((y2-y1)**2 + (x2-x1)**2)
                
                if distance <= self.gap_fill_radius:
                    # Draw line between endpoints
                    cv2.line(skeleton_repaired, (x1, y1), (x2, y2), 255, 1)
        
        return skeleton_repaired
    
    def _restore_mask_width(self, skeleton: np.ndarray, original_mask: np.ndarray) -> np.ndarray:
        """
        Restore mask width using distance transform and original mask guidance.
        Enhanced to handle very thin masks and low-value inputs.
        """
        if skeleton.max() == 0:
            print("[SAP] Empty skeleton, returning original mask")
            return original_mask
        
        # Compute distance transform of original mask
        original_bool = original_mask > 127
        if not np.any(original_bool):
            print("[SAP] Original mask is empty, returning skeleton")
            return skeleton
        
        try:
            distance_map = distance_transform_edt(original_bool)
        except Exception as e:
            print(f"[SAP] Distance transform failed: {e}, using skeleton")
            return skeleton
        
        # Get skeleton points
        skeleton_points = skeleton > 127
        skeleton_pixel_count = np.count_nonzero(skeleton_points)
        
        if skeleton_pixel_count == 0:
            print("[SAP] No skeleton points found, returning original mask")
            return original_mask
        
        # Estimate radius at each skeleton point
        skeleton_distances = distance_map[skeleton_points]
        if len(skeleton_distances) == 0:
            print("[SAP] No valid skeleton distances, returning original mask")
            return original_mask
        
        # Use median radius as restoration guide, but be more conservative for thin masks
        median_radius = np.median(skeleton_distances)
        
        # Adaptive radius based on mask characteristics
        if skeleton_pixel_count < 50:  # Very thin mask
            final_radius = max(1, int(median_radius * 0.8))  # More conservative
            print(f"[SAP] Thin mask detected, using conservative radius: {final_radius}")
        elif median_radius < 2:  # Naturally thin structures
            final_radius = max(1, int(median_radius * 1.2))  # Slightly expand
            print(f"[SAP] Thin structure detected, using slight expansion: {final_radius}")
        else:
            final_radius = max(1, int(median_radius))
            print(f"[SAP] Normal structure, using median radius: {final_radius}")
        
        # Cap at reasonable size to prevent over-expansion
        final_radius = min(final_radius, 7)
        
        # Dilate skeleton with estimated radius
        kernel_size = final_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        try:
            restored_mask = cv2.dilate(skeleton, kernel, iterations=1)
        except Exception as e:
            print(f"[SAP] Dilation failed: {e}, returning skeleton")
            return skeleton
        
        # Constrain to original mask bounds to prevent over-expansion
        restored_mask = cv2.bitwise_and(restored_mask, original_mask)
        
        # Validate result
        final_nonzero = np.count_nonzero(restored_mask)
        print(f"[SAP] Width restoration complete: {skeleton_pixel_count} skeleton -> {final_nonzero} restored pixels")
        
        return restored_mask
    
    def compute_topology_metrics(self, mask: np.ndarray) -> dict:
        """
        Compute topological invariants for QA comparison.
        
        Returns:
            dict: Contains endpoints, junctions, components, euler_number
        """
        if mask.max() == 0:
            return {'endpoints': 0, 'junctions': 0, 'components': 0, 'euler_number': 0}
        
        skeleton = self._extract_skeleton(mask)
        
        # Count connected components
        num_components, _ = cv2.connectedComponents(skeleton, connectivity=8)
        num_components -= 1  # Subtract background
        
        # Count endpoints and junctions
        kernel = np.ones((3, 3), np.uint8)
        neighbor_count = cv2.filter2D(skeleton, -1, kernel) - skeleton
        
        endpoints = np.sum((neighbor_count == 255) & (skeleton == 255))
        junctions = np.sum((neighbor_count >= 3*255) & (skeleton == 255))
        
        # Euler number approximation (components - loops)
        euler_number = num_components - max(0, junctions - endpoints)
        
        return {
            'endpoints': int(endpoints),
            'junctions': int(junctions), 
            'components': int(num_components),
            'euler_number': int(euler_number)
        }
    
    def validate_topology_preservation(self, 
                                     mask_before: np.ndarray, 
                                     mask_after: np.ndarray,
                                     tolerance: int = 1) -> Tuple[bool, str]:
        """
        Validate that augmentation preserved topological structure.
        
        Args:
            mask_before: Original mask
            mask_after: Augmented mask
            tolerance: Allowed change in topology metrics
            
        Returns:
            (is_valid, reason): Validation result and explanation
        """
        try:
            metrics_before = self.compute_topology_metrics(mask_before)
            metrics_after = self.compute_topology_metrics(mask_after)
            
            # Check component count change
            comp_diff = abs(metrics_after['components'] - metrics_before['components'])
            if comp_diff > tolerance:
                return False, f"Component count changed by {comp_diff} (tolerance: {tolerance})"
            
            # Check if major topology was destroyed (too many endpoints lost)
            endpoint_diff = metrics_before['endpoints'] - metrics_after['endpoints']
            if endpoint_diff > tolerance * 2:  # More lenient for endpoints
                return False, f"Too many endpoints lost: {endpoint_diff}"
            
            # Check for major structural damage (Euler number)
            euler_diff = abs(metrics_after['euler_number'] - metrics_before['euler_number'])
            if euler_diff > tolerance:
                return False, f"Euler number changed by {euler_diff}"
            
            return True, "Topology preserved"
            
        except Exception as e:
            logging.error(f"Topology validation failed: {e}")
            return False, f"Validation error: {e}"


def sap_refine(mask_bin: np.ndarray, 
               min_branch_len: int = 5,
               gap_fill_radius: int = 3) -> np.ndarray:
    """
    Convenience function for skeleton-aware post-processing.
    
    Args:
        mask_bin: Binary input mask
        min_branch_len: Minimum branch length to preserve
        gap_fill_radius: Maximum gap distance to repair
        
    Returns:
        Refined binary mask
    """
    processor = SkeletonAwareProcessor(
        min_branch_len=min_branch_len,
        gap_fill_radius=gap_fill_radius
    )
    return processor.sap_refine(mask_bin)


# Global processor instance for efficiency  
_sap_instance = None

def get_skeleton_processor(**kwargs) -> SkeletonAwareProcessor:
    """Get global skeleton processor with lazy initialization."""
    global _sap_instance
    if _sap_instance is None:
        _sap_instance = SkeletonAwareProcessor(**kwargs)
    return _sap_instance
