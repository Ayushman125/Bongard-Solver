import numpy as np
import cv2
import logging
from skimage.morphology import footprint_rectangle, closing, opening, medial_axis, skeletonize
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.draw import polygon as skpolygon
from shapely.geometry import Polygon # For robust area/perimeter calculations

# Ensure calculate_confidence is available
from .confidence_scoring import calculate_confidence

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_clean_mask_and_skeleton(vertices_np, img_size_px=64):
    """
    Extracts a clean binary mask and skeleton from vertices, with contour hierarchy, adaptive thresholding,
    morphological cleaning, and fallback logic. Returns mask, skeleton, and QA metrics.
    """
    if vertices_np is None or vertices_np.size == 0 or vertices_np.shape[0] < 1:
        return np.zeros((img_size_px, img_size_px), dtype=np.uint8), np.zeros((img_size_px, img_size_px), dtype=np.uint8), {'area': 0, 'skeleton_area': 0, 'fill_ratio': 0.0}
    img = _rasterize_vertices_to_image(vertices_np, img_size_px=img_size_px)
    thresh_val = threshold_otsu(img) if np.any(img) else 0
    mask = (img > thresh_val).astype(np.uint8)
    sq_footprint = np.ones((3, 3), dtype=bool)
    mask = closing(opening(mask, sq_footprint), sq_footprint)
    try:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        contours, hierarchy = [], None
    skeleton = skeletonize(mask > 0)
    area = np.sum(mask)
    skeleton_area = np.sum(skeleton)
    fill_ratio = skeleton_area / (area + 1e-6)
    if area == 0 or fill_ratio < 0.05:
        edges = cv2.Canny(img, 50, 150)
        mask = closing(opening(edges, sq_footprint), sq_footprint)
        skeleton = skeletonize(mask > 0)
    return mask, skeleton, {'area': area, 'skeleton_area': skeleton_area, 'fill_ratio': fill_ratio}


def _rasterize_vertices_to_image(vertices_np, img_size_px=64, thickness=2, is_closed=False):
    """
    Helper to safely rasterize a set of vertices onto a small image.
    Handles single points, lines, and polygons.
    Ensures consistent image generation for various feature extractors.
    """
    if vertices_np is None or vertices_np.size == 0 or vertices_np.shape[0] < 1:
        return np.zeros((img_size_px, img_size_px), dtype=np.uint8)

    pts = np.round(vertices_np).astype(np.int32)

    # Shift points to positive coordinates with a margin
    min_coords = pts.min(axis=0)
    max_coords = pts.max(axis=0)

    # Calculate required image dimensions based on point spread + margin
    span_x = max_coords[0] - min_coords[0]
    span_y = max_coords[1] - min_coords[1]
    
    # Ensure min size, but scale larger if needed
    scaling_factor = 1.0
    if max(span_x, span_y) > img_size_px - 20: # If it's larger than nominal internal area
        scaling_factor = (img_size_px - 20) / max(span_x, span_y)
    
    scaled_pts = (pts - min_coords) * scaling_factor + 10 # Add 10px margin
    
    img_h = int(span_y * scaling_factor) + 20
    img_w = int(span_x * scaling_factor) + 20
    
    # Ensure minimum image size for processing
    target_img_h = max(img_h, img_size_px)
    target_img_w = max(img_w, img_size_px)

    img = np.zeros((target_img_h, target_img_w), dtype=np.uint8)

    if scaled_pts.shape[0] == 1:
        # Ensure coordinates are integers for OpenCV
        center = tuple(int(round(x)) for x in scaled_pts[0])
        cv2.circle(img, center, thickness, 255, -1) # Draw a filled circle for a point
    else:
        # Ensure correct shape for OpenCV: (N, 1, 2)
        pts_for_cv = scaled_pts.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img, [pts_for_cv], isClosed=is_closed, color=255, thickness=thickness)
        
        # If it's meant to be a filled polygon, also fill it
        if is_closed and scaled_pts.shape[0] >= 3:
            # Use skimage.draw.polygon for robust filling, handles self-intersections better
            try:
                rr, cc = skpolygon(scaled_pts[:, 1], scaled_pts[:, 0], img.shape)
                img[rr, cc] = 255 # Fill the shape
            except Exception as e:
                logging.warning(f"[Rasterize] Failed to fill polygon with skpolygon: {e}. Falling back to outline.")

    return img





def _normalize_contours_hierarchy(cnts_result):
    """
    Helper to normalize the output of cv2.findContours across OpenCV versions.
    Returns (flat_contours_list, hierarchy_array or None)
    """
    # Peel off any singleton tuple nesting (common in older OpenCV versions)
    while isinstance(cnts_result, (list, tuple)) and len(cnts_result) == 1 and isinstance(cnts_result[0], (list, tuple)):
        cnts_result = cnts_result[0]
    
    # Extract contours and hierarchy - they are always the last two elements
    if isinstance(cnts_result, (list, tuple)) and len(cnts_result) >= 2:
        raw_contours, raw_hierarchy = cnts_result[-2], cnts_result[-1]
    else:
        raise ValueError(f"findContours returned unexpected structure: {type(cnts_result)}, {cnts_result}")

    # Flatten any nested structure in contours (sometimes list of lists of arrays)
    flat_contours = []
    def _flatten_recursive(conts):
        if isinstance(conts, np.ndarray) and conts.ndim >= 2:
            flat_contours.append(conts)
        elif isinstance(conts, (list, tuple)):
            for c in conts:
                _flatten_recursive(c)
    _flatten_recursive(raw_contours)

    hierarchy = raw_hierarchy if isinstance(raw_hierarchy, np.ndarray) and raw_hierarchy.ndim >= 2 else None
    
    return flat_contours, hierarchy


def image_processing_features(vertices_np):
    """
    Extracts image-processing based features from shape vertices.
    Combines various techniques for a comprehensive set of features.
    Returns a dictionary of features.
    """
    import logging
    features = {}

    logging.debug(f"[ImageFeatures] Input vertices_np: shape={getattr(vertices_np, 'shape', None)}, dtype={getattr(vertices_np, 'dtype', None)}, sample={vertices_np[:5] if hasattr(vertices_np, '__getitem__') else vertices_np}")

    if vertices_np.size == 0 or vertices_np.shape[0] < 1:
        logging.debug("[ImageFeatures] Empty input vertices. Returning empty features.")
        return features

    # --- Robust mask extraction, skeletonization, QA ---
    mask, skeleton, qa = extract_clean_mask_and_skeleton(vertices_np, img_size_px=128)
    features['robust_mask_area'] = int(np.sum(mask))
    features['robust_skeleton_area'] = int(np.sum(skeleton))
    features['robust_fill_ratio'] = qa.get('fill_ratio', 0.0)
    if features['robust_mask_area'] == 0:
        features['rasterization_empty'] = True
        logging.debug("[ImageFeatures] Rasterization produced empty mask. Returning early.")
        return features

    img_binary = mask

    try:
        # --- Basic Image Properties ---
        features['image_area_pixels'] = int(np.sum(img_binary > 0)) # Total white pixels
        features['image_density'] = float(features['image_area_pixels'] / (img_binary.size + 1e-6))

        # --- Contour Analysis ---
        cnts_raw = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = _normalize_contours_hierarchy(cnts_raw)
        
        features['n_contours'] = len(contours)
        features['contour_lengths'] = []
        features['contour_areas'] = []
        
        simplified_contours_list = []
        simplified_n_vertices_list = []

        for cnt in contours:
            if cv2.contourArea(cnt) > 0: # Only process non-empty contours
                features['contour_lengths'].append(float(cv2.arcLength(cnt, True)))
                features['contour_areas'].append(float(cv2.contourArea(cnt)))

                # Contour simplification (Douglas-Peucker)
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                if epsilon > 0:
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    # Ensure approx is a list of 2D points for JSON serialization
                    if approx is not None:
                        approx_flat = [[float(pt[0][0]), float(pt[0][1])] for pt in approx.reshape(-1, 1, 2)]
                        simplified_contours_list.append(approx_flat)
                        simplified_n_vertices_list.append(len(approx_flat))
                else: # Degenerate contour or very short line, cannot simplify meaningfully
                    simplified_contours_list.append([])
                    simplified_n_vertices_list.append(0)
        
        features['simplified_contours'] = simplified_contours_list
        features['simplified_n_vertices'] = simplified_n_vertices_list

        # --- Hole Counting (using hierarchy) ---
        n_holes = 0
        if hierarchy is not None and hierarchy.shape[1] > 0:
            for h_idx in range(hierarchy.shape[1]):
                # A contour is a hole if its parent is not -1 (i.e., it has an outer contour)
                if hierarchy[0, h_idx, 3] != -1: 
                    n_holes += 1
        features['n_holes'] = n_holes

        # --- Skeletonization / Medial Axis ---
        # `skel` is the skeleton, `dist` is the distance transform
        skel, dist = medial_axis(img_binary, return_distance=True)
        features['skeleton_sum_pixels'] = int(np.sum(skel)) # Total pixels in skeleton
        features['max_medial_axis_distance'] = float(np.max(dist)) if dist.size > 0 else 0.0

        # --- Morphological Features ---
        if np.sum(img_binary) > 0: # Only if image is not empty
            # Use fixed 3x3 square footprint for general purpose cleaning
            sq_footprint = np.ones((3, 3), dtype=bool)
            
            # Close gaps and fill small holes
            closed_img = closing(img_binary, sq_footprint)
            features['morph_closed_area'] = int(np.sum(closed_img))

            # Remove small objects or noise
            opened_img = opening(closed_img, sq_footprint)
            features['morph_opened_area'] = int(np.sum(opened_img))
        else:
            features['morph_closed_area'] = 0
            features['morph_opened_area'] = 0

        # --- Region Properties (from skimage.measure.regionprops) ---
        if np.sum(img_binary) > 0: # Only if image is not empty
            labeled_img = label(img_binary)
            props = regionprops(labeled_img)
            if props:
                main_prop = props[0] # Assuming largest region is the main shape
                features['bbox_area'] = int(main_prop.bbox_area) # Area of bounding box
                features['eccentricity'] = float(main_prop.eccentricity) # How elongated the shape is
                features['major_axis_length'] = float(main_prop.major_axis_length)
                features['minor_axis_length'] = float(main_prop.minor_axis_length)
                features['orientation_rad'] = float(main_prop.orientation) # Orientation in radians
                features['perimeter'] = float(main_prop.perimeter) # From regionprops (not arcLength)
                features['solidity'] = float(main_prop.solidity) # Area / Convex Hull Area
                features['extent'] = float(main_prop.extent) # Area / Bounding Box Area
                features['form_factor'] = (4 * np.pi * main_prop.area) / (main_prop.perimeter**2 + 1e-6) # Circle is 1, jagged/elongated < 1
            else:
                # Default values if no regionprops found
                features['bbox_area'] = 0
                features['eccentricity'] = 0.0
                features['major_axis_length'] = 0.0
                features['minor_axis_length'] = 0.0
                features['orientation_rad'] = 0.0
                features['perimeter'] = 0.0
                features['solidity'] = 0.0
                features['extent'] = 0.0
                features['form_factor'] = 0.0
        else:
            features['bbox_area'] = 0
            features['eccentricity'] = 0.0
            features['major_axis_length'] = 0.0
            features['minor_axis_length'] = 0.0
            features['orientation_rad'] = 0.0
            features['perimeter'] = 0.0
            features['solidity'] = 0.0
            features['extent'] = 0.0
            features['form_factor'] = 0.0

        # --- Hu Moments (Shape descriptors, invariant to scale, rotation, reflection) ---
        # Convert binary image to CV_8UC1 for moments
        img_uint8 = (img_binary * 255).astype(np.uint8)
        moments = cv2.moments(img_uint8)
        if moments['m00'] != 0: # Only if shape has area
            hu_moments = cv2.HuMoments(moments)
            # Log transform for better scale-invariance and smaller numbers
            features['hu_moments'] = [-np.sign(h) * np.log10(abs(h) + 1e-6) if abs(h) > 0 else 0.0 for h in hu_moments.flatten()]
        else:
            features['hu_moments'] = [0.0] * 7 # 7 Hu moments

        # --- Canny Edge Density ---
        edges = cv2.Canny(img_uint8, 50, 150) # Apply Canny on the 8-bit image
        features['canny_edge_pixels'] = int(np.sum(edges > 0))
        features['canny_edge_density'] = float(features['canny_edge_pixels'] / (edges.size + 1e-6))

        # --- Otsu threshold (for analyzing intensity distribution of underlying pixel data) ---
        # Note: 'img_binary' is already binary. This is more useful if original pixel data were grayscale.
        # For a binary shape, it will mostly be 1s and 0s. The threshold will usually be 0 or 1.
        # Keeping for completeness if the input image were not pre-binarized.
        try:
            if np.unique(img_uint8).size > 1:
                thresh_val, _ = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                features['otsu_threshold_value'] = float(thresh_val)
            else:
                features['otsu_threshold_value'] = 0.0
        except Exception:
            features['otsu_threshold_value'] = 0.0 # Default if Otsu fails (e.g., all same pixel value)
        
        logging.debug(f"[ImageFeatures] Extracted features: {features}")
        return features
    except Exception as e:
        features['image_processing_error'] = str(e)
        logging.error(f"[ImageProcessing] Critical Error: {e}", exc_info=True)
        return features

def compute_euler_characteristic(vertices_np):
    """
    Computes the Euler characteristic for a simple polygon.
    For a simple polygon (no holes, single connected component), V - E + F = 1.
    If it has holes, it changes.
    """
    # This is a topological property best derived from a valid polygon structure.
    # For now, a simple closed polygon has V - E + F = 1.
    # The complexity comes from multiple components or holes.
    # This function is somewhat redundant with n_holes from image_processing_features
    # and persistent_homology_features (Betti numbers).
    
    # Simple case: V - E + F = 1 for a single, closed, hole-less loop.
    # Euler characteristic = (Num Connected Components) - (Num Holes)
    # This is better derived from the image or topology features.
    
    # For now, we'll return a placeholder based on general expectations.
    # Better to use n_holes from image_processing_features and Betti[0] for connected components.
    
    # Let's align with the definition from topology: V - E + F = 1 for a planar graph of a polygon.
    # If using as (Connected Components - Holes), then it needs those inputs.
    
    # As this specific function doesn't compute components/holes, we'll provide a basic estimate.
    if vertices_np.shape[0] < 3: return 0.0 # Not a polygon
    
    # If the shape is closed (implicitly from original use case)
    V = len(vertices_np)
    E = V # Assuming it forms a closed polyline/polygon
    F = 1 # One face (the interior of the polygon)
    
    # A confidence here is hard without explicit hole/component info.
    # This function is likely to be subsumed by image_processing_features or persistent_homology.
    return float(V - E + F) # Should be 1.0 for a simple polygon.

def persistent_homology_features(vertices_np):
    """
    Computes persistent homology features (Betti numbers).
    Requires gudhi or ripser. Provides robust fallback.
    Returns (betti_numbers, confidence)
    Betti[0]: number of connected components
    Betti[1]: number of holes
    """
    try:
        import gudhi as gd
        pts = vertices_np.astype(float)
        if pts.shape[0] < 2: return [1, 0], 0.1 # 1 connected component, 0 holes for < 2 points, low confidence

        # Create a Rips complex for point clouds
        # Alpha complex might also be suitable if points define boundaries of shapes.
        rips = gd.RipsComplex(points=pts)
        st = rips.create_simplex_tree(max_dimension=2) # Max dimension 2 to capture connected components and holes
        
        # Compute persistence
        st.compute_persistence()
        
        # Get Betti numbers
        betti = st.betti_numbers() 
        # Betti numbers are typically for specific dimensions, e.g., betti[0] for 0-dim (connected components)
        # betti[1] for 1-dim (loops/holes). Ensure consistency in output length.
        
        # Ensure betti has at least 2 elements for B0 and B1
        b0 = betti[0] if len(betti) > 0 else 1 # Default to 1 connected component
        b1 = betti[1] if len(betti) > 1 else 0 # Default to 0 holes

        # Confidence based on simplicity of topology:
        # Lower B0 and B1 usually means higher confidence in clean topology.
        # Penalty for high number of components or holes.
        complexity_score = (b0 - 1) + b1 * 2 # Penalize holes more than extra components
        confidence = calculate_confidence(complexity_score, max_value=5.0, min_value=0.0, is_higher_better=False) # 0 defects is perfect, higher is worse
        confidence = np.clip(confidence, 0.0, 1.0) # Ensure within [0,1]
        
        return [b0, b1], confidence
    except ImportError:
        logging.debug(f"[ImageFeatures] Gudhi not installed. Returning default persistent homology features.")
        return [1, 0], 0.0 # Default for 1 component, 0 holes
    except Exception as e:
        logging.error(f"[ImageFeatures] Gudhi error in persistent_homology_features: {e}. Returning default.")
        return [1, 0], 0.0


# Removed 'count_holes_shapely' and 'contour_hierarchy_holes'
# as their functionality is now integrated and enhanced within 'image_processing_features'.

# Removed 'canny_edge_density' as its functionality is now integrated and enhanced within 'image_processing_features'.

