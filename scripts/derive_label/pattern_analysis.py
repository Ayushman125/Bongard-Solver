# --- Robust Symmetry Score from Vertices ---
def symmetry_score_from_vertices(vertices):
    """
    Estimate the degree of reflection symmetry for a polygon/contour given by vertices.
    Returns (score, axis_angle_deg):
        score: float in [0,1], higher is more symmetric
        axis_angle_deg: angle (in degrees) of best symmetry axis (0 = horizontal, 90 = vertical)
    """
    import numpy as np
    if vertices is None or len(vertices) < 3:
        return 0.0, None
    verts = np.array(vertices)
    centroid = np.mean(verts, axis=0)
    verts_centered = verts - centroid
    best_score = 0.0
    best_axis = None
    # Test axes from 0 to 180 deg (step 10 deg for speed, can refine if needed)
    for angle_deg in np.arange(0, 180, 10):
        theta = np.deg2rad(angle_deg)
        axis = np.array([np.cos(theta), np.sin(theta)])
        # Reflect points across axis
        proj = np.dot(verts_centered, axis)
        reflected = verts_centered - 2 * np.outer(proj, axis)
        # For each reflected point, find closest original point
        dists = np.linalg.norm(reflected[:, None, :] - verts_centered[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        # Symmetry score: 1 - (mean normalized distance)
        norm = np.linalg.norm(verts_centered, axis=1).mean() + 1e-6
        score = 1.0 - (min_dists.mean() / norm)
        if score > best_score:
            best_score = score
            best_axis = angle_deg
    return float(np.clip(best_score, 0.0, 1.0)), float(best_axis) if best_axis is not None else None

# --- Boundary Autocorrelation Symmetry ---
def boundary_autocorr_symmetry(vertices):
    """
    Estimate symmetry by autocorrelation of boundary distances (rotation/reflection invariant).
    Returns a float in [0,1], higher is more symmetric.
    """
    import numpy as np
    if vertices is None or len(vertices) < 3:
        return 0.0
    verts = np.array(vertices)
    # Compute pairwise distances along the boundary (closed loop)
    n = len(verts)
    dists = np.linalg.norm(verts - np.roll(verts, -1, axis=0), axis=1)
    # Autocorrelation of the distance sequence
    dists_mean = dists - np.mean(dists)
    autocorr = np.correlate(dists_mean, dists_mean, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    # Normalize: peak at lag 0 is always max, look for secondary peaks
    if len(autocorr) < 2:
        return 0.0
    main_peak = autocorr[0]
    if main_peak == 0:
        return 0.0
    # Find the highest non-zero-lag peak (periodicity/symmetry)
    sec_peak = np.max(autocorr[1:])
    score = sec_peak / main_peak
    return float(np.clip(score, 0.0, 1.0))
import numpy as np
import logging
from shapely.geometry import Polygon, MultiPoint, LineString
from shapely.ops import unary_union
from scipy.spatial.distance import cdist # For efficient distance calculations
from itertools import combinations
import math

# Ensure calculate_confidence is available
from derive_label.confidence_scoring import calculate_confidence

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def detect_grid_pattern(centers, alignment_tolerance=10, uniformity_threshold=0.15):
    """
    Detects if a set of centers forms a grid pattern using clustering on projected axes.

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    This version focuses on robustness and explicit alignment checks.
    Returns (is_grid, grid_info, confidence)
    """
    try:
        from sklearn.cluster import MiniBatchKMeans # More scalable than KMeans

        centers_np = np.array(centers)
        n_points = len(centers_np)
        if n_points < 4: # Need at least 2x2 for a grid
            return False, None, 0.0

        xs = centers_np[:, 0].reshape(-1, 1)
        ys = centers_np[:, 1].reshape(-1, 1)

        best_grid_conf = 0.0
        best_grid_info = None
        is_best_grid = False

        # Iterate through possible number of rows/columns
        # Sensible max: up to sqrt(n_points) or 10, whichever is smaller, to avoid overfitting to noise
        max_dim_k = min(int(np.sqrt(n_points)) + 1, 10)

        for k_x in range(2, max_dim_k + 1):
            for k_y in range(2, max_dim_k + 1):
                if k_x * k_y > n_points * 2: # Avoid trying too many clusters for sparse points
                    continue

                try:
                    # Cluster X and Y coordinates
                    kmeans_x = MiniBatchKMeans(n_clusters=k_x, n_init='auto', random_state=0, batch_size=256).fit(xs)
                    kmeans_y = MiniBatchKMeans(n_clusters=k_y, n_init='auto', random_state=0, batch_size=256).fit(ys)
                except ValueError: # Occurs if n_clusters > n_samples
                    continue

                grid_x_lines = np.sort(kmeans_x.cluster_centers_.flatten())
                grid_y_lines = np.sort(kmeans_y.cluster_centers_.flatten())

                # Check for uniformity of spacing between grid lines
                dx_spacings = np.diff(grid_x_lines)
                dy_spacings = np.diff(grid_y_lines)

                uniformity_x = np.std(dx_spacings) / (np.mean(dx_spacings) + 1e-6) if len(dx_spacings) > 0 else 0.0
                uniformity_y = np.std(dy_spacings) / (np.mean(dy_spacings) + 1e-6) if len(dy_spacings) > 0 else 0.0

                is_uniform_x = uniformity_x < uniformity_threshold
                is_uniform_y = uniformity_y < uniformity_threshold

                if not is_uniform_x or not is_uniform_y:
                    continue # Not a uniform grid

                # Check how many points are aligned to grid intersections
                aligned_points_count = 0
                for c in centers_np:
                    is_aligned_x = np.min(np.abs(c[0] - grid_x_lines)) < alignment_tolerance
                    is_aligned_y = np.min(np.abs(c[1] - grid_y_lines)) < alignment_tolerance
                    if is_aligned_x and is_aligned_y:
                        aligned_points_count += 1

                alignment_ratio = aligned_points_count / n_points

                # Confidence combines uniformity, alignment, and density of grid points
                # If the actual number of points is close to k_x * k_y (a full grid) then confidence is higher
                density_conf = calculate_confidence(n_points, max_value=k_x * k_y, min_value=0.0) # Higher if more points fill the grid cells

                current_conf = (
                    (1 - uniformity_x) * (1 - uniformity_y) * # Penalize non-uniformity
                    alignment_ratio * # Reward aligned points
                    (0.5 + 0.5 * density_conf) # Reward density of points in grid cells
                )
                current_conf = np.clip(current_conf, 0.0, 1.0)

                if current_conf > best_grid_conf:
                    best_grid_conf = current_conf
                    best_grid_info = {
                        'n_rows': k_y, 'n_cols': k_x, # Rows typically map to Y, Cols to X
                        'grid_spacing_x': float(np.mean(dx_spacings)) if len(dx_spacings) > 0 else 0.0,
                        'grid_spacing_y': float(np.mean(dy_spacings)) if len(dy_spacings) > 0 else 0.0,
                        'alignment_ratio': float(alignment_ratio),
                        'uniformity_x': float(uniformity_x),
                        'uniformity_y': float(uniformity_y)
                    }
                    if alignment_ratio > 0.7 and current_conf > 0.6: # Stricter condition for 'is_grid' flag
                        is_best_grid = True

        # Final decision on is_grid
        if is_best_grid: # Only return True if the best grid is highly confident and well-aligned
            return True, best_grid_info, best_grid_conf

        return False, best_grid_info, best_grid_conf # Always return best info, even if not 'is_grid'
    except ImportError:
        logging.debug("[PatternAnalysis] sklearn (MiniBatchKMeans) not installed, skipping grid pattern detection.")
        return False, None, 0.0
    except Exception as e:
        logging.error(f"[GridPattern] Error in detect_grid_pattern: {e}")
        return False, None, 0.0


def detect_periodicity(centers, frequency_threshold=0.1):
    """
    Detects periodicity in a set of centers using Fast Fourier Transform (FFT) on 1D projections.
    More robust for various periodicities than simple autocorrelation.
    Returns (is_periodic, dominant_periods, confidence)
    """

    centers_np = np.array(centers)
    n_points = len(centers_np)
    if n_points < 3: return False, [], 0.0

    all_periods = []
    confidences = []

    alignment_tolerance = 5.0  # Default value for alignment tolerance (pixels)

    for dim_idx, coords in enumerate([centers_np[:, 0], centers_np[:, 1]]):
        if len(coords) < 3: continue

        # Sort coordinates to represent spatial sequence
        sorted_coords = np.sort(coords)

        # Consider the distances between points instead of raw coordinates for periodicity
        diffs = np.diff(sorted_coords)
        if len(diffs) < 2 or np.sum(diffs) == 0: continue # Need enough differences

        # Use FFT on a histogram/density estimate of the differences
        # Create a "signal" representing the distribution of point differences
        hist, bin_edges = np.histogram(diffs, bins='auto', density=True)

        if len(hist) < 2: continue

        # Perform FFT on the histogram
        fft_result = np.fft.fft(hist)
        power_spectrum = np.abs(fft_result)**2

        # Frequencies corresponding to bins
        sample_freq = 1.0 / (bin_edges[1] - bin_edges[0]) # Frequency of sampling in diffs space
        frequencies = np.fft.fftfreq(len(hist), d=1/sample_freq) # Frequencies in 1/pixel units

        # Find dominant frequencies (excluding DC component, which is fft_result[0])
        # Look for peaks in the power spectrum, corresponding to periods
        # Max frequency index (ignoring negative frequencies and DC component)
        positive_freq_idx = np.where(frequencies > 0)[0]
        if len(positive_freq_idx) == 0: continue

        dominant_freq_idx = positive_freq_idx[np.argmax(power_spectrum[positive_freq_idx])]
        dominant_freq = frequencies[dominant_freq_idx]

        if dominant_freq > 1e-6: # Avoid division by zero and near-zero frequencies
            dominant_period = 1.0 / dominant_freq

            # Confidence for periodicity: strength of the dominant peak relative to total power
            total_power = np.sum(power_spectrum)
            if total_power > 0:
                peak_power = power_spectrum[dominant_freq_idx]
                current_conf = calculate_confidence(peak_power, max_value=total_power, min_value=0.0)

                # Further refine confidence: how many points are actually aligned to this period
                aligned_points_count = 0
                for i in range(1, len(sorted_coords)):
                    if np.isclose((sorted_coords[i] - sorted_coords[0]) % dominant_period, 0, atol=alignment_tolerance) or \
                       np.isclose((sorted_coords[i] - sorted_coords[0]) % dominant_period, dominant_period, atol=alignment_tolerance):
                        aligned_points_count += 1

                alignment_ratio = aligned_points_count / n_points
                current_conf *= alignment_ratio # Penalize if points don't align well

                if current_conf > frequency_threshold: # Only add if confident enough
                    all_periods.append({'period': float(dominant_period), 'dimension': 'x' if dim_idx == 0 else 'y', 'confidence': float(current_conf)})
                    confidences.append(current_conf)

    if not all_periods:
        return False, [], 0.0
    
    # Aggregate confidence: take the max confidence from either dimension
    overall_confidence = max(confidences) if confidences else 0.0
    is_periodic = overall_confidence > 0.6 # A higher threshold for overall periodicity

    return is_periodic, all_periods, overall_confidence


def detect_tiling(polygons, scene_bbox, fill_threshold=0.8, overlap_threshold=0.05):
    """
    Checks if objects fill the scene area densely and are non-overlapping, forming a tiling.
    Accepts a list of shapely Polygon objects.
    Returns (is_tiling, fill_ratio, overlap_ratio, confidence)
    """
    valid_polygons = [p for p in polygons if p is not None and p.is_valid and not p.is_empty]
    
    if not valid_polygons:
        return False, 0.0, 0.0, 0.0

    # Calculate total area of individual valid polygons
    total_obj_area = sum(p.area for p in valid_polygons)

    # Use unary_union for precise overlap detection
    # The area of the union will be less than the sum of individual areas if there's overlap.
    union_poly = unary_union(valid_polygons)
    union_area = union_poly.area
    
    # Calculate overlap area (sum of individual areas - area of their union)
    total_overlap_area = total_obj_area - union_area
    total_overlap_area = max(0.0, total_overlap_area) # Ensure non-negative

    # Determine the effective scene area based on the bounding box of all polygons, not external.
    # This makes the fill ratio more meaningful if the scene_bbox provided is much larger than the objects.
    # Or use the provided scene_bbox if it's more specific.
    if union_poly.is_empty:
        effective_scene_area = 0.0
    else:
        # Use the union's bounding box for effective scene area
        minx, miny, maxx, maxy = union_poly.bounds
        effective_scene_area = (maxx - minx) * (maxy - miny)
        # If an external scene_bbox is provided and is larger, use its area
        if scene_bbox and (scene_bbox[2]-scene_bbox[0])*(scene_bbox[3]-scene_bbox[1]) > effective_scene_area:
             effective_scene_area = (scene_bbox[2]-scene_bbox[0])*(scene_bbox[3]-scene_bbox[1])


    fill_ratio = union_area / (effective_scene_area + 1e-6) if effective_scene_area > 0 else 0.0
    
    # overlap_ratio: proportion of total object area that is overlapping
    # This metric is better as total_overlap_area / total_obj_area.
    overlap_ratio = total_overlap_area / (total_obj_area + 1e-6) if total_obj_area > 0 else 0.0
    
    # Tiling implies high fill ratio and low overlap ratio
    is_tiling = fill_ratio > fill_threshold and overlap_ratio < overlap_threshold
    
    # Confidence: weighted average of fill_ratio and (1 - overlap_ratio)
    confidence = (calculate_confidence(fill_ratio, max_value=1.0, min_value=0.5) * 0.7 + 
                  calculate_confidence(overlap_ratio, max_value=0.3, min_value=0.0, is_higher_better=False) * 0.3)
    
    return is_tiling, float(fill_ratio), float(overlap_ratio), float(confidence)

def detect_rotation_scale_reflection(vertices_np, ref_shape_vertices_np, max_disparity=0.1):
    """
    Compares two shapes for transformation (rotation, scale, reflection) using Procrustes analysis.
    Returns (transformation_info, confidence).
    Confidence is inversely related to disparity.
    """
    if vertices_np is None or ref_shape_vertices_np is None or \
       vertices_np.shape[0] < 2 or ref_shape_vertices_np.shape[0] < 2:
        return None, 0.0
    
    try:
        from scipy.spatial import procrustes
        
        n_pts_current = vertices_np.shape[0]
        n_pts_ref = ref_shape_vertices_np.shape[0]
        
        if n_pts_current == 0 or n_pts_ref == 0: return None, 0.0

        # Resample to the minimum number of points for Procrustes analysis
        min_pts = min(n_pts_current, n_pts_ref)
        if min_pts < 2: return None, 0.0 # Procrustes needs at least 2 points

        # Linear interpolation for resampling. Ensure indices are floats for interp.
        interp_x_current = np.interp(np.linspace(0, n_pts_current - 1, min_pts), np.arange(n_pts_current), vertices_np[:, 0])
        interp_y_current = np.interp(np.linspace(0, n_pts_current - 1, min_pts), np.arange(n_pts_current), vertices_np[:, 1])
        resampled_current = np.column_stack([interp_x_current, interp_y_current])

        interp_x_ref = np.interp(np.linspace(0, n_pts_ref - 1, min_pts), np.arange(n_pts_ref), ref_shape_vertices_np[:, 0])
        interp_y_ref = np.interp(np.linspace(0, n_pts_ref - 1, min_pts), np.arange(n_pts_ref), ref_shape_vertices_np[:, 1])
        resampled_ref = np.column_stack([interp_x_ref, interp_y_ref])

        # Procrustes analysis
        mtx1, mtx2, disparity = procrustes(resampled_ref, resampled_current)
        
        # Estimate rotation angle, scale, and reflection from the transformation matrix
        # SVD can be unstable if matrices are singular (e.g., all points collinear or identical)
        if mtx1.shape[0] < 2 or mtx2.shape[0] < 2: return None, 0.0
        
        try:
            # Ensure matrices are full rank or handle gracefully.
            # Adding a small epsilon to diagonal can sometimes help with singular matrices in SVD,
            # but it might also distort results. Best to catch the error.
            u, s, vh = np.linalg.svd(np.dot(mtx1.T, mtx2))
            r = np.dot(u, vh)
            
            det_r = np.linalg.det(r)
            if abs(det_r) < 1e-6: # Treat as no rotation/reflection if singular or near singular
                angle = 0.0
                reflection = False
            else:
                angle = np.arctan2(r[1, 0], r[0, 0]) * 180 / np.pi
                reflection = det_r < 0 # Reflection if determinant is negative
        except np.linalg.LinAlgError:
            logging.warning(f"[Transformation] SVD failed due to degenerate matrix. Setting rotation/reflection to default.")
            angle = 0.0
            reflection = False

        norm_mtx1 = np.linalg.norm(mtx1)
        norm_mtx2 = np.linalg.norm(mtx2)
        scale = norm_mtx2 / (norm_mtx1 + 1e-6) if norm_mtx1 > 0 else 1.0

        # Confidence: inversely related to disparity (lower disparity = higher confidence)
        # Scale disparity to a meaningful range for confidence (0 = perfect match, max_disparity = no match)
        confidence = calculate_confidence(disparity, max_value=max_disparity, min_value=0.0, is_higher_better=False)
        
        transformation_info = {
            'rotation': float(angle),
            'scale': float(scale),
            'reflection': bool(reflection),
            'disparity': float(disparity)
        }
        return transformation_info, confidence
    except ImportError:
        logging.debug("[PatternAnalysis] scipy not installed, skipping transformation detection.")
        return None, 0.0
    except Exception as e:
        logging.warning(f"[Transformation] Error during transformation detection: {e}")
        return None, 0.0

def triplet_relations(centers, collinearity_tolerance=5.0, angle_tolerance=10.0, distance_ratio_tolerance=0.1):
    """
    Detects various spatial triplet relations (collinear, equidistant, right-angle).
    Returns (triplets_list, confidence)
    """
    n = len(centers)
    triplets_detected = []
    centers_np = np.array(centers)

    if n < 3:
        return [], 0.0

    total_possible_triplets = 0
    num_strong_relations = 0

    # Iterate through all unique combinations of three points
    for i, j, k in combinations(range(n), 3):
        p1, p2, p3 = centers_np[i], centers_np[j], centers_np[k]
        
        total_possible_triplets += 1
        
        current_triplet_relations = []
        
        # 1. Collinearity Check
        # Area of triangle formed by three points should be close to zero.
        area = 0.5 * np.abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))
        
        # Maximum possible area for typical ranges (e.g., 256x256 image, can be much larger)
        # Using a relative tolerance might be better, or absolute for pixels.
        # For simplicity, using a fixed pixel area threshold.
        if area < collinearity_tolerance: # Threshold for collinearity (area in px^2)
            current_triplet_relations.append({'type': 'collinear', 'area_error': float(area)})
            
        # 2. Equidistance Check (Isosceles or Equilateral Triangle)
        d12 = np.linalg.norm(p1 - p2)
        d23 = np.linalg.norm(p2 - p3)
        d31 = np.linalg.norm(p3 - p1)
        distances = np.sort([d12, d23, d31])

        # Check for approximate equality of side lengths
        if distances[0] > 1e-6: # Avoid division by zero
            if abs(distances[1] - distances[0]) / distances[0] < distance_ratio_tolerance: # Isosceles
                current_triplet_relations.append({'type': 'equidistant_isosceles'})
                if abs(distances[2] - distances[0]) / distances[0] < distance_ratio_tolerance: # Equilateral
                    current_triplet_relations[-1]['type'] = 'equidistant_equilateral'
            elif abs(distances[2] - distances[1]) / distances[1] < distance_ratio_tolerance: # Isosceles (other pair)
                current_triplet_relations.append({'type': 'equidistant_isosceles'})

        # 3. Right-Angle Check
        # Calculate angles at each vertex
        angles_deg = []
        vectors = [p2 - p1, p3 - p2, p1 - p3] # Vectors forming the triangle
        points_in_order = [p1, p2, p3, p1] # For iterating edges
        
        for idx_vertex in range(3):
            v_in = points_in_order[idx_vertex + 1] - points_in_order[idx_vertex]
            v_out = points_in_order[idx_vertex + 2] - points_in_order[idx_vertex + 1]
            
            # Use dot product to find angle at vertex (p_curr)
            # p_prev -> p_curr -> p_next
            p_prev_tri = centers_np[[i,j,k][(idx_vertex-1+3)%3]]
            p_curr_tri = centers_np[[i,j,k][idx_vertex]]
            p_next_tri = centers_np[[i,j,k][(idx_vertex+1)%3]]

            vec1 = p_prev_tri - p_curr_tri
            vec2 = p_next_tri - p_curr_tri
            
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            
            if norm_vec1 > 1e-6 and norm_vec2 > 1e-6:
                cosine_angle = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
                angle_deg = np.degrees(np.arccos(np.clip(cosine_angle, -1, 1)))
                angles_deg.append(angle_deg)
                if abs(angle_deg - 90) < angle_tolerance:
                    current_triplet_relations.append({'type': 'right_angle', 'angle_deg': float(angle_deg)})
        
        if current_triplet_relations:
            triplets_detected.append({
                'points_indices': [i, j, k],
                'relations': current_triplet_relations
            })
            num_strong_relations += 1

    # Confidence: ratio of detected strong triplets to total possible triplets
    confidence = calculate_confidence(num_strong_relations, max_value=total_possible_triplets, min_value=0) if total_possible_triplets > 0 else 0.0
    
    return triplets_detected, confidence

