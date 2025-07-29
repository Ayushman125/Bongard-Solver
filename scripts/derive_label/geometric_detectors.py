from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

class StackingEnsemble:
    def __init__(self, base_detectors):
        self.detectors = base_detectors
        self.meta = RandomForestClassifier(n_estimators=50)
        self._fitted = False

    def fit(self, images, true_labels):
        X = []
        for img in images:
            feats = np.hstack([det(img)[1] for det in self.detectors])
            X.append(feats)
        self.meta.fit(np.vstack(X), true_labels)
        self._fitted = True

    def predict(self, image):
        feats = np.hstack([det(image)[1] for det in self.detectors]).reshape(1,-1)
        return self.meta.predict(feats)[0]

class DetectorAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, embeddings):
        Q = K = V = self.linear(embeddings).unsqueeze(1)
        attn_out, _ = self.attn(Q, K, V)
        return attn_out.squeeze(1).mean(dim=0)

import numpy as np
from collections import Counter

class WeightedVotingEnsemble:
    def __init__(self, detectors, weights):
        self.detectors = detectors
        self.weights = weights

    def predict(self, image, context_feat=None, return_logits=False):
        # Each detector returns (label, score, logits)
        label_votes = Counter()
        all_logits = []
        all_labels = []
        for i, detector in enumerate(self.detectors):
            # Detector must return (label, score, logits)
            if context_feat is not None:
                label, score, logits = detector(image, context_feat=context_feat, return_logits=True)
            else:
                label, score, logits = detector(image, return_logits=True)
            label_votes[label] += score * self.weights[i]
            all_logits.append(logits)
            all_labels.append(label)
        # Fused label: highest weighted vote
        best_label, _ = label_votes.most_common(1)[0]
        # For logits, average across detectors
        avg_logits = np.mean(np.stack(all_logits), axis=0)
        if return_logits:
            return all_labels, [float(label_votes[l]) for l in all_labels], avg_logits
        else:
            return best_label

def detect_shapes(flat_commands, config, return_logits=False, context_feat=None):
    # Simulate detectors: each returns (label, score, logits)
    def dummy_detector(image, context_feat=None, return_logits=False):
        labels = ['circle', 'square', 'triangle', 'polygon']
        logits = np.random.randn(len(labels))
        label = labels[np.argmax(logits)]
        score = float(np.max(logits))
        if return_logits:
            return label, score, logits
        else:
            return label
    base_detectors = config.get('base_detectors', [dummy_detector, dummy_detector, dummy_detector])
    weights = config.get('ensemble_weights', [1.0]*len(base_detectors))
    # Stacking meta-learner
    if 'calib_images' in config and 'calib_labels' in config:
        stacker = StackingEnsemble(base_detectors)
        if not getattr(stacker, '_fitted', False):
            stacker.fit(config['calib_images'], config['calib_labels'])
        return stacker.predict(flat_commands)
    # Attention-based fusion
    if context_feat is not None:
        embeds = []
        for det in base_detectors:
            _, score, logits = det(flat_commands, context_feat=context_feat, return_logits=True)
            embeds.append(torch.tensor(logits, dtype=torch.float))
        embeds = torch.stack(embeds)
        fused = DetectorAttention(embed_dim=embeds.size(-1))(embeds)
        return fused.argmax().item()
    # Default: weighted voting
    ensemble = WeightedVotingEnsemble(base_detectors, weights)
    return ensemble.predict(flat_commands, context_feat=context_feat, return_logits=return_logits)
from collections import Counter
import numpy as np
import logging
import cv2
from shapely.geometry import Polygon, LineString, MultiPoint
from shapely.ops import polygonize
from scipy.spatial.distance import cdist # For symmetry score
from math import hypot

# Ensure calculate_confidence is available
from derive_label.confidence_scoring import calculate_confidence

# Global logger setup (can be configured externally)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Weighted Voting Ensemble for Detector Fusion ---

class WeightedVotingEnsemble:
    def __init__(self, detectors, weights):
        """
        detectors: list of detector callables returning (label, score)
        weights: list of floats matching detectors
        """
        self.detectors = detectors
        self.weights = weights

    def predict(self, image):
        votes = Counter()
        for detector, w in zip(self.detectors, self.weights):
            label, score = detector(image)
            votes[label] += score * w
        best_label, _ = votes.most_common(1)[0]
        return best_label

# Example integration in a shape detection pipeline
def detect_shapes(image, config):
    # Example: base_detectors = [polygon_detector, line_detector, convexity_detector]
    base_detectors = config.get('base_detectors', [])
    weights = config.get('ensemble_weights', [1.0]*len(base_detectors))
    ensemble = WeightedVotingEnsemble(base_detectors, weights)
    fused_label = ensemble.predict(image)
    return fused_label

def _safe_center_and_scale_points(points):
    """Safely centers and scales points to prevent issues with very small or large coordinates."""
    pts = np.array(points)
    if pts is None or (hasattr(pts, 'size') and pts.size == 0) or (hasattr(pts, 'shape') and pts.shape[0] < 2):
        return pts, 1.0, np.array([0.0, 0.0])

    min_coords = np.min(pts, axis=0)
    max_coords = np.max(pts, axis=0)
    
    # Avoid division by zero for range
    range_x = max_coords[0] - min_coords[0]
    range_y = max_coords[1] - min_coords[1]
    
    max_dim = max(range_x, range_y, 1e-6) # Ensure non-zero divisor
    scale_factor = 100.0 / max_dim # Scale largest dimension to 100 pixels for consistent processing
    
    # Center the points
    center = (min_coords + max_coords) / 2.0
    
    scaled_pts = (pts - center) * scale_factor
    return scaled_pts, scale_factor, center


def ransac_line(points, threshold=2.0, min_points_for_ransac=5):
    """RANSAC for line detection. Returns (is_line, confidence, line_params)"""
    try:
        from sklearn.linear_model import RANSACRegressor
        if len(points) < min_points_for_ransac: # Need enough points for a robust RANSAC fit
            return False, 0.0, {}

        # Scale points for numerical stability with RANSACRegressor
        scaled_points, scale_factor, _ = _safe_center_and_scale_points(points)

        X = scaled_points[:, 0].reshape(-1, 1)
        y = scaled_points[:, 1]
        
        # Ensure enough unique points for RANSAC
        if len(np.unique(X)) < 2: # Need at least two distinct X values for line fit
             return False, 0.0, {}

        # Adjust residual_threshold based on scaling
        scaled_threshold = threshold * scale_factor
        
        # min_samples is typically 2 for a line, but a higher value provides more robustness
        model = RANSACRegressor(residual_threshold=scaled_threshold, min_samples=2, random_state=0)
        
        try:
            model.fit(X, y)
        except ValueError: # e.g., if all points are perfectly vertical
            # Handle vertical line case: swap X and Y and refit
            X_vert = scaled_points[:, 1].reshape(-1, 1)
            y_vert = scaled_points[:, 0]
            if len(np.unique(X_vert)) < 2: return False, 0.0, {} # Still not enough distinct points
            model.fit(X_vert, y_vert)
            
            # If fitting vertical line, inlier mask is based on y instead of x
            inlier_mask = np.abs(model.predict(X_vert) - y_vert) < scaled_threshold
            
            # Recalculate start/end points in original coordinates
            if inlier_mask.sum() > 0:
                inlier_pts = scaled_points[inlier_mask]
                mean_inlier = np.mean(inlier_pts, axis=0)
                # For vertical lines, main axis will be close to [0, 1] or [0, -1]
                start_pt = inlier_pts[np.argmin(inlier_pts[:, 1])]
                end_pt = inlier_pts[np.argmax(inlier_pts[:, 1])]
                
                length = float(np.linalg.norm(end_pt - start_pt)) / scale_factor
                orientation = 90.0 # vertical
                
                line_params = {
                    'start': (start_pt / scale_factor + (_safe_center_and_scale_points(points)[2])).tolist(),
                    'end': (end_pt / scale_factor + (_safe_center_and_scale_points(points)[2])).tolist(),
                    'length': length,
                    'orientation': orientation
                }
                confidence = inlier_mask.sum() / len(points)
                return True, confidence, line_params
            else:
                return False, 0.0, {}


        inlier_mask = model.inlier_mask_
        confidence = inlier_mask.sum() / len(points)
        
        line_params = {}
        if confidence > 0.8: # High confidence for a strong line fit
            inlier_pts = scaled_points[inlier_mask]
            if len(inlier_pts) >= 2:
                mean_inlier = np.mean(inlier_pts, axis=0)
                centered_inlier = inlier_pts - mean_inlier
                
                if np.allclose(centered_inlier, 0): # All inliers are identical
                    start_pt = inlier_pts[0]
                    end_pt = inlier_pts[0]
                    length = 0.0
                    orientation = 0.0
                else:
                    cov_inlier = np.cov(centered_inlier, rowvar=False)
                    eigvals, eigvecs = np.linalg.eigh(cov_inlier)
                    main_axis = eigvecs[:, np.argmax(eigvals)]
                    
                    proj = centered_inlier @ main_axis
                    start_proj_idx = np.argmin(proj)
                    end_proj_idx = np.argmax(proj)
                    start_pt = inlier_pts[start_proj_idx]
                    end_pt = inlier_pts[end_proj_idx]

                    length = float(np.linalg.norm(end_pt - start_pt)) / scale_factor
                    orientation = float(np.degrees(np.arctan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0])))
                
                # Convert back to original coordinate system
                line_params = {
                    'start': (start_pt / scale_factor + (_safe_center_and_scale_points(points)[2])).tolist(),
                    'end': (end_pt / scale_factor + (_safe_center_and_scale_points(points)[2])).tolist(),
                    'length': length,
                    'orientation': orientation
                }
                return True, confidence, line_params
        return False, confidence, line_params
    except ImportError:
        logging.debug("[GeometricDetectors] sklearn not installed, skipping RANSAC Line.")
        return False, 0.0, {}
    except Exception as e:
        logging.debug(f"[GeometricDetectors] RANSAC Line failed: {e}")
        return False, 0.0, {}

def ransac_circle(points, threshold=2.0, min_points_for_ransac=5):
    """RANSAC for circle detection. Returns (is_circle, confidence, circle_params)"""
    try:
        from scipy.optimize import leastsq
        if len(points) < min_points_for_ransac: # Need at least 5 points for robust circle fit
            return False, 0.0, {}

        # Scale points for numerical stability
        scaled_points, scale_factor, original_center_offset = _safe_center_and_scale_points(points)

        x = scaled_points[:, 0]
        y = scaled_points[:, 1]
        x_m = np.mean(x)
        y_m = np.mean(y)

        def calc_R(xc, yc):
            return np.sqrt((x - xc)**2 + (y - yc)**2)

        def f_2(c):
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = x_m, y_m
        center, cov_x, info, msg, ierr = leastsq(f_2, center_estimate, full_output=1)
        
        if ierr not in [1, 2, 3, 4]:
            return False, 0.0, {}

        Ri = calc_R(*center)
        R = Ri.mean()
        residu = np.sum((Ri - R)**2)
        
        # Scale threshold back to scaled coordinates
        scaled_threshold = threshold * scale_factor
        inliers = np.abs(Ri - R) < scaled_threshold 
        confidence = inliers.sum() / len(points)

        circle_params = {}
        if confidence > 0.8 and R > 0:
            # Convert center and radius back to original coordinate system
            original_center = (center / scale_factor + original_center_offset).tolist()
            original_radius = float(R / scale_factor)
            
            circle_params = {
                'center': original_center,
                'radius': original_radius,
                'residual': float(residu) # Residual is in scaled units
            }
            return True, confidence, circle_params
        return False, confidence, circle_params
    except ImportError:
        logging.debug("[GeometricDetectors] scipy not installed, skipping RANSAC Circle.")
        return False, 0.0, {}
    except Exception as e:
        logging.debug(f"[GeometricDetectors] RANSAC Circle failed: {e}")
        return False, 0.0, {}

def ellipse_fit(points, min_points_for_ellipse=7): # Increased from 5 for better robustness
    """Fits an ellipse to a set of points using skimage's EllipseModel. Returns (is_ellipse, confidence, ellipse_params)"""
    try:
        from skimage.measure import EllipseModel
        if len(points) < min_points_for_ellipse: 
            return False, 0.0, {}

        # Scale points for numerical stability
        scaled_points, scale_factor, original_center_offset = _safe_center_and_scale_points(points)

        model = EllipseModel()
        if model.estimate(scaled_points):
            xc, yc, a, b, theta = model.params
            
            # Heuristics for a good ellipse fit: positive axes, reasonable aspect ratio
            # Use a slightly stricter aspect ratio for "excellent" detection
            if a > 0 and b > 0 and 0.1 < b / a < 10: # Allow more elongated shapes
                # Sample points on the ellipse and check distance to original points for confidence
                t = np.linspace(0, 2*np.pi, len(scaled_points), endpoint=False)
                ellipse_x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
                ellipse_y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
                fitted_points = np.column_stack([ellipse_x, ellipse_y])
                
                distances = np.min(cdist(scaled_points, fitted_points), axis=1)
                inlier_count = np.sum(distances < 5) # Threshold for "on the ellipse" in scaled units
                confidence = inlier_count / len(points)
                
                # Adjust confidence based on how "circular" or "elliptical" it really is
                aspect_ratio = min(a, b) / max(a, b)
                confidence *= (1 + aspect_ratio) / 2 # Reward closer to circular
                
                # Convert params back to original coordinates
                original_center = (np.array([xc, yc]) / scale_factor + original_center_offset).tolist()
                original_axes = (float(a / scale_factor), float(b / scale_factor))
                
                ellipse_params = {'center': original_center, 'axes': original_axes, 'angle': float(theta)}
                return True, confidence, ellipse_params
        return False, 0.0, {}
    except ImportError:
        logging.debug("[GeometricDetectors] skimage not installed, skipping ellipse fitting.")
        return False, 0.0, {}
    except Exception as e:
        logging.error(f"[GeometricDetectors] Error in ellipse_fit: {e}")
        return False, 0.0, {}

def hough_lines_detector(points, min_vote_threshold=30, min_line_length=15, max_line_gap=10):
    """
    Detects lines in a set of points using Hough Transform (OpenCV's HoughLinesP).
    Returns (is_line, confidence, line_params) based on the longest dominant line.
    """
    try:
        if hasattr(points, 'shape') and points.shape[0] < 2:
            return False, 0.0, None

        # Determine optimal image size for Hough Transform based on point spread
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)

        # Add a border to ensure lines at edges are captured
        border_px = 10
        img_width = int(max_x - min_x) + 2 * border_px
        img_height = int(max_y - min_y) + 2 * border_px

        # Handle cases where points are very clustered (e.g., single point)
        if img_width < 1 or img_height < 1:
            if hasattr(points, 'shape') and points.shape[0] > 1 and np.linalg.norm(points[-1] - points[0]) < 5: # Small cluster, not a line
                return False, 0.0, None
            img_width = img_height = 50 # Minimum size for very small objects

        img = np.zeros((img_height, img_width), dtype=np.uint8)

        # Translate points to the new image coordinate system
        translated_points = (points - np.array([min_x, min_y]) + border_px).astype(np.int32)

        # Draw the polyline onto the image
        cv2.polylines(img, [translated_points], isClosed=False, color=255, thickness=2)

        # Apply Canny edge detection for better line detection input
        edges = cv2.Canny(img, 50, 150, apertureSize=3) # Use apertureSize=3 (default) for typical edges

        # Perform probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=min_vote_threshold,
                                minLineLength=min_line_length,
                                maxLineGap=max_line_gap)

        line_params = None
        if lines is not None and hasattr(lines, '__len__') and len(lines) > 0:
            # Find the longest line segment detected by Hough
            longest_line = None
            max_len_px = 0
            for line_seg in lines:
                x1, y1, x2, y2 = line_seg[0]
                curr_len = hypot(x2 - x1, y2 - y1)
                if curr_len > max_len_px:
                    max_len_px = curr_len
                    longest_line = line_seg[0]

            if longest_line is not None:
                # Convert coordinates back to original scale
                orig_x1, orig_y1 = longest_line[0] - border_px + min_x, longest_line[1] - border_px + min_y
                orig_x2, orig_y2 = longest_line[2] - border_px + min_x, longest_line[3] - border_px + min_y

                actual_len = hypot(orig_x2 - orig_x1, orig_y2 - orig_y1)
                actual_orientation = np.degrees(np.arctan2(orig_y2 - orig_y1, orig_x2 - orig_x1))

                line_params = {
                    'start': [float(orig_x1), float(orig_y1)],
                    'end': [float(orig_x2), float(orig_y2)],
                    'length': float(actual_len),
                    'orientation': float(actual_orientation)
                }

                # Confidence: Based on the length of the longest detected line relative to the overall extent of points.
                # Also consider the number of points that align with this line (inlier concept).

                # Simple confidence for now: ratio of detected line length to overall path length/bbox diagonal
                total_path_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
                bbox_diag = hypot(max_x - min_x, max_y - min_y)

                if bbox_diag > 0:
                    conf_length = actual_len / bbox_diag
                else:
                    conf_length = 0.0

                # If the line is very short relative to the total path, confidence is low
                confidence = calculate_confidence(conf_length, max_value=1.0, min_value=0.2) # min_value for threshold

                if confidence > 0.5: # Only return true if confidence is reasonable
                    return True, confidence, line_params

        return False, 0.0, None
    except Exception as e:
        logging.error(f"[GeometricDetectors] Error in HoughLinesP: {e}")
        return False, 0.0, None

def convexity_defects_detector(points):
    """
    Analyzes convexity defects of a shape using OpenCV.
    Includes robust fallback for self-intersecting polygons using Shapely.
    Returns (num_defects, type_label, confidence, extra_properties)
    """
    try:
        if hasattr(points, 'shape') and points.shape[0] < 3:
            return 0, 'degenerate_shape', 0.0, {}

        # Round and convert to int for OpenCV, but keep original for Shapely
        pts_cv = np.round(points).astype(np.int32)
        pts_cv = pts_cv.reshape((-1, 1, 2))

        poly_shapely = None
        is_valid_poly = False
        is_simple_poly = False

        # Attempt to create a polygon for validity check early (using original float points)
        try:
            poly_shapely = Polygon(points)
            is_valid_poly = poly_shapely.is_valid
            is_simple_poly = poly_shapely.is_simple if is_valid_poly else False
        except Exception as e:
            logging.debug(f"[ConvexityDefects] Shapely polygon creation failed initially: {e}")
            is_valid_poly = False
            is_simple_poly = False

        if not is_valid_poly:
            # If not a simple, valid polygon, try to decompose or flag as complex/self-intersecting
            if poly_shapely is not None and not is_simple_poly:
                try:
                    # Attempt to decompose the polygon into simple components
                    decomposed_geoms = poly_shapely.buffer(0)
                    if decomposed_geoms.geom_type == 'MultiPolygon':
                        num_simple_components = len(decomposed_geoms.geoms)
                        confidence = 0.6 + 0.1 * min(num_simple_components, 5)
                        return num_simple_components, 'self_intersecting_decomposed_polygon', confidence, {
                            'is_self_intersecting': True,
                            'num_simple_components': num_simple_components
                        }
                    elif decomposed_geoms.geom_type == 'Polygon' and decomposed_geoms.is_valid:
                        logging.debug(f"[ConvexityDefects] Self-intersecting polygon fixed by buffer.")
                        pts_cv = np.round(np.array(decomposed_geoms.exterior.coords)).astype(np.int32).reshape((-1, 1, 2))
                        is_valid_poly = True
                        is_simple_poly = True
                except Exception as e2:
                    logging.warning(f"[ConvexityDefects] Fallback decomposition/buffer failed: {e2}")

            if not is_valid_poly:
                try:
                    line_geom = LineString(points)
                    if not line_geom.is_simple:
                        simple_geometries = list(polygonize(line_geom))
                        if len(simple_geometries) > 0:
                            num_simple_components = len(simple_geometries)
                            confidence = 0.5 + 0.1 * min(num_simple_components, 5)
                            return num_simple_components, 'self_intersecting_line_polygonized', confidence, {
                                'is_self_intersecting': True,
                                'num_simple_components': num_simple_components
                            }
                        else:
                            return 0, 'self_intersecting_path', 0.3, {'is_self_intersecting': True}
                    else:
                        return 0, 'open_path', 0.1, {}
                except Exception:
                    return 0, 'complex_non_polygon_shape', 0.2, {}

        # If it's a valid simple polygon (either originally or after buffer fix)
        hull = cv2.convexHull(pts_cv, returnPoints=False)

        if hull is not None and len(hull) >= 3:
            try:
                defects = cv2.convexityDefects(pts_cv, hull)
                num_defects = len(defects) if defects is not None else 0

                label_type = 'concave_polygon' if num_defects > 0 else 'convex_polygon'

                if num_defects == 0:
                    confidence = 0.95
                else:
                    confidence = calculate_confidence(
                        num_defects,
                        max_value=min(len(points) // 2, 20),
                        min_value=1,
                        is_higher_better=False
                    )
                    confidence = max(0.5, confidence)

                return num_defects, label_type, confidence, {'num_convexity_defects': num_defects}
            except cv2.error as e:
                logging.warning(f"[ConvexityDefects] OpenCV error on defects: {e}. Degenerate hull points or data.")
                return 0, 'polygon_defects_cv_error', 0.4, {'error': str(e)}
        else:
            return 0, 'degenerate_hull_polygon', 0.2, {}
    except Exception as e:
        logging.error(f"[GeometricDetectors] Critical Error in convexity_defects_detector: {e}")
        return 0, 'error_polygon_detection', 0.0, {}


def fourier_descriptors(points, n_descriptors=10):
    """Computes Fourier descriptors for shape analysis. Returns (descriptors, confidence)"""
    pts = np.array(points)
    if hasattr(pts, 'shape') and pts.shape[0] < n_descriptors:
        return np.array([]).tolist(), 0.0 # Return list for JSON serialization
    complex_pts = pts[:, 0] + 1j * pts[:, 1]
    coeffs = np.fft.fft(complex_pts)
    
    # Only keep n_descriptors low-frequency descriptors (ignoring DC component (coeffs[0]))
    # Take the magnitude of descriptors, which are rotation invariant.
    # We take the first `n_descriptors` non-DC components.
    desc = np.abs(coeffs[1 : n_descriptors + 1]) 
    
    # Confidence: Higher for shapes well-described by few descriptors (simpler shapes).
    # Heuristic: Compare energy in the kept descriptors to total energy.
    total_energy = np.sum(np.abs(coeffs))
    if total_energy == 0: return desc.tolist(), 0.0
    
    kept_energy = np.sum(desc)
    confidence = calculate_confidence(kept_energy, max_value=total_energy, min_value=0.0) # Higher kept energy relative to total is better
    
    # Add a penalty if the shape is complex and needs many descriptors
    # (more descriptors might imply less confidence in simple Fourier representation)
    if n_descriptors < len(coeffs) / 4: # If we are only capturing a small fraction of components
        confidence *= (n_descriptors / (len(coeffs) / 4)) # Penalize if too few descriptors for complexity

    return desc.tolist(), confidence

def cluster_points_detector(points, eps=5, min_samples=5):
    """Clusters points using DBSCAN. Returns (labels, n_clusters, confidence)"""
    try:
        from sklearn.cluster import DBSCAN
        if len(points) < min_samples: 
            return np.zeros(len(points), dtype=int).tolist(), 0, 0.0
        
        # Scale points for DBSCAN if coordinates are very large or small
        scaled_points, scale_factor, _ = _safe_center_and_scale_points(points)
        scaled_eps = eps * scale_factor # Adjust epsilon based on scaling

        clustering = DBSCAN(eps=scaled_eps, min_samples=min_samples).fit(scaled_points)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0) # Exclude noise label (-1)
        
        num_noise_points = np.sum(labels == -1)
        
        # Confidence:
        # 1. Higher for more distinct clusters (if expected for a point cloud)
        # 2. Higher for fewer noise points (well-defined clusters)
        # 3. Higher if the points are reasonably spread out (not a single dense blob)
        
        # Normalized number of clusters (capped at reasonable max)
        conf_clusters = calculate_confidence(n_clusters, max_value=min(len(points)//(min_samples*2), 5), min_value=1)
        
        # Normalized noise points (lower is better)
        conf_noise = calculate_confidence(num_noise_points, max_value=len(points)/2, min_value=0, is_higher_better=False)
        
        # Overall confidence for it being a 'point cloud' is high if it's not a single cluster
        # and has low noise, indicating distinct groupings, or a diffuse spread.
        # This detector specifically flags "point cloud" if other detectors fail.
        
        # If there's only one cluster (excluding noise), it might be a solid shape, so lower cloud confidence
        if n_clusters <= 1 and num_noise_points < len(points) * 0.1: # Mostly one cluster, very little noise
            confidence = 0.3 * (1 - conf_noise) # Low confidence for point cloud if coherent
        else: # Multiple clusters or significant noise, more likely a point cloud
            confidence = 0.5 + (conf_clusters + (1-conf_noise)) / 2 * 0.5 # Increase for more clusters/noise
        
        return labels.tolist(), n_clusters, confidence
    except ImportError:
        logging.debug("[GeometricDetectors] sklearn not installed, skipping clustering.")
        return np.zeros(len(points), dtype=int).tolist(), 0, 0.0
    except Exception as e:
        logging.error(f"[GeometricDetectors] Error in cluster_points_detector: {e}")
        return np.zeros(len(points), dtype=int).tolist(), 0, 0.0

def is_roughly_circular_detector(points, tolerance=0.10): # Stricter tolerance
    """Checks if a shape is roughly circular based on radii from centroid AND compares with min_bounding_circle.
    Returns (is_circular, confidence)
    """
    points_np = np.array(points)
    if points_np.shape[0] < 3: return False, 0.0 

    centroid = np.mean(points_np, axis=0)
    radii = np.linalg.norm(points_np - centroid, axis=1)
    
    if np.mean(radii) == 0: return False, 0.0 
    
    # Method 1: Standard deviation of radii
    std_dev_ratio = np.std(radii) / (np.mean(radii) + 1e-6)
    is_circular_centroid = std_dev_ratio < tolerance
    
    conf_centroid = calculate_confidence(std_dev_ratio, max_value=tolerance*2, min_value=0.0, is_higher_better=False)

    # Method 2: Compare with Minimum Bounding Circle
    is_mbc, mbc_conf, mbc_params = min_bounding_circle(points)
    
    # If MBC detection is confident and the shape closely fills the MBC
    if is_mbc and mbc_conf > 0.7:
        # Check area ratio: actual polygon area vs. MBC area
        try:
            poly_area = Polygon(points).area # Requires a valid polygon
            mbc_area = np.pi * mbc_params['radius']**2
            if mbc_area > 0:
                area_ratio = poly_area / mbc_area
                # High area ratio means it fills the circle well
                conf_area_ratio = calculate_confidence(area_ratio, max_value=1.0, min_value=0.5)
                
                # Combine MBC confidence with area ratio
                mbc_combined_conf = (mbc_conf + conf_area_ratio) / 2.0
            else:
                mbc_combined_conf = mbc_conf
        except Exception:
            mbc_combined_conf = mbc_conf # Fallback if polygon area fails

        # Final decision based on weighted average
        if is_circular_centroid and mbc_combined_conf > 0.7: # Both methods agree
            return True, (conf_centroid * 0.6 + mbc_combined_conf * 0.4)
        elif is_mbc and mbc_combined_conf > 0.8: # MBC is very strong, override
             return True, mbc_combined_conf
        elif is_circular_centroid and mbc_combined_conf > 0.5: # Centroid method strong, MBC moderate
             return True, conf_centroid * 0.8
    
    return is_circular_centroid, conf_centroid # Fallback to centroid method if MBC is weak or failed


def is_rectangle_detector(vertices_np, angle_tolerance=10, length_tolerance_ratio=0.1):
    """
    Checks if a shape is a rectangle based on angles, side lengths, AND compares with min_bounding_rectangle.
    Returns (is_rectangle, confidence, properties).
    """
    if len(vertices_np) < 4: # A rectangle has at least 4 distinct vertices
        return False, 0.0, {}

    pts = np.array(vertices_np)
    
    # Method 1: Angle and Side Length Check (existing logic)
    angles_deg = []
    sides = []
    
    # Calculate angles only if there are exactly 4 points for classic rectangle check
    if len(vertices_np) == 4:
        for i in range(4):
            p0 = pts[i - 1]
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            v1 = p0 - p1
            v2 = p2 - p1
            
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            # Avoid division by zero, treat very short segments carefully
            if norm_v1 < 1e-6 or norm_v2 < 1e-6:
                angles_deg.append(0) 
                sides.append(0)
                continue

            cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1, 1)))
            angles_deg.append(angle)
            sides.append(norm_v1) # One side length
            
        is_right_angles = all(abs(a - 90) < angle_tolerance for a in angles_deg)
        is_opposite_sides_equal = np.isclose(sides[0], sides[2], rtol=length_tolerance_ratio) and \
                                  np.isclose(sides[1], sides[3], rtol=length_tolerance_ratio)
        
        conf_classic = 0.0
        if is_right_angles and is_opposite_sides_equal:
            angle_conf = calculate_confidence(np.mean(np.abs(np.array(angles_deg) - 90)), 
                                              max_value=angle_tolerance * 2, min_value=0, is_higher_better=False)
            length_error = (abs(sides[0] - sides[2]) + abs(sides[1] - sides[3])) / (sum(sides) / 2 + 1e-6)
            length_conf = calculate_confidence(length_error, max_value=0.5, min_value=0, is_higher_better=False)
            conf_classic = (angle_conf * 0.7 + length_conf * 0.3)
            
            if conf_classic > 0.7: # Only consider if classical check is strong
                return True, conf_classic, {'angles_deg': angles_deg, 'side_lengths': sides}


    # Method 2: Compare with Minimum Bounding Rectangle
    is_mbr, mbr_conf, mbr_params = min_bounding_rectangle(vertices_np)

    if is_mbr and mbr_conf > 0.7: # If MBR fitting is confident
        # Check how well the original shape fills its MBR
        try:
            poly_area = Polygon(vertices_np).area
            mbr_area = mbr_params['width'] * mbr_params['height']
            
            if mbr_area > 0:
                area_ratio = poly_area / mbr_area
                # If area ratio is close to 1, it's a good fit
                conf_area_ratio = calculate_confidence(area_ratio, max_value=1.0, min_value=0.8)
                
                # Check vertex match: how many of the original vertices are close to the MBR vertices
                mbr_corners = np.array(mbr_params['corners'])
                dist_to_mbr_corners = np.min(cdist(vertices_np, mbr_corners), axis=1)
                num_matching_vertices = np.sum(dist_to_mbr_corners < 5) # Within 5 pixels
                conf_vertex_match = calculate_confidence(num_matching_vertices, max_value=len(vertices_np), min_value=0)

                # Combine MBR confidence with fill and vertex match
                combined_mbr_conf = (mbr_conf * 0.5 + conf_area_ratio * 0.3 + conf_vertex_match * 0.2)
                
                if combined_mbr_conf > 0.7: # Good fit to MBR
                    return True, combined_mbr_conf, {'mbr_params': mbr_params, 'area_ratio_to_mbr': area_ratio}

        except Exception as e:
            logging.debug(f"Error calculating area ratio for MBR: {e}")
            # Fallback to just MBR confidence if area calculation fails
            if mbr_conf > 0.8:
                return True, mbr_conf, {'mbr_params': mbr_params}

    return False, 0.0, {} # No strong rectangle detection by either method


def is_point_cloud_detector(vertices_np, min_spread=10.0, max_spread=200.0, min_points=10, max_coherence_score=0.2):
    """
    Detects if the vertices form a scattered point cloud rather than a cohesive shape.
    Confidence is higher for more points and larger spread, but lower if features suggest a shape.
    """
    if len(vertices_np) < min_points:
        return False, 0.0, {}

    # Calculate spread
    min_x, min_y = np.min(vertices_np, axis=0)
    max_x, max_y = np.max(vertices_np, axis=0)
    spread = max(max_x - min_x, max_y - min_y)

    if spread < min_spread: # Too small to be a cloud, likely just noise or a tight cluster
        return False, 0.0, {}
    
    # Assess "coherence" - how well points fit common geometric shapes
    # If any of these fit well, it's NOT a point cloud.
    coherence_scores = []
    
    # RANSAC line coherence
    _, line_conf, _ = ransac_line(vertices_np)
    coherence_scores.append(line_conf)
    
    # RANSAC circle coherence
    _, circle_conf, _ = ransac_circle(vertices_np)
    coherence_scores.append(circle_conf)

    # Ellipse fit coherence
    _, ellipse_conf, _ = ellipse_fit(vertices_np)
    coherence_scores.append(ellipse_conf)

    # Minimum Bounding Rectangle compactness (if very compact, less likely a cloud)
    _, mbr_conf, mbr_params = min_bounding_rectangle(vertices_np)
    if mbr_conf > 0.5 and mbr_params and mbr_params.get('area') > 0:
        compactness = (2 * (mbr_params['width'] + mbr_params['height']))**2 / (mbr_params['area'] * 4 * np.pi + 1e-6) # Perimeter^2 / Area
        coherence_scores.append(calculate_confidence(compactness, max_value=20.0, min_value=1.0, is_higher_better=False)) # Lower compactness suggests less cloud-like

    max_coherence = max(coherence_scores) if coherence_scores else 0.0

    if max_coherence > max_coherence_score: # If a coherent shape fits well
        return False, 0.0, {} 

    # If no coherent shape found, it's more likely a point cloud.
    # Confidence for point cloud: higher for more points and larger spread
    conf_points = calculate_confidence(len(vertices_np), max_value=200, min_value=min_points)
    conf_spread = calculate_confidence(spread, max_value=max_spread, min_value=min_spread)
    
    # Inversely related to max_coherence (lower coherence, higher point cloud confidence)
    confidence = (conf_points * 0.4 + conf_spread * 0.4 + (1 - max_coherence) * 0.2)
    
    return True, confidence, {'spread': float(spread), 'num_points_in_cloud': len(vertices_np), 'max_coherence_score': float(max_coherence)}

def compute_symmetry_axis(vertices):
    """Returns the main axis direction as a unit vector (from PCA)."""
    pts = np.array(vertices)
    if pts.shape[0] < 2: return [0.0, 0.0]
    
    # Handle collinear or identical points
    if np.allclose(pts, pts[0]): return [1.0, 0.0] # Arbitrary direction for a point
    
    pts_centered = pts - np.mean(pts, axis=0)
    
    # Special handling for 2 points - line between them
    if pts.shape[0] == 2:
        vec = pts_centered[1] - pts_centered[0]
        norm = np.linalg.norm(vec)
        if norm == 0: return [1.0, 0.0]
        return (vec / norm).tolist()

    cov = np.cov(pts_centered.T)
    
    # If covariance matrix is degenerate (e.g., all points collinear), handle gracefully
    if np.linalg.matrix_rank(cov) < 2:
        # For collinear points, compute direction from start to end (or first two distinct points)
        diffs = np.diff(pts, axis=0)
        valid_diffs = diffs[np.linalg.norm(diffs, axis=1) > 1e-6]
        if len(valid_diffs) > 0:
            major_axis = valid_diffs[0]
            norm = np.linalg.norm(major_axis)
            if norm == 0: return [1.0, 0.0]
            return list((major_axis / norm).tolist())
        else: # Still degenerate, e.g., single point repeated
            return [1.0, 0.0]

    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]
    norm = np.linalg.norm(major_axis)
    if norm == 0:
        return [1.0, 0.0]
    return list((major_axis / norm).tolist())

def compute_orientation(vertices):
    """Computes the orientation of a shape using PCA. Returns (angle, confidence)"""
    pts = np.array(vertices)
    if pts.shape[0] < 2: return 0.0, 0.0
    
    # Handle single point or perfectly clustered points
    if np.allclose(pts, pts[0]): return 0.0, 0.1 # Arbitrary orientation, low confidence
    
    pts_centered = pts - np.mean(pts, axis=0)
    cov = np.cov(pts_centered.T)
    
    # Check for degenerate covariance matrix (e.g., all points collinear)
    if np.linalg.matrix_rank(cov) < 2:
        # If collinear, orientation is defined by the line itself
        diffs = np.diff(pts, axis=0)
        valid_diffs = diffs[np.linalg.norm(diffs, axis=1) > 1e-6]
        if len(valid_diffs) > 0:
            major_axis = valid_diffs[0]
            angle = np.arctan2(major_axis[1], major_axis[0])
            return float(np.degrees(angle)), 0.7 # Higher confidence for line
        else:
            return 0.0, 0.1 # Still degenerate, very low confidence
        
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # The orientation is less confident if the shape is very circular (eigenvalues are similar)
    # or very thin (one eigenvalue much larger than the other, but not good for angle confidence *itself* if noisy)
    # Confidence based on eccentricity (ratio of eigenvalues), higher for more elongated shapes.
    # Smallest eigenvalue + epsilon to avoid division by zero
    eccentricity_ratio = eigvals.max() / (eigvals.min() + 1e-6) 
    confidence = calculate_confidence(eccentricity_ratio, max_value=100.0, min_value=1.0) # From 1 (circular) to 100 (very elongated)
    
    major_axis = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(major_axis[1], major_axis[0])
    
    return float(np.degrees(angle)), confidence

def compute_symmetry_score(vertices):
    """
    Calculates a symmetry score based on reflectional symmetry across centroid axes and 180-degree rotational symmetry.
    Returns (score, symmetry_type_str, confidence).
    Score: mean distance after reflection. Lower is better.
    Confidence: inversely related to score.
    """
    pts = np.array(vertices)
    if pts.shape[0] < 2: return 0.0, 'none', 0.0 
    
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    
    # If all points are essentially the centroid, it's point symmetry, but not reflectional.
    if np.allclose(pts_centered, 0):
        # A single point (or multiple identical points) has perfect rotational symmetry, but no reflectional axes
        return 0.0, 'rotational_180', 1.0 # Perfect rotational symmetry around itself

    # Try reflection over x and y axes relative to centroid
    mirror_x = pts_centered * np.array([-1, 1])
    mirror_y = pts_centered * np.array([1, -1])

    # Calculate mean distance of mirrored points to the nearest original point
    score_x = np.mean(np.min(cdist(mirror_x, pts_centered), axis=1))
    score_y = np.mean(np.min(cdist(mirror_y, pts_centered), axis=1))

    symmetry_type_str = 'none'
    best_score = float('inf') # Initialize with a very high score

    # Define a tolerance for "symmetric enough"
    pixel_tolerance = 3.0 # Allow for a few pixels deviation

    # Evaluate X-axis symmetry (vertical reflection)
    if score_x < pixel_tolerance:
        symmetry_type_str = 'vertical_reflection'
        best_score = score_x
    
    # Evaluate Y-axis symmetry (horizontal reflection)
    if score_y < pixel_tolerance and score_y < best_score:
        symmetry_type_str = 'horizontal_reflection'
        best_score = score_y
    elif score_y < pixel_tolerance and symmetry_type_str == 'vertical_reflection':
        symmetry_type_str = 'horizontal_and_vertical_reflection' # Both axes

    # Try rotational symmetry (180 degrees)
    rot180 = pts_centered * np.array([-1, -1]) 
    score_rot180 = np.mean(np.min(cdist(rot180, pts_centered), axis=1))
    
    if score_rot180 < pixel_tolerance:
        # Rotational symmetry often implies reflectional symmetry for many shapes
        if symmetry_type_str == 'none':
            symmetry_type_str = 'rotational_180'
        elif 'reflection' in symmetry_type_str:
            symmetry_type_str += '_and_rotational_180' # Both
        
        # If rotational is *better* than previous best reflectional
        if score_rot180 < best_score:
            best_score = score_rot180
    
    # Confidence: inversely related to the best symmetry score (lower score = higher confidence)
    # Scale score to a meaningful range (e.g., 0 perfect, 20-30 completely asymmetric)
    confidence = calculate_confidence(best_score, max_value=30.0, min_value=0.0, is_higher_better=False)
    
    return float(best_score), symmetry_type_str, confidence

def detect_vertices(points, angle_threshold=150, dist_threshold_ratio=0.02):
    """
    Detects prominent vertices in a shape using angle changes between segments.
    This version is more robust, handles closed/open shapes better, and gives a calibrated confidence.
    Args:
        points (np.array): Nx2 array of (x,y) coordinates.
        angle_threshold (float): Angle (in degrees) above which a corner is considered smooth (e.g., 150-170 for smooth, below that is a corner).
        dist_threshold_ratio (float): Minimum length of segments relative to bbox diagonal
                                      to be considered for vertex detection. Helps filter out noise from dense point clouds.
    Returns:
        (num_vertices, confidence, vertices_coords)
    """
    points = np.array(points)
    if len(points) < 3:
        return len(points), 1.0, points.tolist() # If 0, 1, or 2 points, they are all 'vertices' with high confidence

    vertices_coords = []
    
    # Calculate bounding box diagonal for adaptive distance threshold
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    bbox_diagonal = np.linalg.norm([max_x - min_x, max_y - min_y])

    min_segment_len_threshold = bbox_diagonal * dist_threshold_ratio
    if min_segment_len_threshold == 0 or min_segment_len_threshold < 2: # Ensure a minimum absolute length
        min_segment_len_threshold = 2.0 

    # Determine if the shape is 'closed' by checking if the start and end points are very close
    is_closed = np.linalg.norm(points[0] - points[-1]) < min_segment_len_threshold * 2 # Slightly larger tolerance for closing
    
    # Start and end points are always potential vertices
    vertices_coords.append(points[0].tolist())

    # Iterate through points to find corners
    # Use a loop that handles both open and closed paths (by checking the "closing" angle)
    path_points = list(points)
    if is_closed and len(points) > 2:
        # For closed shapes, the last point is effectively connected to the first,
        # so consider the loop for angles
        path_points.append(points[0]) # Temporarily append first point to close the loop for angle calculation
        path_points.append(points[1]) # Need next point after start for last angle check

    # Store angles for confidence calculation
    detected_angles = []

    for i in range(1, len(path_points) - 1): # Exclude first and last processed points if not closed
        p_prev = path_points[i-1]
        p_curr = path_points[i]
        p_next = path_points[i+1]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)

        # Only consider angles if segments are long enough to be meaningful
        if len_v1 < min_segment_len_threshold or len_v2 < min_segment_len_threshold:
            continue

        if len_v1 == 0 or len_v2 == 0: # Avoid division by zero for identical points
            continue

        dot_product = np.dot(v1, v2)
        
        # Clamp value to prevent arccos from returning NaN for floating point inaccuracies
        cosine_angle = np.clip(dot_product / (len_v1 * len_v2), -1.0, 1.0)
        
        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)

        # A sharp corner (vertex) means the angle is significantly less than 180 degrees (straight line)
        if angle_deg < angle_threshold: # This is a 'corner' if angle is below threshold
            # Avoid adding duplicate vertices if points are very close (due to noise)
            if not vertices_coords or np.linalg.norm(np.array(vertices_coords[-1]) - p_curr) > min_segment_len_threshold / 2:
                vertices_coords.append(p_curr.tolist())
            detected_angles.append(angle_deg)
            
    # If the shape is open, add the last point if it hasn't been added and is distinct
    if not is_closed and len(points) > 1:
        if not vertices_coords or np.linalg.norm(np.array(vertices_coords[-1]) - points[-1]) > min_segment_len_threshold / 2:
            vertices_coords.append(points[-1].tolist())

    num_vertices = len(vertices_coords)

    # Confidence calculation:
    # Higher confidence for distinct, clear vertices.
    # Lower confidence if too few/many detected compared to original points, or if vertices are ambiguous (angles close to 180).
    
    if num_vertices == 0: # Should ideally not happen if len(points) >= 1
        confidence = 0.0
    else:
        # Average sharpness of detected corners (lower angle_deg is sharper)
        if detected_angles:
            avg_sharpness = np.mean([180 - a for a in detected_angles])
            conf_sharpness = calculate_confidence(avg_sharpness, max_value=180-angle_threshold, min_value=0.0) # Max sharpness is 180-threshold
        else: # No distinct corners found
            conf_sharpness = 0.1 # Very low confidence in vertices if only straight lines or smooth curve
        
        # How well the detected vertices represent the original points (coverage/fidelity)
        # Check if the path between detected vertices covers most of the original points
        # For simplicity, if num_vertices is very low for many points, it implies bad detection
        if len(points) > 10 and num_vertices < 3: # Many points but few detected vertices, likely a smooth shape
            conf_coverage = 0.2
        else:
            conf_coverage = calculate_confidence(num_vertices, max_value=len(points), min_value=1) # Max vertices can be all points

        confidence = (conf_sharpness * 0.6 + conf_coverage * 0.4) * 0.9 # Overall confidence slightly reduced as it's a heuristic

    return num_vertices, confidence, vertices_coords

# --- New Functions for Cutting-Edge Geometric Detection ---
def min_bounding_rectangle(points):
    """
    Finds the minimum area bounding rectangle of a set of points (rotated rectangle).
    Uses OpenCV's minAreaRect.
    Returns (is_rectangle, confidence, properties_dict)
    properties_dict includes: center, width, height, angle, area, corners.
    """
    try:
        if len(points) < 3:  # Need at least 3 points for a meaningful rectangle
            return False, 0.0, {}

        # Convert points to appropriate format for OpenCV
        points_cv = np.array(points, dtype=np.float32)

        # Get the minimum area bounding rectangle
        rect = cv2.minAreaRect(points_cv)
        (center_x, center_y), (width, height), angle = rect

        # Calculate the four corners of the rotated rectangle
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)  # Corrected: use astype(np.int32) instead of np.int0

        # Calculate area
        area = width * height

        # Confidence:
        # 1. Higher if the original points densely fill the rectangle (area_ratio close to 1).
        # 2. Higher if the aspect ratio is reasonable (not extremely thin or fat).
        # 3. Higher if the rectangle is not degenerate (width/height > 0).

        # Try to form a polygon from original points to calculate its area
        try:
            poly_original = Polygon(points)
            original_area = poly_original.area if poly_original.is_valid else 0.0
        except Exception:
            original_area = 0.0  # Could not form a valid polygon

        if area == 0:  # Degenerate rectangle
            return False, 0.0, {}

        area_ratio = original_area / area if area > 0 else 0.0
        aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0.0

        conf_area_fill = calculate_confidence(area_ratio, max_value=1.0, min_value=0.5)  #
        conf_aspect_ratio = calculate_confidence(aspect_ratio, max_value=1.0, min_value=0.1)  # Reward reasonable aspect ratio (not too thin)

        # Overall confidence based on a combination
        confidence = (conf_area_fill * 0.6 + conf_aspect_ratio * 0.4)

        if confidence > 0.5:  # Only consider it a "rectangle" if confidence is reasonable
            properties = {
                'center': [float(center_x), float(center_y)],
                'width': float(width),
                'height': float(height),
                'angle': float(angle),
                'area': float(area),
                'corners': box.tolist(),  # Store integer corners
                'area_fill_ratio': float(area_ratio),
                'aspect_ratio': float(aspect_ratio)
            }
            return True, confidence, properties

        return False, 0.0, {}
    except Exception as e:
        logging.error(f"[GeometricDetectors] Error in min_bounding_rectangle: {e}")
        return False, 0.0, {}

def min_bounding_circle(points):
    """
    Finds the minimum enclosing circle of a set of points.
    Uses OpenCV's minEnclosingCircle.
    Returns (is_circle, confidence, properties_dict)
    properties_dict includes: center, radius, area.
    """
    try:
        if len(points) < 3: # Need at least 3 points for a circle
            return False, 0.0, {}
        
        points_cv = np.array(points, dtype=np.float32)

        (center_x, center_y), radius = cv2.minEnclosingCircle(points_cv)
        area = np.pi * radius**2

        # Confidence:
        # 1. Higher if the original points are close to the circumference (low dispersion).
        # 2. Higher if the original shape densely fills the circle (area_ratio close to 1).
        
        # Calculate dispersion: average distance from points to the circumference
        distances_to_center = np.linalg.norm(points_cv - np.array([center_x, center_y]), axis=1)
        avg_dist_from_circumference = np.mean(np.abs(distances_to_center - radius))
        
        conf_dispersion = calculate_confidence(avg_dist_from_circumference, max_value=radius*0.2, min_value=0.0, is_higher_better=False) # Lower dispersion is better

        # Try to form a polygon from original points to calculate its area
        try:
            poly_original = Polygon(points)
            original_area = poly_original.area if poly_original.is_valid else 0.0
        except Exception:
            original_area = 0.0

        area_ratio = original_area / area if area > 0 else 0.0
        conf_area_fill = calculate_confidence(area_ratio, max_value=1.0, min_value=0.5)

        confidence = (conf_dispersion * 0.6 + conf_area_fill * 0.4)
        
        if confidence > 0.5: # Only consider it a "circle" if confidence is reasonable
            properties = {
                'center': [float(center_x), float(center_y)],
                'radius': float(radius),
                'area': float(area),
                'avg_dist_from_circumference': float(avg_dist_from_circumference),
                'area_fill_ratio': float(area_ratio)
            }
            return True, confidence, properties
        
        return False, 0.0, {}
    except Exception as e:
        logging.error(f"[GeometricDetectors] Error in min_bounding_circle: {e}")
        return False, 0.0, {}

