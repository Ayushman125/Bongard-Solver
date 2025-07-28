import argparse

import argparse
import sys
import os
import ijson
import json
import logging
import numpy as np
from shapely.geometry import Polygon

# Ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline.loader import BongardLoader
from src.data_pipeline.logo_parser import BongardLogoParser
from src.data_pipeline.physics_infer import PhysicsInference

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def main():
    def flatten_action_program(prog):
        flat = []
        if isinstance(prog, list):
            for item in prog:
                if isinstance(item, list):
                    flat.extend(flatten_action_program(item))
                elif isinstance(item, str):
                    flat.append(item)
        elif isinstance(prog, str):
            flat.append(prog)
        return flat

    def extract_shapes_from_action_program(flat_commands):
        shapes = []
        fallback_reasons = []
        current_stroke = []
        stroke_id = 0
        for cmd in flat_commands:
            if isinstance(cmd, str):
                if cmd.strip().upper() == 'PU':
                    if current_stroke:
                        result = create_shape_from_stroke(current_stroke, stroke_id)
                        if isinstance(result, tuple):
                            shape_obj, fallback_reason = result
                        else:
                            shape_obj, fallback_reason = result, None
                        if shape_obj:
                            shapes.append(shape_obj)
                        else:
                            fallback_reasons.append(fallback_reason)
                        stroke_id += 1
                        current_stroke = []
                else:
                    current_stroke.append(cmd)
        if current_stroke:
            result = create_shape_from_stroke(current_stroke, stroke_id)
            if isinstance(result, tuple):
                shape_obj, fallback_reason = result
            else:
                shape_obj, fallback_reason = result, None
            if shape_obj:
                shapes.append(shape_obj)
            else:
                fallback_reasons.append(fallback_reason)
        return shapes, fallback_reasons
    """
    Parse a LOGO action program (list of commands) to extract individual shape objects.
    Returns a list of shape dicts with vertices and attributes.
    """
    # Removed erroneous shape extraction loop over undefined 'action_program'.

    # --- Advanced helpers moved to global scope for use in infer_shape_type ---
    def ransac_line(points, threshold=2.0):
        try:
            from sklearn.linear_model import RANSACRegressor
            X = points[:,0].reshape(-1,1)
            y = points[:,1]
            model = RANSACRegressor(residual_threshold=threshold)
            model.fit(X, y)
            inlier_mask = model.inlier_mask_
            return inlier_mask.sum() / len(points) > 0.8
        except Exception:
            return False

    def ransac_circle(points, threshold=2.0):
        try:
            from sklearn.linear_model import RANSACRegressor
            x = points[:,0]
            y = points[:,1]
            x_m = np.mean(x)
            y_m = np.mean(y)
            def calc_R(xc, yc):
                return np.sqrt((x-xc)**2 + (y-yc)**2)
            def f_2(c):
                Ri = calc_R(*c)
                return Ri - Ri.mean()
            from scipy.optimize import leastsq
            center_estimate = x_m, y_m
            center, _ = leastsq(f_2, center_estimate)
            Ri = calc_R(*center)
            R = Ri.mean()
            residu = np.sum((Ri - R)**2)
            inliers = np.abs(Ri - R) < threshold
            return inliers.sum() / len(points) > 0.8
        except Exception:
            return False

    def detect_vertices(points, angle_thresh=30):
        pts = np.array(points)
        n = len(pts)
        if n < 3:
            return []
        angles = []
        for i in range(n):
            p0 = pts[i-1]
            p1 = pts[i]
            p2 = pts[(i+1)%n]
            v1 = p0 - p1
            v2 = p2 - p1
            angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8), -1, 1)))
            angles.append(angle)
        return [i for i, a in enumerate(angles) if a < angle_thresh]

    def create_shape_from_stroke(stroke_cmds, stroke_id):
        # --- Topology, hole, skeleton, and image-processing helpers ---
        def image_processing_features(vertices_np):
            features = {}
            features = {}
            import cv2
            from skimage.morphology import skeletonize, medial_axis
            from skimage.measure import label, regionprops
            from skimage.filters import threshold_otsu
            from skimage.morphology import closing, opening, square
            # Rasterize stroke
            pts = np.round(vertices_np).astype(np.int32)
            pts = pts - pts.min(axis=0) + 10
            img = np.zeros((256, 256), dtype=np.uint8)
            try:
                cv2.polylines(img, [pts], isClosed=True, color=255, thickness=2)
                # 1. Contour detection
                contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                features['n_contours'] = int(len(contours))
                features['contour_lengths'] = [float(cv2.arcLength(cnt, True)) for cnt in contours]
                # 2. Hole-punching (count holes via hierarchy)
                n_holes = 0
                if hierarchy is not None:
                    for h in hierarchy[0]:
                        if h[3] != -1:
                            n_holes += 1
                features['n_holes'] = int(n_holes)
                # 3. Thinning/skeletonization
                bin_img = (img > 0).astype(np.uint8)
                skel = skeletonize(bin_img)
                features['skeleton_sum'] = int(np.sum(skel))
                # 4. Morphological cleaning
                closed = closing(bin_img, square(3))
                opened = opening(closed, square(3))
                features['morph_cleaned_sum'] = int(np.sum(opened))
                # 5. Contour simplification (Douglas-Peucker)
                simplified_contours = []
                for cnt in contours:
                    epsilon = 0.01 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    # Convert to list of [x, y] floats
                    if hasattr(approx, 'tolist'):
                        simplified_contours.append([[float(pt[0][0]), float(pt[0][1])] for pt in approx])
                    else:
                        simplified_contours.append([float(x) for x in approx])
                features['simplified_contours'] = simplified_contours
                features['simplified_n_vertices'] = [len(approx) for approx in simplified_contours]
                # 6. Area/solidity/extent from regionprops
                labeled = label(bin_img)
                props = regionprops(labeled)
                if props:
                    features['area'] = int(props[0].area)
                    features['solidity'] = float(props[0].solidity) if hasattr(props[0], 'solidity') else None
                    features['extent'] = float(props[0].extent) if hasattr(props[0], 'extent') else None
                # 7. Otsu threshold (for noisy strokes)
                try:
                    thresh = threshold_otsu(img)
                    features['otsu_thresh'] = int(thresh)
                except Exception:
                    features['otsu_thresh'] = None
            except Exception as e:
                features['image_processing_error'] = str(e)
            return features
        def compute_euler_characteristic(vertices_np):
            # For polygons: χ = V - E + F (F=1 for simple polygon)
            V = len(vertices_np)
            E = V
            F = 1
            return V - E + F

        def count_holes_shapely(vertices_np):
            try:
                from shapely.geometry import Polygon
                poly = Polygon(vertices_np)
                if poly.is_valid:
                    return len(poly.interiors)
            except Exception:
                pass
            return 0

        def contour_hierarchy_holes(vertices_np):
            try:
                import cv2
                pts = np.round(vertices_np).astype(np.int32)
                pts = pts - pts.min(axis=0) + 10
                img = np.zeros((256, 256), dtype=np.uint8)
                cv2.polylines(img, [pts], isClosed=True, color=255, thickness=2)
                contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                n_holes = 0
                if hierarchy is not None:
                    for h in hierarchy[0]:
                        if h[3] != -1:
                            n_holes += 1
                return n_holes, len(contours)
            except Exception:
                return 0, 0

        def medial_axis_features(vertices_np):
            try:
                from skimage.morphology import medial_axis
                from skimage.draw import polygon as skpolygon
                pts = np.round(vertices_np).astype(np.int32)
                pts = pts - pts.min(axis=0) + 10
                img = np.zeros((256, 256), dtype=bool)
                rr, cc = skpolygon(pts[:,1], pts[:,0], img.shape)
                img[rr, cc] = True
                skel, dist = medial_axis(img, return_distance=True)
                n_branches = int(np.sum(skel))
                return n_branches, float(np.max(dist))
            except Exception:
                return 0, 0.0

        def persistent_homology_features(vertices_np):
            # Optional: requires gudhi or ripser, skip if not available
            try:
                import gudhi as gd
                pts = vertices_np.astype(float)
                rips = gd.RipsComplex(points=pts)
                st = rips.create_simplex_tree(max_dimension=2)
                betti = st.betti_numbers()
                # betti[0]: connected components, betti[1]: holes
                return betti
            except Exception:
                return []

        def canny_edge_density(vertices_np):
            try:
                import cv2
                pts = np.round(vertices_np).astype(np.int32)
                pts = pts - pts.min(axis=0) + 10
                img = np.zeros((256, 256), dtype=np.uint8)
                cv2.polylines(img, [pts], isClosed=True, color=255, thickness=2)
                edges = cv2.Canny(img, 50, 150)
                density = float(np.sum(edges > 0)) / edges.size
                return density
            except Exception:
                return 0.0

        print(f"[DIAG] stroke_id={stroke_id} | input stroke_cmds: {stroke_cmds}")
        try:
            parser = BongardLogoParser()
            vertices = parser.parse_action_program(stroke_cmds, scale=120)
        except Exception as e:
            print(f"[DIAG] stroke_id={stroke_id} | Exception in parse_action_program: {e}")
            vertices = None
        print(f"[DIAG] stroke_id={stroke_id} | parsed vertices: {vertices}")

        # Guarantee vertices_np is always defined and valid, and all early returns return a degenerate object
        if not vertices or not isinstance(vertices, (list, tuple)) or len(vertices) < 1:
            vertices_np = np.array([[0.0, 0.0]])
            deg_obj = {
                'id': f'shape_{stroke_id}',
                'vertices': [[0.0, 0.0]],
                'shape_type': 'point',
                'label': 'point',
                'category': 'degenerate',
                'degenerate': True,
                'num_points': 1,
                'degenerate_reason': 'no valid vertices parsed',
                'stroke_cmds': stroke_cmds,
            }
            print(f"[DIAG] stroke_id={stroke_id} | output (degenerate): {deg_obj}")
            return deg_obj
        vertices_np = np.array(vertices)
        # If after conversion, vertices_np is empty or has invalid shape, return degenerate
        if vertices_np.size == 0 or vertices_np.shape[-1] != 2 or len(vertices_np) < 1:
            vertices_np = np.array([[0.0, 0.0]])
            deg_obj = {
                'id': f'shape_{stroke_id}',
                'vertices': [[0.0, 0.0]],
                'shape_type': 'point',
                'label': 'point',
                'category': 'degenerate',
                'degenerate': True,
                'num_points': 1,
                'degenerate_reason': 'invalid vertices_np after conversion',
                'stroke_cmds': stroke_cmds,
            }
            print(f"[DIAG] stroke_id={stroke_id} | output (degenerate): {deg_obj}")
            return deg_obj

        # --- Compute topology, hole, skeleton, edge, and image-processing features ---
        euler_char = compute_euler_characteristic(vertices_np)
        n_holes_shapely = count_holes_shapely(vertices_np)
        n_holes_cv, n_contours_cv = contour_hierarchy_holes(vertices_np)
        n_skel_branches, skel_max_dist = medial_axis_features(vertices_np)
        betti_nums = persistent_homology_features(vertices_np)
        edge_density = canny_edge_density(vertices_np)
        imgproc = image_processing_features(vertices_np)

        # --- Use these features to enrich labeling (main logic, not fallback) ---
        # Labeling logic: prefer more precise, topology-aware, and image-based labels
        topo_label = None
        label_reason = []
        # 1. Image-based: If a single contour with many vertices and no holes, likely a simple polygon
        if imgproc.get('n_contours', 0) == 1 and imgproc.get('n_holes', 0) == 0 and imgproc.get('simplified_n_vertices', [0])[0] >= 4:
            topo_label = 'polygon_imgproc'
            label_reason.append('single contour, no holes, sufficient vertices (image)')
        # 2. Image-based: If multiple contours or holes, label as complex or holed polygon
        elif imgproc.get('n_holes', 0) > 0:
            topo_label = 'polygon_with_holes_imgproc'
            label_reason.append('holes detected (image)')
        elif imgproc.get('n_contours', 0) > 1:
            topo_label = 'multi_contour_shape_imgproc'
            label_reason.append('multiple contours detected (image)')
        # 3. Skeleton: If skeleton sum is high, likely a branched or complex shape
        elif imgproc.get('skeleton_sum', 0) > 50:
            topo_label = 'branched_shape_imgproc'
            label_reason.append('skeleton sum high (image)')
        # 4. Morphological cleaning: If cleaned area is much larger than original, likely noisy or fragmented
        elif imgproc.get('morph_cleaned_sum', 0) > 1.5 * imgproc.get('area', 0):
            topo_label = 'noisy_or_fragmented_imgproc'
            label_reason.append('morphological cleaning increased area (image)')
        # 5. Topology-aware (geometric): holes, euler, betti
        elif n_holes_shapely > 0 or n_holes_cv > 0 or (len(betti_nums) > 1 and betti_nums[1] > 0):
            topo_label = 'polygon_with_holes'
            label_reason.append('holes detected (geometry)')
        elif n_skel_branches > 2:
            topo_label = 'branched_shape'
            label_reason.append('many skeleton branches (geometry)')
        elif euler_char < 1:
            topo_label = 'multi_component_or_holey'
            label_reason.append('euler characteristic < 1 (geometry)')
        elif edge_density > 0.05 and n_contours_cv > 1:
            topo_label = 'complex_edge_shape'
            label_reason.append('high edge density and multiple contours (geometry)')

        # If a strong image-based label is found, return it as the main label
        if topo_label is not None:
            return {
                'id': f'shape_{stroke_id}',
                'vertices': [list(map(float, pt)) for pt in vertices_np],
                'shape_type': topo_label,
                'label': topo_label,
                'category': 'object',
                'degenerate': False,
                'num_points': len(vertices_np),
                'degenerate_reason': '; '.join(label_reason),
                'stroke_cmds': stroke_cmds,
                'imgproc': imgproc,
            }
        n_points = len(vertices_np)
        unique_points = np.unique(vertices_np, axis=0)
        n_unique = len(unique_points)
        # Robust fallback for too few points for a polygon
        if n_unique < 4:
            # 1 unique point: always point
            if n_unique < 2:
                return {
                    'id': f'shape_{stroke_id}',
                    'vertices': [list(map(float, pt)) for pt in vertices_np],
                    'shape_type': 'point',
                    'label': 'point',
                    'category': 'degenerate',
                    'degenerate': True,
                    'num_points': n_unique,
                    'degenerate_reason': 'only one unique point',
                    'stroke_cmds': stroke_cmds,
                }
            # 2 unique points: try RANSAC line, PCA, fallback to line/point
            elif n_unique == 2:
                pt1, pt2 = unique_points[0], unique_points[1]
                length = float(np.linalg.norm(pt2 - pt1))
                orientation = float(np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])))
                # RANSAC line fit (trivial for 2 points, but for completeness)
                ransac_success = False
                try:
                    from sklearn.linear_model import RANSACRegressor
                    X = unique_points[:,0].reshape(-1,1)
                    y = unique_points[:,1]
                    model = RANSACRegressor(residual_threshold=1.0)
                    model.fit(X, y)
                    inlier_mask = model.inlier_mask_
                    ransac_success = inlier_mask.sum() == 2
                except Exception:
                    pass
                if ransac_success:
                    label = 'line_ransac'
                    degenerate_reason = 'RANSAC line fit (2 points)'
                elif length < 1e-2:
                    label = 'point'
                    degenerate_reason = 'two points but very short length'
                elif length < 5.0:
                    label = 'short_line'
                    degenerate_reason = 'line but short length'
                else:
                    label = 'line'
                    degenerate_reason = 'two unique points'
                return {
                    'id': f'shape_{stroke_id}',
                    'vertices': [list(map(float, pt)) for pt in unique_points],
                    'shape_type': 'line',
                    'label': label,
                    'category': 'stroke',
                    'length': length,
                    'orientation': orientation,
                    'degenerate': True,
                    'num_points': n_unique,
                    'degenerate_reason': degenerate_reason,
                    'stroke_cmds': stroke_cmds,
                }
            # 3 unique points: try triangle, RANSAC line, PCA, min bounding rect, fallback to polyline
            elif n_unique == 3:
                a = np.linalg.norm(unique_points[0] - unique_points[1])
                b = np.linalg.norm(unique_points[1] - unique_points[2])
                c = np.linalg.norm(unique_points[2] - unique_points[0])
                s = (a + b + c) / 2.0
                try:
                    area = max(0.0, (s * (s - a) * (s - b) * (s - c))) ** 0.5
                except Exception:
                    area = 0.0
                perimeter = a + b + c
                colinear = area < 1e-2 or area / (perimeter + 1e-6) < 1e-3
                # 1. Try valid triangle
                if not colinear:
                    return {
                        'id': f'shape_{stroke_id}',
                        'vertices': [list(map(float, pt)) for pt in unique_points],
                        'shape_type': 'triangle',
                        'label': 'triangle',
                        'category': 'object',
                        'area': area,
                        'perimeter': perimeter,
                        'degenerate': False,
                        'num_points': n_unique,
                        'degenerate_reason': 'three points, valid triangle',
                        'stroke_cmds': stroke_cmds,
                    }
                # 2. Try RANSAC line fit
                ransac_success = False
                try:
                    from sklearn.linear_model import RANSACRegressor
                    X = unique_points[:,0].reshape(-1,1)
                    y = unique_points[:,1]
                    model = RANSACRegressor(residual_threshold=1.0)
                    model.fit(X, y)
                    inlier_mask = model.inlier_mask_
                    ransac_success = inlier_mask.sum() == 3
                except Exception:
                    pass
                if ransac_success:
                    # Return as line
                    idx = np.argsort(unique_points[:,0])
                    pts_sorted = unique_points[idx]
                    length = float(np.linalg.norm(pts_sorted[2] - pts_sorted[0]))
                    orientation = float(np.degrees(np.arctan2(pts_sorted[2][1] - pts_sorted[0][1], pts_sorted[2][0] - pts_sorted[0][0])))
                    return {
                        'id': f'shape_{stroke_id}',
                        'vertices': [list(map(float, pt)) for pt in pts_sorted],
                        'shape_type': 'line',
                        'label': 'line_ransac',
                        'category': 'degenerate',
                        'length': length,
                        'orientation': orientation,
                        'degenerate': True,
                        'num_points': 3,
                        'degenerate_reason': 'RANSAC line fit (3 points, colinear)',
                        'stroke_cmds': stroke_cmds,
                    }
                # 3. Try PCA min area rectangle
                try:
                    pts = unique_points
                    mean = np.mean(pts, axis=0)
                    cov = np.cov(pts - mean, rowvar=False)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    order = np.argsort(eigvals)[::-1]
                    eigvecs = eigvecs[:, order]
                    rot = eigvecs
                    pts_rot = (pts - mean) @ rot
                    min_x, min_y = np.min(pts_rot, axis=0)
                    max_x, max_y = np.max(pts_rot, axis=0)
                    rect = np.array([
                        [min_x, min_y],
                        [max_x, min_y],
                        [max_x, max_y],
                        [min_x, max_y],
                        [min_x, min_y]
                    ])
                    rect = rect @ rot.T + mean
                    return {
                        'id': f'shape_{stroke_id}',
                        'vertices': [list(map(float, pt)) for pt in rect],
                        'shape_type': 'min_area_rect',
                        'label': 'min_area_rect',
                        'category': 'degenerate',
                        'degenerate': True,
                        'num_points': 4,
                        'degenerate_reason': 'PCA min area rectangle (3 colinear points)',
                        'stroke_cmds': stroke_cmds,
                    }
                except Exception:
                    pass
                # 4. Try minimum bounding line (convex hull)
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(unique_points)
                    hull_points = unique_points[hull.vertices]
                    if len(hull_points) == 2:
                        length = float(np.linalg.norm(hull_points[1] - hull_points[0]))
                        orientation = float(np.degrees(np.arctan2(hull_points[1][1] - hull_points[0][1], hull_points[1][0] - hull_points[0][0])))
                        return {
                            'id': f'shape_{stroke_id}',
                            'vertices': [list(map(float, pt)) for pt in hull_points],
                            'shape_type': 'line',
                            'label': 'line_hull',
                            'category': 'degenerate',
                            'length': length,
                            'orientation': orientation,
                            'degenerate': True,
                            'num_points': 2,
                            'degenerate_reason': 'Convex hull line (3 colinear points)',
                            'stroke_cmds': stroke_cmds,
                        }
                    else:
                        return {
                            'id': f'shape_{stroke_id}',
                            'vertices': [list(map(float, pt)) for pt in hull_points],
                            'shape_type': 'polyline',
                            'label': 'polyline_hull',
                            'category': 'degenerate',
                            'degenerate': True,
                            'num_points': len(hull_points),
                            'degenerate_reason': 'Convex hull polyline (3 points)',
                            'stroke_cmds': stroke_cmds,
                        }
                except Exception:
                    pass
                # 5. Fallback: return as polyline with explicit degenerate reason
                return {
                    'id': f'shape_{stroke_id}',
                    'vertices': [list(map(float, pt)) for pt in unique_points],
                    'shape_type': 'polyline',
                    'label': 'polyline',
                    'category': 'degenerate',
                    'degenerate': True,
                    'num_points': n_unique,
                    'degenerate_reason': 'all fallback strategies for 3 points failed: colinear or numerically unstable',
                    'stroke_cmds': stroke_cmds,
                }
        # If Polygon creation fails due to too few points, try advanced fallback techniques before polyline
        last_fallback_reason = None
        try:
            from shapely.geometry import Polygon
            poly = Polygon(vertices_np)
            if not poly.is_valid or len(vertices_np) < 4:
                # 1. Try alpha shape (concave hull)
                try:
                    import alphashape
                    alpha = 0.0
                    for a in [0.1, 0.5, 1.0, 2.0]:
                        ashape = alphashape.alphashape(vertices_np, a)
                        if ashape and hasattr(ashape, 'exterior'):
                            coords = list(ashape.exterior.coords)
                            if len(coords) >= 4:
                                logging.debug(f"[Fallback] stroke_id={stroke_id} | alpha_shape succeeded with alpha={a}")
                                return {
                                    'id': f'shape_{stroke_id}',
                                    'vertices': [list(map(float, pt)) for pt in coords],
                                    'shape_type': 'alpha_shape',
                                    'label': 'alpha_shape',
                                    'category': 'object',
                                    'degenerate': False,
                                    'num_points': len(coords),
                                    'degenerate_reason': 'alpha shape fallback',
                                    'stroke_cmds': stroke_cmds,
                                }
                    last_fallback_reason = 'alpha shape failed: not enough points or alphashape not available'
                    logging.debug(f"[Fallback] stroke_id={stroke_id} | alpha_shape failed")
                except Exception as e:
                    last_fallback_reason = f'alpha shape exception: {e}'
                    logging.debug(f"[Fallback] stroke_id={stroke_id} | alpha_shape exception: {e}")
                # 2. Try PCA minimum area rectangle
                try:
                    from scipy.spatial import ConvexHull
                    from numpy.linalg import eig, inv
                    pts = vertices_np
                    mean = np.mean(pts, axis=0)
                    cov = np.cov(pts - mean, rowvar=False)
                    eigvals, eigvecs = eig(cov)
                    order = np.argsort(eigvals)[::-1]
                    eigvecs = eigvecs[:, order]
                    rot = eigvecs
                    pts_rot = (pts - mean) @ rot
                    min_x, min_y = np.min(pts_rot, axis=0)
                    max_x, max_y = np.max(pts_rot, axis=0)
                    rect = np.array([
                        [min_x, min_y],
                        [max_x, min_y],
                        [max_x, max_y],
                        [min_x, max_y],
                        [min_x, min_y]
                    ])
                    rect = rect @ rot.T + mean
                    logging.debug(f"[Fallback] stroke_id={stroke_id} | PCA min area rectangle succeeded")
                    return {
                        'id': f'shape_{stroke_id}',
                        'vertices': [list(map(float, pt)) for pt in rect],
                        'shape_type': 'min_area_rect',
                        'label': 'min_area_rect',
                        'category': 'object',
                        'degenerate': False,
                        'num_points': 4,
                        'degenerate_reason': 'PCA min area rectangle fallback',
                        'stroke_cmds': stroke_cmds,
                    }
                except Exception as e:
                    last_fallback_reason = f'PCA min area rectangle exception: {e}'
                    logging.debug(f"[Fallback] stroke_id={stroke_id} | PCA min area rectangle exception: {e}")
                # 3. Try minimum area enclosing triangle (Welzl's algorithm)
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(vertices_np)
                    hull_points = vertices_np[hull.vertices]
                    if len(hull_points) == 3:
                        a = np.linalg.norm(hull_points[0] - hull_points[1])
                        b = np.linalg.norm(hull_points[1] - hull_points[2])
                        c = np.linalg.norm(hull_points[2] - hull_points[0])
                        s = (a + b + c) / 2.0
                        try:
                            area = max(0.0, (s * (s - a) * (s - b) * (s - c))) ** 0.5
                        except Exception:
                            area = 0.0
                        perimeter = a + b + c
                        logging.debug(f"[Fallback] stroke_id={stroke_id} | min area triangle succeeded")
                        return {
                            'id': f'shape_{stroke_id}',
                            'vertices': [list(map(float, pt)) for pt in hull_points],
                            'shape_type': 'triangle',
                            'label': 'triangle',
                            'category': 'object',
                            'area': area,
                            'perimeter': perimeter,
                            'degenerate': False,
                            'num_points': 3,
                            'degenerate_reason': 'min area triangle fallback',
                            'stroke_cmds': stroke_cmds,
                        }
                    last_fallback_reason = 'min area triangle failed: hull not 3 points'
                    logging.debug(f"[Fallback] stroke_id={stroke_id} | min area triangle failed: hull not 3 points")
                except Exception as e:
                    last_fallback_reason = f'min area triangle exception: {e}'
                    logging.debug(f"[Fallback] stroke_id={stroke_id} | min area triangle exception: {e}")
                # 4. Try circle/ellipse fit
                try:
                    pts = vertices_np
                    x = pts[:,0]
                    y = pts[:,1]
                    x_m = np.mean(x)
                    y_m = np.mean(y)
                    def calc_R(xc, yc):
                        return np.sqrt((x-xc)**2 + (y-yc)**2)
                    def f_2(c):
                        Ri = calc_R(*c)
                        return Ri - Ri.mean()
                    from scipy.optimize import leastsq
                    center_estimate = x_m, y_m
                    center, _ = leastsq(f_2, center_estimate)
                    Ri = calc_R(*center)
                    R = Ri.mean()
                    residu = np.sum((Ri - R)**2)
                    if np.std(Ri) / (R + 1e-6) < 0.2:
                        logging.debug(f"[Fallback] stroke_id={stroke_id} | circle fit succeeded")
                        return {
                            'id': f'shape_{stroke_id}',
                            'vertices': [list(map(float, pt)) for pt in pts],
                            'shape_type': 'circle',
                            'label': 'circle',
                            'category': 'object',
                            'degenerate': False,
                            'num_points': len(pts),
                            'degenerate_reason': 'circle fit fallback',
                            'center': list(map(float, center)),
                            'radius': float(R),
                            'residual': float(residu),
                            'stroke_cmds': stroke_cmds,
                        }
                    last_fallback_reason = 'circle fit failed: not circular enough'
                    logging.debug(f"[Fallback] stroke_id={stroke_id} | circle fit failed: not circular enough")
                except Exception as e:
                    last_fallback_reason = f'circle fit exception: {e}'
                    logging.debug(f"[Fallback] stroke_id={stroke_id} | circle fit exception: {e}")
                # 5. If all else fails, fallback to polyline
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(vertices_np)
                    hull_points = vertices_np[hull.vertices]
                    logging.debug(f"[Fallback] stroke_id={stroke_id} | polyline fallback (after advanced techniques)")
                    return {
                        'id': f'shape_{stroke_id}',
                        'vertices': [list(map(float, pt)) for pt in hull_points],
                        'shape_type': 'polyline',
                        'label': 'polyline',
                        'category': 'degenerate',
                        'degenerate': True,
                        'num_points': len(hull_points),
                        'degenerate_reason': 'polygon fallback to polyline (after advanced techniques)',
                        'stroke_cmds': stroke_cmds,
                    }
                except Exception as e:
                    last_fallback_reason = f'polyline fallback exception: {e}'
                    logging.debug(f"[Fallback] stroke_id={stroke_id} | polyline fallback exception: {e}")
                    return None, last_fallback_reason
        except Exception as e:
            last_fallback_reason = f'polygon creation failed: {e}'
            logging.debug(f"[Fallback] stroke_id={stroke_id} | polygon creation failed: {e}")
            return None, last_fallback_reason
        # --- Advanced degenerate structure analysis (real geometric labeling, not bypass) ---
        # 1. Colinear Polyline (PCA or RANSAC)
        try:
            pts = vertices_np
            if len(pts) >= 2:
                # PCA colinearity check
                mean = np.mean(pts, axis=0)
                centered = pts - mean
                cov = np.cov(centered, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(cov)
                ratio = eigvals[-1] / (eigvals[0] + 1e-8)
                # If ratio is very large, points are nearly colinear
                if ratio > 1e3:
                    # Project points onto main axis
                    axis = eigvecs[:, -1]
                    proj = centered @ axis
                    idx_min = np.argmin(proj)
                    idx_max = np.argmax(proj)
                    start_pt = pts[idx_min]
                    end_pt = pts[idx_max]
                    orientation = float(np.degrees(np.arctan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0])))
                    length = float(np.linalg.norm(end_pt - start_pt))
                    logging.debug(f"[DegenerateLabel] stroke_id={stroke_id} | colinear_polyline (PCA) detected")
                    return {
                        'id': f'shape_{stroke_id}',
                        'vertices': [list(map(float, start_pt)), list(map(float, end_pt))],
                        'shape_type': 'colinear_polyline',
                        'label': 'colinear_polyline',
                        'category': 'degenerate',
                        'degenerate': True,
                        'num_points': len(pts),
                        'degenerate_reason': 'all points colinear (PCA)',
                        'orientation': orientation,
                        'length': length,
                        'stroke_cmds': stroke_cmds,
                    }
                # RANSAC line fit (robust to outliers)
                try:
                    from sklearn.linear_model import RANSACRegressor
                    X = pts[:,0].reshape(-1,1)
                    y = pts[:,1]
                    model = RANSACRegressor(residual_threshold=2.0)
                    model.fit(X, y)
                    inlier_mask = model.inlier_mask_
                    if inlier_mask.sum() / len(pts) > 0.95:
                        inlier_pts = pts[inlier_mask]
                        idx_min = np.argmin(inlier_pts[:,0])
                        idx_max = np.argmax(inlier_pts[:,0])
                        start_pt = inlier_pts[idx_min]
                        end_pt = inlier_pts[idx_max]
                        orientation = float(np.degrees(np.arctan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0])))
                        length = float(np.linalg.norm(end_pt - start_pt))
                        logging.debug(f"[DegenerateLabel] stroke_id={stroke_id} | colinear_polyline (RANSAC) detected")
                        return {
                            'id': f'shape_{stroke_id}',
                            'vertices': [list(map(float, start_pt)), list(map(float, end_pt))],
                            'shape_type': 'colinear_polyline',
                            'label': 'colinear_polyline',
                            'category': 'degenerate',
                            'degenerate': True,
                            'num_points': len(pts),
                            'degenerate_reason': 'all points colinear (RANSAC)',
                            'orientation': orientation,
                            'length': length,
                            'stroke_cmds': stroke_cmds,
                        }
                except Exception:
                    pass
        except Exception:
            pass

        # 2. Piecewise Polyline (RDP simplification)
        try:
            if len(vertices_np) > 3:
                def rdp(points, epsilon=2.0):
                    from math import hypot
                    def _rdp(points, epsilon):
                        if len(points) < 3:
                            return points
                        start, end = points[0], points[-1]
                        max_dist, idx = 0, 0
                        for i in range(1, len(points)-1):
                            px, py = points[i]
                            sx, sy = start
                            ex, ey = end
                            num = abs((ey-sy)*px - (ex-sx)*py + ex*sy - ey*sx)
                            den = hypot(ex-sx, ey-sy)
                            dist = num / (den+1e-8)
                            if dist > max_dist:
                                max_dist = dist
                                idx = i
                        if max_dist > epsilon:
                            left = _rdp(points[:idx+1], epsilon)
                            right = _rdp(points[idx:], epsilon)
                            return left[:-1] + right
                        else:
                            return [start, end]
                    return np.array(_rdp(list(points), epsilon))
                simplified = rdp(vertices_np, epsilon=2.0)
                if len(simplified) < len(vertices_np) and len(simplified) >= 2:
                    logging.debug(f"[DegenerateLabel] stroke_id={stroke_id} | piecewise_polyline (RDP) detected")
                    return {
                        'id': f'shape_{stroke_id}',
                        'vertices': [list(map(float, pt)) for pt in simplified],
                        'shape_type': 'piecewise_polyline',
                        'label': 'piecewise_polyline',
                        'category': 'degenerate',
                        'degenerate': True,
                        'num_points': len(simplified),
                        'degenerate_reason': 'piecewise linear structure (RDP)',
                        'stroke_cmds': stroke_cmds,
                    }
        except Exception:
            pass

        # 3. Principal Curve/Fourier (principal_axis_curve)
        try:
            if len(vertices_np) > 3:
                pts = np.array(vertices_np)
                complex_pts = pts[:,0] + 1j*pts[:,1]
                coeffs = np.fft.fft(complex_pts)
                # Only keep 3 lowest frequencies (principal axis)
                low_freq = np.fft.ifft(coeffs[:3], n=len(pts))
                curve_pts = np.column_stack([low_freq.real, low_freq.imag])
                if np.std(curve_pts - pts) < 5.0:
                    logging.debug(f"[DegenerateLabel] stroke_id={stroke_id} | principal_axis_curve (Fourier) detected")
                    return {
                        'id': f'shape_{stroke_id}',
                        'vertices': [list(map(float, pt)) for pt in curve_pts],
                        'shape_type': 'principal_axis_curve',
                        'label': 'principal_axis_curve',
                        'category': 'degenerate',
                        'degenerate': True,
                        'num_points': len(curve_pts),
                        'degenerate_reason': 'principal axis curve (Fourier)',
                        'stroke_cmds': stroke_cmds,
                    }
        except Exception:
            pass

        # 4. Clustered Line Segments (DBSCAN + line fit)
        try:
            from sklearn.cluster import DBSCAN
            if len(vertices_np) > 3:
                clustering = DBSCAN(eps=5, min_samples=3).fit(vertices_np)
                labels = clustering.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters > 1:
                    segments = []
                    for cl in set(labels):
                        if cl == -1:
                            continue
                        cl_pts = vertices_np[labels == cl]
                        if len(cl_pts) >= 2:
                            # Fit line to cluster
                            mean = np.mean(cl_pts, axis=0)
                            centered = cl_pts - mean
                            cov = np.cov(centered, rowvar=False)
                            eigvals, eigvecs = np.linalg.eigh(cov)
                            axis = eigvecs[:, -1]
                            proj = centered @ axis
                            idx_min = np.argmin(proj)
                            idx_max = np.argmax(proj)
                            start_pt = cl_pts[idx_min]
                            end_pt = cl_pts[idx_max]
                            segments.append([list(map(float, start_pt)), list(map(float, end_pt))])
                    if segments:
                        logging.debug(f"[DegenerateLabel] stroke_id={stroke_id} | multi_segment_line (DBSCAN) detected")
                        return {
                            'id': f'shape_{stroke_id}',
                            'vertices': segments,
                            'shape_type': 'multi_segment_line',
                            'label': 'multi_segment_line',
                            'category': 'degenerate',
                            'degenerate': True,
                            'num_points': len(vertices_np),
                            'degenerate_reason': 'multi-segment line (DBSCAN)',
                            'stroke_cmds': stroke_cmds,
                        }
        except Exception:
            pass

        # 5. Jitter and Refit
        try:
            if len(vertices_np) > 2:
                jitter = np.random.normal(0, 1e-2, vertices_np.shape)
                jittered = vertices_np + jitter
                mean = np.mean(jittered, axis=0)
                centered = jittered - mean
                cov = np.cov(centered, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(cov)
                ratio = eigvals[-1] / (eigvals[0] + 1e-8)
                if ratio > 1e3:
                    axis = eigvecs[:, -1]
                    proj = centered @ axis
                    idx_min = np.argmin(proj)
                    idx_max = np.argmax(proj)
                    start_pt = jittered[idx_min]
                    end_pt = jittered[idx_max]
                    orientation = float(np.degrees(np.arctan2(end_pt[1] - start_pt[1], end_pt[0] - start_pt[0])))
                    length = float(np.linalg.norm(end_pt - start_pt))
                    logging.debug(f"[DegenerateLabel] stroke_id={stroke_id} | jittered_polyline detected")
                    return {
                        'id': f'shape_{stroke_id}',
                        'vertices': [list(map(float, start_pt)), list(map(float, end_pt))],
                        'shape_type': 'jittered_polyline',
                        'label': 'jittered_polyline',
                        'category': 'degenerate',
                        'degenerate': True,
                        'num_points': len(vertices_np),
                        'degenerate_reason': 'jittered colinear structure',
                        'orientation': orientation,
                        'length': length,
                        'stroke_cmds': stroke_cmds,
                    }
        except Exception:
            pass

        # 6. Explicit degenerate geometry: point cloud
        try:
            if len(vertices_np) > 1:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(vertices_np)
                hull_points = vertices_np[hull.vertices]
                logging.debug(f"[DegenerateLabel] stroke_id={stroke_id} | degenerate_point_cloud (convex hull) detected")
                return {
                    'id': f'shape_{stroke_id}',
                    'vertices': [list(map(float, pt)) for pt in hull_points],
                    'shape_type': 'degenerate_point_cloud',
                    'label': 'degenerate_point_cloud',
                    'category': 'degenerate',
                    'degenerate': True,
                    'num_points': len(hull_points),
                    'degenerate_reason': 'degenerate point cloud (convex hull)',
                    'stroke_cmds': stroke_cmds,
                }
        except Exception:
            pass

        # If we reach here, all fallbacks failed, ensure a fallback reason is set
        if last_fallback_reason is None:
            last_fallback_reason = "all fallback strategies failed: unknown or unhandled degenerate input (e.g., all points colinear, identical, or numerical instability)"
        logging.debug(f"[Fallback] stroke_id={stroke_id} | all fallback strategies failed: {last_fallback_reason}")
        return None, last_fallback_reason
        # 1. Smoothing (Savitzky–Golay filter)
        try:
            from scipy.signal import savgol_filter
            if n_points > 7:
                vertices_np[:,0] = savgol_filter(vertices_np[:,0], window_length=min(11, n_points//2*2+1), polyorder=2)
                vertices_np[:,1] = savgol_filter(vertices_np[:,1], window_length=min(11, n_points//2*2+1), polyorder=2)
                logging.debug(f"[ShapeDebug] stroke_id={stroke_id} | Savitzky–Golay smoothing applied")
        except ImportError:
            logging.debug("[ShapeDebug] scipy not installed, skipping smoothing.")

        # 2. Attempt to close the shape if nearly closed
        if n_points > 2 and np.linalg.norm(vertices_np[0] - vertices_np[-1]) < 1e-1:
            vertices_np[-1] = vertices_np[0]
            logging.debug(f"[ShapeDebug] stroke_id={stroke_id} | shape auto-closed (first and last point merged)")

        # 3. Simplify shape if too many points (Ramer–Douglas–Peucker)
        def rdp(points, epsilon=1.0):
            from math import hypot
            def _rdp(points, epsilon):
                if len(points) < 3:
                    return points
                start, end = points[0], points[-1]
                max_dist, idx = 0, 0
                for i in range(1, len(points)-1):
                    px, py = points[i]
                    sx, sy = start
                    ex, ey = end
                    num = abs((ey-sy)*px - (ex-sx)*py + ex*sy - ey*sx)
                    den = hypot(ex-sx, ey-sy)
                    dist = num / (den+1e-8)
                    if dist > max_dist:
                        max_dist = dist
                        idx = i
                if max_dist > epsilon:
                    left = _rdp(points[:idx+1], epsilon)
                    right = _rdp(points[idx:], epsilon)
                    return left[:-1] + right
                else:
                    return [start, end]
            return np.array(_rdp(list(points), epsilon))
        if n_points > 30:
            orig_n = n_points
            vertices_np = rdp(vertices_np, epsilon=2.0)
            n_points = len(vertices_np)
            logging.debug(f"[ShapeDebug] stroke_id={stroke_id} | shape simplified from {orig_n} to {n_points} points")

        # 4. RANSAC-based line/circle/ellipse fitting (out-of-the-box robust fitting)
        def ransac_line(points, threshold=2.0):
            from sklearn.linear_model import RANSACRegressor
            X = points[:,0].reshape(-1,1)
            y = points[:,1]
            model = RANSACRegressor(residual_threshold=threshold)
            model.fit(X, y)
            inlier_mask = model.inlier_mask_
            return inlier_mask.sum() / len(points) > 0.8
        def ransac_circle(points, threshold=2.0):
            from sklearn.linear_model import RANSACRegressor
            x = points[:,0]
            y = points[:,1]
            x_m = np.mean(x)
            y_m = np.mean(y)
            def calc_R(xc, yc):
                return np.sqrt((x-xc)**2 + (y-yc)**2)
            def f_2(c):
                Ri = calc_R(*c)
                return Ri - Ri.mean()
            from scipy.optimize import leastsq
            center_estimate = x_m, y_m
            center, _ = leastsq(f_2, center_estimate)
            Ri = calc_R(*center)
            R = Ri.mean()
            residu = np.sum((Ri - R)**2)
            inliers = np.abs(Ri - R) < threshold
            return inliers.sum() / len(points) > 0.8

        # Ellipse fitting using EllipseModel (scikit-image)
        def ellipse_fit(points):
            try:
                from skimage.measure import EllipseModel
            except ImportError:
                logging.debug("[ShapeDebug] skimage not installed, skipping ellipse fitting.")
                return None
            try:
                model = EllipseModel()
                if model.estimate(points):
                    xc, yc, a, b, theta = model.params
                    if a > 0 and b > 0 and 0.2 < b/a < 5:
                        return {'center': (xc, yc), 'axes': (a, b), 'angle': theta}
            except Exception:
                pass
            return None

        # Hough Transform for line/arc detection (OpenCV)
        def hough_lines(points):
            try:
                import cv2
                img = np.zeros((256, 256), dtype=np.uint8)
                pts = np.round(points).astype(np.int32)
                pts = pts - pts.min(axis=0)  # shift to positive
                pts = pts + 10  # margin
                cv2.polylines(img, [pts], isClosed=False, color=255, thickness=2)
                lines = cv2.HoughLinesP(img, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
                if lines is not None and len(lines) > 0:
                    return True, lines
            except ImportError:
                logging.debug("[ShapeDebug] OpenCV not installed, skipping Hough Transform.")
            return False, None

        # Convexity defect analysis (OpenCV) with robust fallback for self-intersecting polygons
        def convexity_defects(points):
            try:
                import cv2
                from shapely.geometry import Polygon, LineString
                from shapely.ops import polygonize
                pts = np.round(points).astype(np.int32)
                pts = pts.reshape((-1,1,2))
                hull = cv2.convexHull(pts, returnPoints=False)
                if hull is not None and len(hull) > 3:
                    try:
                        defects = cv2.convexityDefects(pts, hull)
                        if defects is not None:
                            return len(defects), None, None
                    except cv2.error as e:
                        # OpenCV error: likely self-intersecting polygon
                        logging.warning(f"[ShapeDebug] Convexity defect OpenCV error: {e}. Attempting robust fallback.")
                        poly = Polygon(points)
                        is_simple = poly.is_simple if poly.is_valid else False
                        if not is_simple:
                            # Try to decompose into simple polygons
                            try:
                                # Use LineString to polygonize
                                lines = LineString(points)
                                simple_polys = list(polygonize(lines))
                                if simple_polys:
                                    labels = []
                                    for simp in simple_polys:
                                        simp_label = 'polygon'
                                        if simp.is_valid and simp.is_simple:
                                            n_verts = len(list(simp.exterior.coords)) - 1
                                            if n_verts == 3:
                                                simp_label = 'triangle'
                                            elif n_verts == 4:
                                                simp_label = 'quadrilateral'
                                            elif n_verts == 5:
                                                simp_label = 'pentagon'
                                            elif n_verts == 6:
                                                simp_label = 'hexagon'
                                            elif n_verts > 6:
                                                simp_label = 'polygon'
                                        labels.append({'label': simp_label, 'area': simp.area, 'perimeter': simp.length, 'num_vertices': len(list(simp.exterior.coords))-1})
                                    return None, labels, {'self_intersecting': True, 'num_simple_components': len(simple_polys)}
                            except Exception as e2:
                                logging.warning(f"[ShapeDebug] Fallback decomposition failed: {e2}")
                        # If all else fails, label as self_intersecting_polygon
                        return None, None, {'label': 'self_intersecting_polygon', 'self_intersecting': True, 'area': poly.area, 'perimeter': poly.length}
            except ImportError:
                logging.debug("[ShapeDebug] OpenCV or shapely not installed, skipping convexity defect analysis.")
            return 0, None, None

        # Fourier Descriptors for advanced shape analysis
        def fourier_descriptors(points, n_descriptors=10):
            pts = np.array(points)
            complex_pts = pts[:,0] + 1j*pts[:,1]
            coeffs = np.fft.fft(complex_pts)
            # Only keep n_descriptors low-frequency descriptors
            desc = coeffs[:n_descriptors]
            return np.abs(desc)

        # DBSCAN clustering for multi-shape separation
        def cluster_points(points, eps=5, min_samples=5):
            try:
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
                labels = clustering.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                return labels, n_clusters
            except ImportError:
                logging.debug("[ShapeDebug] sklearn not installed, skipping clustering.")
            return np.zeros(len(points)), 1

        # 5. Angle-based vertex detection for polygons
        def detect_vertices(points, angle_thresh=30):
            # Returns indices of vertices based on angle threshold (degrees)
            pts = np.array(points)
            n = len(pts)
            if n < 3:
                return []
            angles = []
            for i in range(n):
                p0 = pts[i-1]
                p1 = pts[i]
                p2 = pts[(i+1)%n]
                v1 = p0 - p1
                v2 = p2 - p1
                angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8), -1, 1)))
                angles.append(angle)
            # Vertex if angle < threshold
            return [i for i, a in enumerate(angles) if a < angle_thresh]

        logging.debug(f"[ShapeDebug] stroke_id={stroke_id} | n_points={n_points} | vertices={vertices_np.tolist()}")
        poly = Polygon(vertices_np)
        area = poly.area if poly.is_valid else 0.0
        convexity = poly.convex_hull.area / area if area > 0 else 0.0
        centroid = poly.centroid
        bbox = [float(np.min(vertices_np[:,0])), float(np.min(vertices_np[:,1])),
                float(np.max(vertices_np[:,0])), float(np.max(vertices_np[:,1]))]
        logging.debug(f"[ShapeDebug] stroke_id={stroke_id} | first_stroke_cmds={stroke_cmds[:5]}")
        shape_type = infer_shape_type(vertices_np)
        perimeter = poly.length if poly.is_valid else float(np.sum(np.linalg.norm(np.diff(vertices_np, axis=0), axis=1)))
        closure = bool(np.linalg.norm(vertices_np[0] - vertices_np[-1]) < 1e-2)
        orientation = compute_orientation(vertices_np)
        symmetry = compute_symmetry_score(vertices_np)
        symmetry_axis = compute_symmetry_axis(vertices_np)
        edge_lengths = np.linalg.norm(np.diff(vertices_np, axis=0, append=vertices_np[:1]), axis=1)
        roughness = float(np.std(edge_lengths) / (np.mean(edge_lengths) + 1e-6))
        stroke_length = float(np.sum(edge_lengths))
        ellipse_params = ellipse_fit(vertices_np)
        hough_found, hough_lines_result = hough_lines(vertices_np)
        n_defects, simple_poly_labels, self_intersecting_info = convexity_defects(vertices_np)
        fourier_desc = fourier_descriptors(vertices_np)
        cluster_labels, n_clusters = cluster_points(vertices_np)

        # --- Hybrid labeling: combine geometric, topology, skeleton, and edge features ---
        hybrid_label = None
        if topo_label is not None:
            hybrid_label = topo_label
        elif simple_poly_labels is not None:
            hybrid_label = 'multi_simple_polygons'
        elif self_intersecting_info is not None:
            hybrid_label = self_intersecting_info.get('label', 'self_intersecting_polygon')
        elif shape_type in ['circle', 'ellipse']:
            hybrid_label = 'ellipse'
        elif shape_type in ['rectangle', 'quadrilateral', 'square']:
            hybrid_label = 'quadrilateral'
        elif shape_type in ['triangle']:
            hybrid_label = 'triangle'
        elif shape_type in ['pentagon', 'hexagon', 'polygon']:
            hybrid_label = 'regular_polygon'
        elif shape_type == 'line':
            hybrid_label = 'open_shape'
        else:
            hybrid_label = 'other_shape'

        high_category = 'object' if closure and area > 0 else 'stroke'
        logging.debug(
            f"[ShapeDebug] stroke_id={stroke_id} | "
            f"shape_type={shape_type}, hybrid_label={hybrid_label}, category={high_category}, bbox={bbox}, area={area:.4f}, "
            f"convexity={convexity:.4f}, cx={centroid.x:.2f}, cy={centroid.y:.2f}, orientation={orientation:.2f}, "
            f"perimeter={perimeter:.4f}, closure={closure}, symmetry={symmetry:.4f}, symmetry_axis={symmetry_axis}, "
            f"roughness={roughness:.4f}, stroke_length={stroke_length:.4f}, num_points={len(vertices_np)}, "
            f"ellipse_params={ellipse_params}, hough_lines_found={hough_found}, n_convexity_defects={n_defects}, "
            f"fourier_desc={fourier_desc.tolist() if hasattr(fourier_desc, 'tolist') else fourier_desc}, n_clusters={n_clusters}, "
            f"euler_char={euler_char}, n_holes_shapely={n_holes_shapely}, n_holes_cv={n_holes_cv}, n_skel_branches={n_skel_branches}, skel_max_dist={skel_max_dist}, betti_nums={betti_nums}, edge_density={edge_density:.4f}"
        )
        out_obj = {
            'id': f'shape_{stroke_id}',
            'vertices': [list(map(float, pt)) for pt in vertices_np],
            'shape_type': shape_type,
            'label': hybrid_label,
            'category': high_category,
            'bbox': bbox,
            'area': float(area),
            'convexity': float(convexity),
            'cx': centroid.x,
            'cy': centroid.y,
            'orientation': orientation,
            'perimeter': float(perimeter),
            'closure': closure,
            'symmetry': symmetry,
            'symmetry_axis': symmetry_axis,
            'roughness': roughness,
            'stroke_length': stroke_length,
            'num_points': len(vertices_np),
            'stroke_cmds': stroke_cmds,
            'ellipse_params': ellipse_params,
            'hough_lines_found': hough_found,
            'n_convexity_defects': n_defects,
            'fourier_descriptors': fourier_desc.tolist() if hasattr(fourier_desc, 'tolist') else fourier_desc,
            'cluster_labels': cluster_labels.tolist() if hasattr(cluster_labels, 'tolist') else cluster_labels,
            'n_clusters': n_clusters,
            'euler_characteristic': euler_char,
            'n_holes_shapely': n_holes_shapely,
            'n_holes_cv': n_holes_cv,
            'n_contours_cv': n_contours_cv,
            'n_skeleton_branches': n_skel_branches,
            'skeleton_max_dist': skel_max_dist,
            'betti_numbers': betti_nums,
            'edge_density': edge_density,
        }
        print(f"[DIAG] stroke_id={stroke_id} | output: {out_obj}")
        return out_obj
    def compute_symmetry_axis(vertices):
        # Returns the main axis direction as a unit vector (from PCA)
        pts = np.array(vertices)
        pts = pts - np.mean(pts, axis=0)
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_axis = eigvecs[:, np.argmax(eigvals)]
        norm = np.linalg.norm(major_axis)
        if norm == 0:
            return [1.0, 0.0]
        return list((major_axis / norm).tolist())
    def compute_orientation(vertices):
        # PCA: direction of max variance
        pts = np.array(vertices)
        pts = pts - np.mean(pts, axis=0)
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_axis = eigvecs[:, np.argmax(eigvals)]
        angle = np.arctan2(major_axis[1], major_axis[0])
        return float(np.degrees(angle))

    def compute_symmetry_score(vertices):
        # Reflectional symmetry: mean distance between points and their mirror across centroid axis
        pts = np.array(vertices)
        centroid = np.mean(pts, axis=0)
        pts_centered = pts - centroid
        # Try symmetry over x and y axes
        mirror_x = pts_centered * np.array([-1, 1])
        mirror_y = pts_centered * np.array([1, -1])
        # Find best matching (min mean distance)
        score_x = np.mean(np.min(np.linalg.norm(mirror_x[:, None, :] - pts_centered[None, :, :], axis=2), axis=1))
        score_y = np.mean(np.min(np.linalg.norm(mirror_y[:, None, :] - pts_centered[None, :, :], axis=2), axis=1))
        return float(min(score_x, score_y))

    parser = argparse.ArgumentParser()
    # ...existing code...

    parser.add_argument('--input-dir', required=True, help='Input directory containing ShapeBongard_V2 data')
    parser.add_argument('--output', required=True, help='Output JSON file to save extracted shape data')
    parser.add_argument('--problems-list', required=False, help='Optional file with list of problem IDs to process')
    def infer_shape_type(vertices):
        n = len(vertices)
        if n < 2:
            logging.debug("[ShapeDebug] Not enough points to identify shape.")
            return 'unknown'
        pts = np.array(vertices)
        is_closed = np.linalg.norm(pts[0] - pts[-1]) < 1e-1
        # Out-of-the-box: RANSAC line/circle/ellipse fitting, EllipseModel, Hough Transform
        try:
            from skimage.measure import EllipseModel
        except ImportError:
            EllipseModel = None
        try:
            import cv2
        except ImportError:
            cv2 = None
        if n > 7:
            if ransac_line(pts):
                logging.debug("[ShapeDebug] RANSAC: Detected as line.")
                return 'line'
            if ransac_circle(pts):
                logging.debug("[ShapeDebug] RANSAC: Detected as circle.")
                return 'circle'
            if EllipseModel is not None:
                em = EllipseModel()
                if em.estimate(pts):
                    xc, yc, a, b, theta = em.params
                    if a > 0 and b > 0 and 0.2 < b/a < 5:
                        logging.debug("[ShapeDebug] EllipseModel: Detected as ellipse.")
                        return 'ellipse'
            if cv2 is not None:
                img = np.zeros((256, 256), dtype=np.uint8)
                pts_img = np.round(pts).astype(np.int32)
                pts_img = pts_img - pts_img.min(axis=0)
                pts_img = pts_img + 10
                cv2.polylines(img, [pts_img], isClosed=False, color=255, thickness=2)
                lines = cv2.HoughLinesP(img, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
                if lines is not None and len(lines) > 0:
                    logging.debug("[ShapeDebug] HoughLinesP: Detected as line.")
                    return 'line'
        # Area/perimeter ratio for degeneracy
        try:
            poly = Polygon(pts)
            area = poly.area if poly.is_valid else 0.0
            perimeter = poly.length if poly.is_valid else float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        except Exception:
            area = 0.0
            perimeter = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if perimeter > 0 and area / (perimeter ** 2) < 0.001:
            logging.debug(f"[ShapeDebug] Degenerate polygon detected (area={area:.4f}, perimeter={perimeter:.4f})")
            return 'line' if not is_closed else 'open_shape'
        if not is_closed:
            logging.debug("[ShapeDebug] Shape is not closed, cannot be polygon/triangle/rectangle.")
            return 'open_shape'
        # Angle-based vertex detection
        vertices_idx = detect_vertices(pts)
        if len(vertices_idx) == 3:
            logging.debug("[ShapeDebug] Angle-based: Detected triangle.")
            return 'triangle'
        if len(vertices_idx) == 4:
            logging.debug("[ShapeDebug] Angle-based: Detected quadrilateral.")
            vecs = pts - np.roll(pts, 1, axis=0)
            lens = np.linalg.norm(vecs, axis=1)
            if np.allclose(lens[0], lens[2], rtol=0.2) and np.allclose(lens[1], lens[3], rtol=0.2):
                dotprods = [np.dot(vecs[i], vecs[(i+1)%4]) for i in range(4)]
                if all(abs(dp) < 0.2 * (lens[i]*lens[(i+1)%4]) for i, dp in enumerate(dotprods)):
                    return 'rectangle'
            return 'quadrilateral'
        if len(vertices_idx) == 5:
            logging.debug("[ShapeDebug] Angle-based: Detected pentagon.")
            return 'pentagon'
        if len(vertices_idx) == 6:
            logging.debug("[ShapeDebug] Angle-based: Detected hexagon.")
            return 'hexagon'
        if len(vertices_idx) > 6:
            logging.debug(f"[ShapeDebug] Angle-based: Detected polygon with {len(vertices_idx)} vertices.")
            return 'polygon'
        # Convexity defect analysis for concave polygons
        try:
            import cv2
            pts_cv = np.round(pts).astype(np.int32)
            pts_cv = pts_cv.reshape((-1,1,2))
            hull = cv2.convexHull(pts_cv, returnPoints=False)
            if hull is not None and len(hull) > 3:
                defects = cv2.convexityDefects(pts_cv, hull)
                if defects is not None and len(defects) > 0:
                    logging.debug(f"[ShapeDebug] Convexity defects found: {len(defects)}")
                    return 'concave_polygon'
        except ImportError:
            pass
        # Fallback: classic rules
        if is_roughly_circular(vertices):
            return 'circle'
        if n == 2:
            return 'line'
        logging.debug("[ShapeDebug] Could not identify shape type.")
        return 'unknown'

    def is_roughly_circular(points, tolerance=0.15):
        points = np.array(points)
        centroid = np.mean(points, axis=0)
        radii = np.linalg.norm(points - centroid, axis=1)
        return np.std(radii) / (np.mean(radii) + 1e-6) < tolerance


    args = parser.parse_args()


    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    categories = ['bd', 'ff', 'hd']
    base_dir = args.input_dir
    output_path = args.output
    problems_list_path = args.problems_list

    # Read problem IDs from problems-list file
    required_ids = None
    if problems_list_path:
        with open(problems_list_path, 'r') as f:
            required_ids = set(line.strip() for line in f if line.strip())

    all_results = []
    flagged_cases = []
    all_labels = set()
    all_shape_types = set()
    all_categories = set()

    for cat in categories:
        json_path = os.path.join(base_dir, f"ShapeBongard_V2/{cat}/{cat}_action_programs.json")
        if not os.path.exists(json_path):
            logging.warning(f"Missing JSON file: {json_path}")
            continue

        with open(json_path, 'r') as f:
            for problem_id, pos_neg_lists in ijson.kvitems(f, ''):
                if required_ids and problem_id not in required_ids:
                    continue
                for label, group in zip(['category_1', 'category_0'], pos_neg_lists):
                    norm_label = 'positive' if label == 'category_1' else 'negative'
                    for idx, action_program in enumerate(group):
                        img_dir = os.path.join(base_dir, f"ShapeBongard_V2/{cat}/images/{problem_id}", label)
                        img_path = os.path.join(img_dir, f"{idx}.png")

                        flat_commands = [cmd for cmd in flatten_action_program(action_program) if isinstance(cmd, str)]
                        try:
                            # Extract multiple shapes from the action program
                            objects, fallback_reasons = extract_shapes_from_action_program(flat_commands)
                            if not objects:
                                flagged_cases.append({'problem_id': problem_id, 'image_path': img_path, 'error': 'No objects extracted from LOGO commands', 'fallback_reasons': fallback_reasons})
                                logging.warning(f"No objects extracted for problem_id={problem_id}, image={img_path}, fallback_reasons={fallback_reasons}")
                                continue
                            # Collect all unique labels and shape types for traceability
                            for obj in objects:
                                new_label = obj.get('label', '')
                                new_shape_type = obj.get('shape_type', '')
                                new_category = obj.get('category', '')
                                if new_label not in all_labels:
                                    logging.info(f"New label found: {new_label} (problem_id={problem_id}, image={img_path})")
                                if new_shape_type not in all_shape_types:
                                    logging.info(f"New shape_type found: {new_shape_type} (problem_id={problem_id}, image={img_path})")
                                if new_category not in all_categories:
                                    logging.info(f"New category found: {new_category} (problem_id={problem_id}, image={img_path})")
                                all_labels.add(new_label)
                                all_shape_types.add(new_shape_type)
                                all_categories.add(new_category)
                            # Add a summary of all object-level attributes for this image
                            image_record = {
                                'problem_id': problem_id,
                                'category': cat,
                                'label': norm_label,
                                'image_path': img_path,
                                'objects': objects,
                                'action_program': flat_commands,
                                'num_objects': len(objects),
                                'object_labels': list(set(obj.get('label', '') for obj in objects)),
                                'object_shape_types': list(set(obj.get('shape_type', '') for obj in objects)),
                                'object_categories': list(set(obj.get('category', '') for obj in objects)),
                            }
                            logging.info(f"Processed image: {img_path} | problem_id={problem_id} | num_objects={len(objects)} | labels={image_record['object_labels']} | shape_types={image_record['object_shape_types']}")
                            all_results.append(image_record)
                        except Exception as e:
                            flagged_cases.append({'problem_id': problem_id, 'image_path': img_path, 'error': str(e)})
                            logging.error(f"Exception for problem_id={problem_id}, image={img_path}: {e}")

    # Write output with a top-level summary for traceability
    output_data = {
        'summary': {
            'unique_labels': sorted(list(all_labels)),
            'unique_shape_types': sorted(list(all_shape_types)),
            'unique_categories': sorted(list(all_categories)),
            'num_samples': len(all_results),
        },
        'samples': all_results
    }
    with open(output_path, 'w') as out:
        json.dump(output_data, out, indent=2)
    print(f"INFO: Saved {len(all_results)} valid samples to {output_path} (skipped {len(flagged_cases)} of {len(all_results) + len(flagged_cases)})")

    flagged_path = os.path.join(os.path.dirname(output_path), 'flagged_cases.txt')
    with open(flagged_path, 'w') as out:
        for case in flagged_cases:
            out.write(json.dumps(case) + '\n')
    print(f"INFO: Flagged {len(flagged_cases)} cases for review in {flagged_path}")


if __name__ == '__main__':
    main()