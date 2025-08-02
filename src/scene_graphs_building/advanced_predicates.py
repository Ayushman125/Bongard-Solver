import numpy as np
from shapely.geometry import Polygon, LineString

def intersects(node_a, node_b):
    """Checks if the geometries of two nodes intersect without merely touching."""
    try:
        geom_a = Polygon(node_a['vertices']) if node_a.get('is_closed') else LineString(node_a['vertices'])
        geom_b = Polygon(node_b['vertices']) if node_b.get('is_closed') else LineString(node_b['vertices'])
        return geom_a.intersects(geom_b) and not geom_a.touches(geom_b)
    except Exception:
        return False

def contains(node_a, node_b):
    """Checks if the geometry of node_a properly contains node_b."""
    try:
        # Only polygons can contain other objects
        if not node_a.get('is_closed'):
            return False
        geom_a = Polygon(node_a['vertices'])
        geom_b = Polygon(node_b['vertices']) if node_b.get('is_closed') else LineString(node_b['vertices'])
        return geom_a.contains(geom_b)
    except Exception:
        return False

def is_above(node_a, node_b, tol=1e-5):
    """Checks if node_a is predominantly above node_b based on centroid y-coordinates."""
    try:
        centroid_a = np.array(node_a['centroid'])
        centroid_b = np.array(node_b['centroid'])
        # In image coordinates, smaller y is higher
        return centroid_a[1] < centroid_b[1] - tol
    except (KeyError, IndexError):
        return False

def is_parallel(node_a, node_b, tol=7.0):
    """Checks if two line-like objects are parallel within a tolerance."""
    try:
        # Parallelism is most meaningful for non-polygon shapes
        if node_a.get('object_type') == 'polygon' or node_b.get('object_type') == 'polygon':
            return False
        
        orient_a = node_a.get('orientation', 0)
        orient_b = node_b.get('orientation', 0)
        
        angle_diff = abs(orient_a - orient_b)
        return min(angle_diff, 180 - angle_diff) < tol
    except KeyError:
        return False

def same_shape_class(node_a, node_b):
    """Checks if two objects belong to the same abstract shape class based on geometric properties."""
    try:
        # Compare object types
        if node_a.get('object_type') != node_b.get('object_type'):
            return False
        
        # For polygons, compare aspect ratio and compactness
        if node_a.get('object_type') == 'polygon':
            ar_a = node_a.get('aspect_ratio', 1.0)
            ar_b = node_b.get('aspect_ratio', 1.0)
            comp_a = node_a.get('compactness', 0.0)
            comp_b = node_b.get('compactness', 0.0)
            
            ar_similar = abs(ar_a - ar_b) < 0.3
            comp_similar = abs(comp_a - comp_b) < 0.2
            
            return ar_similar and comp_similar
        
        # For lines, compare length ratio
        if node_a.get('object_type') == 'line':
            len_a = node_a.get('length', 0)
            len_b = node_b.get('length', 0)
            if len_a > 0 and len_b > 0:
                ratio = max(len_a, len_b) / min(len_a, len_b)
                return ratio < 2.0
        
        return True
    except Exception:
        return False

def forms_symmetry(node_a, node_b):
    """Checks if two objects form a symmetrical relationship."""
    try:
        centroid_a = np.array(node_a['centroid'])
        centroid_b = np.array(node_b['centroid'])
        
        # Calculate midpoint
        midpoint = (centroid_a + centroid_b) / 2
        
        # Check if objects are equidistant from midpoint
        dist_a = np.linalg.norm(centroid_a - midpoint)
        dist_b = np.linalg.norm(centroid_b - midpoint)
        
        return abs(dist_a - dist_b) < 5.0  # Tolerance for symmetry
    except Exception:
        return False

def similar_size(node_a, node_b, tol=0.5):
    """Checks if two objects have similar sizes based on area or length."""
    try:
        if node_a.get('object_type') == 'polygon' and node_b.get('object_type') == 'polygon':
            area_a = node_a.get('area', 0)
            area_b = node_b.get('area', 0)
            if area_a > 0 and area_b > 0:
                ratio = max(area_a, area_b) / min(area_a, area_b)
                return ratio < (1 + tol)
        else:
            # For lines and curves, compare lengths
            len_a = node_a.get('length', 0)
            len_b = node_b.get('length', 0)
            if len_a > 0 and len_b > 0:
                ratio = max(len_a, len_b) / min(len_a, len_b)
                return ratio < (1 + tol)
        
        return False
    except Exception:
        return False

def near_objects(node_a, node_b, tol=50.0):
    """Checks if two objects are spatially close to each other."""
    try:
        centroid_a = np.array(node_a['centroid'])
        centroid_b = np.array(node_b['centroid'])
        distance = np.linalg.norm(centroid_a - centroid_b)
        return distance < tol
    except Exception:
        return False

def same_orientation(node_a, node_b, tol=10.0):
    """Checks if two objects have similar orientations."""
    try:
        orient_a = node_a.get('orientation', 0)
        orient_b = node_b.get('orientation', 0)
        angle_diff = abs(orient_a - orient_b)
        return min(angle_diff, 180 - angle_diff) < tol
    except Exception:
        return False

def same_stroke_count(node_a, node_b):
    """Checks if two composite shapes have the same number of component strokes."""
    try:
        count_a = node_a.get('stroke_count', 0)
        count_b = node_b.get('stroke_count', 0)
        return count_a > 0 and count_a == count_b
    except Exception:
        return False

# === HIGH-LEVEL SEMANTIC PREDICATES FOR BONGARD PROBLEMS ===

def has_apex_at_left(node_a, node_b):
    """Check if a shape has its apex (topmost/bottommost point) at the left side"""
    try:
        verts_a = node_a.get('vertices', [])
        if len(verts_a) < 3:
            return False
        
        # Find extremal points
        y_coords = [v[1] for v in verts_a]
        x_coords = [v[0] for v in verts_a]
        
        # Find apex (highest point)
        max_y_idx = np.argmax(y_coords)
        min_y_idx = np.argmin(y_coords)
        
        # Use the more extreme point as apex
        apex_idx = max_y_idx if abs(y_coords[max_y_idx]) > abs(y_coords[min_y_idx]) else min_y_idx
        apex_x = x_coords[apex_idx]
        
        # Check if apex is on the left side of the centroid
        centroid = node_a.get('centroid', [0, 0])
        return apex_x < centroid[0]
    except Exception:
        return False

def has_asymmetric_base(node_a, node_b):
    """Check if a shape has an asymmetric base (unequal left/right extensions)"""
    try:
        verts = node_a.get('vertices', [])
        if len(verts) < 3:
            return False
        
        centroid = node_a.get('centroid', [0, 0])
        
        # Find leftmost and rightmost points
        x_coords = [v[0] for v in verts]
        left_extent = centroid[0] - min(x_coords)
        right_extent = max(x_coords) - centroid[0]
        
        # Check for significant asymmetry (>20% difference)
        if max(left_extent, right_extent) > 0:
            asymmetry_ratio = abs(left_extent - right_extent) / max(left_extent, right_extent)
            return asymmetry_ratio > 0.2
        return False
    except Exception:
        return False

def has_tilted_orientation(node_a, node_b):
    """Check if a shape is significantly tilted from horizontal/vertical"""
    try:
        # Use the dominant orientation from the longest stroke
        orientation = node_a.get('orientation', 0.0)
        
        # Check if orientation is significantly away from cardinal directions
        # Normalize to [0, 360)
        norm_orient = orientation % 360
        
        # Check distance from cardinal directions (0, 90, 180, 270)
        cardinal_distances = [
            min(abs(norm_orient - 0), abs(norm_orient - 360)),
            abs(norm_orient - 90),
            abs(norm_orient - 180),
            abs(norm_orient - 270)
        ]
        
        min_distance = min(cardinal_distances)
        return min_distance > 15.0  # More than 15 degrees from cardinal
    except Exception:
        return False

def has_length_ratio_imbalance(node_a, node_b):
    """Check if shape has significant length imbalance between parts"""
    try:
        aspect_ratio = node_a.get('aspect_ratio', 1.0)
        
        # Check for significant deviation from square aspect ratio
        if aspect_ratio > 0:
            imbalance = max(aspect_ratio, 1.0/aspect_ratio)
            return imbalance > 1.5  # 50% or more difference
        return False
    except Exception:
        return False

def forms_open_vs_closed_distinction(node_a, node_b):
    """Check if one shape is open and another is closed"""
    try:
        is_closed_a = node_a.get('is_closed', False)
        is_closed_b = node_b.get('is_closed', False)
        return is_closed_a != is_closed_b
    except Exception:
        return False

def has_geometric_complexity_difference(node_a, node_b):
    """Check if shapes have different geometric complexity (stroke count difference)"""
    try:
        stroke_count_a = node_a.get('stroke_count', len(node_a.get('vertices', [])))
        stroke_count_b = node_b.get('stroke_count', len(node_b.get('vertices', [])))
        
        # Significant complexity difference (>2 strokes)
        return abs(stroke_count_a - stroke_count_b) > 2
    except Exception:
        return False

def has_compactness_difference(node_a, node_b):
    """Check if shapes have significantly different compactness"""
    try:
        comp_a = node_a.get('compactness', 0.0)
        comp_b = node_b.get('compactness', 0.0)
        
        if max(comp_a, comp_b) > 0:
            diff_ratio = abs(comp_a - comp_b) / max(comp_a, comp_b)
            return diff_ratio > 0.3  # 30% difference in compactness
        return False
    except Exception:
        return False

def exhibits_mirror_asymmetry(node_a, node_b):
    """Check if a shape exhibits mirror asymmetry along vertical axis"""
    try:
        verts = node_a.get('vertices', [])
        if len(verts) < 3:
            return False
        
        centroid = node_a.get('centroid', [0, 0])
        
        # Mirror points across vertical line through centroid
        mirrored_verts = []
        for v in verts:
            mirrored_x = 2 * centroid[0] - v[0]
            mirrored_verts.append([mirrored_x, v[1]])
        
        # Check if mirrored shape significantly differs from original
        # Use simple distance-based comparison
        min_distances = []
        for mv in mirrored_verts:
            distances = [np.linalg.norm(np.array(mv) - np.array(ov)) for ov in verts]
            min_distances.append(min(distances))
        
        avg_min_distance = np.mean(min_distances)
        shape_size = np.linalg.norm([max(v[0] for v in verts) - min(v[0] for v in verts),
                                    max(v[1] for v in verts) - min(v[1] for v in verts)])
        
        # Asymmetry if average distance is >10% of shape size
        return shape_size > 0 and (avg_min_distance / shape_size) > 0.1
    except Exception:
        return False

def has_dominant_direction(node_a, node_b):
    """Check if shape has a dominant direction (vertical vs horizontal extent)"""
    try:
        bbox = node_a.get('bounding_box')
        if not bbox or len(bbox) < 4:
            return False
        
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if max(width, height) > 0:
            dominance_ratio = max(width, height) / min(width, height)
            return dominance_ratio > 2.0  # One dimension is 2x the other
        return False
    except Exception:
        return False

# === HIGH-LEVEL SEMANTIC PREDICATES FOR BONGARD PROBLEMS ===

def has_apex_at_left(node_a, node_b):
    """Checks if one shape has an apex positioned to the left relative to its center"""
    try:
        for node in [node_a, node_b]:
            if node.get('apex_relative_to_center') == 'left':
                return True
        return False
    except Exception:
        return False

def has_asymmetric_base(node_a, node_b):
    """Checks if shapes have asymmetric base widths or irregular base structure"""
    try:
        for node in [node_a, node_b]:
            left_extent = node.get('left_extent', 0)
            right_extent = node.get('right_extent', 0)
            if max(left_extent, right_extent) > 0:
                asymmetry = abs(left_extent - right_extent) / max(left_extent, right_extent)
                if asymmetry > 0.3:  # 30% asymmetry threshold
                    return True
        return False
    except Exception:
        return False

def has_tilted_orientation(node_a, node_b):
    """Checks if shapes have tilted orientation (not aligned with cardinal directions)"""
    try:
        for node in [node_a, node_b]:
            orientation = node.get('orientation', 0)
            # Check if orientation is significantly off from 0, 90, 180, 270 degrees
            normalized_angle = orientation % 90
            tilt_amount = min(normalized_angle, 90 - normalized_angle)
            if tilt_amount > 15:  # More than 15 degrees off cardinal direction
                return True
        return False
    except Exception:
        return False

def has_length_ratio_imbalance(node_a, node_b):
    """Checks if there's a significant imbalance in dimensional ratios"""
    try:
        ratio_a = node_a.get('aspect_ratio', 1.0)
        ratio_b = node_b.get('aspect_ratio', 1.0)
        
        if max(ratio_a, ratio_b) > 0:
            imbalance = abs(ratio_a - ratio_b) / max(ratio_a, ratio_b)
            return imbalance > 0.5  # 50% difference in aspect ratios
        return False
    except Exception:
        return False

def forms_open_vs_closed_distinction(node_a, node_b):
    """Checks if one shape is open while another is closed"""
    try:
        closed_a = node_a.get('is_closed', False)
        closed_b = node_b.get('is_closed', False)
        return closed_a != closed_b  # Different closure states
    except Exception:
        return False

def has_geometric_complexity_difference(node_a, node_b):
    """Checks for differences in geometric complexity (vertex count, curvature)"""
    try:
        complexity_a = node_a.get('geometric_complexity', 0)
        complexity_b = node_b.get('geometric_complexity', 0)
        
        if max(complexity_a, complexity_b) > 0:
            diff_ratio = abs(complexity_a - complexity_b) / max(complexity_a, complexity_b)
            return diff_ratio > 0.4  # 40% complexity difference
        return False
    except Exception:
        return False

def has_compactness_difference(node_a, node_b):
    """Checks for significant differences in shape compactness"""
    try:
        comp_a = node_a.get('compactness', 0)
        comp_b = node_b.get('compactness', 0)
        
        if max(comp_a, comp_b) > 0:
            diff_ratio = abs(comp_a - comp_b) / max(comp_a, comp_b)
            return diff_ratio > 0.3  # 30% compactness difference
        return False
    except Exception:
        return False

def exhibits_mirror_asymmetry(node_a, node_b):
    """Checks if shapes exhibit mirror asymmetry patterns"""
    try:
        h_asym_a = node_a.get('horizontal_asymmetry', 0)
        h_asym_b = node_b.get('horizontal_asymmetry', 0)
        v_asym_a = node_a.get('vertical_asymmetry', 0)
        v_asym_b = node_b.get('vertical_asymmetry', 0)
        
        # Check if one has high asymmetry while the other doesn't
        total_asym_a = h_asym_a + v_asym_a
        total_asym_b = h_asym_b + v_asym_b
        
        return abs(total_asym_a - total_asym_b) > 0.4  # Significant asymmetry difference
    except Exception:
        return False

def forms_bridge_pattern(node_a, node_b):
    """Checks if nodes form a bridge-like pattern"""
    try:
        # Look for parallel structures with connecting elements
        if is_parallel(node_a, node_b):
            # Check if there are other nodes that could connect them
            return True
        return False
    except Exception:
        return False

def has_curvature_distinction(node_a, node_b):
    """Checks for curved vs straight distinctions"""
    try:
        # Determine curvature based on object type and compactness
        curved_a = node_a.get('object_type') in ['arc', 'curve'] or node_a.get('compactness', 0) > 0.7
        curved_b = node_b.get('object_type') in ['arc', 'curve'] or node_b.get('compactness', 0) > 0.7
        return curved_a != curved_b  # One curved, one straight
    except Exception:
        return False

def is_arc_of_circle(node_a, node_b):
    """Checks if a shape is a circular arc"""
    try:
        # High compactness and curvature indicators suggest circular arcs
        compactness = node_a.get('compactness', 0)
        curvature_score = node_a.get('curvature_score', 0)
        
        return compactness > 0.6 and curvature_score > 0.5
    except Exception:
        return False

def shares_endpoint(node_a, node_b):
    """Checks if two line segments share an endpoint (T-junction, L-junction, etc.)"""
    try:
        endpoints_a = node_a.get('endpoints', [])
        endpoints_b = node_b.get('endpoints', [])
        
        if not endpoints_a or not endpoints_b:
            return False
        
        # Check if any endpoint from A is close to any endpoint from B
        tolerance = 5.0  # Pixel tolerance for endpoint matching
        for ep_a in endpoints_a:
            for ep_b in endpoints_b:
                if ep_a is not None and ep_b is not None:
                    distance = np.linalg.norm(np.array(ep_a) - np.array(ep_b))
                    if distance < tolerance:
                        return True
        return False
    except Exception:
        return False

def forms_apex(node_a, node_b):
    """Checks if nodes form an apex structure (meeting at a point)"""
    try:
        # Both nodes should share an endpoint and have diverging orientations
        if not shares_endpoint(node_a, node_b):
            return False
        
        # Check if orientations diverge (angle between them > 30 degrees)
        orient_a = node_a.get('orientation', 0)
        orient_b = node_b.get('orientation', 0)
        
        angle_diff = abs(orient_a - orient_b)
        angle_diff = min(angle_diff, 180 - angle_diff)  # Normalize to [0, 90]
        
        return angle_diff > 30  # Significant angular difference
    except Exception:
        return False

def forms_t_junction(node_a, node_b):
    """Checks if nodes form a T-junction pattern"""
    try:
        if not shares_endpoint(node_a, node_b):
            return False
        
        # T-junction: one line perpendicular to another
        orient_a = node_a.get('orientation', 0)
        orient_b = node_b.get('orientation', 0)
        
        angle_diff = abs(orient_a - orient_b)
        angle_diff = min(angle_diff, 180 - angle_diff)
        
        # Near perpendicular (90 ± 15 degrees)
        return abs(angle_diff - 90) < 15
    except Exception:
        return False

def forms_x_junction(node_a, node_b):
    """Checks if nodes intersect to form an X-junction (crossing)"""
    try:
        # Must intersect (not just touch at endpoints)
        if not intersects(node_a, node_b):
            return False
        
        # Should not share endpoints (that would be apex or T-junction)
        if shares_endpoint(node_a, node_b):
            return False
        
        # Orientations should be significantly different
        orient_a = node_a.get('orientation', 0)
        orient_b = node_b.get('orientation', 0)
        
        angle_diff = abs(orient_a - orient_b)
        angle_diff = min(angle_diff, 180 - angle_diff)
        
        return angle_diff > 20  # Significant crossing angle
    except Exception:
        return False

def forms_bridge_arc(node_a, node_b):
    """Checks if nodes form a bridge-like arc structure"""
    try:
        # Look for curved connection between parallel elements
        if is_parallel(node_a, node_b):
            return False  # Parallel elements themselves, not the bridge
        
        # Check if one is curved and connects others
        curved_a = node_a.get('object_type') in ['arc', 'curve'] or node_a.get('compactness', 0) > 0.5
        curved_b = node_b.get('object_type') in ['arc', 'curve'] or node_b.get('compactness', 0) > 0.5
        
        # Bridge arc should be curved and positioned "above" straight elements
        if curved_a:
            return node_a.get('centroid', [0, 0])[1] < node_b.get('centroid', [0, 0])[1]
        elif curved_b:
            return node_b.get('centroid', [0, 0])[1] < node_a.get('centroid', [0, 0])[1]
        
        return False
    except Exception:
        return False

def has_irregular_shape(node_a, node_b):
    """Checks if shape has irregular, non-geometric form"""
    try:
        # High vertex count combined with low compactness suggests irregularity
        vertex_count = len(node_a.get('vertices', []))
        compactness = node_a.get('compactness', 0)
        
        # Many vertices but not compact = irregular
        return vertex_count > 8 and compactness < 0.3
    except Exception:
        return False

def exhibits_rotational_symmetry(node_a, node_b):
    """Checks if shape exhibits rotational symmetry"""
    try:
        # Use orientation variance as indicator
        orientation_variance = node_a.get('orientation_variance', 0)
        compactness = node_a.get('compactness', 0)
        
        # High compactness with low orientation variance suggests rotational symmetry
        return compactness > 0.7 and orientation_variance < 0.2
    except Exception:
        return False

# === STATE-OF-THE-ART: PROGRAM SYNTHESIS PREDICATES ===

def has_stroke_count_pattern(node_a, node_b, target_count=None):
    """Checks if objects have specific stroke count patterns (e.g., 'has_four_lines', 'has_six_lines')"""
    try:
        count_a = node_a.get('stroke_count', len(node_a.get('vertices', [])))
        count_b = node_b.get('stroke_count', len(node_b.get('vertices', [])))
        
        if target_count is not None:
            return count_a == target_count or count_b == target_count
        
        # Check for distinctive count patterns (4 vs 6 is common in BONGARD-LOGO)
        distinctive_counts = [3, 4, 5, 6, 8]
        return count_a in distinctive_counts and count_b in distinctive_counts and count_a != count_b
    except Exception:
        return False

def has_convexity_distinction(node_a, node_b):
    """Checks for convex vs non-convex distinction using convex hull analysis"""
    try:
        from scipy.spatial import ConvexHull
        
        def is_convex(vertices):
            if len(vertices) < 4:
                return True
            try:
                hull = ConvexHull(vertices)
                return len(hull.vertices) == len(vertices)
            except:
                return False
        
        convex_a = is_convex(node_a.get('vertices', []))
        convex_b = is_convex(node_b.get('vertices', []))
        
        return convex_a != convex_b
    except Exception:
        return False

def has_symmetry_axis(node_a, node_b):
    """Checks for bilateral symmetry along vertical or horizontal axis"""
    try:
        def has_bilateral_symmetry(vertices, axis='vertical'):
            if len(vertices) < 3:
                return False
            
            vertices = np.array(vertices)
            center = np.mean(vertices, axis=0)
            
            if axis == 'vertical':
                # Mirror across vertical line through center
                mirrored = vertices.copy()
                mirrored[:, 0] = 2 * center[0] - mirrored[:, 0]
            else:  # horizontal
                mirrored = vertices.copy()
                mirrored[:, 1] = 2 * center[1] - mirrored[:, 1]
            
            # Check if mirrored points match original (within tolerance)
            tolerance = 3.0
            for mp in mirrored:
                min_dist = min(np.linalg.norm(mp - v) for v in vertices)
                if min_dist > tolerance:
                    return False
            return True
        
        sym_a_v = has_bilateral_symmetry(node_a.get('vertices', []), 'vertical')
        sym_a_h = has_bilateral_symmetry(node_a.get('vertices', []), 'horizontal')
        sym_b_v = has_bilateral_symmetry(node_b.get('vertices', []), 'vertical')
        sym_b_h = has_bilateral_symmetry(node_b.get('vertices', []), 'horizontal')
        
        # Return True if one has symmetry and the other doesn't
        return (sym_a_v or sym_a_h) != (sym_b_v or sym_b_h)
    except Exception:
        return False

def has_hole_distinction(node_a, node_b):
    """Checks for shapes with holes vs solid shapes"""
    try:
        # Approximate hole detection: check if there are disconnected components
        def has_hole(vertices):
            if len(vertices) < 6:  # Need sufficient complexity for holes
                return False
            
            # Simple hole heuristic: if compactness is very low despite being closed
            is_closed = len(vertices) > 2 and np.allclose(vertices[0], vertices[-1], atol=1e-5)
            if not is_closed:
                return False
            
            # Calculate area vs perimeter ratio
            try:
                from shapely.geometry import Polygon
                poly = Polygon(vertices)
                compactness = (4 * np.pi * poly.area) / (poly.length ** 2)
                return compactness < 0.3  # Very low compactness suggests holes
            except:
                return False
        
        hole_a = has_hole(node_a.get('vertices', []))
        hole_b = has_hole(node_b.get('vertices', []))
        
        return hole_a != hole_b
    except Exception:
        return False

def has_angle_count_pattern(node_a, node_b):
    """Checks for specific angle count patterns (triangle vs square vs pentagon)"""
    try:
        def count_significant_angles(vertices):
            if len(vertices) < 3:
                return 0
            
            angles = []
            n = len(vertices)
            for i in range(n):
                v1 = np.array(vertices[i])
                v2 = np.array(vertices[(i + 1) % n])
                v3 = np.array(vertices[(i + 2) % n])
                
                # Calculate angle at v2
                vec1 = v1 - v2
                vec2 = v3 - v2
                
                if np.linalg.norm(vec1) > 1e-6 and np.linalg.norm(vec2) > 1e-6:
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.degrees(np.arccos(cos_angle))
                    
                    # Count significant angles (not near 180 degrees)
                    if abs(angle - 180) > 30:  # Significant turn
                        angles.append(angle)
            
            return len(angles)
        
        angles_a = count_significant_angles(node_a.get('vertices', []))
        angles_b = count_significant_angles(node_b.get('vertices', []))
        
        # Common patterns: 3 (triangle), 4 (square), 5 (pentagon), 6 (hexagon)
        return angles_a != angles_b and min(angles_a, angles_b) >= 3
    except Exception:
        return False

def has_intersection_count_pattern(node_a, node_b):
    """Checks for patterns based on self-intersections or line crossings"""
    try:
        def count_self_intersections(vertices):
            if len(vertices) < 4:
                return 0
            
            intersections = 0
            n = len(vertices)
            
            for i in range(n - 1):
                line1 = (vertices[i], vertices[i + 1])
                for j in range(i + 2, n - 1):
                    line2 = (vertices[j], vertices[j + 1])
                    
                    # Skip adjacent segments
                    if abs(i - j) <= 1 or (i == 0 and j == n - 2):
                        continue
                    
                    # Check intersection using cross product method
                    def lines_intersect(p1, p2, p3, p4):
                        def ccw(A, B, C):
                            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
                        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
                    
                    if lines_intersect(line1[0], line1[1], line2[0], line2[1]):
                        intersections += 1
            
            return intersections
        
        intersections_a = count_self_intersections(node_a.get('vertices', []))
        intersections_b = count_self_intersections(node_b.get('vertices', []))
        
        return intersections_a != intersections_b
    except Exception:
        return False

# === STATE-OF-THE-ART: CONTRASTIVE ANALOGICAL PREDICATES ===

def forms_action_sequence_analogy(node_a, node_b):
    """Checks if objects are analogous through their action program sequence patterns"""
    try:
        program_a = node_a.get('action_program', [])
        program_b = node_b.get('action_program', [])
        
        if not program_a or not program_b:
            return False
        
        # Extract command types
        def extract_command_pattern(program):
            pattern = []
            for cmd in program:
                if isinstance(cmd, str) and '_' in cmd:
                    cmd_type = cmd.split('_')[0]
                    pattern.append(cmd_type)
            return pattern
        
        pattern_a = extract_command_pattern(program_a)
        pattern_b = extract_command_pattern(program_b)
        
        # Check for analogous patterns (e.g., line->arc vs arc->line)
        if len(pattern_a) == len(pattern_b):
            # Exact analogy: same sequence
            if pattern_a == pattern_b:
                return True
            
            # Mirror analogy: reversed sequence
            if pattern_a == pattern_b[::-1]:
                return True
            
            # Substitution analogy: one type replaced by another
            diff_count = sum(1 for a, b in zip(pattern_a, pattern_b) if a != b)
            if diff_count == 1:  # Single substitution
                return True
        
        return False
    except Exception:
        return False

def has_scaling_invariant_property(node_a, node_b):
    """Checks for properties that remain invariant under scaling (shape ratios, angles)"""
    try:
        # Compare normalized shape properties
        def get_normalized_properties(node):
            area = node.get('area', 0)
            perimeter = node.get('perimeter', 0)
            bbox = node.get('bounding_box', [0, 0, 0, 0])
            
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                size_measure = max(width, height)
            else:
                size_measure = 1.0
            
            # Normalize by size
            if size_measure > 0 and perimeter > 0:
                normalized_area = area / (size_measure ** 2)
                normalized_perimeter = perimeter / size_measure
                aspect_ratio = node.get('aspect_ratio', 1.0)
                
                return {
                    'norm_area': normalized_area,
                    'norm_perimeter': normalized_perimeter,
                    'aspect_ratio': aspect_ratio
                }
            return None
        
        props_a = get_normalized_properties(node_a)
        props_b = get_normalized_properties(node_b)
        
        if props_a and props_b:
            # Check if normalized properties are similar (indicating same shape, different scale)
            area_sim = abs(props_a['norm_area'] - props_b['norm_area']) < 0.1
            perimeter_sim = abs(props_a['norm_perimeter'] - props_b['norm_perimeter']) < 0.1
            aspect_sim = abs(props_a['aspect_ratio'] - props_b['aspect_ratio']) < 0.2
            
            return area_sim and perimeter_sim and aspect_sim
        
        return False
    except Exception:
        return False

def exhibits_topological_invariance(node_a, node_b):
    """Checks for topological invariants (connectivity, genus) that persist under continuous deformation"""
    try:
        # Compare topological properties
        def get_topology_signature(node):
            vertices = node.get('vertices', [])
            is_closed = node.get('is_closed', False)
            
            # Basic topology: closed vs open
            closure_type = 'closed' if is_closed else 'open'
            
            # Connectivity: count of disconnected components (simplified)
            # For now, assume single component, but this could be extended
            component_count = 1
            
            # Junction count (approximate genus)
            junction_count = 0
            if len(vertices) > 4:
                # Simplified junction detection based on sharp turns
                for i in range(1, len(vertices) - 1):
                    v1 = np.array(vertices[i - 1])
                    v2 = np.array(vertices[i])
                    v3 = np.array(vertices[i + 1])
                    
                    vec1 = v2 - v1
                    vec2 = v3 - v2
                    
                    if np.linalg.norm(vec1) > 1e-6 and np.linalg.norm(vec2) > 1e-6:
                        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                        
                        if abs(angle - 180) > 60:  # Sharp turn indicates junction
                            junction_count += 1
            
            return (closure_type, component_count, junction_count)
        
        topo_a = get_topology_signature(node_a)
        topo_b = get_topology_signature(node_b)
        
        return topo_a == topo_b
    except Exception:
        return False

# === STATE-OF-THE-ART: CONTEXT-DEPENDENT PREDICATES ===

def has_context_dependent_interpretation(node_a, node_b, context_nodes=None):
    """Checks for predicates that depend on surrounding context (e.g., 'inside', 'between')"""
    try:
        if not context_nodes:
            return False
        
        # Example: "between" predicate that depends on three or more objects
        centroid_a = np.array(node_a.get('centroid', [0, 0]))
        centroid_b = np.array(node_b.get('centroid', [0, 0]))
        
        for context_node in context_nodes:
            if context_node['object_id'] != node_a['object_id'] and context_node['object_id'] != node_b['object_id']:
                centroid_c = np.array(context_node.get('centroid', [0, 0]))
                
                # Check if one object is between the other two
                dist_ac = np.linalg.norm(centroid_a - centroid_c)
                dist_bc = np.linalg.norm(centroid_b - centroid_c)
                dist_ab = np.linalg.norm(centroid_a - centroid_b)
                
                # A is between B and C if dist(B,C) ≈ dist(B,A) + dist(A,C)
                if abs(dist_bc - (dist_ab + dist_ac)) < 5.0:
                    return True
                
                # B is between A and C
                if abs(dist_ac - (dist_ab + dist_bc)) < 5.0:
                    return True
        
        return False
    except Exception:
        return False

def forms_grouping_gestalt(node_a, node_b, all_nodes=None):
    """Checks for Gestalt grouping principles (proximity, similarity, continuity)"""
    try:
        if not all_nodes:
            return False
        
        # Proximity grouping
        centroid_a = np.array(node_a.get('centroid', [0, 0]))
        centroid_b = np.array(node_b.get('centroid', [0, 0]))
        dist_ab = np.linalg.norm(centroid_a - centroid_b)
        
        # Check if A and B are closer to each other than to any other object
        closer_to_each_other = True
        for other_node in all_nodes:
            if other_node['object_id'] not in [node_a['object_id'], node_b['object_id']]:
                centroid_other = np.array(other_node.get('centroid', [0, 0]))
                dist_a_other = np.linalg.norm(centroid_a - centroid_other)
                dist_b_other = np.linalg.norm(centroid_b - centroid_other)
                
                if dist_a_other < dist_ab or dist_b_other < dist_ab:
                    closer_to_each_other = False
                    break
        
        if closer_to_each_other:
            return True
        
        # Similarity grouping
        type_a = node_a.get('object_type', '')
        type_b = node_b.get('object_type', '')
        size_a = node_a.get('area', 0) if node_a.get('area', 0) > 0 else node_a.get('length', 0)
        size_b = node_b.get('area', 0) if node_b.get('area', 0) > 0 else node_b.get('length', 0)
        
        type_similar = type_a == type_b
        size_similar = abs(size_a - size_b) / max(size_a, size_b, 1e-6) < 0.3
        
        return type_similar and size_similar
    except Exception:
        return False

# === STATE-OF-THE-ART: PROGRAM-SEMANTIC ALIGNMENT PREDICATES ===

def aligns_with_logo_semantics(node_a, node_b):
    """Checks if spatial relationships align with LOGO program semantics"""
    try:
        # Check if action sequence semantics match spatial arrangement
        program_a = node_a.get('action_program', [])
        program_b = node_b.get('action_program', [])
        
        if not program_a or not program_b:
            return False
        
        # Extract movement semantics
        def extract_movement_semantics(program):
            movements = []
            for cmd in program:
                if isinstance(cmd, str):
                    if 'forward' in cmd or 'line' in cmd:
                        movements.append('linear')
                    elif 'turn' in cmd or 'arc' in cmd:
                        movements.append('angular')
                    elif 'start' in cmd:
                        movements.append('start')
            return movements
        
        semantics_a = extract_movement_semantics(program_a)
        semantics_b = extract_movement_semantics(program_b)
        
        # Check if spatial relationship aligns with program semantics
        if 'angular' in semantics_a and 'linear' in semantics_b:
            # If A has angular movements and B has linear, check if they form junction
            return shares_endpoint(node_a, node_b)
        
        if semantics_a == semantics_b:
            # Similar movement patterns should result in similar shapes
            return same_shape_class(node_a, node_b)
        
        return False
    except Exception:
        return False

def exhibits_rule_compositionality(node_a, node_b, rule_context=None):
    """Checks if objects follow compositional rules that can be learned from examples"""
    try:
        if not rule_context:
            return False
        
        # Example: if rule is "triangles are always red and squares are always blue"
        # This would check color-shape associations
        
        # For now, implement a simple compositional rule:
        # "closed shapes with N sides have property P"
        
        def get_compositional_features(node):
            is_closed = node.get('is_closed', False)
            side_count = len(node.get('vertices', [])) - 1 if is_closed else len(node.get('vertices', []))
            size_category = 'large' if node.get('area', 0) > 100 else 'small'
            orientation_category = 'tilted' if abs(node.get('orientation', 0) % 90) > 15 else 'cardinal'
            
            return (is_closed, side_count, size_category, orientation_category)
        
        features_a = get_compositional_features(node_a)
        features_b = get_compositional_features(node_b)
        
        # Check if they follow the same compositional pattern
        return features_a[0] == features_b[0] and features_a[1] == features_b[1]  # Same closure and side count
    except Exception:
        return False

# BONGARD-SPECIFIC SEMANTIC PREDICATES
def semantic_contains_triangle(node_a, node_b):
    """Checks if the node semantically contains triangular elements."""
    return node_a.get('semantic_features', {}).get('has_triangles', False)

def semantic_contains_square(node_a, node_b):
    """Checks if the node semantically contains square/rectangular elements."""
    return node_a.get('semantic_features', {}).get('has_squares', False)

def semantic_contains_circle(node_a, node_b):
    """Checks if the node semantically contains circular elements."""
    return node_a.get('semantic_features', {}).get('has_circles', False)

def semantic_three_sided(node_a, node_b):
    """Checks if the node represents a three-sided figure."""
    return node_a.get('semantic_features', {}).get('has_three_sides', False)

def semantic_four_sided(node_a, node_b):
    """Checks if the node represents a four-sided figure."""
    return node_a.get('semantic_features', {}).get('has_four_sides', False)

def semantic_has_curves(node_a, node_b):
    """Checks if the node has curved elements."""
    return node_a.get('semantic_features', {}).get('has_curved_elements', False)

def semantic_closed_shape(node_a, node_b):
    """Checks if the node represents a closed shape."""
    return node_a.get('semantic_features', {}).get('is_closed', False)

def semantic_open_shape(node_a, node_b):
    """Checks if the node represents an open shape."""
    return node_a.get('semantic_features', {}).get('is_open', False)

def semantic_simple_shape(node_a, node_b):
    """Checks if the node is a simple (non-composite) shape."""
    return not node_a.get('semantic_features', {}).get('is_composite', True)

def semantic_complex_shape(node_a, node_b):
    """Checks if the node is a complex (composite) shape."""
    return node_a.get('semantic_features', {}).get('is_composite', False)

# Comprehensive Bongard-specific predicate functions that integrate with the semantic action parser
def is_circle(node_a, node_b=None):
    """Checks if shape is a circle based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'circle'
    except:
        return False

def is_triangle(node_a, node_b=None):
    """Checks if shape is a triangle based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'triangle'
    except:
        return False

def is_square(node_a, node_b=None):
    """Checks if shape is a square based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'square'
    except:
        return False

def is_rectangle(node_a, node_b=None):
    """Checks if shape is a rectangle based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'rectangle'
    except:
        return False

def is_pentagon(node_a, node_b=None):
    """Checks if shape is a pentagon based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'pentagon'
    except:
        return False

def is_hexagon(node_a, node_b=None):
    """Checks if shape is a hexagon based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'hexagon'
    except:
        return False

def is_octagon(node_a, node_b=None):
    """Checks if shape is an octagon based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'octagon'
    except:
        return False

def is_line(node_a, node_b=None):
    """Checks if shape is a line based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'line'
    except:
        return False

def is_arc(node_a, node_b=None):
    """Checks if shape is an arc based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'arc'
    except:
        return False

def is_star(node_a, node_b=None):
    """Checks if shape is a star based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'star'
    except:
        return False

def is_cross(node_a, node_b=None):
    """Checks if shape is a cross based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'cross'
    except:
        return False

def is_ellipse(node_a, node_b=None):
    """Checks if shape is an ellipse based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'ellipse'
    except:
        return False

def is_diamond(node_a, node_b=None):
    """Checks if shape is a diamond based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'diamond'
    except:
        return False

def is_arrow(node_a, node_b=None):
    """Checks if shape is an arrow based on comprehensive features."""
    try:
        shape_type = node_a.get('semantic_features', {}).get('primary_shape_type')
        return shape_type == 'arrow'
    except:
        return False

def has_vertical_symmetry(node_a, node_b=None):
    """Checks if shape has vertical symmetry."""
    try:
        symmetry = node_a.get('semantic_features', {}).get('symmetry_type')
        return symmetry in ['vertical', 'bilateral']
    except:
        return False

def has_horizontal_symmetry(node_a, node_b=None):
    """Checks if shape has horizontal symmetry."""
    try:
        symmetry = node_a.get('semantic_features', {}).get('symmetry_type')
        return symmetry in ['horizontal', 'bilateral']
    except:
        return False

def has_rotational_symmetry(node_a, node_b=None):
    """Checks if shape has rotational symmetry."""
    try:
        symmetry = node_a.get('semantic_features', {}).get('symmetry_type')
        return symmetry == 'rotational'
    except:
        return False

def has_point_symmetry(node_a, node_b=None):
    """Checks if shape has point symmetry."""
    try:
        symmetry = node_a.get('semantic_features', {}).get('symmetry_type')
        return symmetry == 'point'
    except:
        return False

def is_convex(node_a, node_b=None):
    """Checks if shape is convex."""
    try:
        topology = node_a.get('semantic_features', {}).get('topology_type')
        return topology == 'convex'
    except:
        return False

def is_concave(node_a, node_b=None):
    """Checks if shape is concave."""
    try:
        topology = node_a.get('semantic_features', {}).get('topology_type')
        return topology == 'concave'
    except:
        return False

def is_closed_shape(node_a, node_b=None):
    """Checks if shape is closed."""
    try:
        topology = node_a.get('semantic_features', {}).get('topology_type')
        return topology in ['closed', 'convex', 'concave']
    except:
        return False

def is_open_shape(node_a, node_b=None):
    """Checks if shape is open."""
    try:
        topology = node_a.get('semantic_features', {}).get('topology_type')
        return topology == 'open'
    except:
        return False

def has_hole(node_a, node_b=None):
    """Checks if shape has a hole."""
    try:
        topology = node_a.get('semantic_features', {}).get('topology_type')
        return topology == 'hole'
    except:
        return False

def has_three_sides(node_a, node_b=None):
    """Checks if shape has three sides."""
    try:
        sides = node_a.get('semantic_features', {}).get('side_count', 0)
        return sides == 3
    except:
        return False

def has_four_sides(node_a, node_b=None):
    """Checks if shape has four sides."""
    try:
        sides = node_a.get('semantic_features', {}).get('side_count', 0)
        return sides == 4
    except:
        return False

def has_five_sides(node_a, node_b=None):
    """Checks if shape has five sides."""
    try:
        sides = node_a.get('semantic_features', {}).get('side_count', 0)
        return sides == 5
    except:
        return False

def has_six_sides(node_a, node_b=None):
    """Checks if shape has six sides."""
    try:
        sides = node_a.get('semantic_features', {}).get('side_count', 0)
        return sides == 6
    except:
        return False

def has_eight_sides(node_a, node_b=None):
    """Checks if shape has eight sides."""
    try:
        sides = node_a.get('semantic_features', {}).get('side_count', 0)
        return sides == 8
    except:
        return False

def is_regular_polygon(node_a, node_b=None):
    """Checks if shape is a regular polygon."""
    try:
        shape_features = node_a.get('semantic_features', {})
        regularity = shape_features.get('regularity_score', 0)
        return regularity > 0.8  # High regularity threshold
    except:
        return False

def is_irregular_polygon(node_a, node_b=None):
    """Checks if shape is an irregular polygon."""
    try:
        shape_features = node_a.get('semantic_features', {})
        regularity = shape_features.get('regularity_score', 0)
        sides = shape_features.get('side_count', 0)
        return sides >= 3 and regularity < 0.5  # Low regularity threshold
    except:
        return False

def is_curved(node_a, node_b=None):
    """Checks if shape contains curved elements."""
    try:
        curvature = node_a.get('semantic_features', {}).get('curvature_score', 0)
        return curvature > 0.3
    except:
        return False

def is_straight(node_a, node_b=None):
    """Checks if shape contains only straight elements."""
    try:
        curvature = node_a.get('semantic_features', {}).get('curvature_score', 0)
        return curvature < 0.1
    except:
        return False

def is_composite(node_a, node_b=None):
    """Checks if shape is composite (made of multiple parts)."""
    try:
        return node_a.get('semantic_features', {}).get('is_composite', False)
    except:
        return False

def is_simple(node_a, node_b=None):
    """Checks if shape is simple (single component)."""
    try:
        return not node_a.get('semantic_features', {}).get('is_composite', True)
    except:
        return False

def has_acute_angles(node_a, node_b=None):
    """Checks if shape has acute angles."""
    try:
        angles = node_a.get('semantic_features', {}).get('angle_types', [])
        return 'acute' in angles
    except:
        return False

def has_right_angles(node_a, node_b=None):
    """Checks if shape has right angles."""
    try:
        angles = node_a.get('semantic_features', {}).get('angle_types', [])
        return 'right' in angles
    except:
        return False

def has_obtuse_angles(node_a, node_b=None):
    """Checks if shape has obtuse angles."""
    try:
        angles = node_a.get('semantic_features', {}).get('angle_types', [])
        return 'obtuse' in angles
    except:
        return False

def is_thick(node_a, node_b=None):
    """Checks if shape has thick lines/strokes."""
    try:
        thickness = node_a.get('semantic_features', {}).get('line_thickness', 0)
        return thickness > 3.0
    except:
        return False

def is_thin(node_a, node_b=None):
    """Checks if shape has thin lines/strokes."""
    try:
        thickness = node_a.get('semantic_features', {}).get('line_thickness', 0)
        return thickness <= 1.5
    except:
        return False

def is_large(node_a, node_b=None):
    """Checks if shape is large in size."""
    try:
        size_category = node_a.get('semantic_features', {}).get('size_category')
        return size_category == 'large'
    except:
        return False

def is_small(node_a, node_b=None):
    """Checks if shape is small in size."""
    try:
        size_category = node_a.get('semantic_features', {}).get('size_category')
        return size_category == 'small'
    except:
        return False

def is_medium(node_a, node_b=None):
    """Checks if shape is medium in size."""
    try:
        size_category = node_a.get('semantic_features', {}).get('size_category')
        return size_category == 'medium'
    except:
        return False

def is_tall(node_a, node_b=None):
    """Checks if shape is tall (height > width)."""
    try:
        aspect_ratio = node_a.get('semantic_features', {}).get('aspect_ratio', 1.0)
        return aspect_ratio > 1.5  # Height significantly larger than width
    except:
        return False

def is_wide(node_a, node_b=None):
    """Checks if shape is wide (width > height)."""
    try:
        aspect_ratio = node_a.get('semantic_features', {}).get('aspect_ratio', 1.0)
        return aspect_ratio < 0.67  # Width significantly larger than height
    except:
        return False

def is_centered(node_a, node_b=None):
    """Checks if shape is centered in the frame."""
    try:
        position = node_a.get('semantic_features', {}).get('position_category')
        return position == 'center'
    except:
        return False

def is_left_positioned(node_a, node_b=None):
    """Checks if shape is positioned on the left."""
    try:
        position = node_a.get('semantic_features', {}).get('position_category')
        return position == 'left'
    except:
        return False

def is_right_positioned(node_a, node_b=None):
    """Checks if shape is positioned on the right."""
    try:
        position = node_a.get('semantic_features', {}).get('position_category')
        return position == 'right'
    except:
        return False

def is_top_positioned(node_a, node_b=None):
    """Checks if shape is positioned at the top."""
    try:
        position = node_a.get('semantic_features', {}).get('position_category')
        return position == 'top'
    except:
        return False

def is_bottom_positioned(node_a, node_b=None):
    """Checks if shape is positioned at the bottom."""
    try:
        position = node_a.get('semantic_features', {}).get('position_category')
        return position == 'bottom'
    except:
        return False

# Registry of advanced predicates to be applied to higher-level objects
ADVANCED_PREDICATE_REGISTRY = {
    # COMPREHENSIVE BONGARD SHAPE DETECTION PREDICATES
    'is_circle': is_circle,
    'is_triangle': is_triangle,
    'is_square': is_square,
    'is_rectangle': is_rectangle,
    'is_pentagon': is_pentagon,
    'is_hexagon': is_hexagon,
    'is_octagon': is_octagon,
    'is_line': is_line,
    'is_arc': is_arc,
    'is_star': is_star,
    'is_cross': is_cross,
    'is_ellipse': is_ellipse,
    'is_diamond': is_diamond,
    'is_arrow': is_arrow,
    
    # COMPREHENSIVE SYMMETRY PREDICATES
    'has_vertical_symmetry': has_vertical_symmetry,
    'has_horizontal_symmetry': has_horizontal_symmetry,
    'has_rotational_symmetry': has_rotational_symmetry,
    'has_point_symmetry': has_point_symmetry,
    
    # COMPREHENSIVE TOPOLOGY PREDICATES
    'is_convex': is_convex,
    'is_concave': is_concave,
    'is_closed_shape': is_closed_shape,
    'is_open_shape': is_open_shape,
    'has_hole': has_hole,
    
    # COMPREHENSIVE SIDE COUNT PREDICATES
    'has_three_sides': has_three_sides,
    'has_four_sides': has_four_sides,
    'has_five_sides': has_five_sides,
    'has_six_sides': has_six_sides,
    'has_eight_sides': has_eight_sides,
    
    # COMPREHENSIVE REGULARITY PREDICATES
    'is_regular_polygon': is_regular_polygon,
    'is_irregular_polygon': is_irregular_polygon,
    
    # COMPREHENSIVE CURVATURE PREDICATES
    'is_curved': is_curved,
    'is_straight': is_straight,
    
    # COMPREHENSIVE COMPOSITION PREDICATES
    'is_composite': is_composite,
    'is_simple': is_simple,
    
    # COMPREHENSIVE ANGLE PREDICATES
    'has_acute_angles': has_acute_angles,
    'has_right_angles': has_right_angles,
    'has_obtuse_angles': has_obtuse_angles,
    
    # COMPREHENSIVE SIZE AND DIMENSION PREDICATES
    'is_thick': is_thick,
    'is_thin': is_thin,
    'is_large': is_large,
    'is_small': is_small,
    'is_medium': is_medium,
    'is_tall': is_tall,
    'is_wide': is_wide,
    
    # COMPREHENSIVE POSITION PREDICATES
    'is_centered': is_centered,
    'is_left_positioned': is_left_positioned,
    'is_right_positioned': is_right_positioned,
    'is_top_positioned': is_top_positioned,
    'is_bottom_positioned': is_bottom_positioned,
    
    # ORIGINAL LOW-LEVEL GEOMETRIC PREDICATES
    'intersects': intersects,
    'contains': contains,
    'is_above': is_above,
    'is_parallel': is_parallel,
    'same_shape_class': same_shape_class,
    'forms_symmetry': forms_symmetry,
    'similar_size': similar_size,
    'near_objects': near_objects,
    'same_orientation': same_orientation,
    'same_stroke_count': same_stroke_count,
    
    # ORIGINAL HIGH-LEVEL SEMANTIC PREDICATES FOR BONGARD PROBLEMS
    'has_apex_at_left': has_apex_at_left,
    'has_asymmetric_base': has_asymmetric_base,
    'has_tilted_orientation': has_tilted_orientation,
    'has_length_ratio_imbalance': has_length_ratio_imbalance,
    'forms_open_vs_closed_distinction': forms_open_vs_closed_distinction,
    'has_geometric_complexity_difference': has_geometric_complexity_difference,
    'has_compactness_difference': has_compactness_difference,
    'exhibits_mirror_asymmetry': exhibits_mirror_asymmetry,
    'has_dominant_direction': has_dominant_direction,
    'forms_bridge_pattern': forms_bridge_pattern,
    'has_curvature_distinction': has_curvature_distinction,
    
    # ORIGINAL CURVATURE AND JUNCTION TYPE PREDICATES
    'is_arc_of_circle': is_arc_of_circle,
    'shares_endpoint': shares_endpoint,
    'forms_apex': forms_apex,
    'forms_t_junction': forms_t_junction,
    'forms_x_junction': forms_x_junction,
    'forms_bridge_arc': forms_bridge_arc,
    'has_irregular_shape': has_irregular_shape,
    'exhibits_rotational_symmetry': exhibits_rotational_symmetry,
    
    # ORIGINAL STATE-OF-THE-ART: PROGRAM SYNTHESIS PREDICATES
    'has_stroke_count_pattern': has_stroke_count_pattern,
    'has_convexity_distinction': has_convexity_distinction,
    'has_symmetry_axis': has_symmetry_axis,
    'has_hole_distinction': has_hole_distinction,
    'has_angle_count_pattern': has_angle_count_pattern,
    'has_intersection_count_pattern': has_intersection_count_pattern,
    
    # ORIGINAL STATE-OF-THE-ART: CONTRASTIVE ANALOGICAL PREDICATES
    'forms_action_sequence_analogy': forms_action_sequence_analogy,
    'has_scaling_invariant_property': has_scaling_invariant_property,
    'exhibits_topological_invariance': exhibits_topological_invariance,
    
    # ORIGINAL STATE-OF-THE-ART: CONTEXT-DEPENDENT PREDICATES
    'has_context_dependent_interpretation': has_context_dependent_interpretation,
    'forms_grouping_gestalt': forms_grouping_gestalt,
    
    # ORIGINAL STATE-OF-THE-ART: PROGRAM-SEMANTIC ALIGNMENT PREDICATES
    'aligns_with_logo_semantics': aligns_with_logo_semantics,
    'exhibits_rule_compositionality': exhibits_rule_compositionality,
    
    # ORIGINAL BONGARD-SPECIFIC: SEMANTIC SHAPE PREDICATES
    'semantic_contains_triangle': semantic_contains_triangle,
    'semantic_contains_square': semantic_contains_square, 
    'semantic_contains_circle': semantic_contains_circle,
    'semantic_three_sided': semantic_three_sided,
    'semantic_four_sided': semantic_four_sided,
    'semantic_has_curves': semantic_has_curves,
    'semantic_closed_shape': semantic_closed_shape,
    'semantic_open_shape': semantic_open_shape,
    'semantic_simple_shape': semantic_simple_shape,
    'semantic_complex_shape': semantic_complex_shape,
}
