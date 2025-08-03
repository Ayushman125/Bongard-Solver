"""
Stroke-specific predicates for Arc vs Line types in Bongard-LOGO dataset
Addresses the need for different predicates and commonsense knowledge for arc and line stroke types
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

# ============================================================================
# ARC-SPECIFIC PREDICATES
# ============================================================================

def arc_connects_parallel_lines(arc_node, other_node):
    """Check if arc connects two parallel line segments (bridge pattern)"""
    try:
        # Arc should be curved, other should be line
        if not (arc_node.get('stroke_type') == 'arc' or 'arc_' in arc_node.get('action_command', '')):
            return False
        if not (other_node.get('stroke_type') == 'line' or 'line_' in other_node.get('action_command', '')):
            return False
            
        # Check if arc is positioned "above" the line (typical bridge pattern)
        arc_centroid = arc_node.get('centroid', [0, 0])
        line_centroid = other_node.get('centroid', [0, 0])
        
        # Arc should be vertically offset from line
        return abs(arc_centroid[1] - line_centroid[1]) > 20
        
    except Exception:
        return False

def arc_forms_semicircle(arc_node, other_node=None):
    """Check if arc forms a semicircle (180-degree arc)"""
    try:
        if not (arc_node.get('stroke_type') == 'arc' or 'arc_' in arc_node.get('action_command', '')):
            return False
            
        # Check curvature and angle span
        curvature = arc_node.get('curvature_score', 0)
        max_curvature = arc_node.get('max_curvature', 0)
        
        # Semicircle should have high, consistent curvature
        return curvature > 0.5 and max_curvature > 0.5
        
    except Exception:
        return False

def arc_forms_quarter_circle(arc_node, other_node=None):
    """Check if arc forms a quarter circle (90-degree arc)"""
    try:
        if not (arc_node.get('stroke_type') == 'arc' or 'arc_' in arc_node.get('action_command', '')):
            return False
            
        # Check for moderate curvature and specific orientation
        curvature = arc_node.get('curvature_score', 0)
        vertices = arc_node.get('vertices', [])
        
        if len(vertices) < 3:
            return False
            
        # Check if arc spans approximately 90 degrees
        start_vec = np.array(vertices[len(vertices)//2]) - np.array(vertices[0])
        end_vec = np.array(vertices[-1]) - np.array(vertices[len(vertices)//2])
        
        if np.linalg.norm(start_vec) > 1e-6 and np.linalg.norm(end_vec) > 1e-6:
            cos_angle = np.dot(start_vec, end_vec) / (np.linalg.norm(start_vec) * np.linalg.norm(end_vec))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            return 70 <= angle <= 110  # Approximately 90 degrees
            
    except Exception:
        return False

def arc_has_high_curvature(arc_node, other_node=None):
    """Check if arc has high curvature (tight curve)"""
    try:
        if not (arc_node.get('stroke_type') == 'arc' or 'arc_' in arc_node.get('action_command', '')):
            return False
            
        curvature = arc_node.get('curvature_score', 0)
        return curvature > 0.7
        
    except Exception:
        return False

def arc_has_uniform_curvature(arc_node, other_node=None):
    """Check if arc has uniform curvature (perfect circle segment)"""
    try:
        if not (arc_node.get('stroke_type') == 'arc' or 'arc_' in arc_node.get('action_command', '')):
            return False
            
        curvature = arc_node.get('curvature_score', 0)
        max_curvature = arc_node.get('max_curvature', 0)
        
        # Uniform curvature means low variance
        curvature_variance = abs(max_curvature - curvature) / max(curvature, 1e-6)
        return curvature_variance < 0.3
        
    except Exception:
        return False

def arc_connects_endpoints(arc_node, line_node):
    """Check if arc connects the endpoints of a line segment"""
    try:
        if not (arc_node.get('stroke_type') == 'arc' or 'arc_' in arc_node.get('action_command', '')):
            return False
        if not (line_node.get('stroke_type') == 'line' or 'line_' in line_node.get('action_command', '')):
            return False
            
        arc_endpoints = arc_node.get('endpoints', [])
        line_endpoints = line_node.get('endpoints', [])
        
        if len(arc_endpoints) != 2 or len(line_endpoints) != 2:
            return False
            
        # Check if arc endpoints match line endpoints
        for arc_ep in arc_endpoints:
            for line_ep in line_endpoints:
                if np.allclose(arc_ep, line_ep, atol=5.0):
                    return True
                    
        return False
        
    except Exception:
        return False

# ============================================================================
# LINE-SPECIFIC PREDICATES  
# ============================================================================

def line_is_perfectly_straight(line_node, other_node=None):
    """Check if line is perfectly straight (no deviation)"""
    try:
        if not (line_node.get('stroke_type') == 'line' or 'line_' in line_node.get('action_command', '')):
            return False
            
        vertices = line_node.get('vertices', [])
        if len(vertices) < 3:
            return True  # 2 points always form straight line
            
        # Check collinearity
        for i in range(2, len(vertices)):
            p1, p2, p3 = vertices[i-2], vertices[i-1], vertices[i]
            # Calculate cross product to check collinearity
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p3) - np.array(p2)
            cross = abs(np.cross(v1, v2))
            if cross > 1e-3:  # Not perfectly straight
                return False
                
        return True
        
    except Exception:
        return False

def line_forms_right_angle(line_node, other_line_node):
    """Check if two lines form a right angle (90 degrees)"""
    try:
        if not (line_node.get('stroke_type') == 'line' or 'line_' in line_node.get('action_command', '')):
            return False
        if not (other_line_node.get('stroke_type') == 'line' or 'line_' in other_line_node.get('action_command', '')):
            return False
            
        orientation1 = line_node.get('orientation', 0)
        orientation2 = other_line_node.get('orientation', 0)
        
        angle_diff = abs(orientation1 - orientation2)
        # Normalize to 0-180 range
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        return 85 <= angle_diff <= 95  # Approximately 90 degrees
        
    except Exception:
        return False

def line_is_horizontal(line_node, other_node=None):
    """Check if line is horizontal (0 or 180 degrees)"""
    try:
        if not (line_node.get('stroke_type') == 'line' or 'line_' in line_node.get('action_command', '')):
            return False
            
        orientation = line_node.get('orientation', 0)
        # Normalize to 0-360
        orientation = orientation % 360
        
        return (orientation < 10 or orientation > 350) or (170 < orientation < 190)
        
    except Exception:
        return False

def line_is_vertical(line_node, other_node=None):
    """Check if line is vertical (90 or 270 degrees)"""
    try:
        if not (line_node.get('stroke_type') == 'line' or 'line_' in line_node.get('action_command', '')):
            return False
            
        orientation = line_node.get('orientation', 0)
        # Normalize to 0-360
        orientation = orientation % 360
        
        return (80 < orientation < 100) or (260 < orientation < 280)
        
    except Exception:
        return False

def line_is_diagonal(line_node, other_node=None):
    """Check if line is diagonal (not horizontal or vertical)"""
    try:
        if not (line_node.get('stroke_type') == 'line' or 'line_' in line_node.get('action_command', '')):
            return False
            
        return not (line_is_horizontal(line_node) or line_is_vertical(line_node))
        
    except Exception:
        return False

def line_has_specific_length(line_node, other_node=None, target_length=None):
    """Check if line has specific length pattern"""
    try:
        if not (line_node.get('stroke_type') == 'line' or 'line_' in line_node.get('action_command', '')):
            return False
            
        length = line_node.get('length', 0)
        
        if target_length is not None:
            return abs(length - target_length) < target_length * 0.1
            
        # Check for standard lengths (common in LOGO programs)
        standard_lengths = [50, 100, 150, 200]  # Common LOGO lengths
        for std_len in standard_lengths:
            if abs(length - std_len) < std_len * 0.1:
                return True
                
        return False
        
    except Exception:
        return False

def lines_form_parallel_pattern(line_node1, line_node2):
    """Check if two lines are parallel"""
    try:
        if not (line_node1.get('stroke_type') == 'line' or 'line_' in line_node1.get('action_command', '')):
            return False
        if not (line_node2.get('stroke_type') == 'line' or 'line_' in line_node2.get('action_command', '')):
            return False
            
        orientation1 = line_node1.get('orientation', 0)
        orientation2 = line_node2.get('orientation', 0)
        
        angle_diff = abs(orientation1 - orientation2)
        # Normalize to 0-180 range  
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        return angle_diff < 10 or angle_diff > 170  # Parallel or anti-parallel
        
    except Exception:
        return False

def lines_form_corner(line_node1, line_node2):
    """Check if two lines meet to form a corner"""
    try:
        if not (line_node1.get('stroke_type') == 'line' or 'line_' in line_node1.get('action_command', '')):
            return False
        if not (line_node2.get('stroke_type') == 'line' or 'line_' in line_node2.get('action_command', '')):
            return False
            
        # Check if they share an endpoint
        endpoints1 = line_node1.get('endpoints', [])
        endpoints2 = line_node2.get('endpoints', [])
        
        for ep1 in endpoints1:
            for ep2 in endpoints2:
                if np.allclose(ep1, ep2, atol=5.0):
                    return True
                    
        return False
        
    except Exception:
        return False

# ============================================================================
# STROKE TYPE COMPARISON PREDICATES
# ============================================================================

def is_arc_line_pair(node1, node2):
    """Check if one node is arc and other is line"""
    try:
        node1_is_arc = node1.get('stroke_type') == 'arc' or 'arc_' in node1.get('action_command', '')
        node1_is_line = node1.get('stroke_type') == 'line' or 'line_' in node1.get('action_command', '')
        
        node2_is_arc = node2.get('stroke_type') == 'arc' or 'arc_' in node2.get('action_command', '')
        node2_is_line = node2.get('stroke_type') == 'line' or 'line_' in node2.get('action_command', '')
        
        return (node1_is_arc and node2_is_line) or (node1_is_line and node2_is_arc)
        
    except Exception:
        return False

def same_stroke_type(node1, node2):
    """Check if both nodes have the same stroke type (both arc or both line)"""
    try:
        node1_is_arc = node1.get('stroke_type') == 'arc' or 'arc_' in node1.get('action_command', '')
        node1_is_line = node1.get('stroke_type') == 'line' or 'line_' in node1.get('action_command', '')
        
        node2_is_arc = node2.get('stroke_type') == 'arc' or 'arc_' in node2.get('action_command', '')
        node2_is_line = node2.get('stroke_type') == 'line' or 'line_' in node2.get('action_command', '')
        
        return (node1_is_arc and node2_is_arc) or (node1_is_line and node2_is_line)
        
    except Exception:
        return False

def arc_completes_line_shape(arc_node, line_node):
    """Check if arc completes a shape started by lines (e.g., arc caps a rectangle)"""
    try:
        if not is_arc_line_pair(arc_node, line_node):
            return False
            
        # Arc should connect line endpoints or bridge parallel lines
        return (arc_connects_endpoints(arc_node, line_node) or 
                arc_connects_parallel_lines(arc_node, line_node))
        
    except Exception:
        return False

# ============================================================================
# STROKE-SPECIFIC PREDICATE REGISTRY
# ============================================================================

ARC_SPECIFIC_PREDICATES = {
    'arc_connects_parallel_lines': arc_connects_parallel_lines,
    'arc_forms_semicircle': arc_forms_semicircle,
    'arc_forms_quarter_circle': arc_forms_quarter_circle,
    'arc_has_high_curvature': arc_has_high_curvature,
    'arc_has_uniform_curvature': arc_has_uniform_curvature,
    'arc_connects_endpoints': arc_connects_endpoints,
}

LINE_SPECIFIC_PREDICATES = {
    'line_is_perfectly_straight': line_is_perfectly_straight,
    'line_forms_right_angle': line_forms_right_angle,
    'line_is_horizontal': line_is_horizontal,
    'line_is_vertical': line_is_vertical,
    'line_is_diagonal': line_is_diagonal,
    'line_has_specific_length': line_has_specific_length,
    'lines_form_parallel_pattern': lines_form_parallel_pattern,
    'lines_form_corner': lines_form_corner,
}

STROKE_COMPARISON_PREDICATES = {
    'is_arc_line_pair': is_arc_line_pair,
    'same_stroke_type': same_stroke_type,
    'arc_completes_line_shape': arc_completes_line_shape,
}

# Combined registry for all stroke-specific predicates
ALL_STROKE_PREDICATES = {
    **ARC_SPECIFIC_PREDICATES,
    **LINE_SPECIFIC_PREDICATES,
    **STROKE_COMPARISON_PREDICATES
}

def get_applicable_predicates(node1, node2=None):
    """Get predicates applicable to the given node(s) based on stroke type"""
    applicable = {}
    
    # Check node1 stroke type
    node1_is_arc = node1.get('stroke_type') == 'arc' or 'arc_' in node1.get('action_command', '')
    node1_is_line = node1.get('stroke_type') == 'line' or 'line_' in node1.get('action_command', '')
    
    if node2 is not None:
        node2_is_arc = node2.get('stroke_type') == 'arc' or 'arc_' in node2.get('action_command', '')
        node2_is_line = node2.get('stroke_type') == 'line' or 'line_' in node2.get('action_command', '')
        
        # Add comparison predicates
        applicable.update(STROKE_COMPARISON_PREDICATES)
        
        # Add specific predicates based on pair types
        if node1_is_arc or node2_is_arc:
            applicable.update(ARC_SPECIFIC_PREDICATES)
        if node1_is_line or node2_is_line:
            applicable.update(LINE_SPECIFIC_PREDICATES)
    else:
        # Single node predicates
        if node1_is_arc:
            applicable.update(ARC_SPECIFIC_PREDICATES)
        if node1_is_line:
            applicable.update(LINE_SPECIFIC_PREDICATES)
    
    return applicable

# ============================================================================
# LOGGING AND DIAGNOSTICS
# ============================================================================

def analyze_stroke_distribution(scene_graph):
    """Analyze the distribution of stroke types in a scene graph"""
    arc_count = 0
    line_count = 0
    unknown_count = 0
    
    for node_id, node_data in scene_graph.nodes(data=True):
        if node_data.get('stroke_type') == 'arc' or 'arc_' in node_data.get('action_command', ''):
            arc_count += 1
        elif node_data.get('stroke_type') == 'line' or 'line_' in node_data.get('action_command', ''):
            line_count += 1
        else:
            unknown_count += 1
    
    total = arc_count + line_count + unknown_count
    
    logging.info(f"Stroke distribution: Arc={arc_count} ({arc_count/total*100:.1f}%), "
                f"Line={line_count} ({line_count/total*100:.1f}%), "
                f"Unknown={unknown_count} ({unknown_count/total*100:.1f}%)")
    
    return {
        'arc_count': arc_count,
        'line_count': line_count, 
        'unknown_count': unknown_count,
        'total': total
    }

if __name__ == "__main__":
    # Test the predicates
    print("Stroke-specific predicates loaded successfully!")
    print(f"Arc predicates: {len(ARC_SPECIFIC_PREDICATES)}")
    print(f"Line predicates: {len(LINE_SPECIFIC_PREDICATES)}")
    print(f"Comparison predicates: {len(STROKE_COMPARISON_PREDICATES)}")
    print(f"Total stroke-specific predicates: {len(ALL_STROKE_PREDICATES)}")
