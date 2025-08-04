import os
import csv
import logging
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import glob
import json
from datetime import datetime

# Add comprehensive visualization and CSV export support for enhanced Bongard Solver

__all__ = [
    'save_scene_graph_visualization',
    'save_scene_graph_csv',
    'save_enhanced_scene_graph_visualization',
    'create_multi_puzzle_visualization',
    'create_missing_data_analysis',
    'analyze_puzzle_completeness',
    'save_comprehensive_bongard_analysis',
    'export_comprehensive_features_csv',
    'visualize_shape_categories',
    'create_predicate_analysis_report'
]

def get_node_value(node_data, field_name, default_value=None):
    """Safely extract node field value with appropriate defaults"""
    value = node_data.get(field_name, default_value)
    
    # Handle special cases
    if field_name == 'vl_embed' and (value is None or value == []):
        return [0.0] * 512
    elif field_name in ['motif_type', 'connectivity_pattern', 'symmetry_type'] and value is None:
        return 'unknown'
    elif field_name in ['motif_complexity_score', 'clip_sim', 'vl_sim'] and value is None:
        return 0.0
    elif field_name == 'predicate' and value is None:
        return ''
    elif field_name == 'semantic_features' and value is None:
        return {}
    
    return value

def get_comprehensive_node_style(data):
    """Enhanced node styling based on comprehensive shape features and 5 discovered Bongard-LOGO shape types"""
    # Extract semantic features
    semantic_features = data.get('semantic_features', {})
    comprehensive_features = data.get('comprehensive_features')
    geometric_analysis = data.get('geometric_analysis', {})
    commonsense_analysis = data.get('commonsense_analysis', {})
    
    # Determine shape type with fallback logic - prioritize discovered Bongard-LOGO types
    shape_type = 'unknown'
    topology = 'unknown'
    symmetry = 'none'
    
    # First check for discovered Bongard-LOGO shape types from current data structure
    if 'shape_type' in data:
        shape_type = data['shape_type']
    elif 'action_command' in data and data['action_command']:
        # Extract shape type from action command for Bongard-LOGO data
        action_cmd = data['action_command']
        if '_' in action_cmd:
            parts = action_cmd.split('_')
            if len(parts) >= 2:
                potential_shape = parts[1]  # e.g., "line_normal" -> "normal"
                if potential_shape in ['normal', 'circle', 'square', 'triangle', 'zigzag']:
                    shape_type = potential_shape
    elif comprehensive_features:
        shape_type = comprehensive_features.shape_type.value if hasattr(comprehensive_features.shape_type, 'value') else str(comprehensive_features.shape_type)
        topology = comprehensive_features.topology.value if hasattr(comprehensive_features.topology, 'value') else str(comprehensive_features.topology)
        symmetry = comprehensive_features.symmetry_type.value if hasattr(comprehensive_features.symmetry_type, 'value') else str(comprehensive_features.symmetry_type)
    else:
        # Fallback to semantic features
        shape_type = semantic_features.get('primary_shape_type', 'unknown')
        topology = semantic_features.get('topology_type', 'unknown')
        symmetry = semantic_features.get('symmetry_type', 'none')
        
        # Further fallback to geometric analysis for unknown shapes
        if shape_type == 'unknown' and geometric_analysis:
            vertex_count = geometric_analysis.get('total_vertices', 0)
            is_closed = geometric_analysis.get('all_closed', False)
            
            # Infer shape type from geometric analysis
            if vertex_count == 3:
                shape_type = 'triangle'
            elif vertex_count == 4:
                shape_type = 'square'
            elif vertex_count == 5:
                shape_type = 'pentagon'
            elif vertex_count == 6:
                shape_type = 'hexagon'
            elif vertex_count > 6:
                shape_type = 'polygon'
            elif vertex_count == 0 and is_closed:
                shape_type = 'circle'
            elif vertex_count == 0:
                shape_type = 'normal'  # Default to normal for lines
    
    # Enhanced color mapping prioritizing the 5 discovered Bongard-LOGO shape types
    shape_colors = {
        # DISCOVERED BONGARD-LOGO SHAPE TYPES (prioritized)
        'normal': '#4CAF50',        # Green - most common type (24,107 occurrences)
        'circle': '#FF6B6B',        # Red - 6,256 occurrences
        'triangle': '#4ECDC4',      # Teal - 5,837 occurrences
        'square': '#45B7D1',        # Blue - 6,519 occurrences
        'zigzag': '#FF9800',        # Orange - 6,729 occurrences
        
        # Legacy and additional shape types for backward compatibility
        'rectangle': '#96CEB4',     # Green
        'pentagon': '#FFEAA7',      # Yellow
        'hexagon': '#DDA0DD',       # Plum
        'heptagon': '#F7B2C1',      # Light pink
        'octagon': '#98D8C8',       # Mint
        'decagon': '#FFB347',       # Peach
        'polygon': '#E6E6FA',       # Lavender
        'line': '#F7DC6F',          # Light yellow
        'ray': '#F9E79F',           # Pale yellow
        'segment': '#FCF3CF',       # Very pale yellow
        'arc': '#BB8FCE',           # Light purple
        'spiral': '#D2B4DE',        # Pale purple
        'crescent': '#AED6F1',      # Light blue
        'semicircle': '#A9CCE3',    # Pale blue
        'star': '#F8C471',          # Orange
        'cross': '#85C1E9',         # Light blue
        'ellipse': '#F1948A',       # Salmon
        'oval': '#F1948A',          # Salmon
        'diamond': '#82E0AA',       # Light green
        'rhombus': '#7DCEA0',       # Medium green
        'trapezoid': '#85C1E9',     # Light blue
        'parallelogram': '#AED6F1', # Pale blue
        'arrow': '#D7BDE2',         # Lavender
        'heart': '#F8BBD9',         # Light pink
        'flower': '#F9E2AF',        # Light cream
        'house': '#A9DFBF',         # Light green
        'blob': '#D5DBDB',          # Light gray
        'freeform': '#CCD1D1',      # Gray
        'irregular': '#B2BABB',     # Medium gray
        'unknown': '#BDC3C7',       # Gray
        'special': '#E8DAEF',       # Light lavender
        'curved': '#FADBD8',        # Light rose
        'polygonal': '#D5F4E6',     # Light mint
    }
    
    # Border styles based on topology with unknown shape handling
    topology_borders = {
        'closed': {'width': 3, 'style': '-'},
        'open': {'width': 2, 'style': ':'},
        'simple': {'width': 2, 'style': '-'},
        'complex': {'width': 3, 'style': '--'},
        'convex': {'width': 3, 'style': '-'},
        'concave': {'width': 2, 'style': '--'},
        'connected': {'width': 2, 'style': '-'},
        'disconnected': {'width': 1, 'style': ':'},
        'hole': {'width': 4, 'style': '-.'},
        'unknown': {'width': 1, 'style': '-'}
    }
    
    # Size based on shape complexity with unknown shape handling
    complexity = 0.5
    if comprehensive_features:
        complexity = comprehensive_features.complexity_score
    elif semantic_features:
        complexity = semantic_features.get('complexity_score', 0.5)
    elif geometric_analysis:
        complexity_level = geometric_analysis.get('shape_complexity_level', 'simple')
        complexity = {'simple': 0.3, 'moderate': 0.6, 'complex': 0.9}.get(complexity_level, 0.5)
    
    node_size = 300 + int(complexity * 500)  # Range: 300-800
    
    # Handle unknown shapes with special styling
    if shape_type == 'unknown':
        # Use geometric analysis to determine visual properties
        if geometric_analysis:
            vertex_count = geometric_analysis.get('total_vertices', 0)
            if vertex_count > 0:
                # Color based on vertex count for unknown shapes
                if vertex_count <= 4:
                    base_color = '#F8C471'  # Orange for simple unknown
                elif vertex_count <= 8:
                    base_color = '#BB8FCE'  # Purple for moderate unknown
                else:
                    base_color = '#85C1E9'  # Blue for complex unknown
            else:
                base_color = shape_colors['unknown']
        else:
            base_color = shape_colors['unknown']
        
        # Special border for unknown shapes
        border_info = {'width': 2, 'style': '--'}
        
        # Lower opacity for unknown shapes
        confidence = 0.4
    else:
        base_color = shape_colors.get(shape_type, shape_colors['unknown'])
        border_info = topology_borders.get(topology, topology_borders['unknown'])
        
        # Confidence based on comprehensive features
        if comprehensive_features:
            confidence = 0.8  # High confidence for recognized shapes
        elif semantic_features:
            confidence = semantic_features.get('confidence_score', 0.7)
        else:
            confidence = 0.5
    
    # Alpha based on confidence
    alpha = 0.4 + 0.6 * confidence
    
    # Special handling for commonsense analysis
    if commonsense_analysis:
        bongard_relevance = commonsense_analysis.get('bongard_relevance', 0.5)
        alpha *= (0.5 + 0.5 * bongard_relevance)  # Adjust alpha by Bongard relevance
    
    return {
        'color': base_color,
        'alpha': alpha,
        'size': node_size,
        'border_width': border_info['width'],
        'border_style': border_info['style'],
        'shape_type': shape_type,
        'topology': topology,
        'symmetry': symmetry,
        'is_unknown': shape_type == 'unknown',
        'complexity': complexity
    }

def get_comprehensive_edge_style(predicate):
    """Enhanced edge styling based on comprehensive predicate types and current system"""
    predicate_styles = {
        # Shape detection predicates for discovered types
        'is_circle': {'color': '#FF6B6B', 'width': 2, 'style': '-'},
        'is_triangle': {'color': '#4ECDC4', 'width': 2, 'style': '-'},
        'is_square': {'color': '#45B7D1', 'width': 2, 'style': '-'},
        'is_rectangle': {'color': '#96CEB4', 'width': 2, 'style': '-'},
        'is_normal': {'color': '#4CAF50', 'width': 2, 'style': '-'},
        'is_zigzag': {'color': '#FF9800', 'width': 2, 'style': '-'},
        'same_shape_class': {'color': '#9C27B0', 'width': 2, 'style': '--'},
        
        # Symmetry predicates
        'has_vertical_symmetry': {'color': '#9B59B6', 'width': 3, 'style': '--'},
        'has_horizontal_symmetry': {'color': '#8E44AD', 'width': 3, 'style': '--'},
        'has_rotational_symmetry': {'color': '#6C3483', 'width': 3, 'style': ':'},
        'forms_symmetry': {'color': '#673AB7', 'width': 2, 'style': '-.'},
        'symmetric_with': {'color': '#3F51B5', 'width': 2, 'style': ':'},
        
        # Topology predicates
        'is_convex': {'color': '#27AE60', 'width': 2, 'style': '-'},
        'is_concave': {'color': '#E74C3C', 'width': 2, 'style': '--'},
        'is_closed_shape': {'color': '#2ECC71', 'width': 2, 'style': '-'},
        'is_open_shape': {'color': '#E67E22', 'width': 2, 'style': ':'},
        'has_convexity_distinction': {'color': '#1B5E20', 'width': 2, 'style': '--'},
        'has_hole_distinction': {'color': '#BF360C', 'width': 2, 'style': '-.'},
        
        # Spatial relationships
        'intersects': {'color': '#FF4757', 'width': 2, 'style': '-'},
        'contains': {'color': '#3742FA', 'width': 2, 'style': '-'},
        'is_above': {'color': '#2ED573', 'width': 1, 'style': '->'},
        'is_below': {'color': '#FFA726', 'width': 1, 'style': '->'},
        'is_left_of': {'color': '#42A5F5', 'width': 1, 'style': '->'},
        'is_right_of': {'color': '#AB47BC', 'width': 1, 'style': '->'},
        'is_parallel': {'color': '#FFA502', 'width': 2, 'style': '||'},
        'near': {'color': '#26C6DA', 'width': 1, 'style': ':'},
        'adjacent_endpoints': {'color': '#66BB6A', 'width': 1, 'style': ':'},
        'shares_endpoint': {'color': '#EF5350', 'width': 1, 'style': ':'},
        
        # Size and measurement predicates
        'larger_than': {'color': '#D32F2F', 'width': 2, 'style': '-'},
        'smaller_than': {'color': '#1976D2', 'width': 2, 'style': '-'},
        'similar_size': {'color': '#388E3C', 'width': 2, 'style': '--'},
        'aspect_sim': {'color': '#F57C00', 'width': 1, 'style': ':'},
        'length_sim': {'color': '#7B1FA2', 'width': 1, 'style': ':'},
        
        # Geometric predicates
        'has_angle_count_pattern': {'color': '#5D4037', 'width': 2, 'style': '-.'},
        'has_stroke_count_pattern': {'color': '#455A64', 'width': 2, 'style': '-.'},
        'orientation_sim': {'color': '#00695C', 'width': 1, 'style': ':'},
        'curvature_sim': {'color': '#BF360C', 'width': 1, 'style': ':'},
        
        # Motif and pattern predicates
        'part_of_motif': {'color': '#FFD700', 'width': 3, 'style': '-'},
        'motif_similarity': {'color': '#FF8C00', 'width': 2, 'style': '--'},
        'part_of': {'color': '#9370DB', 'width': 2, 'style': '-'},
        'visual_similarity': {'color': '#FF1493', 'width': 1, 'style': ':'},
        'vl_sim': {'color': '#00CED1', 'width': 1, 'style': ':'},
        'clip_sim': {'color': '#20B2AA', 'width': 1, 'style': ':'},
        
        # Commonsense predicates
        'related_to': {'color': '#8A2BE2', 'width': 1, 'style': ':'},
        'kb_sim': {'color': '#9932CC', 'width': 1, 'style': ':'},
        'programmatic_sim': {'color': '#4B0082', 'width': 1, 'style': ':'},
        
        # Current system predicates from predicate miner
        'para': {'color': '#FF6347', 'width': 2, 'style': '||'},
        'centroid_dist': {'color': '#32CD32', 'width': 1, 'style': ':'},
        'near_and_para': {'color': '#FF69B4', 'width': 2, 'style': '-.'},
        'same_program_and_near': {'color': '#00BFFF', 'width': 2, 'style': '-.'},
        
        # Pattern and structure predicates
        'forms_bridge_pattern': {'color': '#8B4513', 'width': 2, 'style': '-'},
        'forms_apex': {'color': '#A0522D', 'width': 2, 'style': '-'},
        'forms_t_junction': {'color': '#CD853F', 'width': 2, 'style': '-'},
        'forms_x_junction': {'color': '#DEB887', 'width': 2, 'style': '-'},
        
        # Default
        'default': {'color': '#7F8C8D', 'width': 1, 'style': '-'}
    }
    
    return predicate_styles.get(predicate, predicate_styles['default'])

def export_comprehensive_features_csv(scene_graph, output_path, puzzle_name=""):
    """Export comprehensive shape features and predicates to CSV for analysis"""
    
    # Prepare data for CSV export
    node_data = []
    edge_data = []
    
    # Extract comprehensive node features
    for node_id, node_attrs in scene_graph.nodes(data=True):
        semantic_features = node_attrs.get('semantic_features', {})
        comprehensive_features = node_attrs.get('comprehensive_features')
        geometric_analysis = node_attrs.get('geometric_analysis', {})
        commonsense_analysis = node_attrs.get('commonsense_analysis', {})
        
        # Basic shape information - prioritize discovered Bongard-LOGO types
        shape_type = 'unknown'
        side_count = 0
        angle_count = 0
        vertex_count = 0
        
        # Extract shape type from current data structure (compatible with data loading)
        if 'shape_type' in node_attrs:
            shape_type = node_attrs['shape_type']
        elif 'action_command' in node_attrs and node_attrs['action_command']:
            # Extract from action command for Bongard-LOGO data
            action_cmd = node_attrs['action_command']
            if '_' in action_cmd:
                parts = action_cmd.split('_')
                if len(parts) >= 2:
                    potential_shape = parts[1]  # e.g., "line_normal" -> "normal"
                    if potential_shape in ['normal', 'circle', 'square', 'triangle', 'zigzag']:
                        shape_type = potential_shape
        elif comprehensive_features:
            shape_type = comprehensive_features.shape_type.value if hasattr(comprehensive_features.shape_type, 'value') else str(comprehensive_features.shape_type)
            side_count = comprehensive_features.side_count
            angle_count = comprehensive_features.angle_count
            vertex_count = comprehensive_features.vertex_count
        
        # Extract additional shape properties from current data structure
        is_closed = node_attrs.get('is_closed', False)
        curvature_type = node_attrs.get('curvature_type', 'unknown')
        regularity_score = node_attrs.get('shape_regularity', 0.5)
        complexity_level = node_attrs.get('shape_complexity_level', 1.0)
        
        node_row = {
            'puzzle_name': puzzle_name,
            'node_id': node_id,
            'object_type': node_attrs.get('object_type', 'unknown'),
            'object_id': node_attrs.get('object_id', node_id),
            'parent_shape_id': node_attrs.get('parent_shape_id', ''),
            
            # DISCOVERED BONGARD-LOGO SHAPE TYPES
            'shape_type': shape_type,
            'is_normal_type': 1 if shape_type == 'normal' else 0,
            'is_circle_type': 1 if shape_type == 'circle' else 0,
            'is_square_type': 1 if shape_type == 'square' else 0,
            'is_triangle_type': 1 if shape_type == 'triangle' else 0,
            'is_zigzag_type': 1 if shape_type == 'zigzag' else 0,
            
            # Comprehensive shape features
            'primary_shape_type': semantic_features.get('primary_shape_type', shape_type),
            'secondary_shape_types': ', '.join(semantic_features.get('secondary_shape_types', [])),
            'topology_type': comprehensive_features.topology.value if comprehensive_features else semantic_features.get('topology_type', 'unknown'),
            'symmetry_type': comprehensive_features.symmetry_type.value if comprehensive_features else semantic_features.get('symmetry_type', 'none'),
            
            # Current data structure geometric properties
            'is_closed': is_closed,
            'curvature_type': curvature_type,
            'vertices': str(node_attrs.get('vertices', [])),
            'action_command': node_attrs.get('action_command', ''),
            'action_program': str(node_attrs.get('action_program', [])),
            'stroke_type': node_attrs.get('stroke_type', 'unknown'),
            
            # Geometric properties
            'side_count': side_count,
            'angle_count': angle_count,
            'vertex_count': vertex_count or len(node_attrs.get('vertices', [])),
            'is_connected': comprehensive_features.is_connected if comprehensive_features else semantic_features.get('is_connected', False),
            'has_holes': comprehensive_features.has_holes if comprehensive_features else 0,
            
            # Physics and computed attributes
            'area': node_attrs.get('area', 0),
            'perimeter': node_attrs.get('perimeter', 0),
            'length': node_attrs.get('length', 0),
            'aspect_ratio': node_attrs.get('aspect_ratio', 1.0),
            'compactness': node_attrs.get('compactness', 0),
            'orientation': node_attrs.get('orientation', 0),
            'inertia': node_attrs.get('inertia', 0),
            'convexity': node_attrs.get('convexity', 0),
            
            # Advanced geometric analysis
            'total_edges': geometric_analysis.get('total_edges', 0),
            'total_vertices': geometric_analysis.get('total_vertices', 0),
            'curve_segments': geometric_analysis.get('curve_segments', 0),
            'straight_segments': geometric_analysis.get('straight_segments', 0),
            'all_closed': geometric_analysis.get('all_closed', False),
            'all_open': geometric_analysis.get('all_open', False),
            'mixed_shape_types': geometric_analysis.get('mixed_shape_types', False),
            'contains_curves_and_lines': geometric_analysis.get('contains_curves_and_lines', False),
            
            # Quality measures and scores
            'regularity_score': regularity_score,
            'complexity_level': complexity_level,
            'convexity_score': comprehensive_features.convexity if comprehensive_features else semantic_features.get('convexity_score', 0),
            'complexity_score': comprehensive_features.complexity_score if comprehensive_features else semantic_features.get('complexity_score', 0),
            'symmetry_score': comprehensive_features.symmetry_score if comprehensive_features else semantic_features.get('symmetry_score', 0),
            'confidence_score': semantic_features.get('confidence_score', 0),
            
            # Size and scale
            'size_category': comprehensive_features.size_category if comprehensive_features else semantic_features.get('size_category', 'unknown'),
            'relative_size': comprehensive_features.relative_size if comprehensive_features else semantic_features.get('relative_size', 0),
            'compactness_alt': comprehensive_features.compactness if comprehensive_features else semantic_features.get('compactness', 0),
            
            # Compositional features
            'is_composite': comprehensive_features.is_composite if comprehensive_features else semantic_features.get('is_composite', False),
            'component_shapes': ', '.join(comprehensive_features.component_shapes if comprehensive_features else []),
            'spatial_arrangement': comprehensive_features.spatial_arrangement if comprehensive_features else 'unknown',
            'stroke_count': comprehensive_features.stroke_count if comprehensive_features else node_attrs.get('stroke_count', 1),
            'intersection_count': comprehensive_features.intersection_count if comprehensive_features else 0,
            
            # Boolean features
            'is_regular': semantic_features.get('is_regular', False),
            'is_symmetric': semantic_features.get('is_symmetric', False),
            'contains_curves': comprehensive_features.contains_curves if comprehensive_features else semantic_features.get('has_curves', False),
            'contains_angles': comprehensive_features.contains_angles if comprehensive_features else semantic_features.get('has_angles', False),
            'has_sharp_angles': semantic_features.get('has_sharp_angles', False),
            
            # Mathematical properties
            'rotation_order': comprehensive_features.rotation_order if comprehensive_features else 1,
            'genus': comprehensive_features.genus if comprehensive_features else 0,
            'symmetry_axes_count': len(comprehensive_features.symmetry_axes) if comprehensive_features else 0,
            
            # Bongard-specific features for discovered shape types
            'has_three_sides': side_count == 3 or shape_type == 'triangle',
            'has_four_sides': side_count == 4 or shape_type == 'square',
            'has_five_sides': side_count == 5,
            'has_six_sides': side_count == 6,
            'has_many_sides': side_count > 6,
            'has_odd_sides': side_count > 0 and side_count % 2 == 1,
            'has_even_sides': side_count > 0 and side_count % 2 == 0,
            'has_prime_sides': side_count in [2, 3, 5, 7, 11, 13, 17],
            
            # Feature validity flags from current data structure
            'geometry_valid': node_attrs.get('geometry_valid', False),
            'feature_valid': str(node_attrs.get('feature_valid', {})),
            'fallback_geometry': node_attrs.get('fallback_geometry', False),
            
            # Semantic and KB labels
            'semantic_label': node_attrs.get('semantic_label', ''),
            'shape_label': node_attrs.get('shape_label', ''),
            'kb_concept': node_attrs.get('kb_concept', ''),
            'programmatic_label': node_attrs.get('programmatic_label', ''),
            'pattern_role': node_attrs.get('pattern_role', ''),
            
            # Motif and grouping information
            'motif_id': node_attrs.get('motif_id', ''),
            'motif_type': node_attrs.get('motif_type', ''),
            'motif_label': node_attrs.get('motif_label', ''),
            'connectivity_pattern': node_attrs.get('connectivity_pattern', ''),
            'symmetry_type_motif': node_attrs.get('symmetry_type', 'none'),
            
            # Shape complexity analysis
            'shape_complexity_level': geometric_analysis.get('shape_complexity_level', 'simple'),
            'dominant_shape_type': geometric_analysis.get('dominant_shape_type', 'unknown'),
            'shape_count': geometric_analysis.get('shape_count', 1),
            'exhibits_scaling_pattern': geometric_analysis.get('exhibits_scaling_pattern', False),
            'follows_pattern_rule': geometric_analysis.get('follows_pattern_rule', False),
            'has_nested_structure': geometric_analysis.get('has_nested_structure', False),
            
            # Commonsense analysis
            'shape_category': commonsense_analysis.get('shape_category', 'unknown') if commonsense_analysis else 'unknown',
            'bongard_relevance': commonsense_analysis.get('bongard_relevance', 0.5) if commonsense_analysis else 0.5,
            'reasoning_strategies': ', '.join(commonsense_analysis.get('reasoning_strategies', [])) if commonsense_analysis else '',
            'mathematical_properties': str(commonsense_analysis.get('mathematical_properties', {})) if commonsense_analysis else '',
            'visual_properties': str(commonsense_analysis.get('visual_properties', {})) if commonsense_analysis else '',
            
            # Position data with safe extraction
            'centroid_x': node_attrs.get('centroid', [0, 0])[0] if node_attrs.get('centroid') and len(node_attrs.get('centroid', [])) >= 2 else 0,
            'centroid_y': node_attrs.get('centroid', [0, 0])[1] if node_attrs.get('centroid') and len(node_attrs.get('centroid', [])) >= 2 else 0,
            'cx': node_attrs.get('cx', 0),
            'cy': node_attrs.get('cy', 0),
            
            # VL and embedding features
            'vl_embed': str(node_attrs.get('vl_embed', [])),
            'clip_sim': node_attrs.get('clip_sim', 0.0),
            'vl_sim': node_attrs.get('vl_sim', 0.0),
            
            # Action and program information
            'action_index': node_attrs.get('action_index', -1),
            'repetition_count': node_attrs.get('repetition_count', 1),
            'turn_direction': node_attrs.get('turn_direction', 'none'),
            'turn_angle': node_attrs.get('turn_angle', 0),
            
            # Unknown shape indicators
            'is_unknown_shape': shape_type == 'unknown',
            'requires_geometric_analysis': shape_type == 'unknown' or any(s.get('confidence', 0) < 0.5 for s in node_attrs.get('semantic_shapes', [])),
            'geometric_basis': any(s.get('geometric_basis', False) for s in node_attrs.get('semantic_shapes', [])),
        }
        
        node_data.append(node_row)
    
    # Extract comprehensive edge relationships with current predicate system
    for u, v, edge_attrs in scene_graph.edges(data=True):
        predicate = edge_attrs.get('predicate', 'unknown')
        confidence = edge_attrs.get('confidence', 0.0)
        
        edge_row = {
            'puzzle_name': puzzle_name,
            'source_node': u,
            'target_node': v,
            'predicate': predicate,
            'confidence': confidence,
            'predicate_category': _categorize_predicate(predicate),
            'relationship_strength': edge_attrs.get('relationship_strength', 0.0),
            'spatial_distance': edge_attrs.get('spatial_distance', 0.0),
            
            # Additional edge attributes from current system
            'edge_weight': edge_attrs.get('weight', 1.0),
            'angle_between': edge_attrs.get('angle_between', 0.0),
            'distance': edge_attrs.get('distance', 0.0),
            'overlap_ratio': edge_attrs.get('overlap_ratio', 0.0),
            'semantic_similarity': edge_attrs.get('semantic_similarity', 0.0),
            'spatial_distance': edge_attrs.get('spatial_distance', 0.0),
        }
        
        edge_data.append(edge_row)
    
    # Create DataFrames and save to CSV with standard library
    try:
        if node_data:
            nodes_csv_path = output_path.replace('.csv', '_nodes.csv')
            
            # Write nodes CSV using standard csv module
            with open(nodes_csv_path, 'w', newline='', encoding='utf-8') as f:
                if node_data:
                    fieldnames = node_data[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(node_data)
            
            logging.info(f"[Comprehensive CSV] Saved node features to {nodes_csv_path}")
        
        if edge_data:
            edges_csv_path = output_path.replace('.csv', '_edges.csv')
            
            # Write edges CSV using standard csv module
            with open(edges_csv_path, 'w', newline='', encoding='utf-8') as f:
                if edge_data:
                    fieldnames = edge_data[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(edge_data)
            
            logging.info(f"[Comprehensive CSV] Saved edge relationships to {edges_csv_path}")
        
        # Also create the main CSV file that the test expects
        main_csv_path = output_path
        with open(main_csv_path, 'w', newline='', encoding='utf-8') as f:
            if node_data:
                fieldnames = node_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(node_data)
        
        logging.info(f"[Comprehensive CSV] Saved main CSV to {main_csv_path}")
            
        # Create summary statistics
        _create_summary_statistics(node_data, edge_data, output_path.replace('.csv', '_summary.csv'))
        
        return True
        
    except Exception as e:
        logging.error(f"[Comprehensive CSV] Error saving CSV: {e}")
        print(f"Error details: {e}")  # For debugging
        return False

def _categorize_predicate(predicate):
    """Categorize predicates into semantic groups with current system predicates"""
    # Shape detection predicates for discovered types
    shape_predicates = ['is_circle', 'is_triangle', 'is_square', 'is_rectangle', 'is_pentagon', 'is_hexagon',
                       'is_normal', 'is_zigzag', 'same_shape_class']
    
    # Symmetry predicates
    symmetry_predicates = ['has_vertical_symmetry', 'has_horizontal_symmetry', 'has_rotational_symmetry',
                          'forms_symmetry', 'symmetric_with', 'exhibits_mirror_asymmetry']
    
    # Topology predicates  
    topology_predicates = ['is_convex', 'is_concave', 'is_closed_shape', 'is_open_shape', 'has_hole',
                          'has_convexity_distinction', 'has_hole_distinction', 'forms_open_vs_closed_distinction']
    
    # Spatial relationships
    spatial_predicates = ['intersects', 'contains', 'is_above', 'is_below', 'is_left_of', 'is_right_of',
                         'is_parallel', 'near_objects', 'near', 'adjacent_endpoints', 'shares_endpoint',
                         'forms_bridge_pattern', 'forms_apex', 'forms_t_junction', 'forms_x_junction']
    
    # Size and measurement predicates
    size_predicates = ['is_large', 'is_small', 'is_medium', 'is_tall', 'is_wide', 'larger_than', 'smaller_than',
                      'similar_size', 'aspect_sim', 'length_sim', 'has_length_ratio_imbalance']
    
    # Geometric and mathematical predicates
    geometric_predicates = ['has_angle_count_pattern', 'has_stroke_count_pattern', 'has_intersection_count_pattern',
                           'has_geometric_complexity_difference', 'has_compactness_difference', 'orientation_sim',
                           'curvature_sim', 'has_curvature_distinction']
    
    # Motif and pattern predicates
    motif_predicates = ['part_of_motif', 'motif_similarity', 'part_of', 'visual_similarity', 'vl_sim', 'clip_sim']
    
    # Commonsense and knowledge-based predicates
    commonsense_predicates = ['related_to', 'kb_sim', 'programmatic_sim', 'has_dominant_direction',
                             'has_tilted_orientation', 'has_asymmetric_base', 'has_apex_at_left']
    
    # Current system predicates from predicate miner
    miner_predicates = ['para', 'centroid_dist', 'global_stat_sim', 'stroke_count_sim', 'near_and_para',
                       'same_program_and_near']
    
    if predicate in shape_predicates:
        return 'shape_detection'
    elif predicate in symmetry_predicates:
        return 'symmetry'
    elif predicate in topology_predicates:
        return 'topology'
    elif predicate in spatial_predicates:
        return 'spatial'
    elif predicate in size_predicates:
        return 'size'
    elif predicate in geometric_predicates:
        return 'geometric'
    elif predicate in motif_predicates:
        return 'motif'
    elif predicate in commonsense_predicates:
        return 'commonsense'
    elif predicate in miner_predicates:
        return 'miner'
    else:
        return 'other'

def _create_summary_statistics(node_data, edge_data, summary_path):
    """Create summary statistics for the comprehensive analysis with discovered shape types"""
    summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_nodes': len(node_data),
        'total_edges': len(edge_data),
    }
    
    if node_data:
        # Shape type distribution (prioritize discovered types)
        shape_types = [node.get('shape_type', 'unknown') for node in node_data]
        summary['shape_type_distribution'] = dict(Counter(shape_types))
        
        # Discovered Bongard-LOGO shape type counts
        discovered_types = ['normal', 'circle', 'square', 'triangle', 'zigzag']
        summary['discovered_shape_counts'] = {
            shape: sum(1 for node in node_data if node.get('shape_type') == shape)
            for shape in discovered_types
        }
        
        # Shape type binary indicators summary
        summary['shape_type_indicators'] = {
            'normal_count': sum(node.get('is_normal_type', 0) for node in node_data),
            'circle_count': sum(node.get('is_circle_type', 0) for node in node_data),
            'square_count': sum(node.get('is_square_type', 0) for node in node_data),
            'triangle_count': sum(node.get('is_triangle_type', 0) for node in node_data),
            'zigzag_count': sum(node.get('is_zigzag_type', 0) for node in node_data)
        }
        
        # Topology distribution
        topology_types = [node.get('topology_type', 'unknown') for node in node_data]
        summary['topology_distribution'] = dict(Counter(topology_types))
        
        # Symmetry distribution
        symmetry_types = [node.get('symmetry_type', 'none') for node in node_data]
        summary['symmetry_distribution'] = dict(Counter(symmetry_types))
        
        # Curvature type distribution
        curvature_types = [node.get('curvature_type', 'unknown') for node in node_data]
        summary['curvature_distribution'] = dict(Counter(curvature_types))
        
        # Object type distribution
        object_types = [node.get('object_type', 'unknown') for node in node_data]
        summary['object_type_distribution'] = dict(Counter(object_types))
        
        # Average scores with safe extraction
        scores = ['regularity_score', 'complexity_level', 'complexity_score', 'confidence_score', 
                 'convexity_score', 'symmetry_score']
        for score in scores:
            values = []
            for node in node_data:
                val = node.get(score, 0)
                if val is not None and isinstance(val, (int, float)):
                    values.append(val)
            if values:
                summary[f'avg_{score}'] = float(np.mean(values))
                summary[f'std_{score}'] = float(np.std(values))
                summary[f'min_{score}'] = float(np.min(values))
                summary[f'max_{score}'] = float(np.max(values))
        
        # Geometric properties summary
        summary['geometric_properties'] = {
            'total_closed_shapes': sum(1 for node in node_data if node.get('is_closed', False)),
            'total_open_shapes': sum(1 for node in node_data if not node.get('is_closed', True)),
            'avg_vertex_count': np.mean([node.get('vertex_count', 0) for node in node_data]),
            'avg_area': np.mean([node.get('area', 0) for node in node_data if node.get('area', 0) > 0]),
            'avg_perimeter': np.mean([node.get('perimeter', 0) for node in node_data if node.get('perimeter', 0) > 0])
        }
    
    if edge_data:
        # Predicate distribution
        predicates = [edge.get('predicate', 'unknown') for edge in edge_data]
        summary['predicate_distribution'] = dict(Counter(predicates))
        
        # Predicate category distribution
        categories = [_categorize_predicate(edge.get('predicate', 'unknown')) for edge in edge_data]
        summary['predicate_category_distribution'] = dict(Counter(categories))
    
    # Save summary as JSON
    try:
        with open(summary_path.replace('.csv', '.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        logging.info(f"[Summary] Saved analysis summary to {summary_path.replace('.csv', '.json')}")
    except Exception as e:
        logging.error(f"[Summary] Error saving summary: {e}")

def visualize_shape_categories(scene_graph, output_path, puzzle_name=""):
    """Create comprehensive visualization showing shape categories and relationships"""
    
    plt.figure(figsize=(16, 12))
    
    # Create subplots for different aspects
    gs = plt.GridSpec(3, 3, figure=plt.gcf())
    
    # Main scene graph
    ax_main = plt.subplot(gs[:2, :2])
    _plot_comprehensive_scene_graph(scene_graph, ax_main, puzzle_name)
    
    # Shape distribution
    ax_shapes = plt.subplot(gs[0, 2])
    _plot_shape_distribution(scene_graph, ax_shapes)
    
    # Predicate distribution  
    ax_predicates = plt.subplot(gs[1, 2])
    _plot_predicate_distribution(scene_graph, ax_predicates)
    
    # Feature analysis
    ax_features = plt.subplot(gs[2, :])
    _plot_feature_analysis(scene_graph, ax_features)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"[Comprehensive Visualization] Saved to {output_path}")

def _plot_comprehensive_scene_graph(scene_graph, ax, puzzle_name):
    """Plot the main scene graph with comprehensive styling"""
    pos = nx.spring_layout(scene_graph, k=2, iterations=50)
    
    # Draw nodes with comprehensive styling
    for node_id, node_data in scene_graph.nodes(data=True):
        style = get_comprehensive_node_style(node_data)
        x, y = pos[node_id]
        
        # Draw node
        circle = plt.Circle((x, y), 0.05, 
                           color=style['color'], 
                           alpha=style['alpha'],
                           linewidth=style['border_width'])
        ax.add_patch(circle)
        
        # Add shape type label
        shape_type = style['shape_type']
        ax.text(x, y-0.08, shape_type, ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    # Draw edges with comprehensive styling
    for u, v, edge_data in scene_graph.edges(data=True):
        predicate = edge_data.get('predicate', 'unknown')
        style = get_comprehensive_edge_style(predicate)
        
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        ax.plot([x1, x2], [y1, y2], 
                color=style['color'], 
                linewidth=style['width'],
                linestyle='-', alpha=0.7)
        
        # Add predicate label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, predicate, ha='center', va='center', fontsize=6,
                bbox=dict(boxstyle="round,pad=0.1", facecolor='yellow', alpha=0.6))
    
    ax.set_title(f'Comprehensive Scene Graph: {puzzle_name}', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')

def _plot_shape_distribution(scene_graph, ax):
    """Plot distribution of shape types"""
    shape_types = []
    for _, node_data in scene_graph.nodes(data=True):
        semantic_features = node_data.get('semantic_features', {})
        shape_type = semantic_features.get('primary_shape_type', 'unknown')
        shape_types.append(shape_type)
    
    shape_counts = Counter(shape_types)
    if shape_counts:
        shapes, counts = zip(*shape_counts.most_common())
        colors = plt.cm.Set3(np.linspace(0, 1, len(shapes)))
        
        ax.pie(counts, labels=shapes, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Shape Distribution', fontweight='bold')

def _plot_predicate_distribution(scene_graph, ax):
    """Plot distribution of predicate types"""
    predicates = []
    for _, _, edge_data in scene_graph.edges(data=True):
        predicate = edge_data.get('predicate', 'unknown')
        predicates.append(_categorize_predicate(predicate))
    
    pred_counts = Counter(predicates)
    if pred_counts:
        categories, counts = zip(*pred_counts.most_common())
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(categories)))
        
        ax.bar(categories, counts, color=colors)
        ax.set_title('Predicate Categories', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

def _plot_feature_analysis(scene_graph, ax):
    """Plot feature analysis across all nodes"""
    features = ['regularity_score', 'curvature_score', 'complexity_score', 'confidence_score']
    feature_data = {feature: [] for feature in features}
    
    for _, node_data in scene_graph.nodes(data=True):
        semantic_features = node_data.get('semantic_features', {})
        for feature in features:
            value = semantic_features.get(feature, 0)
            feature_data[feature].append(value)
    
    # Create box plot
    data_to_plot = [feature_data[feature] for feature in features if feature_data[feature]]
    if data_to_plot:
        ax.boxplot(data_to_plot, labels=[f.replace('_', '\n') for f in features])
        ax.set_title('Feature Score Distributions', fontweight='bold')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)

def save_comprehensive_bongard_analysis(scene_graph, output_dir, puzzle_name="", image_path=None):
    """Save comprehensive Bongard analysis including visualizations and CSV exports"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comprehensive visualization
    viz_path = os.path.join(output_dir, f"{puzzle_name}_comprehensive_analysis.png")
    visualize_shape_categories(scene_graph, viz_path, puzzle_name)
    
    # Export comprehensive CSV data
    csv_path = os.path.join(output_dir, f"{puzzle_name}_comprehensive_data.csv")
    export_comprehensive_features_csv(scene_graph, csv_path, puzzle_name)
    
    # Create predicate analysis report
    report_path = os.path.join(output_dir, f"{puzzle_name}_predicate_analysis.json")
    create_predicate_analysis_report(scene_graph, report_path, puzzle_name)
    
    logging.info(f"[Comprehensive Analysis] Saved complete analysis for {puzzle_name} to {output_dir}")
    
    return {
        'visualization': viz_path,
        'csv_data': csv_path,
        'predicate_report': report_path
    }

def create_predicate_analysis_report(scene_graph, output_path, puzzle_name=""):
    """Create detailed predicate analysis report"""
    
    # Analyze predicate patterns
    predicate_analysis = {
        'puzzle_name': puzzle_name,
        'analysis_timestamp': datetime.now().isoformat(),
        'total_nodes': scene_graph.number_of_nodes(),
        'total_edges': scene_graph.number_of_edges(),
    }
    
    # Shape type analysis
    shape_types = []
    symmetry_types = []
    topology_types = []
    
    for _, node_data in scene_graph.nodes(data=True):
        semantic_features = node_data.get('semantic_features', {})
        shape_types.append(semantic_features.get('primary_shape_type', 'unknown'))
        symmetry_types.append(semantic_features.get('symmetry_type', 'none'))
        topology_types.append(semantic_features.get('topology_type', 'unknown'))
    
    predicate_analysis['shape_analysis'] = {
        'unique_shapes': len(set(shape_types)),
        'shape_distribution': dict(Counter(shape_types)),
        'most_common_shape': Counter(shape_types).most_common(1)[0] if shape_types else None,
        'symmetry_distribution': dict(Counter(symmetry_types)),
        'topology_distribution': dict(Counter(topology_types))
    }
    
    # Predicate relationship analysis
    predicates = []
    predicate_confidences = []
    
    for _, _, edge_data in scene_graph.edges(data=True):
        predicate = edge_data.get('predicate', 'unknown')
        confidence = edge_data.get('confidence', 0.0)
        predicates.append(predicate)
        predicate_confidences.append(confidence)
    
    if predicates:
        predicate_analysis['relationship_analysis'] = {
            'unique_predicates': len(set(predicates)),
            'predicate_distribution': dict(Counter(predicates)),
            'most_common_predicate': Counter(predicates).most_common(1)[0],
            'average_confidence': np.mean(predicate_confidences) if predicate_confidences else 0.0,
            'predicate_categories': dict(Counter([_categorize_predicate(p) for p in predicates]))
        }
    
    # Complexity analysis
    complexity_scores = []
    regularity_scores = []
    
    for _, node_data in scene_graph.nodes(data=True):
        semantic_features = node_data.get('semantic_features', {})
        complexity_scores.append(semantic_features.get('complexity_score', 0))
        regularity_scores.append(semantic_features.get('regularity_score', 0))
    
    if complexity_scores:
        predicate_analysis['complexity_analysis'] = {
            'average_complexity': np.mean(complexity_scores),
            'complexity_std': np.std(complexity_scores),
            'average_regularity': np.mean(regularity_scores),
            'regularity_std': np.std(regularity_scores),
            'complexity_range': [min(complexity_scores), max(complexity_scores)],
            'high_complexity_count': sum(1 for c in complexity_scores if c > 0.7)
        }
    
    # Save report
    try:
        with open(output_path, 'w') as f:
            json.dump(predicate_analysis, f, indent=2)
        logging.info(f"[Predicate Analysis] Saved report to {output_path}")
    except Exception as e:
        logging.error(f"[Predicate Analysis] Error saving report: {e}")
    
    return predicate_analysis

def get_edge_style(predicate):
    """Enhanced edge styling based on predicate type"""
    predicate_styles = {
        # Geometric relationships
        'is_above': {'color': 'red', 'style': '-', 'width': 2},
        'is_below': {'color': 'red', 'style': '-', 'width': 2},
        'is_left_of': {'color': 'blue', 'style': '-', 'width': 2},
        'is_right_of': {'color': 'blue', 'style': '-', 'width': 2},
        'contains': {'color': 'green', 'style': '-', 'width': 3},
        'intersects': {'color': 'orange', 'style': '--', 'width': 2},
        
        # Structural relationships
        'part_of': {'color': 'purple', 'style': '-', 'width': 2},
        'adjacent_endpoints': {'color': 'brown', 'style': ':', 'width': 1},
        'part_of_motif': {'color': 'gold', 'style': '-', 'width': 3},
        
        # Similarity relationships
        'same_shape_class': {'color': 'cyan', 'style': '--', 'width': 1},
        'visual_similarity': {'color': 'magenta', 'style': ':', 'width': 1},
        'forms_symmetry': {'color': 'lime', 'style': '-.', 'width': 2},
        
        # Pattern relationships
        'forms_bridge_pattern': {'color': 'navy', 'style': '-', 'width': 2},
        'forms_apex_pattern': {'color': 'maroon', 'style': '-', 'width': 2},
        'forms_symmetry_pattern': {'color': 'teal', 'style': '-', 'width': 2}
    }
    
    return predicate_styles.get(predicate, {'color': 'gray', 'style': '-', 'width': 1})

def get_node_style(data):
    """Enhanced node styling based on object properties and data completeness"""
    node_type = data.get('object_type', 'unknown')
    source = data.get('source', 'unknown')
    completeness = data.get('data_completeness_score', 0.5)
    
    # Base color by type
    color_map = {
        'line': 'lightcoral',
        'curve': 'lightblue', 
        'circle': 'lightgreen',
        'arc': 'lightyellow',
        'polygon': 'lightpink',
        'composite': 'lightgray',
        'motif': 'gold'
    }
    
    base_color = color_map.get(node_type, 'white')
    
    # Adjust alpha based on data completeness
    alpha = 0.3 + 0.7 * completeness  # Range from 0.3 to 1.0
    
    # Border color based on source
    border_colors = {
        'program': 'blue',
        'geometric_grouping': 'green', 
        'motif_mining': 'orange',
        'vision_language': 'purple'
    }
    border_color = border_colors.get(source, 'black')
    
    return {
        'color': base_color,
        'alpha': alpha,
        'border_color': border_color,
        'border_width': 2 if completeness > 0.8 else 1
    }

def _robust_edge_unpack(edges):
    """Yield (u, v, k, data) for all edge tuples, handling 2, 3, 4-item cases and non-dict data."""
    for edge in edges:
        if len(edge) == 4:
            u, v, k, data = edge
        elif len(edge) == 3:
            u, v, data = edge
            k = None
        elif len(edge) == 2:
            u, v = edge
            k = None
            data = {}
        else:
            logging.warning(f"[LOGO Visualization] Skipping unexpectedly short/long edge tuple: {edge}")
            continue
        if not isinstance(data, dict):
            logging.warning(f"[LOGO Visualization] Edge data not dict: {repr(data)} for edge {edge}; using empty dict.")
            data = {}
        yield u, v, k, data

def save_enhanced_scene_graph_visualization(G, image_path, out_dir, problem_id, 
                                          show_missing_data=True, show_patterns=True):
    """
    Save enhanced visualization with comprehensive node/edge styling, 
    missing data highlighting, and multi-view analysis
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Node positions: use centroid, else first vertex, else layout
    pos = {}
    for n, data in G.nodes(data=True):
        if 'centroid' in data and data['centroid'] is not None:
            try:
                centroid = data['centroid']
                if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
                    pos[n] = (float(centroid[0]), float(centroid[1]))
                else:
                    pos[n] = (0, 0)
            except (ValueError, TypeError):
                pos[n] = (0, 0)
        elif 'vertices' in data and data['vertices']:
            try:
                pos[n] = (float(data['vertices'][0][0]), float(data['vertices'][0][1]))
            except (IndexError, ValueError, TypeError):
                pos[n] = (0, 0)
        else:
            pos[n] = (0, 0)
    
    # Use spring layout for nodes without positions
    nodes_without_pos = [n for n, p in pos.items() if p == (0, 0)]
    if nodes_without_pos:
        spring_pos = nx.spring_layout(G.subgraph(nodes_without_pos), k=1, iterations=50)
        for n in nodes_without_pos:
            pos[n] = spring_pos[n]
    
    # Create enhanced visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()
    
    # 1. Full graph with all relationships
    ax = axes[0]
    ax.set_title(f"Complete Scene Graph - {problem_id}", fontsize=14, fontweight='bold')
    
    # Draw nodes with enhanced styling
    for n, data in G.nodes(data=True):
        style = get_node_style(data)
        x, y = pos[n]
        
        # Create fancy node shape
        bbox = FancyBboxPatch(
            (x-0.05, y-0.03), 0.1, 0.06,
            boxstyle="round,pad=0.01",
            facecolor=style['color'],
            edgecolor=style['border_color'],
            linewidth=style['border_width'],
            alpha=style['alpha']
        )
        ax.add_patch(bbox)
        
        # Node label with data quality indicator
        object_type = data.get('object_type', 'obj')
        completeness = data.get('data_completeness_score', 0.0)
        if show_missing_data and completeness < 0.5:
            label = f"{object_type}*"  # Mark incomplete data
        else:
            label = object_type
        
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw edges with enhanced styling
    for u, v, data in G.edges(data=True):
        predicate = data.get('predicate', 'unknown')
        style = get_edge_style(predicate)
        
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        ax.plot([x1, x2], [y1, y2], 
               color=style['color'], 
               linestyle=style['style'],
               linewidth=style['width'],
               alpha=0.7)
        
        # Add edge label for important relationships
        if predicate in ['part_of', 'contains', 'part_of_motif']:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, predicate[:4], fontsize=6, ha='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 2. Relationship network (filtered view)
    ax = axes[1]
    ax.set_title(f"Key Relationships - {problem_id}", fontsize=14, fontweight='bold')
    
    # Filter for important relationships
    important_predicates = ['part_of', 'contains', 'part_of_motif', 'forms_symmetry', 'visual_similarity']
    filtered_edges = [(u, v, data) for u, v, data in G.edges(data=True) 
                     if data.get('predicate', '') in important_predicates]
    
    # Draw filtered graph
    for n, data in G.nodes(data=True):
        style = get_node_style(data)
        x, y = pos[n]
        ax.scatter(x, y, c=style['color'], s=300, alpha=style['alpha'], 
                  edgecolors=style['border_color'], linewidth=style['border_width'])
        ax.text(x, y-0.08, data.get('object_type', 'obj')[:6], ha='center', fontsize=8)
    
    for u, v, data in filtered_edges:
        predicate = data.get('predicate', 'unknown')
        style = get_edge_style(predicate)
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color=style['color'], linewidth=style['width'], alpha=0.8)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 3. Missing data analysis
    ax = axes[2]
    ax.set_title(f"Data Completeness Analysis - {problem_id}", fontsize=14, fontweight='bold')
    
    if show_missing_data:
        completeness_scores = []
        node_types = []
        
        for n, data in G.nodes(data=True):
            completeness_scores.append(data.get('data_completeness_score', 0.0))
            node_types.append(data.get('object_type', 'unknown'))
        
        # Create bar chart of completeness by node type
        type_completeness = defaultdict(list)
        for ntype, score in zip(node_types, completeness_scores):
            type_completeness[ntype].append(score)
        
        types = list(type_completeness.keys())
        avg_scores = [np.mean(type_completeness[t]) for t in types]
        
        bars = ax.bar(types, avg_scores, alpha=0.7)
        
        # Color bars based on completeness
        for bar, score in zip(bars, avg_scores):
            if score < 0.3:
                bar.set_color('red')
            elif score < 0.7:
                bar.set_color('orange') 
            else:
                bar.set_color('green')
        
        ax.set_ylabel('Average Completeness Score')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # Add horizontal line at 0.5 (threshold)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()
    
    # 4. Pattern and motif hierarchy
    ax = axes[3]
    ax.set_title(f"Patterns & Motifs - {problem_id}", fontsize=14, fontweight='bold')
    
    if show_patterns:
        # Show only motif nodes and pattern relationships
        motif_nodes = [n for n, data in G.nodes(data=True) 
                      if data.get('object_type') == 'motif' or 'motif' in data.get('source', '')]
        pattern_edges = [(u, v, data) for u, v, data in G.edges(data=True)
                        if 'pattern' in data.get('predicate', '') or 'motif' in data.get('predicate', '')]
        
        # Draw motif hierarchy
        if motif_nodes:
            for n in motif_nodes:
                data = G.nodes[n]
                x, y = pos[n]
                ax.scatter(x, y, c='gold', s=500, alpha=0.8, edgecolors='orange', linewidth=3)
                
                member_count = data.get('motif_member_count', 0)
                ax.text(x, y, f"M{member_count}", ha='center', va='center', fontweight='bold')
        
        # Draw pattern connections
        for u, v, data in pattern_edges:
            if u in pos and v in pos:
                predicate = data.get('predicate', 'unknown')
                style = get_edge_style(predicate)
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                ax.plot([x1, x2], [y1, y2], color=style['color'], linewidth=3, alpha=0.8)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save enhanced visualization
    out_path = os.path.join(out_dir, f"{problem_id}_enhanced_scene_graph.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    logging.info(f"Saved enhanced scene graph visualization to {out_path}")
    return out_path

def create_multi_puzzle_visualization(csv_dir, out_dir, puzzle_pattern="*", max_puzzles=12):
    """
    Create comprehensive multi-puzzle visualization showing relationships across all images
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Find CSV files matching pattern
    node_files = glob.glob(os.path.join(csv_dir, f"{puzzle_pattern}_nodes.csv"))
    edge_files = glob.glob(os.path.join(csv_dir, f"{puzzle_pattern}_edges.csv"))
    
    if not node_files or not edge_files:
        logging.warning(f"No CSV files found matching pattern {puzzle_pattern} in {csv_dir}")
        return None
    
    # Load data from multiple puzzles
    all_graphs = {}
    puzzle_stats = {}
    
    for i, (node_file, edge_file) in enumerate(zip(node_files[:max_puzzles], edge_files[:max_puzzles])):
        try:
            puzzle_id = os.path.basename(node_file).replace('_nodes.csv', '')
            
            # Load nodes and edges
            nodes_df = pd.read_csv(node_file)
            edges_df = pd.read_csv(edge_file)
            
            # Create graph
            G = nx.MultiDiGraph()
            
            # Add nodes
            for _, row in nodes_df.iterrows():
                node_data = row.to_dict()
                # Handle NaN values
                for key, value in node_data.items():
                    if pd.isna(value):
                        node_data[key] = None
                G.add_node(row['object_id'], **node_data)
            
            # Add edges  
            for _, row in edges_df.iterrows():
                edge_data = row.to_dict()
                for key, value in edge_data.items():
                    if pd.isna(value):
                        edge_data[key] = None
                G.add_edge(row['source'], row['target'], **edge_data)
            
            all_graphs[puzzle_id] = G
            
            # Collect stats
            puzzle_stats[puzzle_id] = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'avg_completeness': np.mean([data.get('data_completeness_score', 0.0) 
                                           for _, data in G.nodes(data=True)]),
                'predicates': len(set([data.get('predicate', 'unknown') 
                                     for _, _, data in G.edges(data=True)]))
            }
            
        except Exception as e:
            logging.error(f"Error loading puzzle {puzzle_id}: {e}")
            continue
    
    if not all_graphs:
        logging.error("No graphs were successfully loaded")
        return None
    
    # Create multi-puzzle visualization
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.ravel()
    
    for i, (puzzle_id, G) in enumerate(list(all_graphs.items())[:12]):
        ax = axes[i]
        
        try:
            # Create layout
            pos = nx.spring_layout(G, k=1, iterations=30)
            
            # Draw nodes
            node_colors = []
            node_sizes = []
            for n, data in G.nodes(data=True):
                completeness = data.get('data_completeness_score', 0.5)
                node_colors.append(completeness)
                node_sizes.append(100 + 200 * completeness)
            
            nx.draw_networkx_nodes(G, pos, ax=ax, 
                                 node_color=node_colors, 
                                 node_size=node_sizes,
                                 cmap='RdYlGn', vmin=0, vmax=1, alpha=0.8)
            
            # Draw edges
            edge_colors = []
            for _, _, data in G.edges(data=True):
                predicate = data.get('predicate', 'unknown')
                if 'pattern' in predicate or 'motif' in predicate:
                    edge_colors.append('red')
                elif predicate in ['part_of', 'contains']:
                    edge_colors.append('blue')
                else:
                    edge_colors.append('gray')
            
            nx.draw_networkx_edges(G, pos, ax=ax, 
                                 edge_color=edge_colors, alpha=0.6, width=0.5)
            
            # Title with stats
            stats = puzzle_stats[puzzle_id]
            ax.set_title(f"{puzzle_id}\n{stats['nodes']}N, {stats['edges']}E, {stats['avg_completeness']:.2f}C",
                        fontsize=10)
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {puzzle_id}", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(all_graphs), 12):
        axes[i].axis('off')
    
    plt.suptitle(f"Multi-Puzzle Scene Graph Overview ({len(all_graphs)} puzzles)", fontsize=16)
    plt.tight_layout()
    
    # Save visualization
    out_path = os.path.join(out_dir, f"multi_puzzle_overview_{puzzle_pattern}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    logging.info(f"Saved multi-puzzle visualization to {out_path}")
    return out_path

def analyze_puzzle_completeness(csv_dir, puzzle_pattern="*"):
    """
    Analyze data completeness across puzzles and generate report
    """
    node_files = glob.glob(os.path.join(csv_dir, f"{puzzle_pattern}_nodes.csv"))
    
    if not node_files:
        logging.warning(f"No node CSV files found matching pattern {puzzle_pattern}")
        return None
    
    completeness_analysis = {
        'puzzles_analyzed': len(node_files),
        'field_completeness': defaultdict(list),
        'puzzle_scores': {},
        'missing_patterns': defaultdict(int)
    }
    
    for node_file in node_files:
        try:
            puzzle_id = os.path.basename(node_file).replace('_nodes.csv', '')
            nodes_df = pd.read_csv(node_file)
            
            # Analyze field completeness
            field_scores = {}
            for col in nodes_df.columns:
                if col == 'object_id':
                    continue
                
                non_null_count = nodes_df[col].notna().sum()
                non_zero_count = (nodes_df[col] != 0).sum() if nodes_df[col].dtype in ['int64', 'float64'] else non_null_count
                
                completeness = non_zero_count / len(nodes_df) if len(nodes_df) > 0 else 0
                field_scores[col] = completeness
                completeness_analysis['field_completeness'][col].append(completeness)
            
            # Overall puzzle score
            puzzle_score = np.mean(list(field_scores.values()))
            completeness_analysis['puzzle_scores'][puzzle_id] = puzzle_score
            
            # Identify missing patterns
            for col, score in field_scores.items():
                if score < 0.1:  # Less than 10% data
                    completeness_analysis['missing_patterns'][col] += 1
                    
        except Exception as e:
            logging.error(f"Error analyzing {node_file}: {e}")
    
    # Generate summary statistics
    summary = {
        'avg_field_completeness': {
            field: np.mean(scores) 
            for field, scores in completeness_analysis['field_completeness'].items()
        },
        'worst_fields': sorted(
            completeness_analysis['field_completeness'].items(),
            key=lambda x: np.mean(x[1])
        )[:10],
        'best_puzzles': sorted(
            completeness_analysis['puzzle_scores'].items(),
            key=lambda x: x[1], reverse=True
        )[:5],
        'worst_puzzles': sorted(
            completeness_analysis['puzzle_scores'].items(),
            key=lambda x: x[1]
        )[:5]
    }
    
    return completeness_analysis, summary

def create_missing_data_analysis(csv_dir, out_dir, puzzle_pattern="*"):
    """
    Create comprehensive missing data analysis visualization
    """
    os.makedirs(out_dir, exist_ok=True)
    
    analysis, summary = analyze_puzzle_completeness(csv_dir, puzzle_pattern)
    if not analysis:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Field completeness heatmap
    ax = axes[0, 0]
    field_names = list(summary['avg_field_completeness'].keys())[:20]  # Top 20 fields
    field_scores = [summary['avg_field_completeness'][f] for f in field_names]
    
    bars = ax.barh(field_names, field_scores)
    
    # Color bars based on completeness
    for bar, score in zip(bars, field_scores):
        if score < 0.3:
            bar.set_color('red')
        elif score < 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    ax.set_xlabel('Average Completeness Score')
    ax.set_title('Field Completeness Across All Puzzles')
    ax.set_xlim(0, 1)
    
    # 2. Puzzle quality distribution
    ax = axes[0, 1]
    puzzle_scores = list(analysis['puzzle_scores'].values())
    ax.hist(puzzle_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(np.mean(puzzle_scores), color='red', linestyle='--', label=f'Mean: {np.mean(puzzle_scores):.3f}')
    ax.set_xlabel('Puzzle Completeness Score')
    ax.set_ylabel('Number of Puzzles')
    ax.set_title('Distribution of Puzzle Quality Scores')
    ax.legend()
    
    # 3. Missing data patterns
    ax = axes[1, 0]
    missing_fields = list(analysis['missing_patterns'].keys())[:15]
    missing_counts = [analysis['missing_patterns'][f] for f in missing_fields]
    
    bars = ax.bar(range(len(missing_fields)), missing_counts, color='red', alpha=0.7)
    ax.set_xticks(range(len(missing_fields)))
    ax.set_xticklabels(missing_fields, rotation=45, ha='right')
    ax.set_ylabel('Number of Puzzles with Missing Data')
    ax.set_title('Most Frequently Missing Fields')
    
    # 4. Best vs Worst puzzles
    ax = axes[1, 1]
    best_puzzles = [p[0] for p in summary['best_puzzles']]
    best_scores = [p[1] for p in summary['best_puzzles']]
    worst_puzzles = [p[0] for p in summary['worst_puzzles']]
    worst_scores = [p[1] for p in summary['worst_puzzles']]
    
    x = range(5)
    ax.bar([i - 0.2 for i in x], best_scores, width=0.4, label='Best', color='green', alpha=0.7)
    ax.bar([i + 0.2 for i in x], worst_scores, width=0.4, label='Worst', color='red', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(best_puzzles, rotation=45, ha='right')
    ax.set_ylabel('Completeness Score')
    ax.set_title('Best vs Worst Puzzle Quality')
    ax.legend()
    
    plt.tight_layout()
    
    # Save analysis
    out_path = os.path.join(out_dir, f"missing_data_analysis_{puzzle_pattern}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Save analysis report
    report_path = os.path.join(out_dir, f"completeness_report_{puzzle_pattern}.json")
    import json
    with open(report_path, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        json_summary = {}
        for key, value in summary.items():
            if isinstance(value, dict):
                json_summary[key] = {str(k): float(v) if isinstance(v, (np.float64, np.float32)) else v 
                                   for k, v in value.items()}
            else:
                json_summary[key] = value
        json.dump(json_summary, f, indent=2)
    
    logging.info(f"Saved missing data analysis to {out_path}")
    logging.info(f"Saved completeness report to {report_path}")
    
    return out_path, report_path

def save_scene_graph_visualization(G, image_path, out_dir, problem_id, abstract_view=True):
    """Save a visualization of the scene graph G overlaid on the real image."""
    logging.info(f"[save_scene_graph_visualization] Called for {problem_id}, image_path={image_path}, out_dir={out_dir}, abstract_view={abstract_view}")
    logging.info(f"[save_scene_graph_visualization] Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    logging.info(f"[save_scene_graph_visualization] Current working directory: {os.getcwd()}")
    logging.info(f"[save_scene_graph_visualization] Output directory absolute path: {os.path.abspath(out_dir)}")
    logging.info(f"[save_scene_graph_visualization] Output directory exists: {os.path.exists(out_dir)}")
    
    os.makedirs(out_dir, exist_ok=True)
    # Node positions: use centroid, else first vertex, else (0,0)
    pos = {}
    for n, data in G.nodes(data=True):
        if 'centroid' in data and data['centroid'] is not None:
            pos[n] = data['centroid']
        elif 'vertices' in data and data['vertices']:
            pos[n] = data['vertices'][0]
        else:
            pos[n] = (0, 0)

    # For visualization: if MultiDiGraph, convert to DiGraph for plotting edge labels
    # This simplifies the visualization by showing one edge between nodes, labeled with all predicates.
    plotG = nx.DiGraph()
    if hasattr(G, 'is_multigraph') and G.is_multigraph():
        plotG.add_nodes_from(G.nodes(data=True))
        # Aggregate edges
        edge_pred_map = defaultdict(list)
        for u, v, data in G.edges(data=True):
            pred = data.get('predicate', 'unknown')
            edge_pred_map[(u, v)].append(pred)
        
        for (u, v), preds in edge_pred_map.items():
            plotG.add_edge(u, v, label=', '.join(sorted(list(set(preds)))))
    else:
        plotG.add_nodes_from(G.nodes(data=True))
        for u, v, data in G.edges(data=True):
            plotG.add_edge(u, v, label=data.get('predicate', 'unknown'))

    # --- Node color/border/label logic ---
    node_labels = {}
    node_colors = []
    node_border_colors = []
    for n, data in plotG.nodes(data=True):
        label = data.get('object_type', 'obj')
        if data.get('source') == 'geometric_grouping':
            node_colors.append('skyblue' if not data.get('is_closed') else 'lightgreen')
            node_border_colors.append('blue')
            label += f"\n(strokes: {data.get('stroke_count', 1)})"
        else: # Primitive
            node_colors.append('lightcoral')
            node_border_colors.append('red')
        node_labels[n] = label

    # --- Edge color/label logic ---
    edge_labels = nx.get_edge_attributes(plotG, 'label')
    edge_colors = []
    color_map = {
        'part_of': 'orange',
        'adjacent_endpoints': 'blue',
        'is_above': 'purple',
        'is_parallel': 'green',
        'contains': 'brown',
        'intersects': 'red',
    }
    for u, v, data in plotG.edges(data=True):
        pred = data.get('label', '').split(',')[0] # Color by first predicate
        edge_colors.append(color_map.get(pred, 'gray'))

    # Filter out labels for self-loops or edges between co-located nodes to prevent drawing errors
    drawable_edge_labels = {
        (u, v): label for (u, v), label in edge_labels.items()
        if u != v and pos.get(u) != pos.get(v)
    }
    if len(drawable_edge_labels) < len(edge_labels):
        logging.warning(f"Skipped {len(edge_labels) - len(drawable_edge_labels)} edge labels for self-loops or co-located nodes in {problem_id}.")

    # 1. Save graph overlayed on real image (if image exists)
    logging.info(f"[save_scene_graph_visualization] Checking image path: {image_path}")
    if image_path and os.path.exists(image_path):
        logging.info(f"[save_scene_graph_visualization] Image exists, creating overlay visualization")
        try:
            with Image.open(image_path) as img:
                plt.figure(figsize=(12, 12))
                plt.imshow(img)
                nx.draw_networkx_nodes(plotG, pos, node_color=node_colors, node_size=500, edgecolors=node_border_colors, linewidths=1.5)
                nx.draw_networkx_edges(plotG, pos, edge_color=edge_colors, arrows=True, arrowstyle='-|>', width=1.5, alpha=0.8)
                nx.draw_networkx_labels(plotG, pos, labels=node_labels, font_size=7)
                nx.draw_networkx_edge_labels(plotG, pos, edge_labels=drawable_edge_labels, font_color='black', font_size=6, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.1))
                plt.title(f"Full Scene Graph - {problem_id}")
                plt.axis('on')
                out_path_full = os.path.join(out_dir, f"{problem_id}_scene_graph.png")
                plt.savefig(out_path_full, bbox_inches='tight', dpi=150)
                plt.close()
                logging.info(f"[save_scene_graph_visualization] Saved overlay visualization to {out_path_full}")
        except Exception as e:
            logging.error(f"Failed to create full visualization for {problem_id}: {e}")
    else:
        logging.info(f"[save_scene_graph_visualization] Image not available, skipping overlay visualization")

    # 2. Save pure matplotlib graph visualization (no image)
    logging.info(f"[save_scene_graph_visualization] Creating graph-only visualization")
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(plotG, pos, node_color=node_colors, node_size=800, edgecolors=node_border_colors, linewidths=1.5)
    nx.draw_networkx_edges(plotG, pos, edge_color=edge_colors, arrows=True, arrowstyle='-|>', width=1.5, alpha=0.8)
    nx.draw_networkx_labels(plotG, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edge_labels(plotG, pos, edge_labels=drawable_edge_labels, font_color='black', font_size=7)
    plt.title(f"Graph Only - {problem_id}")
    plt.axis('off')
    out_path_graph = os.path.join(out_dir, f"{problem_id}_graph_only.png")
    logging.info(f"[save_scene_graph_visualization] Attempting to save graph to: {out_path_graph}")
    try:
        plt.savefig(out_path_graph, bbox_inches='tight')
        plt.close()
        
        # Verify the file was created and get its size
        if os.path.exists(out_path_graph):
            file_size = os.path.getsize(out_path_graph)
            logging.info(f"[save_scene_graph_visualization] Saved graph-only visualization to {out_path_graph} (size: {file_size} bytes)")
        else:
            logging.error(f"[save_scene_graph_visualization] File was not created at {out_path_graph}")
    except Exception as save_error:
        plt.close()  # Ensure we close the figure even if save fails
        logging.error(f"[save_scene_graph_visualization] Failed to save graph: {save_error}")
        import traceback
        logging.error(f"[save_scene_graph_visualization] Save error traceback: {traceback.format_exc()}")
        raise

    # 3. Save abstract graph view (if enabled and applicable)
    logging.info(f"[save_scene_graph_visualization] Abstract view enabled: {abstract_view}")
    if abstract_view:
        abstract_nodes = [n for n, data in G.nodes(data=True) if data.get('source') == 'geometric_grouping']
        logging.info(f"[save_scene_graph_visualization] Found {len(abstract_nodes)} abstract nodes")
        if abstract_nodes:
            abstract_G = G.subgraph(abstract_nodes).copy()
            
            # Since we are creating a new figure, we need to redefine plot elements
            abstract_pos = {n: pos[n] for n in abstract_nodes}
            abstract_node_colors = ['lightgreen' if data.get('is_closed') else 'skyblue' for n, data in abstract_G.nodes(data=True)]
            abstract_node_labels = {n: f"{data.get('object_type')}\n(strokes: {data.get('stroke_count', 1)})" for n, data in abstract_G.nodes(data=True)}
            
            # Aggregate edges for abstract view
            abstract_edge_map = defaultdict(list)
            for u, v, data in abstract_G.edges(data=True):
                abstract_edge_map[(u,v)].append(data.get('predicate', 'unknown'))
            
            abstract_plot_G = nx.DiGraph()
            abstract_plot_G.add_nodes_from(abstract_G.nodes(data=True))
            for (u,v), preds in abstract_edge_map.items():
                abstract_plot_G.add_edge(u, v, label=', '.join(sorted(list(set(preds)))))

            abstract_edge_labels = nx.get_edge_attributes(abstract_plot_G, 'label')
            abstract_edge_colors = [color_map.get(l.split(',')[0], 'gray') for l in abstract_edge_labels.values()]

            plt.figure(figsize=(8, 8))
            nx.draw_networkx_nodes(abstract_plot_G, abstract_pos, node_color=abstract_node_colors, node_size=1200, edgecolors='black', linewidths=1.5)
            nx.draw_networkx_edges(abstract_plot_G, abstract_pos, edge_color=abstract_edge_colors, arrows=True, arrowstyle='-|>', width=1.5, alpha=0.9)
            nx.draw_networkx_labels(abstract_plot_G, abstract_pos, labels=abstract_node_labels, font_size=8)
            nx.draw_networkx_edge_labels(abstract_plot_G, abstract_pos, edge_labels=abstract_edge_labels, font_color='red', font_size=7)
            
            plt.title(f"Abstract View - {problem_id}")
            plt.axis('off')
            out_path_abstract = os.path.join(out_dir, f"{problem_id}_graph_abstract.png")
            plt.savefig(out_path_abstract, bbox_inches='tight')
            plt.close()
            logging.info(f"[save_scene_graph_visualization] Saved abstract visualization to {out_path_abstract}")
        else:
            logging.info(f"[save_scene_graph_visualization] No abstract nodes found, skipping abstract view")
    
    logging.info(f"[save_scene_graph_visualization] Completed visualization for {problem_id}")

def save_scene_graph_csv(G, out_dir, problem_id):
    """Save node and edge data of the scene graph as CSV files."""
    logging.info(f"[LOGO Visualization] save_scene_graph_csv called for problem_id={problem_id}")
    import json
    import numpy as np

    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        else:
            return obj

    nodes = []
    for n, data in G.nodes(data=True):
        d = sanitize(data.copy())
        d['id'] = n
        d['all_data'] = json.dumps(d, ensure_ascii=False)
        nodes.append(d)
    edges = []
    logging.info(f"[LOGO Visualization] (CSV) Graph type: {type(G)}. is_multigraph={isinstance(G, (nx.MultiDiGraph, nx.MultiGraph))}")
    try:
        if hasattr(G, 'is_multigraph') and G.is_multigraph():
            for u, v, k, data in _robust_edge_unpack(G.edges(keys=True, data=True)):
                d = sanitize(data.copy())
                d['source'] = u
                d['target'] = v
                d['key'] = k
                d['all_data'] = json.dumps(d, ensure_ascii=False)
                edges.append(d)
        else:
            for u, v, k, data in _robust_edge_unpack(G.edges(data=True)):
                d = sanitize(data.copy())
                d['source'] = u
                d['target'] = v
                d['all_data'] = json.dumps(d, ensure_ascii=False)
                edges.append(d)
    except Exception as e:
        logging.error(f"[LOGO Visualization] Error unpacking edges for CSV: {e}")
        logging.error(f"[LOGO Visualization] Edges: {list(G.edges(data=True))}")
        raise
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(nodes).to_csv(os.path.join(out_dir, f"{problem_id}_nodes.csv"), index=False)
    if edges:
        pd.DataFrame(edges).to_csv(os.path.join(out_dir, f"{problem_id}_edges.csv"), index=False)
    else:
        logging.warning(f"[LOGO Visualization] No edges found for problem {problem_id}, skipping edge CSV.")
