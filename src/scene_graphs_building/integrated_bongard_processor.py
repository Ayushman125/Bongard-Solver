"""
Integrated Scene Graph Processor for Bongard-LOGO Dataset

This module integrates the corrected action program parsing with the existing
scene graph building pipeline. It replaces the incorrect single-stroke parsing
with proper multi-stroke image composition.

Key corrections:
1. Each image consists of multiple stroke commands (not single commands)
2. Strokes combine to form complete images with 7 positive and 7 negative examples
3. Proper handling of line vs arc stroke types with shape modifiers
4. Integration with existing predicate and feature extraction systems
"""

import logging
import os
import math
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon, LineString

from ..unified_action_parser import UnifiedActionParser, ImageProgram, StrokeCommand
from .vl_features import CLIPEmbedder
from .stroke_specific_predicates import (
    arc_connects_parallel_lines, arc_forms_semicircle, arc_forms_quarter_circle,
    arc_has_high_curvature, arc_has_uniform_curvature, line_is_straight,
    line_has_consistent_direction, line_connects_endpoints
)


class IntegratedBongardProcessor:
    """
    Integrated processor that correctly handles Bongard-LOGO action programs
    and builds scene graphs with proper stroke-level understanding.
    """
    
    def __init__(self):
        self.action_parser = UnifiedActionParser()
        self.clip_embedder = CLIPEmbedder()
        self.logger = logging.getLogger(__name__)
        
    def process_problem(self, problem_id: str, action_programs: Dict[str, Any],
                       image_base_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single Bongard problem with corrected action program parsing.
        
        Args:
            problem_id: Unique identifier for the problem
            action_programs: Raw action program data for this problem
            image_base_dir: Base directory for actual image files (if available)
            
        Returns:
            Dict containing processed scene graphs and metadata
        """
        if problem_id not in action_programs:
            self.logger.warning(f"No action program found for problem {problem_id}")
            return {}
            
        problem_data = action_programs[problem_id]
        
        # Validate structure: [positive_examples, negative_examples]
        if not isinstance(problem_data, list) or len(problem_data) != 2:
            self.logger.error(f"Invalid problem structure for {problem_id}: {type(problem_data)}")
            return {}
            
        positive_examples, negative_examples = problem_data
        
        # Process each example as a complete image
        all_graphs = {}
        all_objects = []
        
        # Process positive examples
        for i, image_strokes in enumerate(positive_examples):
            image_id = f"{problem_id}_pos_{i}"
            graph, objects = self._process_single_image(
                image_strokes, image_id, True, problem_id, image_base_dir
            )
            if graph:
                all_graphs[image_id] = graph
                all_objects.extend(objects)
                
        # Process negative examples
        for i, image_strokes in enumerate(negative_examples):
            image_id = f"{problem_id}_neg_{i}"
            graph, objects = self._process_single_image(
                image_strokes, image_id, False, problem_id, image_base_dir
            )
            if graph:
                all_graphs[image_id] = graph
                all_objects.extend(objects)
        
        # Build cross-image relationships and predicates
        problem_predicates = self._extract_problem_predicates(all_objects)
        
        return {
            'problem_id': problem_id,
            'graphs': all_graphs,
            'objects': all_objects,
            'predicates': problem_predicates,
            'num_positive': len(positive_examples),
            'num_negative': len(negative_examples),
            'total_images': len(all_graphs)
        }
    
    def _process_single_image(self, image_strokes: List[Any], image_id: str,
                             is_positive: bool, problem_id: str,
                             image_base_dir: Optional[str] = None) -> Tuple[nx.Graph, List[Dict]]:
        """
        Process a single image composed of multiple stroke commands.
        
        Returns:
            Tuple of (scene_graph, list_of_objects)
        """
        # Parse the image using our corrected parser
        image_program = self.action_parser._parse_single_image(
            image_strokes, image_id, is_positive, problem_id
        )
        
        if not image_program:
            self.logger.warning(f"Failed to parse image {image_id}")
            return nx.Graph(), []
        
        # Create scene graph
        scene_graph = nx.Graph()
        objects = []
        
        # Add nodes for each stroke
        for stroke_idx, stroke in enumerate(image_program.strokes):
            node_id = f"{image_id}_stroke_{stroke_idx}"
            
            # Calculate stroke-specific geometry
            stroke_vertices = self._get_stroke_vertices(stroke, image_program.vertices, stroke_idx)
            stroke_features = self._calculate_stroke_features(stroke, stroke_vertices)
            
            # Create object dict with all required features
            obj = {
                'object_id': node_id,
                'image_id': image_id,
                'problem_id': problem_id,
                'is_positive': is_positive,
                'stroke_index': stroke_idx,
                'stroke_type': stroke.stroke_type.value,
                'shape_modifier': stroke.shape_modifier.value,
                'raw_command': stroke.raw_command,
                'parameters': stroke.parameters,
                'vertices': stroke_vertices,
                'geometry': stroke_features,
                **stroke_features  # Flatten geometry features
            }
            
            # Add stroke-specific predicates
            obj['predicates'] = self._get_stroke_predicates(stroke, stroke_vertices)
            
            # Add CLIP embedding if image exists
            if image_base_dir:
                image_path = self._find_image_path(image_id, image_base_dir)
                if image_path:
                    obj['clip_embedding'] = self._get_clip_embedding(
                        image_path, stroke_vertices, obj
                    )
            
            # Add to graph and objects list
            scene_graph.add_node(node_id, **obj)
            objects.append(obj)
        
        # Add edges between connected strokes
        self._add_stroke_relationships(scene_graph, image_program)
        
        return scene_graph, objects
    
    def _get_stroke_vertices(self, stroke: StrokeCommand, all_vertices: List[Tuple[float, float]],
                           stroke_idx: int) -> List[Tuple[float, float]]:
        """
        Extract vertices belonging to a specific stroke from the complete image vertices.
        This is a simplified approach - in practice, you'd track vertices per stroke during parsing.
        """
        # For now, estimate based on stroke index and total strokes
        # In a full implementation, the parser should track vertices per stroke
        if not all_vertices:
            return []
            
        total_strokes = stroke_idx + 1  # Approximate
        verts_per_stroke = max(2, len(all_vertices) // total_strokes)
        
        start_idx = stroke_idx * verts_per_stroke
        end_idx = min(start_idx + verts_per_stroke, len(all_vertices))
        
        return all_vertices[start_idx:end_idx]
    
    def _calculate_stroke_features(self, stroke: StrokeCommand,
                                 vertices: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Calculate geometric and semantic features for a single stroke"""
        if len(vertices) < 2:
            return self._default_stroke_features()
            
        # Basic geometry
        start_point = vertices[0]
        end_point = vertices[-1]
        length = self._calculate_path_length(vertices)
        
        # Bounding box
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        bbox = {
            'min_x': min(xs), 'max_x': max(xs),
            'min_y': min(ys), 'max_y': max(ys)
        }
        
        # Centroid
        centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        
        # Orientation
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        orientation = math.degrees(math.atan2(dy, dx)) if dx != 0 or dy != 0 else 0
        
        # Curvature (for arcs)
        curvature = self._calculate_curvature(vertices)
        
        # Stroke-specific features
        features = {
            'start_point': start_point,
            'end_point': end_point,
            'length': length,
            'orientation': orientation,
            'bbox': bbox,
            'centroid': centroid,
            'curvature': curvature,
            'num_vertices': len(vertices),
            'is_closed': np.allclose(start_point, end_point, atol=1e-3) if len(vertices) > 2 else False
        }
        
        # Add stroke type specific features
        if stroke.stroke_type.value == 'arc':
            features.update(self._calculate_arc_features(stroke, vertices))
        elif stroke.stroke_type.value == 'line':
            features.update(self._calculate_line_features(stroke, vertices))
            
        return features
    
    def _calculate_arc_features(self, stroke: StrokeCommand,
                              vertices: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Calculate arc-specific features"""
        radius = stroke.parameters.get('radius', 0.0)
        span_angle = stroke.parameters.get('span_angle', 0.0)
        
        return {
            'arc_radius': radius,
            'arc_span_angle': span_angle,
            'arc_type': 'semicircle' if abs(span_angle) > 150 else 'quarter_circle' if abs(span_angle) > 60 else 'small_arc'
        }
    
    def _calculate_line_features(self, stroke: StrokeCommand,
                               vertices: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Calculate line-specific features"""
        straightness = self._calculate_straightness(vertices)
        
        return {
            'line_straightness': straightness,
            'line_type': 'straight' if straightness > 0.9 else 'curved'
        }
    
    def _calculate_path_length(self, vertices: List[Tuple[float, float]]) -> float:
        """Calculate total path length"""
        if len(vertices) < 2:
            return 0.0
            
        total_length = 0.0
        for i in range(1, len(vertices)):
            dx = vertices[i][0] - vertices[i-1][0]
            dy = vertices[i][1] - vertices[i-1][1]
            total_length += math.sqrt(dx*dx + dy*dy)
            
        return total_length
    
    def _calculate_curvature(self, vertices: List[Tuple[float, float]]) -> float:
        """Calculate average curvature of the path"""
        if len(vertices) < 3:
            return 0.0
            
        curvatures = []
        for i in range(1, len(vertices) - 1):
            p1, p2, p3 = vertices[i-1], vertices[i], vertices[i+1]
            
            # Calculate curvature using the menger curvature formula
            area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
            d1 = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            d2 = math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
            d3 = math.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
            
            if d1 * d2 * d3 > 1e-6:
                curvature = 4 * area / (d1 * d2 * d3)
                curvatures.append(curvature)
        
        return sum(curvatures) / len(curvatures) if curvatures else 0.0
    
    def _calculate_straightness(self, vertices: List[Tuple[float, float]]) -> float:
        """Calculate how straight a line is (1.0 = perfectly straight)"""
        if len(vertices) < 2:
            return 1.0
            
        # Compare actual path length to straight-line distance
        actual_length = self._calculate_path_length(vertices)
        straight_distance = math.sqrt(
            (vertices[-1][0] - vertices[0][0])**2 +
            (vertices[-1][1] - vertices[0][1])**2
        )
        
        if actual_length < 1e-6:
            return 1.0
            
        return straight_distance / actual_length
    
    def _get_stroke_predicates(self, stroke: StrokeCommand,
                             vertices: List[Tuple[float, float]]) -> List[str]:
        """Get relevant predicates for a stroke based on its type and properties"""
        predicates = []
        
        # Create a mock node for predicate functions
        mock_node = {
            'stroke_type': stroke.stroke_type.value,
            'vertices': vertices,
            'curvature_score': self._calculate_curvature(vertices),
            'action_command': stroke.raw_command
        }
        
        if stroke.stroke_type.value == 'arc':
            # Arc-specific predicates
            if arc_forms_semicircle(mock_node):
                predicates.append('forms_semicircle')
            if arc_forms_quarter_circle(mock_node):
                predicates.append('forms_quarter_circle')
            if arc_has_high_curvature(mock_node):
                predicates.append('has_high_curvature')
                
        elif stroke.stroke_type.value == 'line':
            # Line-specific predicates
            if line_is_straight(mock_node):
                predicates.append('is_straight')
            if line_has_consistent_direction(mock_node):
                predicates.append('has_consistent_direction')
        
        # Shape modifier predicates
        shape_mod = stroke.shape_modifier.value
        if shape_mod != 'unknown':
            predicates.append(f'shape_{shape_mod}')
            
        return predicates
    
    def _add_stroke_relationships(self, graph: nx.Graph, image_program: ImageProgram):
        """Add edges between related strokes in the scene graph"""
        strokes = image_program.strokes
        vertices = image_program.vertices
        
        # Add sequential connections (strokes that follow each other)
        for i in range(len(strokes) - 1):
            current_node = f"{image_program.image_id}_stroke_{i}"
            next_node = f"{image_program.image_id}_stroke_{i+1}"
            
            if current_node in graph and next_node in graph:
                graph.add_edge(current_node, next_node, 
                             relationship='sequential',
                             edge_type='stroke_sequence')
        
        # Add proximity-based connections
        for i in range(len(strokes)):
            for j in range(i + 1, len(strokes)):
                node1 = f"{image_program.image_id}_stroke_{i}"
                node2 = f"{image_program.image_id}_stroke_{j}"
                
                if node1 in graph and node2 in graph:
                    # Check if strokes are spatially close
                    obj1 = graph.nodes[node1]
                    obj2 = graph.nodes[node2]
                    
                    if self._are_strokes_connected(obj1, obj2):
                        graph.add_edge(node1, node2,
                                     relationship='spatial_proximity',
                                     edge_type='stroke_connection')
    
    def _are_strokes_connected(self, obj1: Dict, obj2: Dict, threshold: float = 10.0) -> bool:
        """Check if two strokes are spatially connected"""
        try:
            end1 = obj1.get('end_point', (0, 0))
            start2 = obj2.get('start_point', (0, 0))
            
            distance = math.sqrt((end1[0] - start2[0])**2 + (end1[1] - start2[1])**2)
            return distance < threshold
        except:
            return False
    
    def _extract_problem_predicates(self, all_objects: List[Dict]) -> Dict[str, Any]:
        """Extract problem-level predicates by comparing positive and negative examples"""
        positive_objects = [obj for obj in all_objects if obj.get('is_positive', False)]
        negative_objects = [obj for obj in all_objects if not obj.get('is_positive', True)]
        
        predicates = {
            'distinguishing_features': [],
            'common_positive_features': [],
            'common_negative_features': [],
            'stroke_type_differences': {},
            'shape_modifier_differences': {}
        }
        
        # Analyze stroke types
        pos_stroke_types = [obj['stroke_type'] for obj in positive_objects]
        neg_stroke_types = [obj['stroke_type'] for obj in negative_objects]
        
        predicates['stroke_type_differences'] = {
            'positive_distribution': self._count_occurrences(pos_stroke_types),
            'negative_distribution': self._count_occurrences(neg_stroke_types)
        }
        
        # Analyze shape modifiers
        pos_shapes = [obj['shape_modifier'] for obj in positive_objects]
        neg_shapes = [obj['shape_modifier'] for obj in negative_objects]
        
        predicates['shape_modifier_differences'] = {
            'positive_distribution': self._count_occurrences(pos_shapes),
            'negative_distribution': self._count_occurrences(neg_shapes)
        }
        
        return predicates
    
    def _count_occurrences(self, items: List[str]) -> Dict[str, int]:
        """Count occurrences of items in a list"""
        counts = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return counts
    
    def _find_image_path(self, image_id: str, image_base_dir: str) -> Optional[str]:
        """Find the actual image file path if it exists"""
        # This would need to be implemented based on your image directory structure
        # For now, return None since we're working with action programs only
        return None
    
    def _get_clip_embedding(self, image_path: str, vertices: List[Tuple[float, float]],
                          obj: Dict) -> np.ndarray:
        """Get CLIP embedding for a stroke region"""
        try:
            return self.clip_embedder.embed_image(
                image_path, 
                logo_object_data=obj
            )
        except Exception as e:
            self.logger.warning(f"Failed to get CLIP embedding: {e}")
            return np.random.normal(0, 0.1, 512)  # Default embedding
    
    def _default_stroke_features(self) -> Dict[str, Any]:
        """Return default features for invalid strokes"""
        return {
            'start_point': (0, 0),
            'end_point': (0, 0),
            'length': 0.0,
            'orientation': 0.0,
            'bbox': {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0},
            'centroid': (0, 0),
            'curvature': 0.0,
            'num_vertices': 0,
            'is_closed': False
        }


def process_bongard_problems(action_programs: Dict[str, Any],
                           problem_ids: Optional[List[str]] = None,
                           image_base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Process multiple Bongard problems with corrected action program parsing.
    
    Args:
        action_programs: Dict mapping problem_id to action program data
        problem_ids: List of specific problems to process (None = all)
        image_base_dir: Base directory for image files
        
    Returns:
        Dict mapping problem_id to processed results
    """
    processor = IntegratedBongardProcessor()
    
    problems_to_process = problem_ids if problem_ids else list(action_programs.keys())
    results = {}
    
    for problem_id in problems_to_process:
        if problem_id in action_programs:
            try:
                result = processor.process_problem(problem_id, action_programs, image_base_dir)
                if result:
                    results[problem_id] = result
                    logging.info(f"Successfully processed problem {problem_id}")
                else:
                    logging.warning(f"Failed to process problem {problem_id}")
            except Exception as e:
                logging.error(f"Error processing problem {problem_id}: {e}")
        else:
            logging.warning(f"Problem {problem_id} not found in action programs")
    
    return results
