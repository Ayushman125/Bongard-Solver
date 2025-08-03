"""
CommonsenseKB: Comprehensive knowledge base for Bongard problem solving with shape understanding,
geometric relationships, and visual reasoning patterns.
Version: 2.0.0 - Enhanced for comprehensive Bongard reasoning
"""

__version__ = "2.0.0"

import json
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Any

# Import stroke-specific knowledge base
try:
    from src.scene_graphs_building.stroke_specific_kb import stroke_kb, get_stroke_knowledge
    STROKE_KB_AVAILABLE = True
except ImportError:
    STROKE_KB_AVAILABLE = False
    print("Warning: Stroke-specific knowledge base not available")

class ComprehensiveBongardKB:
    """Comprehensive knowledge base for state-of-the-art Bongard problem solving"""
    
    def __init__(self, path: str = None):
        """
        Initialize comprehensive Bongard knowledge base.
        
        Args:
            path: Optional file path to external ConceptNet-lite JSON.
        """
        # Load external knowledge if provided
        self._external_index = defaultdict(list)
        if path:
            try:
                with open(path, "r") as f:
                    edges = json.load(f)
                for subj, pred, obj in edges:
                    self._external_index[pred].append((subj, obj))
            except Exception:
                pass
        
        # Build comprehensive Bongard-specific knowledge base
        self._shape_knowledge = self._build_shape_knowledge()
        self._geometric_knowledge = self._build_geometric_knowledge()
        self._symmetry_knowledge = self._build_symmetry_knowledge()
        self._topology_knowledge = self._build_topology_knowledge()
        self._compositional_knowledge = self._build_compositional_knowledge()
        self._analogical_knowledge = self._build_analogical_knowledge()
        self._visual_reasoning_knowledge = self._build_visual_reasoning_knowledge()
        self._logo_program_knowledge = self._build_logo_program_knowledge()
        self._stroke_specific_knowledge = self._build_stroke_specific_knowledge()
        
    def _build_shape_knowledge(self) -> Dict[str, Any]:
        """Build comprehensive shape category knowledge for 5 discovered Bongard-LOGO shape types"""
        return {
            # DISCOVERED BONGARD-LOGO SHAPE TYPES (Priority Order by Frequency)
            'normal': {
                'properties': ['straight', 'linear', 'simple', 'one_dimensional', 'basic'],
                'variations': ['horizontal_line', 'vertical_line', 'diagonal_line', 'short_line', 'long_line'],
                'related_concepts': ['direction', 'connection', 'boundary', 'path', 'edge'],
                'typical_attributes': {'side_count': 1, 'angle_count': 0, 'regularity': 1.0, 'complexity': 1},
                'frequency': 24107,
                'percentage': 48.7,
                'semantic_roles': ['connector', 'divider', 'border', 'axis']
            },
            'zigzag': {
                'properties': ['irregular', 'angular', 'jagged', 'alternating', 'dynamic'],
                'variations': ['sharp_zigzag', 'smooth_zigzag', 'dense_zigzag', 'sparse_zigzag'],
                'related_concepts': ['chaos', 'variability', 'pattern_breaking', 'texture', 'energy'],
                'typical_attributes': {'side_count': 0, 'angle_count': 5, 'regularity': 0.2, 'complexity': 3},
                'frequency': 6729,
                'percentage': 13.6,
                'semantic_roles': ['texture_maker', 'pattern_breaker', 'noise_generator']
            },
            'square': {
                'properties': ['four_sided', 'equal_sides', 'right_angles', 'regular', 'symmetric'],
                'variations': ['perfect_square', 'rotated_square', 'diamond_orientation', 'filled_square'],
                'related_concepts': ['regularity', 'stability', 'grid_alignment', 'structure', 'containment'],
                'typical_attributes': {'side_count': 4, 'angle_count': 4, 'regularity': 1.0, 'complexity': 2},
                'frequency': 6519,
                'percentage': 13.2,
                'semantic_roles': ['container', 'frame', 'building_block', 'unit']
            },
            'circle': {
                'properties': ['round', 'curved', 'no_angles', 'symmetric', 'closed'],
                'variations': ['perfect_circle', 'ellipse', 'oval', 'arc_segment'],
                'related_concepts': ['rotation', 'symmetry', 'curvature', 'completeness', 'unity'],
                'typical_attributes': {'side_count': 0, 'angle_count': 0, 'regularity': 1.0, 'complexity': 2},
                'frequency': 6256,
                'percentage': 12.6,
                'semantic_roles': ['encloser', 'eye', 'wheel', 'hole']
            },
            'triangle': {
                'properties': ['three_sided', 'angular', 'closed', 'pointed', 'stable'],
                'variations': ['equilateral', 'isosceles', 'scalene', 'right_triangle', 'acute', 'obtuse'],
                'related_concepts': ['stability', 'direction', 'apex', 'pointing', 'hierarchy'],
                'typical_attributes': {'side_count': 3, 'angle_count': 3, 'regularity': 0.8, 'complexity': 2},
                'frequency': 5837,
                'percentage': 11.8,
                'semantic_roles': ['pointer', 'apex', 'roof', 'arrow']
            },
            
            # Legacy shape types for backward compatibility
            'circles': {
                'properties': ['round', 'curved', 'no_angles', 'symmetric', 'closed'],
                'variations': ['perfect_circle', 'ellipse', 'oval'],
                'related_concepts': ['rotation', 'symmetry', 'curvature'],
                'typical_attributes': {'side_count': 3, 'angle_count': 3, 'regularity': 0.8}
            },
            'squares': {
                'properties': ['four_sided', 'equal_sides', 'right_angles', 'regular', 'symmetric'],
                'variations': ['perfect_square', 'rotated_square', 'diamond_orientation'],
                'related_concepts': ['regularity', 'stability', 'grid_alignment'],
                'typical_attributes': {'side_count': 4, 'angle_count': 4, 'regularity': 1.0}
            },
            'rectangles': {
                'properties': ['four_sided', 'right_angles', 'parallel_sides', 'symmetric'],
                'variations': ['landscape', 'portrait', 'long_rectangle', 'wide_rectangle'],
                'related_concepts': ['aspect_ratio', 'directionality', 'framing'],
                'typical_attributes': {'side_count': 4, 'angle_count': 4, 'regularity': 0.7}
            },
            # Note: Pentagon, hexagon, octagon removed - not discovered in Bongard-LOGO dataset
            'lines': {
                'properties': ['straight', 'open', 'no_area', 'directional'],
                'variations': ['horizontal', 'vertical', 'diagonal', 'curved_line', 'broken_line'],
                'related_concepts': ['direction', 'connection', 'boundary'],
                'typical_attributes': {'side_count': 1, 'angle_count': 0, 'regularity': 1.0}
            },
            'arcs': {
                'properties': ['curved', 'open', 'partial_circle', 'smooth'],
                'variations': ['semicircle', 'quarter_circle', 'curved_segment'],
                'related_concepts': ['curvature', 'flow', 'bridge'],
                'typical_attributes': {'side_count': 1, 'angle_count': 0, 'regularity': 0.8}
            },
            'stars': {
                'properties': ['pointed', 'radial', 'complex', 'symmetric'],
                'variations': ['five_pointed', 'six_pointed', 'eight_pointed', 'irregular_star'],
                'related_concepts': ['radiation', 'complexity', 'decoration'],
                'typical_attributes': {'side_count': 10, 'angle_count': 10, 'regularity': 0.6}
            },
            'crosses': {
                'properties': ['orthogonal', 'symmetric', 'composite', 'intersection'],
                'variations': ['plus_sign', 'x_cross', 'thick_cross', 'thin_cross'],
                'related_concepts': ['intersection', 'addition', 'religious_symbol'],
                'typical_attributes': {'side_count': 4, 'angle_count': 12, 'regularity': 0.9}
            }
        }
    
    def _build_geometric_knowledge(self) -> Dict[str, Any]:
        """Build geometric relationship knowledge"""
        return {
            'spatial_relationships': {
                'containment': ['inside', 'outside', 'surrounding', 'enclosed_by'],
                'proximity': ['near', 'far', 'adjacent', 'touching', 'separated'],
                'directional': ['above', 'below', 'left_of', 'right_of', 'in_front', 'behind'],
                'alignment': ['parallel', 'perpendicular', 'aligned', 'misaligned'],
                'intersection': ['crossing', 'overlapping', 'intersecting', 'disjoint']
            },
            'size_relationships': {
                'absolute': ['large', 'medium', 'small', 'tiny', 'huge'],
                'relative': ['larger_than', 'smaller_than', 'same_size', 'proportional'],
                'dimensional': ['tall', 'wide', 'long', 'short', 'thick', 'thin']
            },
            'shape_relationships': {
                'similarity': ['same_shape', 'similar_shape', 'different_shape'],
                'transformation': ['rotated', 'scaled', 'reflected', 'translated'],
                'composition': ['part_of', 'composed_of', 'contains_shape']
            }
        }
    
    def _build_symmetry_knowledge(self) -> Dict[str, Any]:
        """Build symmetry and pattern knowledge"""
        return {
            'symmetry_types': {
                'reflection': {
                    'axes': ['vertical', 'horizontal', 'diagonal'],
                    'properties': ['mirror_image', 'bilateral', 'axis_dependent'],
                    'examples': ['butterfly', 'face', 'building']
                },
                'rotation': {
                    'orders': [2, 3, 4, 5, 6, 8],
                    'properties': ['radial', 'central_point', 'angular'],
                    'examples': ['flower', 'star', 'wheel']
                },
                'translation': {
                    'patterns': ['repetition', 'tiling', 'periodic'],
                    'properties': ['shift_invariant', 'regular_spacing'],
                    'examples': ['wallpaper', 'fence', 'chain']
                },
                'point': {
                    'properties': ['central_symmetry', 'inversion', 'opposite_pairs'],
                    'examples': ['yin_yang', 'hourglass', 'bowtie']
                }
            },
            'pattern_recognition': {
                'regularity': ['regular', 'semi_regular', 'irregular', 'chaotic'],
                'periodicity': ['periodic', 'quasi_periodic', 'aperiodic'],
                'complexity': ['simple', 'compound', 'complex', 'fractal']
            }
        }
    
    def _build_topology_knowledge(self) -> Dict[str, Any]:
        """Build topological understanding knowledge"""
        return {
            'topological_properties': {
                'connectivity': {
                    'simply_connected': ['no_holes', 'single_piece'],
                    'multiply_connected': ['has_holes', 'doughnut_shape'],
                    'disconnected': ['separate_pieces', 'multiple_components']
                },
                'orientation': {
                    'orientable': ['has_consistent_direction', 'two_sided'],
                    'non_orientable': ['mobius_strip', 'klein_bottle']
                },
                'boundary': {
                    'closed': ['no_boundary', 'enclosed'],
                    'open': ['has_boundary', 'incomplete'],
                    'boundary_type': ['smooth', 'piecewise_linear', 'fractal']
                }
            },
            'genus_classification': {
                'genus_0': ['sphere', 'cube', 'simple_closed_curve'],
                'genus_1': ['torus', 'doughnut', 'coffee_cup'],
                'higher_genus': ['pretzel', 'complex_surface']
            }
        }
    
    def _build_compositional_knowledge(self) -> Dict[str, Any]:
        """Build knowledge about shape composition and decomposition"""
        return {
            'composition_rules': {
                'union': ['combining', 'joining', 'merging'],
                'intersection': ['overlapping', 'common_area', 'shared_region'],
                'subtraction': ['cutting', 'removing', 'hole_making'],
                'complement': ['everything_else', 'outside', 'negative_space']
            },
            'part_whole_relationships': {
                'components': ['made_of', 'consists_of', 'contains'],
                'assemblies': ['built_from', 'constructed_of', 'assembled'],
                'hierarchies': ['subsystem', 'component', 'element']
            },
            'construction_patterns': {
                'additive': ['building_up', 'accumulation', 'growth'],
                'subtractive': ['carving', 'removal', 'reduction'],
                'transformative': ['deformation', 'morphing', 'evolution']
            }
        }
    
    def _build_analogical_knowledge(self) -> Dict[str, Any]:
        """Build analogical reasoning knowledge for Bongard problems"""
        return {
            'analogy_types': {
                'geometric_analogy': {
                    'shape_to_shape': ['circle_is_to_sphere', 'triangle_is_to_pyramid'],
                    'size_to_size': ['small_is_to_large', 'thin_is_to_thick'],
                    'position_to_position': ['left_is_to_right', 'top_is_to_bottom']
                },
                'functional_analogy': {
                    'purpose': ['container_holds', 'boundary_separates'],
                    'behavior': ['rolls_like', 'points_like', 'encloses_like']
                },
                'structural_analogy': {
                    'composition': ['built_like', 'structured_like'],
                    'connectivity': ['connected_like', 'separated_like']
                }
            },
            'transformation_patterns': {
                'systematic_change': ['progression', 'series', 'sequence'],
                'alternation': ['every_other', 'pattern_flip', 'oscillation'],
                'recursive': ['self_similar', 'fractal_like', 'nested']
            }
        }
    
    def _build_visual_reasoning_knowledge(self) -> Dict[str, Any]:
        """Build visual reasoning and perception knowledge"""
        return {
            'gestalt_principles': {
                'proximity': ['close_together', 'grouped_by_distance'],
                'similarity': ['same_appearance', 'grouped_by_likeness'],
                'continuity': ['smooth_path', 'connected_flow'],
                'closure': ['complete_shape', 'filled_gaps'],
                'figure_ground': ['foreground_background', 'positive_negative'],
                'common_fate': ['moving_together', 'shared_direction']
            },
            'perceptual_organization': {
                'grouping': ['clustering', 'categorization', 'classification'],
                'segmentation': ['boundary_detection', 'region_separation'],
                'completion': ['gap_filling', 'inference', 'prediction']
            },
            'attention_mechanisms': {
                'salience': ['stands_out', 'conspicuous', 'attention_grabbing'],
                'focus': ['central', 'important', 'primary'],
                'context': ['background', 'supporting', 'environmental']
            }
        }
    
    def _build_logo_program_knowledge(self) -> Dict[str, Any]:
        """Build LOGO programming and procedural knowledge"""
        return {
            'logo_primitives': {
                'movement': ['forward', 'backward', 'left', 'right'],
                'drawing': ['pen_up', 'pen_down', 'set_color'],
                'control': ['repeat', 'loop', 'conditional'],
                'procedures': ['define', 'call', 'parameter']
            },
            'program_patterns': {
                'repetition': ['loop_based', 'recursive', 'iterative'],
                'modular': ['procedure_based', 'function_calls', 'decomposed'],
                'parametric': ['variable_based', 'configurable', 'adaptable']
            },
            'semantic_mapping': {
                'action_to_shape': ['forward_creates_line', 'turn_creates_angle'],
                'program_to_pattern': ['repeat_creates_regularity', 'recursion_creates_self_similarity'],
                'parameter_to_variation': ['size_parameter', 'angle_parameter', 'count_parameter']
            }
        }
    
    def _build_stroke_specific_knowledge(self) -> Dict[str, Any]:
        """Build stroke-specific knowledge for arc vs line differentiation"""
        return {
            'stroke_types': {
                'line': {
                    'geometric_properties': ['straight', 'angular', 'linear_segments', 'corner_based'],
                    'calculation_methods': ['angle_computation', 'edge_detection', 'line_fitting'],
                    'shape_variants': {
                        'line_normal': {'description': 'straight line segment', 'curvature': 0.0},
                        'line_triangle': {'description': 'triangular shape with straight edges', 'angle_sum': 180},
                        'line_square': {'description': 'square shape with straight edges', 'angle_sum': 360},
                        'line_zigzag': {'description': 'zigzag pattern with sharp corners', 'curvature': 0.0}
                    },
                    'semantic_roles': ['connector', 'divider', 'boundary_maker', 'structure_builder'],
                    'typical_features': {
                        'has_corners': True,
                        'has_curvature': False,
                        'supports_angles': True,
                        'edge_based': True
                    }
                },
                'arc': {
                    'geometric_properties': ['curved', 'smooth', 'continuous_curvature', 'flow_based'],
                    'calculation_methods': ['curvature_analysis', 'arc_fitting', 'smooth_interpolation'],
                    'shape_variants': {
                        'arc_normal': {'description': 'curved line segment', 'curvature': 'variable'},
                        'arc_triangle': {'description': 'triangular shape with curved edges', 'smoothness': 'high'},
                        'arc_square': {'description': 'square-like shape with rounded corners', 'corner_radius': 'variable'},
                        'arc_circle': {'description': 'circular or curved shape', 'curvature': 'constant'}
                    },
                    'semantic_roles': ['flow_creator', 'smoother', 'organic_shaper', 'curve_generator'],
                    'typical_features': {
                        'has_corners': False,
                        'has_curvature': True,
                        'supports_angles': False,
                        'curve_based': True
                    }
                }
            },
            'stroke_type_mappings': {
                # Map stroke type + shape type combinations to semantic concepts
                'line_normal': ['straight_line', 'edge', 'boundary', 'connector'],
                'arc_normal': ['curved_line', 'arc_segment', 'flow', 'smooth_curve'],
                'line_triangle': ['angular_triangle', 'sharp_triangle', 'geometric_triangle'],
                'arc_triangle': ['rounded_triangle', 'smooth_triangle', 'organic_triangle'],
                'line_square': ['angular_square', 'sharp_square', 'geometric_square'],
                'arc_square': ['rounded_square', 'smooth_square', 'organic_square'],
                'line_circle': ['polygonal_circle', 'angular_approximation'],  # Rare case
                'arc_circle': ['smooth_circle', 'true_circle', 'curved_circle'],
                'line_zigzag': ['sharp_zigzag', 'angular_zigzag', 'jagged_pattern'],
                'arc_zigzag': ['smooth_zigzag', 'wave_pattern', 'curved_zigzag']
            },
            'calculation_strategies': {
                'line_based_shapes': {
                    'primary_features': ['angles', 'edge_lengths', 'vertex_positions', 'corner_detection'],
                    'measurement_methods': ['angle_calculation', 'distance_measurement', 'slope_analysis'],
                    'geometric_invariants': ['angle_sum', 'edge_count', 'vertex_count'],
                    'symmetry_detection': ['reflection_axes', 'rotational_symmetry_order']
                },
                'arc_based_shapes': {
                    'primary_features': ['curvature', 'arc_length', 'radius_of_curvature', 'smoothness'],
                    'measurement_methods': ['curvature_calculation', 'arc_length_integration', 'smooth_fitting'],
                    'geometric_invariants': ['total_curvature', 'area_enclosed', 'perimeter_length'],
                    'symmetry_detection': ['radial_symmetry', 'smooth_reflection_axes']
                }
            },
            'differentiation_criteria': {
                'stroke_type_detection': {
                    'line_indicators': ['sharp_corners', 'straight_segments', 'angular_features'],
                    'arc_indicators': ['smooth_curves', 'continuous_derivatives', 'rounded_features']
                },
                'shape_analysis_differences': {
                    'line_analysis': 'Focus on angles, edges, and discrete geometric properties',
                    'arc_analysis': 'Focus on curvature, smoothness, and continuous geometric properties'
                }
            }
        }
    
    def query_shape_properties(self, shape_type: str) -> Dict[str, Any]:
        """Query properties of a specific shape type"""
        return self._shape_knowledge.get(shape_type, {})
    
    def query_geometric_relationships(self, relationship_type: str) -> List[str]:
        """Query geometric relationships of a specific type"""
        for category, relationships in self._geometric_knowledge.items():
            if relationship_type in relationships:
                return relationships[relationship_type]
        return []
    
    def query_symmetry_types(self, symmetry_category: str) -> Dict[str, Any]:
        """Query symmetry information"""
        return self._symmetry_knowledge.get('symmetry_types', {}).get(symmetry_category, {})
    
    def query_analogical_patterns(self, pattern_type: str) -> Dict[str, Any]:
        """Query analogical reasoning patterns"""
        return self._analogical_knowledge.get('analogy_types', {}).get(pattern_type, {})
    
    def query_visual_principles(self, principle: str) -> List[str]:
        """Query visual reasoning principles"""
        return self._visual_reasoning_knowledge.get('gestalt_principles', {}).get(principle, [])
    
    def query_stroke_specific_knowledge(self, stroke_type: str) -> Dict[str, Any]:
        """Query stroke-specific knowledge for arc vs line differentiation"""
        return self._stroke_specific_knowledge.get('stroke_types', {}).get(stroke_type, {})
    
    def get_stroke_shape_mapping(self, stroke_type: str, shape_type: str) -> List[str]:
        """Get semantic mapping for stroke_type + shape_type combination"""
        combination_key = f"{stroke_type}_{shape_type}"
        return self._stroke_specific_knowledge.get('stroke_type_mappings', {}).get(combination_key, [])
    
    def get_calculation_strategy_for_stroke(self, stroke_type: str) -> Dict[str, Any]:
        """Get calculation strategy specific to stroke type"""
        strategy_key = f"{stroke_type}_based_shapes"
        return self._stroke_specific_knowledge.get('calculation_strategies', {}).get(strategy_key, {})
    
    def differentiate_stroke_types(self, features: Dict[str, Any]) -> str:
        """Determine stroke type based on geometric features"""
        criteria = self._stroke_specific_knowledge.get('differentiation_criteria', {})
        detection = criteria.get('stroke_type_detection', {})
        
        line_indicators = detection.get('line_indicators', [])
        arc_indicators = detection.get('arc_indicators', [])
        
        # Check for line indicators
        line_score = 0
        arc_score = 0
        
        for indicator in line_indicators:
            if indicator in str(features).lower():
                line_score += 1
                
        for indicator in arc_indicators:
            if indicator in str(features).lower():
                arc_score += 1
        
        # Check specific feature values
        if features.get('curvature_score', 0) > 0.1:
            arc_score += 2
        if features.get('max_curvature', 0) > 0.1:
            arc_score += 2
        if features.get('is_highly_curved', False):
            arc_score += 3
            
        # Check for angular features
        if features.get('num_junctions', 0) > 0:
            line_score += 1
        if features.get('orientation_variance', 0) > 30:  # High variance suggests corners
            line_score += 1
            
        return 'arc' if arc_score > line_score else 'line'
    
    def get_comprehensive_shape_knowledge(self, shape_type: str) -> Dict[str, Any]:
        """Get comprehensive knowledge about a shape including all aspects"""
        knowledge = {
            'basic_properties': self.query_shape_properties(shape_type),
            'symmetry_expectations': self._infer_symmetry_expectations(shape_type),
            'topological_features': self._infer_topological_features(shape_type),
            'compositional_possibilities': self._infer_compositional_possibilities(shape_type),
            'analogical_mappings': self._infer_analogical_mappings(shape_type)
        }
        return knowledge
    
    def _infer_symmetry_expectations(self, shape_type: str) -> List[str]:
        """Infer expected symmetries for discovered Bongard-LOGO shape types"""
        symmetry_map = {
            # Discovered shape types with specific symmetry patterns
            'normal': ['reflection_perpendicular', 'translation_along_axis'],
            'circle': ['rotational_infinite', 'reflection_all_axes', 'radial_symmetry'],
            'square': ['rotational_4', 'reflection_4_axes', 'point_symmetry'],
            'triangle': ['reflection_3_axes', 'rotational_3'],  # Assuming equilateral
            'zigzag': ['translation_periodic', 'approximate_reflection'],  # Weak symmetry
            
            # Legacy types for backward compatibility
            'rectangle': ['reflection_2_axes', 'point_symmetry'],
            'line': ['reflection_perpendicular'],
            'star': ['rotational', 'reflection_radial']
        }
        return symmetry_map.get(shape_type, ['no_clear_symmetry'])
    
    def _infer_topological_features(self, shape_type: str) -> List[str]:
        """Infer topological features for discovered Bongard-LOGO shape types"""
        topology_map = {
            # Discovered shape types with specific topological properties
            'normal': ['open', 'one_dimensional', 'linear_manifold'],
            'circle': ['closed', 'simply_connected', 'genus_0', 'smooth_boundary'],
            'square': ['closed', 'simply_connected', 'genus_0', 'piecewise_linear'],
            'triangle': ['closed', 'simply_connected', 'genus_0', 'piecewise_linear'],
            'zigzag': ['open', 'one_dimensional', 'irregular_curvature'],
            
            # Legacy types for backward compatibility
            'line': ['open', 'one_dimensional'],
            'arc': ['open', 'curved', 'one_dimensional'],
            'rectangle': ['closed', 'simply_connected', 'genus_0']
        }
        return topology_map.get(shape_type, ['unknown_topology'])
    
    def _infer_compositional_possibilities(self, shape_type: str) -> List[str]:
        """Infer compositional possibilities for discovered Bongard-LOGO shape types"""
        composition_map = {
            # Discovered shape types with specific compositional roles
            'normal': ['can_connect', 'can_divide', 'can_bound', 'can_align', 'forms_grid'],
            'circle': ['can_enclose', 'can_be_wheel', 'can_be_eye', 'can_be_dot', 'forms_pattern'],
            'square': ['can_frame', 'can_be_building', 'can_tile', 'can_contain', 'forms_grid'],
            'triangle': ['can_point', 'can_be_arrow', 'can_be_roof', 'can_indicate', 'forms_pattern'],
            'zigzag': ['can_texture', 'can_energize', 'can_break_pattern', 'creates_noise', 'adds_complexity'],
            
            # Legacy types for backward compatibility
            'line': ['can_connect', 'can_divide', 'can_bound'],
            'cross': ['can_intersect', 'can_mark', 'can_add'],
            'rectangle': ['can_frame', 'can_contain', 'can_structure']
        }
        return composition_map.get(shape_type, ['unknown_composition'])
    
    def _infer_analogical_mappings(self, shape_type: str) -> Dict[str, List[str]]:
        """Infer analogical mappings for discovered Bongard-LOGO shape types"""
        analogy_map = {
            # Discovered shape types with rich analogical mappings
            'normal': {
                'similar_concepts': ['line', 'edge', 'path', 'connection', 'boundary'],
                'opposite_concepts': ['curve', 'zigzag', 'circle', 'complex'],
                'transformation_targets': ['curve', 'broken_line', 'zigzag', 'ray'],
                'metaphorical_roles': ['bridge', 'divider', 'axis', 'trajectory']
            },
            'circle': {
                'similar_concepts': ['wheel', 'eye', 'coin', 'moon', 'ring'],
                'opposite_concepts': ['square', 'angular', 'pointed', 'linear'],
                'transformation_targets': ['ellipse', 'oval', 'ring', 'spiral'],
                'metaphorical_roles': ['completeness', 'unity', 'eye', 'hole']
            },
            'square': {
                'similar_concepts': ['box', 'frame', 'window', 'tile', 'building'],
                'opposite_concepts': ['circle', 'curved', 'round', 'organic'],
                'transformation_targets': ['cube', 'rectangle', 'diamond', 'grid'],
                'metaphorical_roles': ['container', 'structure', 'order', 'stability']
            },
            'triangle': {
                'similar_concepts': ['arrow', 'mountain', 'roof', 'wedge', 'pyramid'],
                'opposite_concepts': ['circle', 'round', 'smooth', 'flat'],
                'transformation_targets': ['pyramid', 'cone', 'spike', 'arrow'],
                'metaphorical_roles': ['direction', 'hierarchy', 'pointing', 'focus']
            },
            'zigzag': {
                'similar_concepts': ['lightning', 'sawtooth', 'wave', 'texture', 'chaos'],
                'opposite_concepts': ['smooth', 'regular', 'straight', 'simple'],
                'transformation_targets': ['wave', 'spiral', 'texture', 'noise'],
                'metaphorical_roles': ['energy', 'disruption', 'complexity', 'motion']
            }
        }
        return analogy_map.get(shape_type, {
            'similar_concepts': [],
            'opposite_concepts': [],
            'transformation_targets': [],
            'metaphorical_roles': []
        })
    
    def get_bongard_logo_shape_knowledge(self, shape_type: str) -> Dict[str, Any]:
        """Get specialized knowledge for the 5 discovered Bongard-LOGO shape types"""
        discovered_types = ['normal', 'circle', 'square', 'triangle', 'zigzag']
        
        if shape_type not in discovered_types:
            # Fallback to general shape knowledge
            return self.get_comprehensive_shape_knowledge(shape_type)
        
        # Get comprehensive knowledge for discovered types
        base_knowledge = self.get_comprehensive_shape_knowledge(shape_type)
        
        # Add specific Bongard-LOGO insights
        shape_data = self._shape_knowledge.get(shape_type, {})
        
        # Enhanced knowledge with frequency and semantic role information
        enhanced_knowledge = {
            **base_knowledge,
            'dataset_frequency': shape_data.get('frequency', 0),
            'dataset_percentage': shape_data.get('percentage', 0.0),
            'semantic_roles': shape_data.get('semantic_roles', []),
            'bongard_priority': self._get_bongard_priority(shape_type),
            'complexity_level': shape_data.get('typical_attributes', {}).get('complexity', 1),
            'regularity_score': shape_data.get('typical_attributes', {}).get('regularity', 0.5)
        }
        
        return enhanced_knowledge
    
    def _get_bongard_priority(self, shape_type: str) -> int:
        """Get priority ranking for Bongard-LOGO shape types based on frequency"""
        priority_map = {
            'normal': 1,    # Highest priority - most frequent
            'zigzag': 2,    # Second priority  
            'square': 3,    # Third priority
            'circle': 4,    # Fourth priority
            'triangle': 5   # Fifth priority
        }
        return priority_map.get(shape_type, 10)  # Low priority for unknown types
    
    def query_shape_relationships(self, shape1: str, shape2: str) -> Dict[str, Any]:
        """Query relationships between two shape types"""
        relationships = {
            'similarity_score': self._compute_shape_similarity(shape1, shape2),
            'complementarity': self._check_complementarity(shape1, shape2),
            'analogy_potential': self._assess_analogy_potential(shape1, shape2),
            'compositional_compatibility': self._check_compositional_compatibility(shape1, shape2)
        }
        return relationships
    
    def _compute_shape_similarity(self, shape1: str, shape2: str) -> float:
        """Compute similarity score between two shape types"""
        if shape1 == shape2:
            return 1.0
        
        # Get properties for both shapes
        props1 = self._shape_knowledge.get(shape1, {}).get('properties', [])
        props2 = self._shape_knowledge.get(shape2, {}).get('properties', [])
        
        if not props1 or not props2:
            return 0.0
        
        # Compute Jaccard similarity
        common = set(props1) & set(props2)
        total = set(props1) | set(props2)
        
        return len(common) / len(total) if total else 0.0
    
    def _check_complementarity(self, shape1: str, shape2: str) -> bool:
        """Check if two shapes are complementary (opposites)"""
        analogies1 = self._infer_analogical_mappings(shape1)
        opposites1 = analogies1.get('opposite_concepts', [])
        
        return shape2 in opposites1 or any(shape2 in opp for opp in opposites1)
    
    def _assess_analogy_potential(self, shape1: str, shape2: str) -> float:
        """Assess potential for analogical reasoning between shapes"""
        # Get analogical mappings
        analogies1 = self._infer_analogical_mappings(shape1)
        analogies2 = self._infer_analogical_mappings(shape2)
        
        # Check for shared transformation targets or similar concepts
        targets1 = set(analogies1.get('transformation_targets', []))
        targets2 = set(analogies2.get('transformation_targets', []))
        
        concepts1 = set(analogies1.get('similar_concepts', []))
        concepts2 = set(analogies2.get('similar_concepts', []))
        
        shared_targets = len(targets1 & targets2)
        shared_concepts = len(concepts1 & concepts2)
        
        total_possible = len(targets1 | targets2) + len(concepts1 | concepts2)
        
        return (shared_targets + shared_concepts) / max(1, total_possible)
    
    def _check_compositional_compatibility(self, shape1: str, shape2: str) -> bool:
        """Check if two shapes can be composed together effectively"""
        comp1 = self._infer_compositional_possibilities(shape1)
        comp2 = self._infer_compositional_possibilities(shape2)
        
        # Look for complementary compositional roles
        can_contain = any('contain' in role for role in comp1)
        can_be_contained = any('frame' in role or 'enclose' in role for role in comp2)
        
        can_connect = any('connect' in role for role in comp1)
        can_be_connected = any('connect' in role for role in comp2)
        
        return can_contain and can_be_contained or can_connect and can_be_connected

    def get_bongard_solving_strategies(self) -> Dict[str, List[str]]:
        """Get comprehensive strategies for solving Bongard problems"""
        return {
            'shape_analysis': [
                'identify_basic_shapes',
                'count_sides_and_angles',
                'measure_regularity',
                'detect_symmetries',
                'analyze_topology'
            ],
            'relationship_analysis': [
                'find_spatial_relationships',
                'detect_containment',
                'identify_intersections',
                'measure_distances',
                'analyze_alignments'
            ],
            'pattern_recognition': [
                'look_for_repetition',
                'identify_progressions',
                'detect_alternations',
                'find_transformations',
                'recognize_analogies'
            ],
            'compositional_analysis': [
                'decompose_complex_shapes',
                'identify_components',
                'analyze_construction',
                'find_hierarchies',
                'detect_emergence'
            ],
            'semantic_reasoning': [
                'map_to_concepts',
                'use_analogical_reasoning',
                'apply_gestalt_principles',
                'leverage_common_sense',
                'integrate_multimodal_cues'
            ]
        }
    
    def query(self, predicate: str) -> List[Tuple[str, str]]:
        """Query external ConceptNet knowledge"""
        return self._external_index.get(predicate, [])

    # STROKE-SPECIFIC KNOWLEDGE INTEGRATION
    def get_stroke_specific_knowledge(self, stroke_type: str) -> Dict[str, Any]:
        """Get stroke-specific knowledge for arc vs line types"""
        if not STROKE_KB_AVAILABLE:
            return {"error": "Stroke-specific knowledge not available"}
        
        return get_stroke_knowledge(stroke_type)
    
    def get_stroke_semantic_meaning(self, stroke_type: str, context: str = None) -> str:
        """Get semantic meaning for stroke type in specific context"""
        if not STROKE_KB_AVAILABLE:
            return "General stroke properties"
        
        return stroke_kb.get_semantic_meaning(stroke_type, context)
    
    def get_stroke_predicates(self, stroke_type: str, context: str = None) -> List[str]:
        """Get relevant predicates for stroke type"""
        if not STROKE_KB_AVAILABLE:
            return []
        
        return stroke_kb.suggest_predicates(stroke_type, context)
    
    def analyze_stroke_combination(self, stroke_types: List[str]) -> Dict[str, Any]:
        """Analyze semantic effect of combining different stroke types"""
        if not STROKE_KB_AVAILABLE:
            return {"error": "Stroke analysis not available"}
        
        return stroke_kb.analyze_stroke_combination(stroke_types)
    
    def get_stroke_reasoning_strategy(self, stroke_types: Set[str]) -> Dict[str, Any]:
        """Get reasoning strategy based on stroke types present"""
        if not STROKE_KB_AVAILABLE:
            return {"error": "Stroke reasoning not available"}
        
        return stroke_kb.get_reasoning_strategy(stroke_types)
    
    def get_comprehensive_shape_knowledge_with_strokes(self, shape_type: str, stroke_types: List[str] = None) -> Dict[str, Any]:
        """Get comprehensive knowledge including stroke-specific information"""
        # Get base comprehensive knowledge
        knowledge = self.get_comprehensive_shape_knowledge(shape_type)
        
        # Add stroke-specific knowledge if available and relevant
        if STROKE_KB_AVAILABLE and stroke_types:
            stroke_analysis = self.analyze_stroke_combination(stroke_types)
            stroke_strategies = self.get_stroke_reasoning_strategy(set(stroke_types))
            
            knowledge['stroke_analysis'] = stroke_analysis
            knowledge['stroke_reasoning'] = stroke_strategies
            
            # Add stroke-specific properties for each stroke type
            stroke_properties = {}
            for stroke_type in stroke_types:
                stroke_properties[stroke_type] = self.get_stroke_specific_knowledge(stroke_type)
            
            knowledge['stroke_properties'] = stroke_properties
        
        return knowledge

class CommonsenseKB(ComprehensiveBongardKB):
    """Backward compatibility wrapper"""
    def __init__(self, path: str):
        super().__init__(path)
