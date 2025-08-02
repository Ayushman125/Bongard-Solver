"""
CommonsenseKB: Comprehensive knowledge base for Bongard problem solving with shape understanding,
geometric relationships, and visual reasoning patterns.
Version: 2.0.0 - Enhanced for comprehensive Bongard reasoning
"""

__version__ = "2.0.0"

import json
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Any

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
        
    def _build_shape_knowledge(self) -> Dict[str, Any]:
        """Build comprehensive shape category knowledge"""
        return {
            # Basic shape categories with properties
            'circles': {
                'properties': ['round', 'curved', 'no_angles', 'symmetric', 'closed'],
                'variations': ['perfect_circle', 'ellipse', 'oval'],
                'related_concepts': ['rotation', 'symmetry', 'curvature'],
                'typical_attributes': {'side_count': 0, 'angle_count': 0, 'regularity': 1.0}
            },
            'triangles': {
                'properties': ['three_sided', 'angular', 'closed', 'pointed'],
                'variations': ['equilateral', 'isosceles', 'scalene', 'right_triangle', 'acute', 'obtuse'],
                'related_concepts': ['stability', 'direction', 'apex'],
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
            'pentagons': {
                'properties': ['five_sided', 'angular', 'closed'],
                'variations': ['regular_pentagon', 'irregular_pentagon', 'star_pentagon'],
                'related_concepts': ['complexity', 'asymmetry'],
                'typical_attributes': {'side_count': 5, 'angle_count': 5, 'regularity': 0.6}
            },
            'hexagons': {
                'properties': ['six_sided', 'angular', 'closed', 'tessellating'],
                'variations': ['regular_hexagon', 'elongated_hexagon'],
                'related_concepts': ['efficiency', 'natural_patterns', 'honeycomb'],
                'typical_attributes': {'side_count': 6, 'angle_count': 6, 'regularity': 0.8}
            },
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
        """Infer expected symmetries for a shape type"""
        symmetry_map = {
            'circle': ['rotational', 'reflection_all_axes'],
            'square': ['rotational_4', 'reflection_4_axes'],
            'triangle': ['reflection_3_axes'] if 'equilateral' in shape_type else ['reflection_1_axis'],
            'rectangle': ['reflection_2_axes', 'point_symmetry'],
            'line': ['reflection_perpendicular'],
            'star': ['rotational', 'reflection_radial']
        }
        return symmetry_map.get(shape_type, [])
    
    def _infer_topological_features(self, shape_type: str) -> List[str]:
        """Infer topological features for a shape type"""
        topology_map = {
            'circle': ['closed', 'simply_connected', 'genus_0'],
            'triangle': ['closed', 'simply_connected', 'genus_0'],
            'square': ['closed', 'simply_connected', 'genus_0'],
            'line': ['open', 'one_dimensional'],
            'arc': ['open', 'curved', 'one_dimensional']
        }
        return topology_map.get(shape_type, [])
    
    def _infer_compositional_possibilities(self, shape_type: str) -> List[str]:
        """Infer compositional possibilities for a shape type"""
        composition_map = {
            'circle': ['can_enclose', 'can_be_wheel', 'can_be_eye'],
            'triangle': ['can_point', 'can_be_arrow', 'can_be_roof'],
            'square': ['can_frame', 'can_be_building', 'can_tile'],
            'line': ['can_connect', 'can_divide', 'can_bound'],
            'cross': ['can_intersect', 'can_mark', 'can_add']
        }
        return composition_map.get(shape_type, [])
    
    def _infer_analogical_mappings(self, shape_type: str) -> Dict[str, List[str]]:
        """Infer analogical mappings for a shape type"""
        analogy_map = {
            'circle': {
                'similar_concepts': ['wheel', 'eye', 'coin', 'moon'],
                'opposite_concepts': ['square', 'angular', 'pointed'],
                'transformation_targets': ['ellipse', 'oval', 'ring']
            },
            'triangle': {
                'similar_concepts': ['arrow', 'mountain', 'roof', 'wedge'],
                'opposite_concepts': ['circle', 'round', 'smooth'],
                'transformation_targets': ['pyramid', 'cone', 'spike']
            },
            'square': {
                'similar_concepts': ['box', 'frame', 'window', 'tile'],
                'opposite_concepts': ['circle', 'curved', 'round'],
                'transformation_targets': ['cube', 'rectangle', 'diamond']
            }
        }
        return analogy_map.get(shape_type, {})
    
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

class CommonsenseKB(ComprehensiveBongardKB):
    """Backward compatibility wrapper"""
    def __init__(self, path: str):
        super().__init__(path)
