"""
Stroke-specific commonsense knowledge base for Arc vs Line types
Different semantic properties and behaviors for arc and line stroke types
"""

import logging
from typing import Dict, List, Any, Set

class StrokeSpecificKnowledgeBase:
    """
    Commonsense knowledge base specialized for arc vs line stroke types
    Provides different semantic properties, behaviors, and reasoning patterns
    """
    
    def __init__(self):
        self.arc_knowledge = self._build_arc_knowledge()
        self.line_knowledge = self._build_line_knowledge()
        self.stroke_interactions = self._build_stroke_interactions()
        
    def _build_arc_knowledge(self) -> Dict[str, Any]:
        """Build comprehensive knowledge about arc strokes"""
        return {
            'geometric_properties': {
                'has_curvature': True,
                'has_radius': True,
                'has_center_point': True,
                'has_sweep_angle': True,
                'forms_circular_segments': True,
                'smooth_transitions': True,
                'continuous_curvature': True
            },
            
            'semantic_meanings': {
                'bridges': "Arcs often represent bridges connecting separate elements",
                'flow': "Arcs suggest smooth flow or transition between states",
                'partial_enclosure': "Arcs can form partial boundaries or enclosures",
                'softness': "Arcs add softness to otherwise angular designs",
                'motion': "Arcs can suggest circular or curved motion paths",
                'completeness': "Arcs often complete shapes started by lines"
            },
            
            'common_patterns': {
                'bridge_pattern': {
                    'description': "Arc connects two parallel lines",
                    'frequency': 'high',
                    'semantic_role': 'connection'
                },
                'cap_pattern': {
                    'description': "Arc caps the end of linear elements",
                    'frequency': 'medium',
                    'semantic_role': 'completion'
                },
                'semicircle_pattern': {
                    'description': "Arc forms half circle (180 degrees)",
                    'frequency': 'medium',
                    'semantic_role': 'dome/arch'
                },
                'quarter_circle_pattern': {
                    'description': "Arc forms quarter circle (90 degrees)",
                    'frequency': 'high',
                    'semantic_role': 'corner_rounding'
                }
            },
            
            'behavioral_traits': {
                'connects_smoothly': "Arcs create smooth connections between elements",
                'adds_curvature': "Arcs introduce curvature to angular designs",
                'bridges_gaps': "Arcs can span gaps between disconnected elements",
                'softens_corners': "Arcs can replace sharp corners with rounded ones",
                'creates_flow': "Arcs guide visual flow in curved paths"
            },
            
            'spatial_relationships': {
                'tangent_to_lines': "Arcs are often tangent to line segments",
                'perpendicular_to_radii': "Arc elements are perpendicular to their radii",
                'concentric_with_circles': "Arcs can be concentric with circular elements",
                'bridges_parallel_elements': "Arcs often bridge parallel line segments"
            },
            
            'reasoning_patterns': {
                'curvature_analysis': "Analyze curvature to understand arc properties",
                'connectivity_check': "Check what elements the arc connects",
                'completion_detection': "Determine if arc completes a shape",
                'flow_direction': "Determine the direction of flow suggested by arc"
            },
            
            'typical_functions': [
                'bridge_connector',
                'shape_completer', 
                'corner_rounder',
                'flow_guide',
                'boundary_softener',
                'motion_path'
            ],
            
            'distinguishing_features': {
                'curvature_score': "High curvature scores distinguish arcs",
                'radius_consistency': "Consistent radius indicates uniform arc",
                'endpoint_connectivity': "Arc endpoints often connect to other elements",
                'tangent_relationships': "Arcs maintain tangent relationships"
            }
        }
    
    def _build_line_knowledge(self) -> Dict[str, Any]:
        """Build comprehensive knowledge about line strokes"""
        return {
            'geometric_properties': {
                'has_curvature': False,
                'perfectly_straight': True,
                'has_endpoints': True,
                'has_orientation': True,
                'forms_angles': True,
                'creates_intersections': True,
                'defines_directions': True
            },
            
            'semantic_meanings': {
                'boundaries': "Lines often represent clear boundaries or edges",
                'structure': "Lines provide structural framework",
                'division': "Lines divide space into regions",
                'connection': "Lines connect discrete points",
                'direction': "Lines indicate direction or movement",
                'precision': "Lines suggest precision and exactness"
            },
            
            'common_patterns': {
                'parallel_lines': {
                    'description': "Multiple lines with same orientation",
                    'frequency': 'high',
                    'semantic_role': 'structure/pattern'
                },
                'perpendicular_lines': {
                    'description': "Lines meeting at 90 degrees",
                    'frequency': 'high',
                    'semantic_role': 'framework/grid'
                },
                'diagonal_lines': {
                    'description': "Lines at 45-degree angles",
                    'frequency': 'medium',
                    'semantic_role': 'dynamic/movement'
                },
                'intersecting_lines': {
                    'description': "Lines crossing at any angle",
                    'frequency': 'high',
                    'semantic_role': 'connection/crossing'
                }
            },
            
            'behavioral_traits': {
                'forms_angles': "Lines create precise angles when they meet",
                'defines_edges': "Lines define clear edges and boundaries",
                'creates_structure': "Lines provide structural framework",
                'enables_precision': "Lines enable precise geometric constructions",
                'guides_alignment': "Lines guide alignment of other elements"
            },
            
            'spatial_relationships': {
                'parallel_relationships': "Lines can be parallel to each other",
                'perpendicular_relationships': "Lines can meet at right angles",
                'intersection_points': "Lines create intersection points when crossing",
                'collinear_relationships': "Multiple points can be collinear on lines"
            },
            
            'reasoning_patterns': {
                'angle_analysis': "Analyze angles formed by line intersections",
                'parallelism_check': "Check for parallel line relationships",
                'length_comparison': "Compare lengths of different line segments",
                'orientation_analysis': "Analyze line orientations and directions"
            },
            
            'typical_functions': [
                'boundary_definer',
                'structure_provider',
                'space_divider',
                'angle_former',
                'intersection_creator',
                'direction_indicator'
            ],
            
            'distinguishing_features': {
                'zero_curvature': "Lines have zero curvature throughout",
                'consistent_direction': "Lines maintain consistent direction",
                'sharp_endpoints': "Lines have distinct, sharp endpoints",
                'angle_formation': "Lines form precise angles at intersections"
            }
        }
    
    def _build_stroke_interactions(self) -> Dict[str, Any]:
        """Build knowledge about how arcs and lines interact"""
        return {
            'arc_line_combinations': {
                'arc_bridges_lines': {
                    'description': "Arc connects two separate line segments",
                    'semantic_effect': "Creates unified structure from separate parts",
                    'visual_impact': "Adds flow and continuity"
                },
                'arc_caps_line': {
                    'description': "Arc provides rounded cap to line end",
                    'semantic_effect': "Softens hard edges",
                    'visual_impact': "Reduces visual tension"
                },
                'arc_rounds_corner': {
                    'description': "Arc replaces sharp corner between lines",
                    'semantic_effect': "Creates smooth transition",
                    'visual_impact': "Improves visual flow"
                },
                'lines_frame_arc': {
                    'description': "Lines provide structural context for arc",
                    'semantic_effect': "Arc becomes focal curved element",
                    'visual_impact': "Creates contrast between straight and curved"
                }
            },
            
            'complementary_roles': {
                'structure_and_flow': "Lines provide structure, arcs provide flow",
                'precision_and_softness': "Lines provide precision, arcs provide softness",
                'framework_and_completion': "Lines create framework, arcs complete shapes",
                'division_and_connection': "Lines divide space, arcs connect across divisions"
            },
            
            'design_principles': {
                'balance': "Mix of lines and arcs creates visual balance",
                'rhythm': "Alternating lines and arcs creates visual rhythm",
                'hierarchy': "Lines vs arcs can establish visual hierarchy",
                'unity': "Consistent stroke types create unity"
            }
        }
    
    def get_stroke_properties(self, stroke_type: str) -> Dict[str, Any]:
        """Get all properties for a specific stroke type"""
        if stroke_type == 'arc':
            return self.arc_knowledge
        elif stroke_type == 'line':
            return self.line_knowledge
        else:
            return {}
    
    def get_semantic_meaning(self, stroke_type: str, context: str = None) -> str:
        """Get semantic meaning for stroke type in specific context"""
        knowledge = self.get_stroke_properties(stroke_type)
        meanings = knowledge.get('semantic_meanings', {})
        
        if context and context in meanings:
            return meanings[context]
        
        # Return general semantic summary
        return f"General {stroke_type} properties: " + ", ".join(meanings.values())
    
    def get_applicable_patterns(self, stroke_type: str) -> List[str]:
        """Get common patterns for stroke type"""
        knowledge = self.get_stroke_properties(stroke_type)
        patterns = knowledge.get('common_patterns', {})
        return list(patterns.keys())
    
    def analyze_stroke_combination(self, stroke_types: List[str]) -> Dict[str, Any]:
        """Analyze the semantic effect of combining different stroke types"""
        has_arc = 'arc' in stroke_types
        has_line = 'line' in stroke_types
        
        if has_arc and has_line:
            return {
                'combination_type': 'mixed_strokes',
                'semantic_effects': self.stroke_interactions['complementary_roles'],
                'design_opportunities': self.stroke_interactions['arc_line_combinations'],
                'visual_impact': 'High contrast between curved and straight elements'
            }
        elif has_arc and not has_line:
            return {
                'combination_type': 'arc_only',
                'semantic_effects': {'flow_dominance': 'Smooth, flowing visual experience'},
                'design_opportunities': {'circular_patterns': 'Opportunity for circular motifs'},
                'visual_impact': 'Soft, organic appearance'
            }
        elif has_line and not has_arc:
            return {
                'combination_type': 'line_only', 
                'semantic_effects': {'structure_dominance': 'Precise, structured appearance'},
                'design_opportunities': {'geometric_patterns': 'Opportunity for geometric motifs'},
                'visual_impact': 'Clean, architectural appearance'
            }
        else:
            return {
                'combination_type': 'unknown',
                'semantic_effects': {},
                'design_opportunities': {},
                'visual_impact': 'Unclear'
            }
    
    def suggest_predicates(self, stroke_type: str, context: str = None) -> List[str]:
        """Suggest relevant predicates for stroke type"""
        if stroke_type == 'arc':
            base_predicates = [
                'arc_connects_parallel_lines',
                'arc_forms_semicircle', 
                'arc_forms_quarter_circle',
                'arc_has_high_curvature',
                'arc_has_uniform_curvature',
                'arc_connects_endpoints'
            ]
        elif stroke_type == 'line':
            base_predicates = [
                'line_is_perfectly_straight',
                'line_forms_right_angle',
                'line_is_horizontal',
                'line_is_vertical',
                'line_is_diagonal',
                'lines_form_parallel_pattern',
                'lines_form_corner'
            ]
        else:
            base_predicates = []
        
        # Add context-specific predicates
        if context == 'bridge':
            base_predicates.extend(['arc_connects_parallel_lines', 'arc_completes_line_shape'])
        elif context == 'structure':
            base_predicates.extend(['line_forms_right_angle', 'lines_form_parallel_pattern'])
        
        return base_predicates
    
    def get_reasoning_strategy(self, stroke_types: Set[str]) -> Dict[str, Any]:
        """Get reasoning strategy based on stroke types present"""
        strategies = {}
        
        if 'arc' in stroke_types:
            strategies['arc_reasoning'] = {
                'focus_on_curvature': 'Analyze curvature properties and patterns',
                'check_connectivity': 'Examine what elements arcs connect',
                'flow_analysis': 'Consider flow and transition effects'
            }
        
        if 'line' in stroke_types:
            strategies['line_reasoning'] = {
                'focus_on_angles': 'Analyze angles and intersections',
                'check_parallelism': 'Look for parallel and perpendicular relationships',
                'structure_analysis': 'Consider structural and framework aspects'
            }
        
        if 'arc' in stroke_types and 'line' in stroke_types:
            strategies['combination_reasoning'] = {
                'contrast_analysis': 'Analyze contrast between curved and straight',
                'complementary_roles': 'Consider how arcs and lines complement each other',
                'completion_patterns': 'Look for patterns where arcs complete line-based structures'
            }
        
        return strategies

# Global instance
stroke_kb = StrokeSpecificKnowledgeBase()

def get_stroke_knowledge(stroke_type: str) -> Dict[str, Any]:
    """Convenience function to get stroke-specific knowledge"""
    return stroke_kb.get_stroke_properties(stroke_type)

def analyze_stroke_semantics(stroke_types: List[str]) -> Dict[str, Any]:
    """Analyze semantic implications of stroke type combination"""
    return stroke_kb.analyze_stroke_combination(stroke_types)

if __name__ == "__main__":
    # Test the knowledge base
    kb = StrokeSpecificKnowledgeBase()
    
    print("Arc knowledge summary:")
    arc_knowledge = kb.get_stroke_properties('arc')
    print(f"- Semantic meanings: {len(arc_knowledge['semantic_meanings'])}")
    print(f"- Common patterns: {len(arc_knowledge['common_patterns'])}")
    print(f"- Typical functions: {len(arc_knowledge['typical_functions'])}")
    
    print("\nLine knowledge summary:")
    line_knowledge = kb.get_stroke_properties('line')
    print(f"- Semantic meanings: {len(line_knowledge['semantic_meanings'])}")
    print(f"- Common patterns: {len(line_knowledge['common_patterns'])}")
    print(f"- Typical functions: {len(line_knowledge['typical_functions'])}")
    
    print("\nStroke combination analysis:")
    combination = kb.analyze_stroke_combination(['arc', 'line'])
    print(f"- Combination type: {combination['combination_type']}")
    print(f"- Visual impact: {combination['visual_impact']}")
