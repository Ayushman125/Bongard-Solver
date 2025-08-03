import re
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import math
from dataclasses import dataclass
from enum import Enum
import logging

# Real Bongard Shape Categories - Only the 5 discovered types
class BongardShapeType(Enum):
    # DISCOVERED BONGARD-LOGO SHAPE TYPES (5 total from dataset analysis)
    # These are the actual shape types found in the dataset with frequencies:
    NORMAL = "normal"        # 24,107 occurrences - most common, straight lines
    CIRCLE = "circle"        # 6,256 occurrences - circular shapes/arcs  
    SQUARE = "square"        # 6,519 occurrences - square-based shapes
    TRIANGLE = "triangle"    # 5,837 occurrences - triangular shapes
    ZIGZAG = "zigzag"        # 6,729 occurrences - zigzag patterns
    
    # Irregular and Abstract
    BLOB = "blob"
    FREEFORM = "freeform"
    IRREGULAR = "irregular"
    
    # Special Bongard Categories
    HOUSE = "house"
    ARROW = "arrow"
    HEART = "heart"
    FLOWER = "flower"
    
    UNKNOWN = "unknown"

class BongardTopology(Enum):
    CLOSED = "closed"
    OPEN = "open"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    SIMPLE = "simple"
    COMPLEX = "complex"

class BongardSymmetry(Enum):
    NONE = "none"
    BILATERAL = "bilateral"
    ROTATIONAL = "rotational"
    POINT = "point"
    MULTIPLE = "multiple"

@dataclass
class ComprehensiveShapeFeatures:
    """Comprehensive shape features for Bongard reasoning"""
    # Basic geometry
    shape_type: BongardShapeType
    side_count: int
    angle_count: int
    vertex_count: int
    
    # Topology
    topology: BongardTopology
    is_closed: bool
    is_connected: bool
    has_holes: int
    genus: int
    
    # Symmetry
    symmetry_type: BongardSymmetry
    symmetry_axes: List[float]
    rotation_order: int
    
    # Geometric properties
    convexity: float
    regularity: float
    compactness: float
    aspect_ratio: float
    
    # Size and scale
    size_category: str  # tiny, small, medium, large, huge
    relative_size: float
    
    # Complexity
    complexity_score: float
    stroke_count: int
    intersection_count: int
    
    # Compositional
    is_composite: bool
    component_shapes: List[str]
    spatial_arrangement: str
    
    # Bongard-specific attributes
    contains_curves: bool
    contains_angles: bool
    regularity_score: float
    symmetry_score: float

class SemanticActionParser:
    """Enhanced comprehensive parser for Bongard problem solving with singleton pattern"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SemanticActionParser, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if SemanticActionParser._initialized:
            return
            
        # Enhanced comprehensive shape pattern matching for Bongard LOGO dataset
        # Updated to handle all 5 discovered shape types: normal, circle, square, triangle, zigzag
        self.shape_patterns = {
            # Discovered Bongard-LOGO shape types (exact matches from dataset analysis)
            'normal': re.compile(r'normal|straight|line|regular', re.IGNORECASE),
            'circle': re.compile(r'circle|round|disc|disk|circular|o-shape', re.IGNORECASE), 
            'square': re.compile(r'square|quad(?:rangle)?|four.*side|4.*side', re.IGNORECASE),
            'triangle': re.compile(r'triangle|tri(?:angle)?|three.*side|3.*side|equilateral|isosceles|scalene', re.IGNORECASE),
            'zigzag': re.compile(r'zigzag|zig.*zag|jagged|serrated|wavy|irregular', re.IGNORECASE),
            'normal': re.compile(r'normal|standard|basic|regular|simple', re.IGNORECASE),
            
            # Curves and arcs
            'arc': re.compile(r'arc|curve|curved|bend|bow|arch', re.IGNORECASE),
            'spiral': re.compile(r'spiral|coil|twist|helix|vortex', re.IGNORECASE),
            'crescent': re.compile(r'crescent|moon|c.*shape|sickle', re.IGNORECASE),
            'semicircle': re.compile(r'semicircle|half.*circle|half.*round', re.IGNORECASE),
            # REMOVED: 'quarter_circle' pattern - use action program types only
            
            # Lines and segments
            'line': re.compile(r'line|straight|linear|stroke', re.IGNORECASE),
            'ray': re.compile(r'ray|beam|half.*line', re.IGNORECASE),
            'segment': re.compile(r'segment|seg|line.*segment', re.IGNORECASE),
            'zigzag': re.compile(r'zigzag|zig.*zag|jagged|sawtooth', re.IGNORECASE),
            
            # Valid Bongard shape types only - discovered from dataset analysis
            'normal': re.compile(r'normal|line|straight|linear|simple', re.IGNORECASE),
            'circle': re.compile(r'circle|round|circular|sphere|oval|ellipse', re.IGNORECASE),
            'square': re.compile(r'square|rectangle|rect|box|quadrilateral', re.IGNORECASE),
            'triangle': re.compile(r'triangle|triangular|tri.*angle|three.*side', re.IGNORECASE),
            'zigzag': re.compile(r'zigzag|zig.*zag|jagged|sawtooth|wave', re.IGNORECASE),
            
            # Complex geometric shapes (NOT Bongard shape types)
            'diamond': re.compile(r'diamond|rhomb|lozenge', re.IGNORECASE),
            'trapezoid': re.compile(r'trapezoid|trap|trapez', re.IGNORECASE),
            'parallelogram': re.compile(r'parallelogram|parallel.*quad', re.IGNORECASE),
            'rhombus': re.compile(r'rhombus|rhomb', re.IGNORECASE),
            
            # Composite geometric patterns
            'house': re.compile(r'house|home|building|roof|shelter', re.IGNORECASE),
            'arrow': re.compile(r'arrow|pointer|direction|chevron', re.IGNORECASE),
            'flower': re.compile(r'flower|petal|bloom|blossom|flora', re.IGNORECASE),
            'crown': re.compile(r'crown|tiara|royal.*hat|jagged.*top', re.IGNORECASE),
            'bridge': re.compile(r'bridge|span|arch.*connect|connect.*arch', re.IGNORECASE),
            'funnel': re.compile(r'funnel|cone|taper|narrow.*wide', re.IGNORECASE),
            'lamp': re.compile(r'lamp|light|bulb.*stand|stand.*bulb', re.IGNORECASE),
            'jar': re.compile(r'jar|container|vessel|pot', re.IGNORECASE),
            'sector': re.compile(r'sector|pie.*slice|wedge|fan.*shape', re.IGNORECASE),
            'bird': re.compile(r'bird|wing|avian|flying.*shape', re.IGNORECASE),
            
            # Abstract and irregular shapes
            'blob': re.compile(r'blob|amorphous|shapeless|organic', re.IGNORECASE),
            'freeform': re.compile(r'freeform|free.*form|organic|irregular.*curve', re.IGNORECASE),
            'irregular': re.compile(r'irregular|uneven|asymmetric.*shape|deformed', re.IGNORECASE),
            'unbala': re.compile(r'unbala|unbalanced|asymmetric|lopsided', re.IGNORECASE),
            'mismatch': re.compile(r'mismatch|mixed|combination|hybrid', re.IGNORECASE),
            
            # Curved variations
            'threequarter_circle': re.compile(r'threequarter.*circle|three.*quarter.*circle|270.*degree.*arc', re.IGNORECASE),
            # REMOVED: 'open_curve' pattern - use action program types only
            'closed_curve': re.compile(r'closed.*curve|complete.*curve|full.*arc', re.IGNORECASE),
            
            # Special geometric patterns
            'intersect': re.compile(r'intersect|cross.*lines|overlap|intersection', re.IGNORECASE),
            'parallel': re.compile(r'parallel|side.*by.*side|equidistant', re.IGNORECASE),
            'concentric': re.compile(r'concentric|nested|inside.*outside', re.IGNORECASE),
            'tangent': re.compile(r'tangent|touching|adjacent', re.IGNORECASE),
        }
        
        # Enhanced property patterns for complex shape analysis
        self.property_patterns = {
            # Size and scale patterns
            'size': re.compile(r'(\d+\.?\d*)-(\d+\.?\d*)', re.IGNORECASE),
            'large': re.compile(r'large|big|huge|giant|massive|wide', re.IGNORECASE),
            'small': re.compile(r'small|tiny|little|mini|narrow', re.IGNORECASE),
            'medium': re.compile(r'medium|average|normal|standard', re.IGNORECASE),
            'thick': re.compile(r'thick|fat|wide|bold|heavy', re.IGNORECASE),
            'thin': re.compile(r'thin|narrow|skinny|fine|slim', re.IGNORECASE),
            
            # Topological properties
            'open': re.compile(r'open|incomplete|partial|broken|cut|gap', re.IGNORECASE),
            'closed': re.compile(r'closed|complete|full|solid|sealed', re.IGNORECASE),
            'connected': re.compile(r'connect|join|link|attach|continuous', re.IGNORECASE),
            'separate': re.compile(r'separate|apart|disconnect|isolated|detached', re.IGNORECASE),
            'intersecting': re.compile(r'intersect|cross|overlap|meet|collision', re.IGNORECASE),
            'parallel': re.compile(r'parallel|side.*by.*side|equidistant', re.IGNORECASE),
            'nested': re.compile(r'nested|inside|within|concentric|enclosed', re.IGNORECASE),
            
            # Geometric deformations
            'deformed': re.compile(r'deform|irregular|distort|bent|warped|skewed', re.IGNORECASE),
            'stretched': re.compile(r'stretch|elongated|extended|pulled', re.IGNORECASE),
            'compressed': re.compile(r'compress|squeezed|flattened|squashed', re.IGNORECASE),
            'rotated': re.compile(r'rotat|turn|spin|angled|tilted', re.IGNORECASE),
            'inverted': re.compile(r'invert|flip|upside|reverse|mirror', re.IGNORECASE),
            'scaled': re.compile(r'scaled|resized|enlarged|shrunk', re.IGNORECASE),
            
            # Symmetry and regularity
            'symmetric': re.compile(r'symm|regular|balanced|even|uniform', re.IGNORECASE),
            'asymmetric': re.compile(r'asymm|irregular|unbalanced|uneven|lopsided', re.IGNORECASE),
            'bilateral': re.compile(r'bilateral|mirror.*symm|reflection', re.IGNORECASE),
            'radial': re.compile(r'radial|rotational.*symm|circular.*symm', re.IGNORECASE),
            
            # Convexity and curvature
            'convex': re.compile(r'convex|bulging|outward|rounded.*out', re.IGNORECASE),
            'concave': re.compile(r'concave|dented|inward|rounded.*in', re.IGNORECASE),
            'curved': re.compile(r'curved|round|circular|bent|arc', re.IGNORECASE),
            'angular': re.compile(r'angular|sharp|pointed|edged|cornered', re.IGNORECASE),
            'smooth': re.compile(r'smooth|flowing|continuous|seamless', re.IGNORECASE),
            'jagged': re.compile(r'jagged|rough|serrated|zigzag', re.IGNORECASE),
            
            # Complexity indicators
            'simple': re.compile(r'simple|basic|plain|clean|minimal', re.IGNORECASE),
            'complex': re.compile(r'complex|complicated|intricate|detailed|elaborate', re.IGNORECASE),
            'composite': re.compile(r'composite|combined|multiple|compound|mixed', re.IGNORECASE),
            'multi_part': re.compile(r'multi.*part|several.*part|multiple.*element', re.IGNORECASE),
            
            # Positional and orientational
            'horizontal': re.compile(r'horizontal|sideways|lying.*down', re.IGNORECASE),
            'vertical': re.compile(r'vertical|upright|standing', re.IGNORECASE),
            'diagonal': re.compile(r'diagonal|slanted|angled|tilted', re.IGNORECASE),
            'centered': re.compile(r'centered|middle|central|balanced', re.IGNORECASE),
            'offset': re.compile(r'offset|shifted|displaced|moved', re.IGNORECASE),
            
            # Specific Bongard patterns
            'has_hole': re.compile(r'hole|hollow|empty.*inside|void', re.IGNORECASE),
            'has_parts': re.compile(r'parts|pieces|segments|components', re.IGNORECASE),
            'mismatch': re.compile(r'mismatch|different|varying|mixed', re.IGNORECASE),
            'necked': re.compile(r'necked|narrow.*connection|bottleneck', re.IGNORECASE),
            'acute': re.compile(r'acute|sharp.*angle|pointed', re.IGNORECASE),
            'obtuse': re.compile(r'obtuse|wide.*angle|blunt', re.IGNORECASE),
            'right_angle': re.compile(r'right.*angle|90.*degree|perpendicular', re.IGNORECASE),
        }
        
        # Enhanced topology classification patterns
        self.topology_patterns = {
            # Basic topology
            'closed': re.compile(r'closed|complete|loop|cycle|sealed|full.*circuit', re.IGNORECASE),
            'open': re.compile(r'open|incomplete|partial|broken|cut|gap', re.IGNORECASE),
            'connected': re.compile(r'connect|continuous|unbroken|linked|joined', re.IGNORECASE),
            'disconnected': re.compile(r'disconnect|separate|isolated|split|broken.*apart', re.IGNORECASE),
            
            # Complexity topology
            'simple': re.compile(r'simple|basic|plain|clean|minimal|elementary', re.IGNORECASE),
            'complex': re.compile(r'complex|complicated|intricate|elaborate|multi.*part', re.IGNORECASE),
            'compound': re.compile(r'compound|composite|combined|multiple.*shape', re.IGNORECASE),
            'nested': re.compile(r'nested|inside|within|concentric|layered', re.IGNORECASE),
            
            # Geometric topology
            'convex': re.compile(r'convex|outward.*curve|bulging|rounded.*out', re.IGNORECASE),
            'concave': re.compile(r'concave|inward.*curve|dented|rounded.*in', re.IGNORECASE),
            'self_intersecting': re.compile(r'self.*intersect|cross.*itself|overlap.*itself', re.IGNORECASE),
            
            # Connectivity patterns
            'branching': re.compile(r'branch|fork|tree.*like|radiating', re.IGNORECASE),
            'linear': re.compile(r'linear|straight.*line|sequential|chain', re.IGNORECASE),
            'circular': re.compile(r'circular|round|ring.*like|cyclic', re.IGNORECASE),
            'mesh': re.compile(r'mesh|grid|network|interconnected', re.IGNORECASE),
            
            # Hole and genus topology
            'hole': re.compile(r'hole|hollow|void|empty.*space|donut.*like', re.IGNORECASE),
            'no_hole': re.compile(r'solid|filled|no.*hole|complete', re.IGNORECASE),
            'multi_hole': re.compile(r'multi.*hole|several.*hole|many.*hole', re.IGNORECASE),
            
            # Boundary properties
            'smooth_boundary': re.compile(r'smooth.*boundary|curved.*edge|flowing.*border', re.IGNORECASE),
            'rough_boundary': re.compile(r'rough.*boundary|jagged.*edge|serrated.*border', re.IGNORECASE),
            'mixed_boundary': re.compile(r'mixed.*boundary|curved.*and.*straight|varying.*edge', re.IGNORECASE),
        }
        
        # Enhanced symmetry detection patterns
        self.symmetry_patterns = {
            # Bilateral symmetry
            'bilateral': re.compile(r'bilateral|mirror|reflection|symmetric|left.*right.*symm', re.IGNORECASE),
            'vertical_symmetry': re.compile(r'vertical.*symm|up.*down.*symm|mirror.*vertical', re.IGNORECASE),
            'horizontal_symmetry': re.compile(r'horizontal.*symm|left.*right.*symm|mirror.*horizontal', re.IGNORECASE),
            'diagonal_symmetry': re.compile(r'diagonal.*symm|slanted.*symm|tilted.*symm', re.IGNORECASE),
            
            # Rotational symmetry
            'rotational': re.compile(r'rotational|circular|radial|spinning.*symm', re.IGNORECASE),
            'point_symmetry': re.compile(r'point.*symmetry|central.*symm|180.*degree.*symm', re.IGNORECASE),
            'three_fold': re.compile(r'three.*fold|triple|120.*degree.*symm', re.IGNORECASE),
            'four_fold': re.compile(r'four.*fold|quadruple|90.*degree.*symm', re.IGNORECASE),
            
            # Asymmetry
            'none': re.compile(r'asymmetric|irregular|random|no.*symm|unbalanced', re.IGNORECASE),
            'broken_symmetry': re.compile(r'broken.*symm|partial.*symm|almost.*symm', re.IGNORECASE),
            
            # Complex symmetry patterns
            'multiple_axes': re.compile(r'multiple.*axes|several.*symm|many.*symm.*line', re.IGNORECASE),
            'fractal': re.compile(r'fractal|self.*similar|recursive.*pattern', re.IGNORECASE),
            'translational': re.compile(r'translational|repeated|periodic|tiled', re.IGNORECASE),
            'glide_reflection': re.compile(r'glide.*reflection|sliding.*mirror', re.IGNORECASE),
        }
        
        SemanticActionParser._initialized = True
    
    def identify_stroke_specific_shape(self, node: Dict[str, Any]) -> str:
        """
        Identify stroke-specific shape combining shape type and stroke type.
        
        Returns format: '{stroke_type}_{shape_type}' for valid Bongard shapes.
        Examples: 'line_triangle', 'arc_circle', 'line_normal', etc.
        """
        stroke_type = node.get('stroke_type', 'unknown')
        shape_type = node.get('shape_type', 'unknown')
        
        # Only combine if both are valid
        if stroke_type in ['line', 'arc'] and shape_type in ['normal', 'circle', 'square', 'triangle', 'zigzag']:
            return f"{stroke_type}_{shape_type}"
        
        # Fallback to individual types
        if stroke_type in ['line', 'arc']:
            return stroke_type
        if shape_type in ['normal', 'circle', 'square', 'triangle', 'zigzag']:
            return shape_type
            
        return 'unknown'
    
    def get_stroke_specific_properties(self, stroke_shape_combo: str) -> Dict[str, Any]:
        """Get properties specific to stroke+shape combination"""
        properties = {
            'line_normal': {'curvature': 0.0, 'angular': True, 'smooth': False, 'corners': True},
            'arc_normal': {'curvature': 'variable', 'angular': False, 'smooth': True, 'corners': False},
            'line_triangle': {'curvature': 0.0, 'angular': True, 'corners': 3, 'angle_sum': 180},
            'arc_triangle': {'curvature': 'variable', 'angular': False, 'corners': 0, 'smooth_edges': 3},
            'line_square': {'curvature': 0.0, 'angular': True, 'corners': 4, 'angle_sum': 360},
            'arc_square': {'curvature': 'variable', 'angular': False, 'corners': 0, 'smooth_edges': 4},
            'line_circle': {'curvature': 0.0, 'angular': True, 'approximation': True},  # Polygonal approximation
            'arc_circle': {'curvature': 'constant', 'angular': False, 'smooth': True, 'perfect_round': True},
            'line_zigzag': {'curvature': 0.0, 'angular': True, 'jagged': True, 'irregular': True},
            'arc_zigzag': {'curvature': 'variable', 'angular': False, 'wave_like': True, 'smooth_irregular': True}
        }
        
        return properties.get(stroke_shape_combo, {})
    
    def differentiate_stroke_types_in_reasoning(self, node_a: Dict, node_b: Dict = None) -> List[str]:
        """Generate predicates that differentiate stroke types"""
        predicates = []
        
        if node_b is None:
            # Single node analysis
            stroke_shape = self.identify_stroke_specific_shape(node_a)
            if 'line_' in stroke_shape:
                predicates.extend(['has_angular_features', 'has_sharp_edges', 'is_linear_based'])
            elif 'arc_' in stroke_shape:
                predicates.extend(['has_curved_features', 'has_smooth_edges', 'is_arc_based'])
        else:
            # Comparative analysis
            stroke_shape_a = self.identify_stroke_specific_shape(node_a)
            stroke_shape_b = self.identify_stroke_specific_shape(node_b)
            
            stroke_a = stroke_shape_a.split('_')[0] if '_' in stroke_shape_a else 'unknown'
            stroke_b = stroke_shape_b.split('_')[0] if '_' in stroke_shape_b else 'unknown'
            
            if stroke_a != stroke_b:
                if stroke_a == 'line' and stroke_b == 'arc':
                    predicates.extend(['different_stroke_types', 'angular_vs_curved', 'sharp_vs_smooth'])
                elif stroke_a == 'arc' and stroke_b == 'line':
                    predicates.extend(['different_stroke_types', 'curved_vs_angular', 'smooth_vs_sharp'])
            else:
                predicates.append('same_stroke_type')
                
            # Check for shape differences within same stroke type
            if stroke_a == stroke_b:
                shape_a = stroke_shape_a.split('_')[1] if '_' in stroke_shape_a else stroke_shape_a
                shape_b = stroke_shape_b.split('_')[1] if '_' in stroke_shape_b else stroke_shape_b
                if shape_a != shape_b:
                    predicates.append(f'same_{stroke_a}_different_shapes')
        
        return predicates
    
    def extract_semantic_intent(self, action_program: List[str]) -> Dict[str, Any]:
        """Extract comprehensive semantic shape intent from action program"""
        if not action_program:
            return {'shapes': [], 'properties': {}, 'comprehensive_features': None}

        # Handle nested action program structure from Bongard-LOGO dataset
        # Action programs can be: List[str] or nested structure
        flattened_commands = self._flatten_action_program(action_program)
        
        if not flattened_commands:
            return {'shapes': [], 'properties': {}, 'comprehensive_features': None}

        # Analyze program name and structure
        program_name = str(flattened_commands) if isinstance(flattened_commands, list) else str(action_program)
        
        semantic_shapes = []
        global_properties = defaultdict(int)
        
        # Extract shape types from action commands
        shape_counts = defaultdict(int)
        topology_features = defaultdict(int)
        symmetry_features = defaultdict(int)
        
        for cmd in flattened_commands:
            if not isinstance(cmd, str):
                continue
                
            # Check for semantic shape indicators
            for shape_type, pattern in self.shape_patterns.items():
                if pattern.search(cmd):
                    shape_counts[shape_type] += 1
                    
            # Extract properties
            for prop_type, pattern in self.property_patterns.items():
                if pattern.search(cmd):
                    global_properties[prop_type] += 1
                    
            # Extract topology
            for topo_type, pattern in self.topology_patterns.items():
                if pattern.search(cmd):
                    topology_features[topo_type] += 1
                    
            # Extract symmetry
            for symm_type, pattern in self.symmetry_patterns.items():
                if pattern.search(cmd):
                    symmetry_features[symm_type] += 1
        
        # Convert shape counts to semantic shapes
        for shape_type, count in shape_counts.items():
            if count > 0:
                semantic_shapes.append({
                    'type': shape_type,
                    'count': count,
                    'confidence': min(1.0, count / len(flattened_commands))
                })
        
        # If no explicit shapes found, infer from geometric structure
        if not semantic_shapes:
            inferred_shapes = self._infer_shapes_from_structure(flattened_commands)
            semantic_shapes.extend(inferred_shapes)
        
        # For unknown or low-confidence shapes, perform geometric analysis
        if not semantic_shapes or any(s.get('confidence', 0) < 0.5 for s in semantic_shapes):
            geometric_analysis = self._perform_geometric_analysis(flattened_commands)
            semantic_shapes = self._merge_geometric_analysis(semantic_shapes, geometric_analysis)
        
        # Compute comprehensive features
        comprehensive_features = self._compute_comprehensive_features(
            semantic_shapes, global_properties, topology_features, symmetry_features, flattened_commands
        )
        
        # Apply commonsense knowledge for unknown shapes
        commonsense_analysis = None
        if comprehensive_features and comprehensive_features.shape_type == BongardShapeType.UNKNOWN:
            predicate_engine = BongardPredicateEngine()
            mock_node = {'comprehensive_features': comprehensive_features, 'semantic_shapes': semantic_shapes}
            commonsense_analysis = predicate_engine.analyze_with_commonsense(mock_node)
        
        return {
            'shapes': semantic_shapes,
            'properties': dict(global_properties),
            'topology': dict(topology_features),
            'symmetry': dict(symmetry_features),
            'complexity': len(flattened_commands),
            'semantic_features': self._compute_semantic_features(semantic_shapes, global_properties),
            'comprehensive_features': comprehensive_features,
            'geometric_analysis': self._extract_detailed_geometric_attributes(semantic_shapes, flattened_commands),
            'commonsense_analysis': commonsense_analysis
        }
    
    def _flatten_action_program(self, action_program) -> List[str]:
        """
        Flatten nested action program structure to a list of command strings.
        Handles Bongard-LOGO dataset structure where action_program can be nested.
        """
        flattened = []
        
        def flatten_recursive(item):
            if isinstance(item, str):
                flattened.append(item)
            elif isinstance(item, list):
                for sub_item in item:
                    flatten_recursive(sub_item)
            # Skip non-string, non-list items
        
        try:
            flatten_recursive(action_program)
        except Exception as e:
            logging.warning(f"Error flattening action program: {e}")
            # Fallback to simple conversion
            if isinstance(action_program, list):
                flattened = [str(cmd) for cmd in action_program if cmd is not None]
            else:
                flattened = [str(action_program)]
        
        # Filter out empty strings and normalize commands
        flattened = [cmd.strip() for cmd in flattened if cmd and str(cmd).strip()]
        
        logging.debug(f"Flattened action program: {len(flattened)} commands from input {type(action_program)}")
        return flattened
    
    def _perform_geometric_analysis(self, action_program: List[str]) -> Dict[str, Any]:
        """Perform detailed geometric analysis for unknown shapes"""
        analysis = {
            'vertex_count': 0,
            'edge_count': 0,
            'angle_count': 0,
            'curve_count': 0,
            'intersection_count': 0,
            'symmetry_axes': [],
            'convexity_score': 0.5,
            'regularity_score': 0.5,
            'closure_detected': False,
            'shape_complexity': 'simple'
        }
        
        # Analyze command patterns for geometric features
        line_commands = [cmd for cmd in action_program if isinstance(cmd, str) and 'line' in cmd.lower()]
        arc_commands = [cmd for cmd in action_program if isinstance(cmd, str) and any(word in cmd.lower() for word in ['arc', 'curve', 'circle'])]
        turn_commands = [cmd for cmd in action_program if isinstance(cmd, str) and any(word in cmd.lower() for word in ['turn', 'angle', 'corner'])]
        
        analysis['edge_count'] = len(line_commands) + len(arc_commands)
        analysis['angle_count'] = len(turn_commands)
        analysis['curve_count'] = len(arc_commands)
        
        # Estimate vertices from edges and turns
        if len(line_commands) > 0:
            analysis['vertex_count'] = len(line_commands) + 1  # Lines have 2 vertices each, shared
            if len(turn_commands) > 0:
                analysis['vertex_count'] = max(analysis['vertex_count'], len(turn_commands))
        
        # Detect closure patterns
        first_command = action_program[0] if action_program else ""
        last_command = action_program[-1] if action_program else ""
        
        if isinstance(first_command, str) and isinstance(last_command, str):
            # Check if program returns to starting position
            if any(word in last_command.lower() for word in ['return', 'close', 'home', 'start']):
                analysis['closure_detected'] = True
            # Check if it's a closed shape based on edge count and angles
            elif analysis['edge_count'] == analysis['angle_count'] and analysis['edge_count'] >= 3:
                analysis['closure_detected'] = True
        
        # Estimate regularity from pattern consistency
        if len(set(line_commands)) == 1 and len(line_commands) > 1:  # All line commands identical
            analysis['regularity_score'] = 0.9
        elif len(line_commands) > 0:
            # Analyze variation in commands
            unique_commands = len(set(line_commands))
            analysis['regularity_score'] = max(0.1, 1.0 - (unique_commands / len(line_commands)))
        
        # Estimate convexity from turn patterns
        if len(turn_commands) == 0:
            analysis['convexity_score'] = 1.0  # Straight line or circle
        else:
            # Complex heuristic: consistent turn directions suggest convexity
            analysis['convexity_score'] = 0.7 if len(turn_commands) <= 4 else 0.4
        
        # Determine complexity
        total_commands = len(action_program)
        if total_commands <= 3:
            analysis['shape_complexity'] = 'simple'
        elif total_commands <= 8:
            analysis['shape_complexity'] = 'moderate'
        else:
            analysis['shape_complexity'] = 'complex'
        
        return analysis
    
    def _merge_geometric_analysis(self, semantic_shapes: List[Dict], geometric_analysis: Dict) -> List[Dict]:
        """Merge geometric analysis with semantic shape detection"""
        if not semantic_shapes:
            # Create shape from geometric analysis
            vertex_count = geometric_analysis.get('vertex_count', 0)
            is_closed = geometric_analysis.get('closure_detected', False)
            has_curves = geometric_analysis.get('curve_count', 0) > 0
            
            if has_curves and is_closed:
                shape_type = 'circle' if geometric_analysis.get('regularity_score', 0) > 0.7 else 'normal'
            elif is_closed and vertex_count >= 3:
                shape_type = {3: 'triangle', 4: 'square'}.get(vertex_count, 'normal')
            elif not is_closed and vertex_count >= 2:
                shape_type = 'zigzag' if vertex_count > 2 else 'normal'
            else:
                shape_type = 'normal'
            
            semantic_shapes = [{
                'type': shape_type,
                'count': 1,
                'confidence': 0.6 if shape_type != 'unknown' else 0.3,
                'geometric_basis': True,
                'vertex_count': vertex_count,
                'is_closed': is_closed
            }]
        else:
            # Enhance existing shapes with geometric data
            for shape in semantic_shapes:
                shape.update({
                    'geometric_vertex_count': geometric_analysis.get('vertex_count', 0),
                    'geometric_closure': geometric_analysis.get('closure_detected', False),
                    'geometric_regularity': geometric_analysis.get('regularity_score', 0.5),
                    'geometric_convexity': geometric_analysis.get('convexity_score', 0.5)
                })
        
        return semantic_shapes
    
    def _extract_detailed_geometric_attributes(self, shapes: List[Dict], action_program: List[str]) -> Dict[str, Any]:
        """Extract comprehensive geometric attributes for Bongard reasoning"""
        attributes = {
            # Basic counts
            'total_edges': 0,
            'total_vertices': 0,
            'total_angles': 0,
            'curve_segments': 0,
            'straight_segments': 0,
            
            # Shape properties
            'side_count_distribution': {},
            'angle_type_distribution': {},
            'symmetry_properties': {},
            'topological_features': {},
            
            # Compositional properties
            'shape_count': len(shapes),
            'shape_types': [s.get('type', 'unknown') for s in shapes],
            'dominant_shape_type': None,
            'shape_complexity_level': 'simple',
            
            # Relational properties
            'has_regular_shapes': False,
            'has_irregular_shapes': False,
            'mixed_shape_types': len(set(s.get('type', 'unknown') for s in shapes)) > 1,
            'all_closed': all(s.get('is_closed', False) for s in shapes if 'is_closed' in s),
            'all_open': all(not s.get('is_closed', True) for s in shapes if 'is_closed' in s),
            
            # Advanced attributes
            'contains_curves_and_lines': False,
            'exhibits_scaling_pattern': False,
            'has_nested_structure': False,
            'follows_pattern_rule': False
        }
        
        # Aggregate basic counts
        for shape in shapes:
            attributes['total_vertices'] += shape.get('vertex_count', shape.get('sides', 0))
            attributes['total_edges'] += shape.get('sides', 0)
            attributes['total_angles'] += shape.get('angles', shape.get('sides', 0))
            
            if shape.get('type') in ['circle', 'arc', 'oval', 'spiral']:
                attributes['curve_segments'] += 1
            else:
                attributes['straight_segments'] += 1
        
        # Analyze side count distribution
        side_counts = [s.get('sides', 0) for s in shapes if s.get('sides', 0) > 0]
        if side_counts:
            for count in side_counts:
                attributes['side_count_distribution'][str(count)] = attributes['side_count_distribution'].get(str(count), 0) + 1
        
        # Determine dominant shape type
        if shapes:
            shape_type_counts = Counter(s.get('type', 'unknown') for s in shapes)
            attributes['dominant_shape_type'] = shape_type_counts.most_common(1)[0][0]
        
        # Analyze regularity
        regular_shapes = {'triangle', 'square', 'circle'}
        irregular_shapes = {'normal', 'zigzag', 'unknown'}
        
        for shape in shapes:
            if shape.get('type') in regular_shapes:
                attributes['has_regular_shapes'] = True
            elif shape.get('type') in irregular_shapes:
                attributes['has_irregular_shapes'] = True
        
        # Detect mixed content
        curve_shapes = sum(1 for s in shapes if s.get('type') in ['circle', 'arc', 'oval', 'spiral'])
        line_shapes = sum(1 for s in shapes if s.get('type') in ['triangle', 'square', 'normal', 'zigzag'])
        attributes['contains_curves_and_lines'] = curve_shapes > 0 and line_shapes > 0
        
        # Estimate complexity
        total_complexity = sum(1 for s in shapes) + len(action_program) / 5
        if total_complexity <= 2:
            attributes['shape_complexity_level'] = 'simple'
        elif total_complexity <= 6:
            attributes['shape_complexity_level'] = 'moderate'
        else:
            attributes['shape_complexity_level'] = 'complex'
        
        # Pattern analysis
        if len(shapes) > 1:
            # Check for scaling patterns
            if len(set(s.get('type', 'unknown') for s in shapes)) == 1:
                attributes['exhibits_scaling_pattern'] = True
            
            # Check for pattern rules
            side_counts = [s.get('sides', 0) for s in shapes if s.get('sides', 0) > 0]
            if len(side_counts) >= 2:
                differences = [side_counts[i+1] - side_counts[i] for i in range(len(side_counts)-1)]
                if len(set(differences)) == 1:  # Arithmetic progression
                    attributes['follows_pattern_rule'] = True
        
        return attributes
    
    def _compute_comprehensive_features(self, shapes: List[Dict], properties: Dict, 
                                      topology: Dict, symmetry: Dict, action_program: List[str]) -> ComprehensiveShapeFeatures:
        """Compute comprehensive features for Bongard reasoning"""
        # Determine primary shape type - prioritize discovered Bongard-LOGO shape types
        primary_shape = BongardShapeType.UNKNOWN
        if shapes:
            # Priority order: discovered Bongard-LOGO shape types first
            bongard_priorities = ['circle', 'triangle', 'square', 'zigzag', 'normal']
            
            # Find the highest priority shape type
            primary_shape_name = None
            for priority_shape in bongard_priorities:
                for shape in shapes:
                    if shape['type'] == priority_shape:
                        primary_shape_name = priority_shape
                        break
                if primary_shape_name:
                    break
            
            # If no Bongard priority shape found, default to first available shape
            if not primary_shape_name and shapes:
                primary_shape_name = shapes[0]['type']
            
            # Final fallback to first shape
            if not primary_shape_name and shapes:
                primary_shape_name = shapes[0]['type']
            
            # Convert to BongardShapeType enum
            try:
                primary_shape = BongardShapeType(primary_shape_name)
            except ValueError:
                primary_shape = BongardShapeType.UNKNOWN
        
        # Count sides based on shape type
        side_count = self._estimate_side_count(shapes, action_program)
        
        # Determine topology
        topo = BongardTopology.SIMPLE
        if topology.get('closed', 0) > 0:
            topo = BongardTopology.CLOSED
        elif topology.get('open', 0) > 0:
            topo = BongardTopology.OPEN
        elif topology.get('complex', 0) > 0:
            topo = BongardTopology.COMPLEX
        
        # Determine symmetry
        symm = BongardSymmetry.NONE
        if symmetry.get('bilateral', 0) > 0:
            symm = BongardSymmetry.BILATERAL
        elif symmetry.get('rotational', 0) > 0:
            symm = BongardSymmetry.ROTATIONAL
        elif symmetry.get('point', 0) > 0:
            symm = BongardSymmetry.POINT
        
        # Compute geometric properties
        convexity = self._estimate_convexity(shapes, properties)
        regularity = self._estimate_regularity(shapes, properties, symmetry)
        complexity_score = self._compute_complexity_score(shapes, action_program)
        
        # Size categorization
        size_category = self._categorize_size(properties)
        
        return ComprehensiveShapeFeatures(
            shape_type=primary_shape,
            side_count=side_count,
            angle_count=side_count if side_count > 0 else 0,
            vertex_count=side_count if side_count > 0 else 0,
            topology=topo,
            is_closed=topology.get('closed', 0) > 0,
            is_connected=topology.get('connected', 0) > 0,
            has_holes=0,  # TODO: implement hole detection
            genus=0,
            symmetry_type=symm,
            symmetry_axes=[],  # TODO: compute actual axes
            rotation_order=self._estimate_rotation_order(symm, primary_shape),
            convexity=convexity,
            regularity=regularity,
            compactness=0.5,  # TODO: implement compactness calculation
            aspect_ratio=1.0,  # TODO: implement aspect ratio calculation
            size_category=size_category,
            relative_size=0.5,  # TODO: implement relative size
            complexity_score=complexity_score,
            stroke_count=len(action_program),
            intersection_count=0,  # TODO: implement intersection detection
            is_composite=len(shapes) > 1,
            component_shapes=[s['type'] for s in shapes],
            spatial_arrangement='linear',  # TODO: implement arrangement detection
            contains_curves=any(s['type'] in ['circle'] for s in shapes),
            contains_angles=any(s['type'] in ['triangle', 'square'] for s in shapes),
            regularity_score=regularity,
            symmetry_score=1.0 if symm != BongardSymmetry.NONE else 0.0
        )
    
    def _estimate_side_count(self, shapes: List[Dict], action_program: List[str]) -> int:
        """Estimate the number of sides based on shapes and action program"""
        if not shapes:
            return 0
        
        # Map shape types to side counts
        side_count_map = {
            'triangle': 3, 'square': 4, 'circle': 0, 'normal': 0, 'zigzag': 0
        }
        
        for shape in shapes:
            shape_type = shape['type']
            if shape_type in side_count_map:
                return side_count_map[shape_type]
        
        # Try to infer from action program
        line_count = sum(1 for cmd in action_program if 'line' in str(cmd).lower())
        if line_count >= 3:
            return line_count
        
        return 0
    
    def _estimate_convexity(self, shapes: List[Dict], properties: Dict) -> float:
        """Estimate convexity score"""
        if properties.get('convex', 0) > 0:
            return 1.0
        elif properties.get('concave', 0) > 0:
            return 0.0
        else:
            # Default estimation based on shape type
            convex_shapes = {'circle', 'triangle', 'square'}
            for shape in shapes:
                if shape['type'] in convex_shapes:
                    return 0.8
            return 0.5
    
    def _estimate_regularity(self, shapes: List[Dict], properties: Dict, symmetry: Dict) -> float:
        """Estimate regularity score"""
        if properties.get('symmetric', 0) > 0 or symmetry.get('bilateral', 0) > 0:
            return 0.9
        elif properties.get('asymmetric', 0) > 0 or properties.get('deformed', 0) > 0:
            return 0.1
        else:
            regular_shapes = {'circle', 'square', 'triangle'}
            for shape in shapes:
                if shape['type'] in regular_shapes:
                    return 0.7
            return 0.5
    
    def _compute_complexity_score(self, shapes: List[Dict], action_program: List[str]) -> float:
        """Compute complexity score based on shapes and actions"""
        base_complexity = len(shapes) * 0.3
        action_complexity = len(action_program) * 0.1
        # Only consider valid Bongard shape types - higher complexity for zigzag
        shape_complexity = sum(0.3 if s['type'] == 'zigzag' else 0.1 for s in shapes)
        return min(1.0, base_complexity + action_complexity + shape_complexity)
    
    def _categorize_size(self, properties: Dict) -> str:
        """Categorize size based on properties"""
        if properties.get('large', 0) > 0 or properties.get('huge', 0) > 0:
            return 'large'
        elif properties.get('small', 0) > 0 or properties.get('tiny', 0) > 0:
            return 'small'
        else:
            return 'medium'
    
    def _estimate_rotation_order(self, symmetry: BongardSymmetry, shape: BongardShapeType) -> int:
        """Estimate rotational symmetry order"""
        if symmetry == BongardSymmetry.ROTATIONAL:
            rotation_orders = {
                BongardShapeType.TRIANGLE: 3,
                BongardShapeType.SQUARE: 4,
                BongardShapeType.CIRCLE: 360,
                BongardShapeType.NORMAL: 1,
                BongardShapeType.ZIGZAG: 1
            }
            return rotation_orders.get(shape, 1)
        return 1
    
    def _infer_shapes_from_structure(self, action_program: List[str]) -> List[Dict[str, Any]]:
        """Infer semantic shapes from action program structure with comprehensive unknown shape analysis"""
        inferred = []
        
        line_count = sum(1 for cmd in action_program if isinstance(cmd, str) and 'line' in cmd.lower())
        arc_count = sum(1 for cmd in action_program if isinstance(cmd, str) and 'arc' in cmd.lower())
        turn_count = sum(1 for cmd in action_program if isinstance(cmd, str) and any(word in cmd.lower() for word in ['turn', 'angle', 'bend', 'corner']))
        
        # First, check for the 5 discovered Bongard-LOGO shape types directly
        shape_type_indicators = []
        for cmd in action_program:
            if isinstance(cmd, str):
                cmd_lower = cmd.lower()
                if 'normal' in cmd_lower:
                    shape_type_indicators.append('normal')
                elif 'circle' in cmd_lower:
                    shape_type_indicators.append('circle') 
                elif 'square' in cmd_lower:
                    shape_type_indicators.append('square')
                elif 'triangle' in cmd_lower:
                    shape_type_indicators.append('triangle')
                elif 'zigzag' in cmd_lower:
                    shape_type_indicators.append('zigzag')
        
        # If we found direct shape type indicators, use them
        if shape_type_indicators:
            most_common_type = max(set(shape_type_indicators), key=shape_type_indicators.count)
            confidence = shape_type_indicators.count(most_common_type) / len(shape_type_indicators)
            
            # Shape-specific properties based on discovered types
            shape_properties = {
                'normal': {'sides': 0, 'angles': 0, 'is_closed': False, 'complexity': 1},
                'circle': {'sides': 0, 'angles': 0, 'is_closed': True, 'complexity': 2},
                'square': {'sides': 4, 'angles': 4, 'is_closed': True, 'complexity': 2},
                'triangle': {'sides': 3, 'angles': 3, 'is_closed': True, 'complexity': 2},
                'zigzag': {'sides': 0, 'angles': 0, 'is_closed': False, 'complexity': 3}
            }
            
            props = shape_properties.get(most_common_type, {'sides': 0, 'angles': 0, 'is_closed': None, 'complexity': 1})
            inferred.append({
                'type': most_common_type, 
                'count': 1, 
                'confidence': confidence, 
                'sides': props['sides'], 
                'angles': props['angles'], 
                'is_closed': props['is_closed'],
                'shape_complexity': props['complexity'],
                'detection_method': 'direct_shape_type'
            })
            return inferred
        
        # Fall back to structural inference if no direct indicators
        # Comprehensive shape inference based on discovered Bongard-LOGO shape types
        if line_count == 3:  # Likely triangle
            inferred.append({'type': 'triangle', 'count': 1, 'confidence': 0.8, 'sides': 3, 'angles': 3, 'is_closed': True})
        elif line_count == 4:  # Likely square
            inferred.append({'type': 'square', 'count': 1, 'confidence': 0.8, 'sides': 4, 'angles': 4, 'is_closed': True})
        elif arc_count > 0 and line_count == 0:  # Likely circle
            inferred.append({'type': 'circle', 'count': 1, 'confidence': 0.8, 'sides': 0, 'angles': 0, 'is_closed': True})
        elif line_count > 4:  # Irregular multi-line shape - map to zigzag
            inferred.append({'type': 'zigzag', 'count': 1, 'confidence': 0.6, 'sides': line_count, 'angles': turn_count, 'is_closed': False})
        elif line_count > 0 and arc_count > 0:  # Mixed shape - map to normal
            inferred.append({'type': 'normal', 'count': 1, 'confidence': 0.5, 'sides': line_count, 'angles': turn_count, 'is_closed': None})
        elif line_count <= 2:  # Simple lines - map to normal
            inferred.append({'type': 'normal', 'count': 1, 'confidence': 0.7, 'sides': line_count, 'angles': 0, 'is_closed': False})
        else:
            # Unknown shape - analyze pattern
            unknown_analysis = self._analyze_unknown_shape_pattern(action_program)
            inferred.append(unknown_analysis)
        
        return inferred

    def _analyze_unknown_shape_pattern(self, action_program: List[str]) -> Dict[str, Any]:
        """Comprehensive analysis for unknown shapes with enhanced complex shape detection"""
        pattern_analysis = {
            'type': 'unknown',
            'count': 1,
            'confidence': 0.3,
            'sides': 0,
            'angles': 0,
            'is_closed': None,
            'analysis_method': 'pattern_recognition'
        }
        
        # Count different action types
        move_commands = sum(1 for cmd in action_program if isinstance(cmd, str) and any(word in cmd.lower() for word in ['move', 'goto', 'jump']))
        draw_commands = sum(1 for cmd in action_program if isinstance(cmd, str) and any(word in cmd.lower() for word in ['line', 'draw', 'stroke']))
        curve_commands = sum(1 for cmd in action_program if isinstance(cmd, str) and any(word in cmd.lower() for word in ['arc', 'curve', 'circle', 'round']))
        # Only count commands for valid Bongard shape types
        complex_commands = sum(1 for cmd in action_program if isinstance(cmd, str) and any(word in cmd.lower() for word in ['zigzag', 'jagged', 'wave']))
        
        # Enhanced: Check for complex composite shapes first
        composite_analysis = self._detect_complex_composite_shapes(action_program)
        if composite_analysis['is_composite']:
            pattern_analysis.update({
                'type': composite_analysis['composite_type'],
                'confidence': composite_analysis['confidence'],
                'sides': 0,  # Composite shapes don't have simple side counts
                'angles': 0,
                'is_closed': True,  # Most composite shapes are closed
                'analysis_method': 'composite_shape_detection',
                'composite_components': composite_analysis['components'],
                'pattern_characteristics': composite_analysis['pattern_characteristics']
            })
            return pattern_analysis
        
        # REMOVED: quarter_circle detection from line segments - use action program types only
        # All quarter_circle detection should come from action commands

        # Enhanced: Check for semicircle patterns
        semicircle_analysis = self._detect_semicircle_from_lines(action_program)
        if semicircle_analysis['is_semicircle']:
            pattern_analysis.update({
                'type': 'semicircle',
                'confidence': semicircle_analysis['confidence'],
                'sides': 0,
                'angles': 0,
                'is_closed': False,
                'analysis_method': 'semicircle_detection',
                'arc_angle': semicircle_analysis['arc_angle'],
                'radius_estimate': semicircle_analysis['radius']
            })
            return pattern_analysis
        
        # Enhanced: Check for general arc patterns
        arc_analysis = self._detect_arc_from_lines(action_program)
        if arc_analysis['is_arc']:
            pattern_analysis.update({
                'type': 'arc',
                'confidence': arc_analysis['confidence'],
                'sides': 0,
                'angles': 0,
                'is_closed': False,
                'analysis_method': 'arc_detection',
                'arc_angle': arc_analysis['arc_angle'],
                'curvature_score': arc_analysis['curvature']
            })
            return pattern_analysis
        
        # Enhanced: Check for irregular complex shapes
        irregular_analysis = self._detect_irregular_complex_shapes(action_program)
        if irregular_analysis['is_irregular']:
            pattern_analysis.update({
                'type': irregular_analysis['irregular_type'],
                'confidence': irregular_analysis['confidence'],
                'sides': 0,
                'angles': 0,
                'is_closed': None,
                'analysis_method': 'irregular_shape_detection',
                'complexity_metrics': irregular_analysis['complexity_metrics'],
                'pattern_analysis': irregular_analysis['pattern_analysis']
            })
            return pattern_analysis
        
        # Analyze pattern complexity
        complexity_score = len(action_program) / 10.0  # Normalize by typical program length
        
        # Determine shape characteristics
        if complex_commands > 0:
            pattern_analysis.update({
                'type': 'zigzag',
                'confidence': 0.6,
                'complexity_score': complexity_score,
                'complex_elements': complex_commands
            })
        elif curve_commands > draw_commands:
            pattern_analysis.update({
                'type': 'curved',
                'confidence': 0.5,
                'sides': 0,
                'curved_segments': curve_commands
            })
        elif draw_commands > 0:
            pattern_analysis.update({
                'type': 'normal',
                'confidence': 0.4,
                'sides': draw_commands,
                'angles': max(0, draw_commands - 1)
            })
        
        # Estimate closure based on move vs draw ratio
        if move_commands > 0 and draw_commands > 0:
            closure_likelihood = draw_commands / (move_commands + draw_commands)
            pattern_analysis['is_closed'] = closure_likelihood > 0.7
            pattern_analysis['closure_confidence'] = closure_likelihood
        
        return pattern_analysis
    
    # REMOVED: _detect_quarter_circle_from_lines - Use action program types only
    # All quarter_circle detection should come from action commands, not line approximations
    
    def _detect_semicircle_from_lines(self, action_program: List[str]) -> Dict[str, Any]:
        """Detect semicircle patterns from connected line segments"""
        result = {
            'is_semicircle': False,
            'confidence': 0.0,
            'arc_angle': 0.0,
            'radius': 0.0
        }
        
        line_coords = self._extract_line_coordinates(action_program)
        if len(line_coords) < 5:  # Need more segments for semicircle
            return result
            
        if not self._are_lines_connected(line_coords):
            return result
            
        path_points = self._extract_path_points(line_coords)
        if len(path_points) < 6:
            return result
            
        analysis = self._analyze_path_curvature(path_points)
        
        # Semicircle specific checks (approximately 180 degrees)
        if analysis['total_angle'] >= 160 and analysis['total_angle'] <= 200:
            if analysis['curvature_consistency'] > 0.6:
                if analysis['radius_consistency'] > 0.5:
                    result.update({
                        'is_semicircle': True,
                        'confidence': min(0.8, analysis['curvature_consistency'] * analysis['radius_consistency']),
                        'arc_angle': analysis['total_angle'],
                        'radius': analysis['estimated_radius']
                    })
        
        return result
    
    def _detect_arc_from_lines(self, action_program: List[str]) -> Dict[str, Any]:
        """Detect general arc patterns from connected line segments"""
        result = {
            'is_arc': False,
            'confidence': 0.0,
            'arc_angle': 0.0,
            'curvature': 0.0
        }
        
        line_coords = self._extract_line_coordinates(action_program)
        if len(line_coords) < 2:
            return result
            
        if not self._are_lines_connected(line_coords):
            return result
            
        path_points = self._extract_path_points(line_coords)
        if len(path_points) < 3:
            return result
            
        analysis = self._analyze_path_curvature(path_points)
        
        # General arc checks (any significant curvature)
        if analysis['total_angle'] >= 30:  # Minimum arc angle
            if analysis['curvature_consistency'] > 0.4:  # Some curvature consistency
                result.update({
                    'is_arc': True,
                    'confidence': min(0.7, analysis['curvature_consistency'] * 0.8),
                    'arc_angle': analysis['total_angle'],
                    'curvature': analysis['average_curvature']
                })
        
        return result
    
    def _extract_line_coordinates(self, action_program: List[str]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Extract line coordinates from action program commands"""
        line_coords = []
        
        for cmd in action_program:
            if isinstance(cmd, str) and 'line_normal' in cmd:
                # Extract coordinates from line_normal commands
                # Expected format: line_normal_x.xxx-y.yyy
                try:
                    parts = cmd.split('_')
                    if len(parts) >= 3:
                        coord_part = parts[2]  # Should be like "1.000-0.500"
                        if '-' in coord_part:
                            x_str, y_str = coord_part.split('-')
                            dx, dy = float(x_str), float(y_str)
                            # This gives us the direction vector, we need start/end points
                            # For now, we'll work with relative coordinates
                            line_coords.append(((0.0, 0.0), (dx, dy)))
                except (ValueError, IndexError):
                    continue
        
        return line_coords
    
    def _are_lines_connected(self, line_coords: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> bool:
        """Check if line segments form a connected path"""
        if len(line_coords) < 2:
            return False
            
        tolerance = 0.001
        
        for i in range(len(line_coords) - 1):
            end_current = line_coords[i][1]
            start_next = line_coords[i + 1][0]
            
            # Check if end of current line connects to start of next line
            distance = math.sqrt((end_current[0] - start_next[0])**2 + (end_current[1] - start_next[1])**2)
            if distance > tolerance:
                return False
                
        return True
    
    def _extract_path_points(self, line_coords: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> List[Tuple[float, float]]:
        """Extract all points from connected line segments"""
        if not line_coords:
            return []
            
        points = [line_coords[0][0]]  # Start with first point
        
        for line in line_coords:
            points.append(line[1])  # Add end point of each line
            
        return points
    
    def _analyze_path_curvature(self, path_points: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze curvature characteristics of a path"""
        if len(path_points) < 3:
            return {'total_angle': 0, 'curvature_consistency': 0, 'radius_consistency': 0, 
                   'estimated_radius': 0, 'direction': 'unknown', 'average_curvature': 0}
        
        angles = []
        curvatures = []
        radii = []
        
        # Calculate angles between consecutive segments
        for i in range(1, len(path_points) - 1):
            p1, p2, p3 = path_points[i-1], path_points[i], path_points[i+1]
            
            # Vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Angle between vectors
            angle = self._angle_between_vectors(v1, v2)
            angles.append(abs(angle))
            
            # Curvature estimation
            segment_length = math.sqrt(v1[0]**2 + v1[1]**2)
            if segment_length > 0:
                curvature = abs(angle) / segment_length
                curvatures.append(curvature)
                
                # Radius estimation (rough approximation)
                if curvature > 0:
                    radius = 1.0 / curvature
                    radii.append(radius)
        
        # Calculate consistency metrics
        total_angle = sum(angles)
        
        curvature_consistency = 0.0
        if curvatures:
            avg_curvature = sum(curvatures) / len(curvatures)
            variance = sum((c - avg_curvature)**2 for c in curvatures) / len(curvatures)
            curvature_consistency = max(0, 1.0 - math.sqrt(variance) / (avg_curvature + 0.001))
        
        radius_consistency = 0.0
        estimated_radius = 0.0
        if radii:
            estimated_radius = sum(radii) / len(radii)
            variance = sum((r - estimated_radius)**2 for r in radii) / len(radii)
            radius_consistency = max(0, 1.0 - math.sqrt(variance) / (estimated_radius + 0.001))
        
        # Determine direction (clockwise/counterclockwise)
        direction = 'clockwise' if total_angle > 0 else 'counterclockwise'
        
        return {
            'total_angle': abs(total_angle),
            'curvature_consistency': curvature_consistency,
            'radius_consistency': radius_consistency,
            'estimated_radius': estimated_radius,
            'direction': direction,
            'average_curvature': sum(curvatures) / len(curvatures) if curvatures else 0
        }
    
    def _angle_between_vectors(self, v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """Calculate angle between two vectors in degrees"""
        # Normalize vectors
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 == 0 or len2 == 0:
            return 0.0
            
        norm_v1 = (v1[0] / len1, v1[1] / len1)
        norm_v2 = (v2[0] / len2, v2[1] / len2)
        
        # Dot product
        dot_product = norm_v1[0] * norm_v2[0] + norm_v1[1] * norm_v2[1]
        
        # Clamp to avoid numerical errors
        dot_product = max(-1.0, min(1.0, dot_product))
        
        # Angle in radians, then convert to degrees
        angle_rad = math.acos(dot_product)
        angle_deg = math.degrees(angle_rad)
        
        # Determine sign using cross product
        cross_product = norm_v1[0] * norm_v2[1] - norm_v1[1] * norm_v2[0]
        if cross_product < 0:
            angle_deg = -angle_deg
            
        return angle_deg
    
    def _compute_semantic_features(self, shapes: List[Dict], properties: Dict) -> Dict[str, Any]:
        """Compute high-level semantic features"""
        features = {
            # Shape-based features
            'has_triangles': any(s['type'] == 'triangle' for s in shapes),
            'has_squares': any(s['type'] == 'square' for s in shapes),
            'has_circles': any(s['type'] == 'circle' for s in shapes),
            'has_lines': any(s['type'] == 'line' for s in shapes),
            'shape_count': len(shapes),
            'shape_diversity': len(set(s['type'] for s in shapes)),
            
            # Geometric properties
            'is_open': properties.get('open', 0) > 0,
            'is_closed': properties.get('closed', 0) > 0,
            'is_deformed': properties.get('deformed', 0) > 0,
            'is_symmetric': properties.get('symmetric', 0) > 0,
            
            # Compositional features
            'is_composite': len(shapes) > 1,
            'complexity_score': sum(s['count'] for s in shapes),
            
            # Bongard-relevant features
            'has_three_sides': any(s['type'] == 'triangle' for s in shapes),
            'has_four_sides': any(s['type'] == 'square' for s in shapes),
            'has_curved_elements': any(s['type'] in ['circle', 'arc'] for s in shapes),
            'has_straight_elements': any(s['type'] in ['triangle', 'square', 'line'] for s in shapes)
        }
        
        return features

    def _detect_complex_composite_shapes(self, action_program: List[str]) -> Dict[str, Any]:
        """Detect complex composite shapes like stars, crosses, houses, etc."""
        result = {
            'is_composite': False,
            'composite_type': 'unknown',
            'confidence': 0.0,
            'components': [],
            'pattern_characteristics': {}
        }
        
        # For now, return basic analysis until full implementation
        # Check for common composite shape indicators
        special_commands = sum(1 for cmd in action_program if isinstance(cmd, str) and any(word in cmd.lower() for word in ['star', 'cross', 'house', 'arrow', 'flower']))
        
        if special_commands > 0:
            result.update({
                'is_composite': True,
                'composite_type': 'special',
                'confidence': 0.6,
                'components': special_commands,
                'pattern_characteristics': {'special_elements': special_commands}
            })
        
        return result

    def _detect_irregular_complex_shapes(self, action_program: List[str]) -> Dict[str, Any]:
        """Detect irregular complex shapes like blobs, freeform shapes, etc."""
        result = {
            'is_irregular': False,
            'irregular_type': 'unknown',
            'confidence': 0.0,
            'complexity_metrics': {},
            'pattern_analysis': {}
        }
        
        # Check for irregular shape indicators
        irregular_commands = sum(1 for cmd in action_program if isinstance(cmd, str) and any(word in cmd.lower() for word in ['blob', 'freeform', 'irregular', 'random', 'wiggle']))
        
        if irregular_commands > 0:
            result.update({
                'is_irregular': True,
                'irregular_type': 'freeform',
                'confidence': 0.7,
                'complexity_metrics': {'irregular_elements': irregular_commands},
                'pattern_analysis': {'randomness': irregular_commands / len(action_program)}
            })
        
        return result


class BongardPredicateEngine:
    """Comprehensive predicate engine for Bongard problem reasoning"""
    
    def __init__(self):
        self.predicates = {
            # Core shape predicates - only Bongard-LOGO discovered shapes
            'contains_triangle': self._contains_triangle,
            'contains_square': self._contains_square,
            'contains_circle': self._contains_circle,
            'contains_normal': self._contains_normal,
            'contains_zigzag': self._contains_zigzag,
            
            # Side count predicates
            'has_three_sides': self._has_three_sides,
            'has_four_sides': self._has_four_sides,
            'has_no_sides': self._has_no_sides,
            
            # Topology predicates
            'is_closed_figure': self._is_closed_figure,
            'is_open_figure': self._is_open_figure,
            'is_connected': self._is_connected,
            'is_disconnected': self._is_disconnected,
            'has_holes': self._has_holes,
            'is_simple_shape': self._is_simple_shape,
            'is_complex_shape': self._is_complex_shape,
            
            # Symmetry predicates
            'has_bilateral_symmetry': self._has_bilateral_symmetry,
            'has_rotational_symmetry': self._has_rotational_symmetry,
            'has_point_symmetry': self._has_point_symmetry,
            'is_symmetric': self._is_symmetric,
            'is_asymmetric': self._is_asymmetric,
            
            # Geometric property predicates
            'is_convex': self._is_convex,
            'is_concave': self._is_concave,
            'is_regular': self._is_regular,
            'is_irregular': self._is_irregular,
            'is_compact': self._is_compact,
            'is_elongated': self._is_elongated,
            
            # Size predicates
            'is_large': self._is_large,
            'is_small': self._is_small,
            'is_medium_size': self._is_medium_size,
            'has_similar_size': self._has_similar_size,
            
            # Compositional predicates
            'single_shape': self._single_shape,
            'multiple_shapes': self._multiple_shapes,
            'is_composite': self._is_composite,
            'shapes_connected': self._shapes_connected,
            'shapes_separated': self._shapes_separated,
            
            # Complexity predicates
            'low_complexity': self._low_complexity,
            'high_complexity': self._high_complexity,
            'many_strokes': self._many_strokes,
            'few_strokes': self._few_strokes,
            
            # Curve vs angle predicates
            'contains_curves': self._contains_curves,
            'contains_angles': self._contains_angles,
            'all_straight_lines': self._all_straight_lines,
            'all_curved_lines': self._all_curved_lines,
            
            # Relational predicates
            'shapes_similar': self._shapes_similar,
            'shapes_different': self._shapes_different,
            'same_side_count': self._same_side_count,
            'different_side_count': self._different_side_count,
            'same_topology': self._same_topology,
            'different_topology': self._different_topology,
            'same_symmetry': self._same_symmetry,
            'different_symmetry': self._different_symmetry,
            
            # Advanced Bongard predicates
            'exhibits_scaling': self._exhibits_scaling,
            'exhibits_rotation': self._exhibits_rotation,
            'exhibits_reflection': self._exhibits_reflection,
            'has_nested_shapes': self._has_nested_shapes,
            'forms_pattern': self._forms_pattern,
            'breaks_pattern': self._breaks_pattern,
            
            # Stroke-specific predicates for arc vs line differentiation
            'is_line_based': self._is_line_based,
            'is_arc_based': self._is_arc_based,
            'contains_line_normal': self._contains_line_normal,
            'contains_arc_normal': self._contains_arc_normal,
            'contains_line_triangle': self._contains_line_triangle,
            'contains_arc_triangle': self._contains_arc_triangle,
            'contains_line_square': self._contains_line_square,
            'contains_arc_square': self._contains_arc_square,
            'contains_line_circle': self._contains_line_circle,
            'contains_arc_circle': self._contains_arc_circle,
            'contains_line_zigzag': self._contains_line_zigzag,
            'contains_arc_zigzag': self._contains_arc_zigzag,
            'has_angular_features': self._has_angular_features,
            'has_curved_features': self._has_curved_features,
            'has_sharp_edges': self._has_sharp_edges,
            'has_smooth_edges': self._has_smooth_edges,
            'different_stroke_types': self._different_stroke_types,
            'same_stroke_type': self._same_stroke_type,
        }
        
        # Initialize comprehensive commonsense knowledge base
        self.commonsense_knowledge = self._initialize_commonsense_knowledge()
    
    def evaluate_predicates(self, node_a: Dict, node_b: Dict = None) -> List[str]:
        """Evaluate all relevant predicates for node(s)"""
        applicable_predicates = []
        
        for predicate_name, predicate_func in self.predicates.items():
            try:
                if node_b is None:
                    # Unary predicate
                    if predicate_func(node_a):
                        applicable_predicates.append(predicate_name)
                else:
                    # Binary predicate
                    if predicate_func(node_a, node_b):
                        applicable_predicates.append(predicate_name)
            except Exception as e:
                logging.debug(f"Predicate {predicate_name} failed: {e}")
                continue
        
        # Add stroke-specific differentiation predicates
        if node_b is not None:
            try:
                stroke_diff_predicates = self.differentiate_stroke_types_in_reasoning(node_a, node_b)
                applicable_predicates.extend(stroke_diff_predicates)
            except Exception as e:
                logging.debug(f"Stroke differentiation failed: {e}")
        else:
            # Add single-node stroke-specific predicates
            try:
                stroke_predicates = self.differentiate_stroke_types_in_reasoning(node_a)
                applicable_predicates.extend(stroke_predicates)
            except Exception as e:
                logging.debug(f"Single node stroke analysis failed: {e}")
        
        return applicable_predicates
    
    # Shape detection predicates
    def _contains_triangle(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.shape_type == BongardShapeType.TRIANGLE or features.side_count == 3
        return node.get('semantic_features', {}).get('has_triangles', False)
    
    def _contains_square(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.shape_type == BongardShapeType.SQUARE or (features.side_count == 4 and features.regularity > 0.7)
        return node.get('semantic_features', {}).get('has_squares', False)
    
    def _contains_circle(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.shape_type == BongardShapeType.CIRCLE
        return node.get('semantic_features', {}).get('has_circles', False)
    
    def _contains_normal(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.shape_type == BongardShapeType.NORMAL
        return any(s['type'] == 'normal' for s in node.get('semantic_shapes', []))
    
    def _contains_zigzag(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.shape_type == BongardShapeType.ZIGZAG
        return any(s['type'] == 'zigzag' for s in node.get('semantic_shapes', []))
    
    # Side count predicates
    def _has_three_sides(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.side_count == 3
        return node.get('semantic_features', {}).get('has_three_sides', False)
    
    def _has_four_sides(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.side_count == 4
        return node.get('semantic_features', {}).get('has_four_sides', False)
    
    def _has_no_sides(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.side_count == 0
        return node.get('semantic_features', {}).get('has_circles', False) or node.get('semantic_features', {}).get('has_lines', False)
    
    # Topology predicates
    def _is_closed_figure(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.is_closed
        return node.get('semantic_features', {}).get('is_closed', False)
    
    def _is_open_figure(self, node):
        features = node.get('comprehensive_features')
        if features:
            return not features.is_closed
        return node.get('semantic_features', {}).get('is_open', False)
    
    def _is_connected(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.is_connected
        return node.get('semantic_features', {}).get('shape_count', 0) == 1
    
    def _is_disconnected(self, node):
        features = node.get('comprehensive_features')
        if features:
            return not features.is_connected
        return node.get('semantic_features', {}).get('shape_count', 0) > 1
    
    def _has_holes(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.has_holes > 0
        return False  # TODO: implement hole detection
    
    def _is_simple_shape(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.complexity_score < 0.5
        return node.get('semantic_features', {}).get('complexity_score', 0) <= 1
    
    def _is_complex_shape(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.complexity_score >= 0.5
        return node.get('semantic_features', {}).get('complexity_score', 0) > 1
    
    # Symmetry predicates
    def _has_bilateral_symmetry(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.symmetry_type == BongardSymmetry.BILATERAL
        return node.get('semantic_features', {}).get('is_symmetric', False)
    
    def _has_rotational_symmetry(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.symmetry_type == BongardSymmetry.ROTATIONAL
        return False  # TODO: implement rotational symmetry detection
    
    def _has_point_symmetry(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.symmetry_type == BongardSymmetry.POINT
        return False  # TODO: implement point symmetry detection
    
    def _is_symmetric(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.symmetry_type != BongardSymmetry.NONE
        return self._has_bilateral_symmetry(node)
    
    def _is_asymmetric(self, node):
        return not self._is_symmetric(node)
    
    # Geometric property predicates
    def _is_convex(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.convexity > 0.7
        return False  # TODO: implement convexity detection
    
    def _is_concave(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.convexity < 0.3
        return False  # TODO: implement concavity detection
    
    def _is_regular(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.regularity > 0.7
        return node.get('semantic_features', {}).get('is_symmetric', False)
    
    def _is_irregular(self, node):
        return not self._is_regular(node)
    
    def _is_compact(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.compactness > 0.7
        return False  # TODO: implement compactness detection
    
    def _is_elongated(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.aspect_ratio > 2.0 or features.aspect_ratio < 0.5
        return False  # TODO: implement aspect ratio detection
    
    # Size predicates
    def _is_large(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.size_category == 'large'
        return False  # TODO: implement size detection
    
    def _is_small(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.size_category == 'small'
        return False  # TODO: implement size detection
    
    def _is_medium_size(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.size_category == 'medium'
        return False  # TODO: implement size detection
    
    def _has_similar_size(self, node_a, node_b):
        features_a = node_a.get('comprehensive_features')
        features_b = node_b.get('comprehensive_features')
        if features_a and features_b:
            return features_a.size_category == features_b.size_category
        return False
    
    # Compositional predicates
    def _single_shape(self, node):
        features = node.get('comprehensive_features')
        if features:
            return not features.is_composite
        return node.get('semantic_features', {}).get('shape_count', 0) == 1
    
    def _multiple_shapes(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.is_composite
        return node.get('semantic_features', {}).get('shape_count', 0) > 1
    
    def _is_composite(self, node):
        return self._multiple_shapes(node)
    
    def _shapes_connected(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.is_composite and features.is_connected
        return False  # TODO: implement connection detection
    
    def _shapes_separated(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.is_composite and not features.is_connected
        return False  # TODO: implement separation detection
    
    # Complexity predicates
    def _low_complexity(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.complexity_score < 0.3
        return self._is_simple_shape(node)
    
    def _high_complexity(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.complexity_score > 0.7
        return self._is_complex_shape(node)
    
    def _many_strokes(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.stroke_count > 5
        return False  # TODO: implement stroke counting
    
    def _few_strokes(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.stroke_count <= 3
        return False  # TODO: implement stroke counting
    
    # Curve vs angle predicates
    def _contains_curves(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.contains_curves
        return node.get('semantic_features', {}).get('has_curved_elements', False)
    
    def _contains_angles(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.contains_angles
        return node.get('semantic_features', {}).get('has_straight_elements', False)
    
    def _all_straight_lines(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.contains_angles and not features.contains_curves
        return self._contains_angles(node) and not self._contains_curves(node)
    
    def _all_curved_lines(self, node):
        features = node.get('comprehensive_features')
        if features:
            return features.contains_curves and not features.contains_angles
        return self._contains_curves(node) and not self._contains_angles(node)
    
    # Relational predicates (binary)
    def _shapes_similar(self, node_a, node_b):
        features_a = node_a.get('comprehensive_features')
        features_b = node_b.get('comprehensive_features')
        if features_a and features_b:
            return features_a.shape_type == features_b.shape_type
        
        # Fallback to semantic features
        semantic_a = set(node_a.get('semantic_features', {}).keys())
        semantic_b = set(node_b.get('semantic_features', {}).keys())
        return len(semantic_a.intersection(semantic_b)) > 0
    
    def _shapes_different(self, node_a, node_b):
        return not self._shapes_similar(node_a, node_b)
    
    def _same_side_count(self, node_a, node_b):
        features_a = node_a.get('comprehensive_features')
        features_b = node_b.get('comprehensive_features')
        if features_a and features_b:
            return features_a.side_count == features_b.side_count
        return False
    
    def _different_side_count(self, node_a, node_b):
        return not self._same_side_count(node_a, node_b)
    
    def _same_topology(self, node_a, node_b):
        features_a = node_a.get('comprehensive_features')
        features_b = node_b.get('comprehensive_features')
        if features_a and features_b:
            return features_a.topology == features_b.topology
        return False
    
    def _different_topology(self, node_a, node_b):
        return not self._same_topology(node_a, node_b)
    
    def _same_symmetry(self, node_a, node_b):
        features_a = node_a.get('comprehensive_features')
        features_b = node_b.get('comprehensive_features')
        if features_a and features_b:
            return features_a.symmetry_type == features_b.symmetry_type
        return False
    
    def _different_symmetry(self, node_a, node_b):
        return not self._same_symmetry(node_a, node_b)
    
    # Advanced Bongard predicates
    def _exhibits_scaling(self, node):
        # TODO: implement scaling detection
        return False
    
    def _exhibits_rotation(self, node):
        # TODO: implement rotation detection
        return False
    
    def _exhibits_reflection(self, node):
        return self._has_bilateral_symmetry(node)
    
    def _has_nested_shapes(self, node):
        # TODO: implement nested shape detection
        return False
    
    def _forms_pattern(self, node):
        # TODO: implement pattern detection
        return False
    
    def _breaks_pattern(self, node):
        # TODO: implement pattern breaking detection
        return False

    # Stroke-specific predicates for arc vs line differentiation
    def _is_line_based(self, node):
        """Check if shape is primarily line-based (angular)"""
        stroke_type = node.get('stroke_type', 'unknown')
        return stroke_type == 'line'
    
    def _is_arc_based(self, node):
        """Check if shape is primarily arc-based (curved)"""
        stroke_type = node.get('stroke_type', 'unknown')
        return stroke_type == 'arc'
    
    def _contains_line_normal(self, node):
        """Check for line-based normal shapes (straight lines)"""
        stroke_type = node.get('stroke_type', 'unknown')
        shape_type = node.get('shape_type', 'unknown')
        return stroke_type == 'line' and shape_type == 'normal'
    
    def _contains_arc_normal(self, node):
        """Check for arc-based normal shapes (curved lines)"""
        stroke_type = node.get('stroke_type', 'unknown')
        shape_type = node.get('shape_type', 'unknown')
        return stroke_type == 'arc' and shape_type == 'normal'
    
    def _contains_line_triangle(self, node):
        """Check for line-based triangular shapes (sharp-cornered triangles)"""
        stroke_type = node.get('stroke_type', 'unknown')
        shape_type = node.get('shape_type', 'unknown')
        return stroke_type == 'line' and shape_type == 'triangle'
    
    def _contains_arc_triangle(self, node):
        """Check for arc-based triangular shapes (rounded triangles)"""
        stroke_type = node.get('stroke_type', 'unknown')
        shape_type = node.get('shape_type', 'unknown')
        return stroke_type == 'arc' and shape_type == 'triangle'
    
    def _contains_line_square(self, node):
        """Check for line-based square shapes (sharp-cornered squares)"""
        stroke_type = node.get('stroke_type', 'unknown')
        shape_type = node.get('shape_type', 'unknown')
        return stroke_type == 'line' and shape_type == 'square'
    
    def _contains_arc_square(self, node):
        """Check for arc-based square shapes (rounded squares)"""
        stroke_type = node.get('stroke_type', 'unknown')
        shape_type = node.get('shape_type', 'unknown')
        return stroke_type == 'arc' and shape_type == 'square'
    
    def _contains_line_circle(self, node):
        """Check for line-based circular shapes (polygonal approximation)"""
        stroke_type = node.get('stroke_type', 'unknown')
        shape_type = node.get('shape_type', 'unknown')
        return stroke_type == 'line' and shape_type == 'circle'
    
    def _contains_arc_circle(self, node):
        """Check for arc-based circular shapes (smooth circles)"""
        stroke_type = node.get('stroke_type', 'unknown')
        shape_type = node.get('shape_type', 'unknown')
        return stroke_type == 'arc' and shape_type == 'circle'
    
    def _contains_line_zigzag(self, node):
        """Check for line-based zigzag shapes (sharp jagged patterns)"""
        stroke_type = node.get('stroke_type', 'unknown')
        shape_type = node.get('shape_type', 'unknown')
        return stroke_type == 'line' and shape_type == 'zigzag'
    
    def _contains_arc_zigzag(self, node):
        """Check for arc-based zigzag shapes (smooth wave patterns)"""
        stroke_type = node.get('stroke_type', 'unknown')
        shape_type = node.get('shape_type', 'unknown')
        return stroke_type == 'arc' and shape_type == 'zigzag'
    
    def _has_angular_features(self, node):
        """Check if shape has angular/sharp features"""
        stroke_type = node.get('stroke_type', 'unknown')
        if stroke_type == 'line':
            return True
        
        features = node.get('comprehensive_features')
        if features:
            return (features.angle_count > 0 or 
                    features.num_junctions > 0 or
                    features.orientation_variance > 30)
        return False
    
    def _has_curved_features(self, node):
        """Check if shape has curved/smooth features"""
        stroke_type = node.get('stroke_type', 'unknown')
        if stroke_type == 'arc':
            return True
        
        features = node.get('comprehensive_features')
        if features:
            return (features.curvature_score > 0.1 or 
                    features.max_curvature > 0.1 or
                    features.is_highly_curved)
        return False
    
    def _has_sharp_edges(self, node):
        """Check if shape has sharp edges (line-based)"""
        return self._is_line_based(node) and self._has_angular_features(node)
    
    def _has_smooth_edges(self, node):
        """Check if shape has smooth edges (arc-based)"""
        return self._is_arc_based(node) and self._has_curved_features(node)
    
    def _different_stroke_types(self, node_a, node_b):
        """Check if two nodes have different stroke types"""
        stroke_a = node_a.get('stroke_type', 'unknown')
        stroke_b = node_b.get('stroke_type', 'unknown')
        return stroke_a != stroke_b and stroke_a != 'unknown' and stroke_b != 'unknown'
    
    def _same_stroke_type(self, node_a, node_b):
        """Check if two nodes have the same stroke type"""
        stroke_a = node_a.get('stroke_type', 'unknown')
        stroke_b = node_b.get('stroke_type', 'unknown')
        return stroke_a == stroke_b and stroke_a != 'unknown'

    # Unknown shape analysis predicates
    def _is_recognizable_shape(self, node):
        """Check if shape is recognizable from standard categories"""
        features = node.get('comprehensive_features')
        if features:
            return features.shape_type != BongardShapeType.UNKNOWN
        
        semantic_shapes = node.get('semantic_shapes', [])
        return any(s.get('confidence', 0) > 0.7 for s in semantic_shapes)
    
    def _is_unknown_shape(self, node):
        """Check if shape requires further analysis"""
        return not self._is_recognizable_shape(node)
    
    def _requires_geometric_analysis(self, node):
        """Check if shape needs geometric analysis for classification"""
        features = node.get('comprehensive_features')
        if features:
            return features.shape_type == BongardShapeType.UNKNOWN
        
        semantic_shapes = node.get('semantic_shapes', [])
        return not semantic_shapes or any(s.get('confidence', 0) < 0.5 for s in semantic_shapes)
    
    def _has_identifiable_pattern(self, node):
        """Check if unknown shape follows identifiable pattern"""
        geometric_analysis = node.get('geometric_analysis', {})
        return geometric_analysis.get('follows_pattern_rule', False)
    
    def _exhibits_shape_transformation(self, node):
        """Check if shape shows transformation patterns"""
        geometric_analysis = node.get('geometric_analysis', {})
        return geometric_analysis.get('exhibits_scaling_pattern', False)
    
    def _follows_geometric_rules(self, node):
        """Check if shape follows standard geometric rules"""
        features = node.get('comprehensive_features')
        if features:
            shape_type = features.shape_type.value
            return shape_type in self.commonsense_knowledge['shape_hierarchies']['regular_polygons']
        return False
    
    # Advanced side count predicates for unknown shapes
    def _has_odd_sides(self, node):
        """Check if shape has odd number of sides"""
        features = node.get('comprehensive_features')
        if features and features.side_count > 0:
            return features.side_count % 2 == 1
        
        geometric_analysis = node.get('geometric_analysis', {})
        vertex_count = geometric_analysis.get('total_vertices', 0)
        return vertex_count > 0 and vertex_count % 2 == 1
    
    def _has_even_sides(self, node):
        """Check if shape has even number of sides"""
        features = node.get('comprehensive_features')
        if features and features.side_count > 0:
            return features.side_count % 2 == 0
        
        geometric_analysis = node.get('geometric_analysis', {})
        vertex_count = geometric_analysis.get('total_vertices', 0)
        return vertex_count > 0 and vertex_count % 2 == 0
    
    def _has_prime_number_sides(self, node):
        """Check if shape has prime number of sides"""
        features = node.get('comprehensive_features')
        side_count = 0
        
        if features and features.side_count > 0:
            side_count = features.side_count
        else:
            geometric_analysis = node.get('geometric_analysis', {})
            side_count = geometric_analysis.get('total_vertices', 0)
        
        return side_count in self.commonsense_knowledge['mathematical_sequences']['prime_numbers']
    
    def _has_more_than_six_sides(self, node):
        """Check if shape has more than 6 sides"""
        features = node.get('comprehensive_features')
        if features:
            return features.side_count > 6
        
        geometric_analysis = node.get('geometric_analysis', {})
        return geometric_analysis.get('total_vertices', 0) > 6
    
    def _has_fewer_than_four_sides(self, node):
        """Check if shape has fewer than 4 sides"""
        features = node.get('comprehensive_features')
        if features:
            return 0 < features.side_count < 4
        
        geometric_analysis = node.get('geometric_analysis', {})
        vertex_count = geometric_analysis.get('total_vertices', 0)
        return 0 < vertex_count < 4
    
    # Advanced geometric analysis predicates
    def _has_consistent_angles(self, node):
        """Check if shape has consistent internal angles"""
        features = node.get('comprehensive_features')
        if features:
            return features.regularity > 0.7
        
        geometric_analysis = node.get('geometric_analysis', {})
        return geometric_analysis.get('geometric_regularity', 0.5) > 0.7
    
    def _has_varying_edge_lengths(self, node):
        """Check if shape has varying edge lengths"""
        features = node.get('comprehensive_features')
        if features:
            return features.regularity < 0.3
        
        geometric_analysis = node.get('geometric_analysis', {})
        return geometric_analysis.get('geometric_regularity', 0.5) < 0.3
    
    def _exhibits_radial_pattern(self, node):
        """Check if shape exhibits radial symmetry pattern"""
        features = node.get('comprehensive_features')
        if features:
            return features.symmetry_type == BongardSymmetry.ROTATIONAL
        
        # Check for zigzag patterns that might exhibit radial properties
        semantic_shapes = node.get('semantic_shapes', [])
        return any(s.get('type') == 'zigzag' for s in semantic_shapes)
    
    def _has_fractal_properties(self, node):
        """Check if shape exhibits fractal-like properties"""
        features = node.get('comprehensive_features')
        if features:
            return features.is_composite and features.complexity_score > 0.8
        
        geometric_analysis = node.get('geometric_analysis', {})
        return geometric_analysis.get('shape_complexity_level') == 'complex'
    
    def _shows_progressive_complexity(self, node):
        """Check if shape shows progressive complexity patterns"""
        geometric_analysis = node.get('geometric_analysis', {})
        return geometric_analysis.get('follows_pattern_rule', False)
    
    # Commonsense reasoning predicates
    def _follows_natural_progression(self, node):
        """Check if shape follows natural mathematical progression"""
        features = node.get('comprehensive_features')
        if features and features.side_count > 0:
            return features.side_count in self.commonsense_knowledge['mathematical_sequences']['side_progressions']
        return False
    
    def _violates_geometric_expectation(self, node):
        """Check if shape violates expected geometric properties"""
        features = node.get('comprehensive_features')
        if not features:
            return True  # Unknown shapes violate expectations
        
        shape_type = features.shape_type.value
        expected_properties = self.commonsense_knowledge['side_count_properties'].get(shape_type, {})
        
        if expected_properties:
            expected_sides = expected_properties.get('sides', 0)
            return features.side_count != expected_sides
        
        return False
    
    def _exhibits_mathematical_sequence(self, node):
        """Check if shape properties follow mathematical sequences"""
        features = node.get('comprehensive_features')
        if features and features.side_count > 0:
            side_count = features.side_count
            for sequence_name, sequence in self.commonsense_knowledge['mathematical_sequences'].items():
                if side_count in sequence:
                    return True
        return False
    
    def _shows_visual_hierarchy(self, node):
        """Check if shape demonstrates visual hierarchy"""
        features = node.get('comprehensive_features')
        if features:
            return features.is_composite and len(features.component_shapes) > 1
        
        geometric_analysis = node.get('geometric_analysis', {})
        return geometric_analysis.get('shape_count', 1) > 1
    
    def _demonstrates_shape_evolution(self, node):
        """Check if shape shows evolutionary transformation"""
        geometric_analysis = node.get('geometric_analysis', {})
        return (geometric_analysis.get('exhibits_scaling_pattern', False) or 
                geometric_analysis.get('follows_pattern_rule', False))

    def map_unknown_shape_predicates(self, node: Dict[str, Any]) -> List[str]:
        """Map predicates for unknown shapes using comprehensive analysis"""
        applicable_predicates = []
        
        # Get comprehensive features
        features = node.get('comprehensive_features')
        geometric_analysis = node.get('geometric_analysis', {})
        semantic_features = node.get('semantic_features', {})
        
        # Unknown shape specific analysis
        if not features or features.shape_type == BongardShapeType.UNKNOWN:
            # Use geometric analysis for predicate mapping
            vertex_count = geometric_analysis.get('total_vertices', 0)
            edge_count = geometric_analysis.get('total_edges', 0)
            is_closed = geometric_analysis.get('all_closed', False)
            regularity = geometric_analysis.get('geometric_regularity', 0.5)
            convexity = geometric_analysis.get('geometric_convexity', 0.5)
            
            # Side count predicates for discovered shapes only
            if vertex_count == 3:
                applicable_predicates.extend(['has_three_sides', 'contains_triangle'])
            elif vertex_count == 4:
                applicable_predicates.extend(['has_four_sides', 'contains_square'])
            elif vertex_count > 4:
                applicable_predicates.extend(['has_many_sides', 'has_more_than_four_sides'])
            elif vertex_count == 0:
                applicable_predicates.extend(['has_no_sides', 'contains_circle'])
            
            # Odd/even side predicates
            if vertex_count > 0:
                if vertex_count % 2 == 0:
                    applicable_predicates.append('has_even_sides')
                else:
                    applicable_predicates.append('has_odd_sides')
                
                # Prime number sides
                if vertex_count in [2, 3, 5, 7, 11, 13, 17]:
                    applicable_predicates.append('has_prime_number_sides')
            
            # Topology predicates
            if is_closed:
                applicable_predicates.append('is_closed_figure')
            else:
                applicable_predicates.append('is_open_figure')
            
            # Geometric property predicates
            if regularity > 0.7:
                applicable_predicates.append('is_regular')
            elif regularity < 0.3:
                applicable_predicates.append('is_irregular')
            
            if convexity > 0.7:
                applicable_predicates.append('is_convex')
            elif convexity < 0.3:
                applicable_predicates.append('is_concave')
            
            # Complexity analysis
            complexity_level = geometric_analysis.get('shape_complexity_level', 'simple')
            if complexity_level == 'simple':
                applicable_predicates.append('is_simple_shape')
            elif complexity_level in ['moderate', 'complex']:
                applicable_predicates.append('is_complex_shape')
            
            # Unknown shape specific predicates
            applicable_predicates.extend([
                'is_unknown_shape',
                'requires_geometric_analysis'
            ])
            
            # Pattern analysis predicates
            if geometric_analysis.get('follows_pattern_rule', False):
                applicable_predicates.append('has_identifiable_pattern')
            
            if geometric_analysis.get('exhibits_scaling_pattern', False):
                applicable_predicates.append('exhibits_shape_transformation')
        
        # Use commonsense knowledge for additional predicates
        shape_type = features.shape_type.value if features else 'unknown'
        if shape_type in self.commonsense_knowledge['shape_hierarchies']['regular_polygons']:
            applicable_predicates.append('follows_geometric_rules')
        
        return applicable_predicates
    
    def analyze_with_commonsense(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Apply commonsense reasoning to analyze shapes"""
        analysis = {
            'shape_category': 'unknown',
            'reasoning_strategies': [],
            'mathematical_properties': {},
            'visual_properties': {},
            'bongard_relevance': 0.5
        }
        
        features = node.get('comprehensive_features')
        if not features:
            return analysis
        
        shape_type = features.shape_type.value
        side_count = features.side_count
        
        # Categorize shape using commonsense knowledge
        for category, shapes in self.commonsense_knowledge['shape_hierarchies'].items():
            if shape_type in shapes:
                analysis['shape_category'] = category
                break
        
        # Determine applicable reasoning strategies
        if side_count > 0:
            analysis['reasoning_strategies'].append('count_based')
        
        if features.symmetry_type != BongardSymmetry.NONE:
            analysis['reasoning_strategies'].append('property_based')
        
        if features.is_composite:
            analysis['reasoning_strategies'].append('compositional')
        
        # Mathematical property analysis
        if side_count in self.commonsense_knowledge['mathematical_sequences']['prime_numbers']:
            analysis['mathematical_properties']['is_prime_sided'] = True
        
        if side_count in self.commonsense_knowledge['mathematical_sequences']['perfect_squares']:
            analysis['mathematical_properties']['is_perfect_square_sided'] = True
        
        # Visual property analysis
        if shape_type in self.commonsense_knowledge['symmetry_rules']['bilateral']:
            analysis['visual_properties']['expected_bilateral_symmetry'] = True
        
        if shape_type in self.commonsense_knowledge['topological_categories']['simple_closed']:
            analysis['visual_properties']['expected_closure'] = True
        
        # Bongard relevance scoring
        relevance_score = 0.5
        if side_count in [3, 4, 5, 6]:  # Common Bongard shapes
            relevance_score += 0.3
        if features.symmetry_type != BongardSymmetry.NONE:
            relevance_score += 0.2
        if features.is_closed:
            relevance_score += 0.1
        analysis['bongard_relevance'] = min(1.0, relevance_score)
        
        return analysis
    
    def _initialize_commonsense_knowledge(self) -> Dict[str, Any]:
        """Initialize comprehensive commonsense knowledge base for Bongard reasoning"""
        return {
            'shape_hierarchies': {
                # DISCOVERED BONGARD-LOGO SHAPE TYPES (prioritized)
                'bongard_logo_shapes': ['normal', 'circle', 'square', 'triangle', 'zigzag'],
                'regular_polygons': ['triangle', 'square'],  # Only discovered regular polygons
                'curved_shapes': ['circle', 'zigzag'],  # Only discovered curved types
                'linear_shapes': ['normal'],  # Only discovered linear types
                'complex_shapes': ['zigzag'],  # Complex patterns from discovered types
                'irregular_shapes': ['zigzag'],  # Irregular patterns from discovered types
                
                # STROKE-SPECIFIC SHAPE HIERARCHIES
                'line_stroke_shapes': ['line_normal', 'line_triangle', 'line_square', 'line_zigzag'],
                'arc_stroke_shapes': ['arc_normal', 'arc_triangle', 'arc_square', 'arc_circle', 'arc_zigzag'],
                'angular_shapes': ['line_triangle', 'line_square'],  # Shapes with sharp corners
                'curved_shapes_detailed': ['arc_triangle', 'arc_square', 'arc_circle', 'arc_zigzag'],  # Shapes with curved edges
                'hybrid_shapes': []  # No hybrid stroke types in pure Bongard system
            },
            'bongard_shape_properties': {
                # Properties for the 5 discovered shape types
                'normal': {'sides': 0, 'regularity': 'linear', 'closure': 'open', 'complexity': 1, 'frequency': 24107},
                'circle': {'sides': 0, 'regularity': 'perfect', 'closure': 'closed', 'complexity': 2, 'frequency': 6256},
                'square': {'sides': 4, 'regularity': 'regular', 'closure': 'closed', 'complexity': 2, 'frequency': 6519},
                'triangle': {'sides': 3, 'regularity': 'regular', 'closure': 'closed', 'complexity': 2, 'frequency': 5837},
                'zigzag': {'sides': 0, 'regularity': 'irregular', 'closure': 'open', 'complexity': 3, 'frequency': 6729}
            },
            'side_count_properties': {
                3: {'name': 'triangle', 'angle_sum': 180, 'min_angles': 3},
                4: {'name': 'quadrilateral', 'angle_sum': 360, 'min_angles': 4},
                # Note: Only discovered shapes (triangle, square) have validated side counts
            },
            'symmetry_rules': {
                'bilateral': ['triangle', 'square', 'circle'],  # Only real Bongard shapes that have bilateral symmetry
                'rotational': ['square', 'circle'],  # Only discovered shapes with rotational symmetry
                'point': [],  # No discovered shapes have point symmetry
                'none': ['normal', 'zigzag']  # Linear and irregular discovered shapes
            },
            'topological_categories': {
                'simple_closed': ['triangle', 'square', 'circle'],  # Only discovered closed shapes
                'simple_open': ['normal', 'zigzag'],  # Only discovered open shapes
                'complex_closed': ['star', 'cross', 'house'],
                'complex_open': ['zigzag', 'spiral'],
                'compound': ['multiple_shapes', 'nested_shapes']
            },
            'bongard_reasoning_strategies': [
                'count_based',      # Count sides, angles, shapes
                'property_based',   # Size, symmetry, closure
                'relational',       # Relative positions, connections
                'compositional',    # Single vs multiple shapes
                'topological',      # Open vs closed, connected
                'geometric',        # Regular vs irregular, convex vs concave
                'pattern_based',    # Repetition, scaling, transformation
                'exclusion'         # What's NOT present
            ],
            'mathematical_sequences': {
                'prime_numbers': [2, 3, 5, 7, 11, 13, 17, 19, 23],
                'perfect_squares': [1, 4, 9, 16, 25, 36, 49, 64],
                'fibonacci': [1, 1, 2, 3, 5, 8, 13, 21, 34],
                'triangular': [1, 3, 6, 10, 15, 21, 28, 36],
                'powers_of_2': [1, 2, 4, 8, 16, 32, 64]
            },
            'visual_reasoning_principles': [
                'gestalt_closure',           # Tendency to see complete shapes
                'proximity_grouping',        # Group nearby elements
                'similarity_grouping',       # Group similar elements  
                'symmetry_detection',        # Recognize symmetrical patterns
                'figure_ground_separation',  # Distinguish object from background
                'continuity_principle',      # Follow continuous lines/curves
                'common_fate',               # Elements moving together are grouped
                'good_form',                 # Prefer simple, regular shapes
            ],
            'pattern_recognition_templates': {
                'scaling': ['increasing_size', 'decreasing_size', 'alternating_size'],
                'rotation': ['clockwise', 'counterclockwise', 'alternating_rotation'],
                'reflection': ['horizontal_flip', 'vertical_flip', 'diagonal_flip'],
                'translation': ['horizontal_shift', 'vertical_shift', 'diagonal_shift'],
                'color': ['color_progression', 'color_alternation', 'color_exclusion'],
                'shape': ['shape_progression', 'shape_alternation', 'shape_addition']
            }
        }


def enhance_node_with_semantic_features(node_data: Dict[str, Any], action_program: List[str]) -> Dict[str, Any]:
    """Enhance existing node data with semantic features"""
    print(f"🔍 Enhancing node with action_program: {action_program}")
    
    parser = SemanticActionParser()
    semantic_info = parser.extract_semantic_intent(action_program)
    
    print(f"🔍 Semantic info shapes: {semantic_info.get('shapes', [])}")
    
    # Add semantic information to existing node
    enhanced_node = node_data.copy()
    enhanced_node.update({
        'semantic_intent': semantic_info,
        'semantic_features': semantic_info['semantic_features'],
        'bongard_relevant': True,
        'semantic_shapes': semantic_info['shapes'],
        'semantic_properties': semantic_info['properties']
    })
    
    # SET SHAPE_TYPE BASED ON DISCOVERED BONGARD-LOGO SHAPE TYPES
    shapes = semantic_info.get('shapes', [])
    if shapes:
        # Priority order: discovered Bongard-LOGO shape types first
        bongard_priorities = ['circle', 'triangle', 'square', 'zigzag', 'normal']
        shape_type_set = False
        
        # Find the highest priority Bongard-LOGO shape type
        for priority_shape in bongard_priorities:
            for shape in shapes:
                if shape.get('type') == priority_shape:
                    enhanced_node['shape_type'] = priority_shape
                    print(f"🎯 Set shape_type to '{priority_shape}' based on semantic detection")
                    shape_type_set = True
                    break
            if shape_type_set:
                break
                
        # If no Bongard priority shape found, use the first detected shape
        if not shape_type_set and shapes:
            first_shape_type = shapes[0].get('type', 'normal')
            enhanced_node['shape_type'] = first_shape_type
            print(f"🎯 Set shape_type to '{first_shape_type}' as fallback from semantic detection")
    
    # REMOVED: quarter_circle composite shape detection - use action program types only
    for shape in shapes:
        shape_type = shape.get('type', '')
        print(f"🔍 Found shape type: {shape_type}")
        if shape_type in ['semicircle', 'arc']:  # Removed quarter_circle - action programs only
            enhanced_node['composite_type'] = shape_type
            enhanced_node['composite_confidence'] = shape.get('confidence', 0.0)
            enhanced_node['composite_properties'] = {
                'arc_angle': shape.get('arc_angle', 0),
                'radius_estimate': shape.get('radius_estimate', 0),
                'curvature_direction': shape.get('curvature_direction', 'unknown'),
                'analysis_method': shape.get('analysis_method', 'unknown')
            }
            print(f"🎯 Enhanced node with {shape_type} composite_type (confidence: {shape.get('confidence', 0.0):.3f})")
            break
    
    return enhanced_node

