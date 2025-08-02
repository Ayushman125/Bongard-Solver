"""
Semantic Action Parser for NV-Labs Bongard-LOGO Dataset

This module parses action programs from the NV-Labs dataset to extract
semantic shape information rather than geometric primitives.

Key insight: Action programs like 'line_triangle_1.000-0.500' represent
semantic intent (draw a triangle) not geometric coordinates.
"""

import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ShapeType(Enum):
    TRIANGLE = "triangle"
    SQUARE = "square" 
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    POLYGON = "polygon"
    ARC = "arc"
    LINE = "line"
    UNKNOWN = "unknown"

class LineType(Enum):
    NORMAL = "normal"
    ZIGZAG = "zigzag"
    ARROW = "arrow"
    CLASSIC = "classic"
    UNKNOWN = "unknown"

@dataclass
class SemanticShape:
    """Represents a semantic shape extracted from action program"""
    shape_type: ShapeType
    line_type: LineType
    size: float  # 0.0 to 1.0
    thickness: float  # 0.0 to 1.0
    properties: Dict
    action_string: str
    
    def is_geometric_shape(self) -> bool:
        """Check if this is a primary geometric shape"""
        return self.shape_type in [ShapeType.TRIANGLE, ShapeType.SQUARE, 
                                 ShapeType.RECTANGLE, ShapeType.CIRCLE]
    
    def is_decorative_element(self) -> bool:
        """Check if this is a decorative line element"""
        return self.line_type in [LineType.ZIGZAG, LineType.ARROW, LineType.CLASSIC]
    
    def get_semantic_features(self) -> Dict:
        """Extract semantic features for Bongard reasoning"""
        return {
            'shape_category': self.shape_type.value,
            'is_curved': self.shape_type in [ShapeType.CIRCLE, ShapeType.ARC],
            'is_angular': self.shape_type in [ShapeType.TRIANGLE, ShapeType.SQUARE, ShapeType.RECTANGLE],
            'is_closed': self.shape_type in [ShapeType.TRIANGLE, ShapeType.SQUARE, ShapeType.RECTANGLE, ShapeType.CIRCLE],
            'is_decorative': self.is_decorative_element(),
            'size_category': 'large' if self.size > 0.7 else 'medium' if self.size > 0.3 else 'small',
            'line_style': self.line_type.value,
            'complexity': self.get_complexity_score()
        }
    
    def get_complexity_score(self) -> float:
        """Compute complexity score for the shape"""
        base_complexity = {
            ShapeType.LINE: 1.0,
            ShapeType.ARC: 1.5,
            ShapeType.CIRCLE: 2.0,
            ShapeType.TRIANGLE: 3.0,
            ShapeType.SQUARE: 4.0,
            ShapeType.RECTANGLE: 4.0,
            ShapeType.POLYGON: 5.0
        }
        
        complexity = base_complexity.get(self.shape_type, 1.0)
        
        # Add complexity for decorative elements
        if self.line_type in [LineType.ZIGZAG, LineType.ARROW]:
            complexity += 1.0
            
        return complexity

class SemanticActionParser:
    """
    Parses NV-Labs action programs to extract semantic shapes
    
    Action format: 'line_[shape]_[size]-[thickness]'
    Examples:
    - 'line_triangle_1.000-0.500' → Triangle, size=1.0, thickness=0.5
    - 'line_square_0.750-0.250' → Square, size=0.75, thickness=0.25
    - 'line_circle_0.500-0.333' → Circle, size=0.5, thickness=0.33
    """
    
    def __init__(self):
        self.shape_patterns = {
            r'triangle': ShapeType.TRIANGLE,
            r'square': ShapeType.SQUARE,
            r'rectangle': ShapeType.RECTANGLE,
            r'circle': ShapeType.CIRCLE,
            r'polygon': ShapeType.POLYGON,
            r'arc': ShapeType.ARC,
            r'normal': ShapeType.LINE,
        }
        
        self.line_patterns = {
            r'normal': LineType.NORMAL,
            r'zigzag': LineType.ZIGZAG, 
            r'arrow': LineType.ARROW,
            r'classic': LineType.CLASSIC,
        }
    
    def parse_action_string(self, action: str) -> Optional[SemanticShape]:
        """
        Parse a single action string to extract semantic shape
        
        Args:
            action: Action string like 'line_triangle_1.000-0.500'
            
        Returns:
            SemanticShape object or None if parsing fails
        """
        try:
            # Clean the action string
            action = action.strip().lower()
            
            # Extract components using regex
            # Pattern: line_[shape]_[size]-[thickness]
            pattern = r'line_([a-z]+)_([0-9.]+)-([0-9.]+)'
            match = re.match(pattern, action)
            
            if not match:
                # Try alternative pattern without line_ prefix
                pattern = r'([a-z]+)_([0-9.]+)-([0-9.]+)'
                match = re.match(pattern, action)
            
            if not match:
                return None
            
            shape_name, size_str, thickness_str = match.groups()
            
            # Parse shape type
            shape_type = self._identify_shape_type(shape_name)
            line_type = self._identify_line_type(shape_name)
            
            # Parse numeric values
            size = float(size_str)
            thickness = float(thickness_str)
            
            # Create semantic shape
            semantic_shape = SemanticShape(
                shape_type=shape_type,
                line_type=line_type,
                size=size,
                thickness=thickness,
                properties={
                    'raw_shape_name': shape_name,
                    'parsed_successfully': True
                },
                action_string=action
            )
            
            return semantic_shape
            
        except Exception as e:
            print(f"Warning: Failed to parse action '{action}': {e}")
            return None
    
    def _identify_shape_type(self, shape_name: str) -> ShapeType:
        """Identify shape type from shape name"""
        for pattern, shape_type in self.shape_patterns.items():
            if pattern in shape_name:
                return shape_type
        return ShapeType.UNKNOWN
    
    def _identify_line_type(self, shape_name: str) -> LineType:
        """Identify line type from shape name"""
        for pattern, line_type in self.line_patterns.items():
            if pattern in shape_name:
                return line_type
        return LineType.UNKNOWN
    
    def parse_action_sequence(self, action_sequence: List[str]) -> List[SemanticShape]:
        """
        Parse a sequence of actions for one image
        
        Args:
            action_sequence: List of action strings
            
        Returns:
            List of SemanticShape objects
        """
        shapes = []
        
        for action in action_sequence:
            semantic_shape = self.parse_action_string(action)
            if semantic_shape:
                shapes.append(semantic_shape)
        
        return shapes
    
    def parse_problem_data(self, problem_data: List) -> Tuple[List[List[SemanticShape]], List[List[SemanticShape]]]:
        """
        Parse complete problem data (positive and negative examples)
        
        Args:
            problem_data: [positive_examples, negative_examples] from action program
            
        Returns:
            Tuple of (positive_shapes_list, negative_shapes_list)
        """
        if len(problem_data) != 2:
            raise ValueError("Problem data must contain exactly 2 elements: [positive, negative]")
        
        positive_examples, negative_examples = problem_data
        
        # Parse positive examples
        positive_shapes = []
        for example in positive_examples:
            shapes = self.parse_action_sequence(example)
            positive_shapes.append(shapes)
        
        # Parse negative examples  
        negative_shapes = []
        for example in negative_examples:
            shapes = self.parse_action_sequence(example)
            negative_shapes.append(shapes)
        
        return positive_shapes, negative_shapes
    
    def get_parsing_statistics(self, shapes_list: List[List[SemanticShape]]) -> Dict:
        """Get statistics about parsed shapes"""
        total_shapes = sum(len(shapes) for shapes in shapes_list)
        
        if total_shapes == 0:
            return {'total_shapes': 0}
        
        # Count by shape type
        shape_counts = {}
        line_counts = {}
        size_distribution = []
        
        for shapes in shapes_list:
            for shape in shapes:
                # Shape type counts
                shape_type = shape.shape_type.value
                shape_counts[shape_type] = shape_counts.get(shape_type, 0) + 1
                
                # Line type counts
                line_type = shape.line_type.value
                line_counts[line_type] = line_counts.get(line_type, 0) + 1
                
                # Size distribution
                size_distribution.append(shape.size)
        
        return {
            'total_shapes': total_shapes,
            'avg_shapes_per_image': total_shapes / len(shapes_list),
            'shape_type_distribution': shape_counts,
            'line_type_distribution': line_counts,
            'avg_size': np.mean(size_distribution),
            'size_std': np.std(size_distribution),
            'geometric_shapes_ratio': sum(1 for shapes in shapes_list for shape in shapes if shape.is_geometric_shape()) / total_shapes
        }

def test_semantic_parser():
    """Test the semantic action parser with sample data"""
    parser = SemanticActionParser()
    
    # Test individual action parsing
    test_actions = [
        'line_triangle_1.000-0.500',
        'line_square_0.750-0.250', 
        'line_circle_0.500-0.333',
        'line_zigzag_0.600-0.400'
    ]
    
    print("=== SEMANTIC ACTION PARSER TEST ===\n")
    
    print("1. Individual Action Parsing:")
    for action in test_actions:
        shape = parser.parse_action_string(action)
        if shape:
            print(f"   {action}")
            print(f"   → {shape.shape_type.value} ({shape.line_type.value})")
            print(f"   → Size: {shape.size}, Thickness: {shape.thickness}")
            print(f"   → Features: {shape.get_semantic_features()}")
            print()
    
    # Test sequence parsing
    print("2. Sequence Parsing:")
    test_sequence = ['line_triangle_1.000-0.500', 'line_square_0.750-0.250']
    shapes = parser.parse_action_sequence(test_sequence)
    print(f"   Input: {test_sequence}")
    print(f"   Parsed: {len(shapes)} shapes")
    for i, shape in enumerate(shapes):
        print(f"   Shape {i+1}: {shape.shape_type.value}")

if __name__ == "__main__":
    test_semantic_parser()
