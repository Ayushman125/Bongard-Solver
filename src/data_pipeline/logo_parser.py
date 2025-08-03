import re
from math import cos, sin, radians
from typing import List, Tuple, Dict, Optional
import math

# Discovered Bongard-LOGO shape types from dataset analysis
BONGARD_SHAPE_TYPES = {
    'normal': {'count': 24107, 'description': 'Straight lines, most common'},
    'circle': {'count': 6256, 'description': 'Circular shapes and arcs'},
    'square': {'count': 6519, 'description': 'Square-based shapes'},
    'triangle': {'count': 5837, 'description': 'Triangular shapes'},
    'zigzag': {'count': 6729, 'description': 'Zigzag patterns'}
}

class LogoParser:
    def parse_logo_script_from_lines(self, lines):
        """
        Accepts a list of LOGO command strings (as lines), returns ordered vertex list.
        This is used for Bongard-LOGO JSON action programs, where each image's program is a list of command strings.
        Enhanced to detect the 5 discovered shape types during parsing.
        """
        import re
        from math import cos, sin, radians
        pattern = re.compile(r'(F|B|R|L)\s*(-?\d+(?:\.\d+)?)')
        x, y = 0, 0
        angle = 0
        vertices = [(x, y)]
        
        # Track patterns to infer shape type
        direction_changes = 0
        total_distance = 0
        angles_turned = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = pattern.match(line)
            if not match:
                continue
            cmd, val = match.group(1), float(match.group(2))
            if cmd == 'F':
                rad = radians(angle)
                x += val * cos(rad)
                y += val * sin(rad)
                total_distance += val
            elif cmd == 'B':
                rad = radians(angle)
                x -= val * cos(rad)
                y -= val * sin(rad)
                total_distance += val
            elif cmd == 'R':
                angle -= val
                angle %= 360
                angles_turned.append(-val)
                direction_changes += 1
            elif cmd == 'L':
                angle += val
                angle %= 360
                angles_turned.append(val)
                direction_changes += 1
            vertices.append((x, y))
        
        # Infer shape type from parsing patterns
        shape_type = self._infer_shape_type(vertices, direction_changes, angles_turned, total_distance)
        
        return vertices, shape_type
    
    def _infer_shape_type(self, vertices: List[Tuple[float, float]], 
                         direction_changes: int, angles_turned: List[float], 
                         total_distance: float) -> str:
        """Infer one of the 5 discovered Bongard-LOGO shape types from parsing patterns"""
        
        # Check for closed shape (start and end points close)
        if len(vertices) >= 2:
            start, end = vertices[0], vertices[-1]
            is_closed = abs(start[0] - end[0]) < 1.0 and abs(start[1] - end[1]) < 1.0
        else:
            is_closed = False
        
        # Analyze angle patterns
        if angles_turned:
            avg_angle = sum(abs(a) for a in angles_turned) / len(angles_turned)
            angle_variance = sum((abs(a) - avg_angle) ** 2 for a in angles_turned) / len(angles_turned)
        else:
            avg_angle = 0
            angle_variance = 0
        
        # Shape type inference based on discovered patterns
        if direction_changes == 0:
            return 'normal'  # Straight line, no turns
        elif is_closed and len(set(abs(a) for a in angles_turned if abs(a) > 1)) == 1:
            # Regular angles suggest geometric shapes
            if avg_angle > 85 and avg_angle < 95:
                return 'square'  # 90-degree turns
            elif avg_angle > 110 and avg_angle < 130:
                return 'triangle'  # ~120-degree turns
        elif is_closed and avg_angle < 45:
            return 'circle'  # Many small turns suggest circular arc
        elif angle_variance > 1000:  # High variance in turn angles
            return 'zigzag'  # Irregular zigzag pattern
        else:
            return 'normal'  # Default to normal for unclassified patterns
    
class BongardLogoParser:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.angle = 0.0  # degrees
        self.shape_patterns = {
            'normal': 0,
            'circle': 0, 
            'square': 0,
            'triangle': 0,
            'zigzag': 0
        }

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.angle = 0.0
        # Reset shape pattern counters
        self.shape_patterns = {k: 0 for k in self.shape_patterns}
    
    def detect_shape_type(self, action_sequence: List[str]) -> str:
        """Detect one of the 5 discovered Bongard-LOGO shape types from action sequence"""
        # Parse the action sequence to get vertices and patterns
        vertices = self.parse_action_sequence(action_sequence)
        
        # Analyze the parsed vertices for shape characteristics
        return self._classify_shape_from_vertices(vertices)
    
    def _classify_shape_from_vertices(self, vertices: List[Tuple[float, float]]) -> str:
        """Classify vertices into one of the 5 discovered shape types"""
        if len(vertices) < 2:
            return 'normal'
        
        # Calculate geometric properties
        is_closed = self._is_closed_shape(vertices)
        angles = self._calculate_internal_angles(vertices)
        curvature = self._calculate_curvature(vertices)
        symmetry_score = self._calculate_symmetry_score(vertices)
        
        # Classification logic based on discovered patterns
        if not is_closed and len(vertices) == 2:
            return 'normal'  # Simple straight line
        elif is_closed and len(set(round(a, 0) for a in angles if abs(a) > 10)) == 1:
            # Regular polygon with consistent angles
            avg_angle = sum(angles) / len(angles)
            if 85 <= avg_angle <= 95:
                return 'square'
            elif 55 <= avg_angle <= 65:
                return 'triangle'
        elif curvature > 0.8:  # High curvature suggests circle
            return 'circle'
        elif self._has_zigzag_pattern(vertices):
            return 'zigzag'
        else:
            return 'normal'  # Default fallback
    
    def _is_closed_shape(self, vertices: List[Tuple[float, float]]) -> bool:
        """Check if shape is closed (start and end points are close)"""
        if len(vertices) < 3:
            return False
        start, end = vertices[0], vertices[-1]
        return abs(start[0] - end[0]) < 2.0 and abs(start[1] - end[1]) < 2.0
    
    def _calculate_internal_angles(self, vertices: List[Tuple[float, float]]) -> List[float]:
        """Calculate internal angles at each vertex"""
        angles = []
        n = len(vertices)
        for i in range(1, n - 1):
            p1, p2, p3 = vertices[i-1], vertices[i], vertices[i+1]
            v1 = (p1[0] - p2[0], p1[1] - p2[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate angle between vectors
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = (v1[0]**2 + v1[1]**2) ** 0.5
            mag2 = (v2[0]**2 + v2[1]**2) ** 0.5
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = math.degrees(math.acos(cos_angle))
                angles.append(angle)
        
        return angles
    
    def _calculate_curvature(self, vertices: List[Tuple[float, float]]) -> float:
        """Calculate average curvature of the path"""
        if len(vertices) < 3:
            return 0.0
        
        curvatures = []
        for i in range(1, len(vertices) - 1):
            p1, p2, p3 = vertices[i-1], vertices[i], vertices[i+1]
            
            # Calculate curvature using the menger curvature formula
            area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
            d1 = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) ** 0.5
            d2 = ((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2) ** 0.5
            d3 = ((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2) ** 0.5
            
            if d1 * d2 * d3 > 0:
                curvature = 4 * area / (d1 * d2 * d3)
                curvatures.append(curvature)
        
        return sum(curvatures) / len(curvatures) if curvatures else 0.0
    
    def _calculate_symmetry_score(self, vertices: List[Tuple[float, float]]) -> float:
        """Calculate symmetry score for the shape"""
        if len(vertices) < 4:
            return 0.0
        
        # Simple symmetry test - check if shape is symmetric about centroid
        cx = sum(v[0] for v in vertices) / len(vertices)
        cy = sum(v[1] for v in vertices) / len(vertices)
        
        symmetry_score = 0.0
        for x, y in vertices:
            reflected_x = 2 * cx - x
            reflected_y = 2 * cy - y
            
            # Find closest point to reflection
            min_dist = float('inf')
            for vx, vy in vertices:
                dist = ((vx - reflected_x)**2 + (vy - reflected_y)**2) ** 0.5
                min_dist = min(min_dist, dist)
            
            symmetry_score += 1.0 / (1.0 + min_dist)
        
        return symmetry_score / len(vertices)
    
    def _has_zigzag_pattern(self, vertices: List[Tuple[float, float]]) -> bool:
        """Detect if vertices form a zigzag pattern"""
        if len(vertices) < 4:
            return False
        
        # Check for alternating direction changes
        direction_changes = 0
        for i in range(2, len(vertices) - 1):
            p1, p2, p3, p4 = vertices[i-2:i+2]
            
            # Calculate direction vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            v3 = (p4[0] - p3[0], p4[1] - p3[1])
            
            # Check for direction changes
            cross1 = v1[0] * v2[1] - v1[1] * v2[0]
            cross2 = v2[0] * v3[1] - v2[1] * v3[0]
            
            if cross1 * cross2 < 0:  # Opposite signs indicate direction change
                direction_changes += 1
        
        # Zigzag has frequent direction changes
        return direction_changes >= len(vertices) * 0.3
        self.angle = 0.0

    def parse_action_program(self, action_list: List[str], scale: float = 100.0) -> List[Tuple[float, float]]:
        self.reset()
        vertices = [(self.x, self.y)]
        symmetry_axis = None
        predicate = None
        for action_cmd in action_list:
            new_points = self.parse_single_action(action_cmd, scale)
            vertices.extend(new_points)
            # LOGO predicate extraction logic (example: detect symmetry, parallelism, etc.)
            if action_cmd.startswith('line_'):
                predicate = 'line'
                # For lines, symmetry axis is the direction of the line
                if len(new_points) >= 2:
                    x0, y0 = new_points[0]
                    x1, y1 = new_points[-1]
                    symmetry_axis = math.atan2(y1 - y0, x1 - x0)
            elif action_cmd.startswith('arc_'):
                predicate = 'arc'
                # For arcs, symmetry axis is the bisector of the arc
                if len(new_points) >= 2:
                    x0, y0 = new_points[0]
                    x1, y1 = new_points[-1]
                    symmetry_axis = math.atan2(y1 - y0, x1 - x0)
        # Ensure at least 4 vertices for valid polygon
        if len(vertices) < 4:
            x, y = self.x, self.y
            vertices = [(x, y), (x + 5, y), (x + 5, y + 5), (x, y)]
        # Return vertices, symmetry_axis, predicate for downstream use
        self.last_symmetry_axis = symmetry_axis
        self.last_predicate = predicate
        return vertices

    def parse_single_action(self, action_cmd: str, scale: float) -> List[Tuple[float, float]]:
        if action_cmd.startswith('line_'):
            return self.parse_line_command(action_cmd, scale)
        elif action_cmd.startswith('arc_'):
            return self.parse_arc_command(action_cmd, scale)
        else:
            return []

    def parse_line_command(self, cmd: str, scale: float) -> List[Tuple[float, float]]:
        pattern = r'line_(\w+)_([0-9.]+)-([0-9.]+)'
        match = re.match(pattern, cmd)
        if not match:
            return self._generate_minimal_vertices()
        style, length_str, angle_str = match.groups()
        length = float(length_str) * scale
        if length < 1:
            length = 1  # enforce minimal movement
        if length < 5:
            return self._generate_minimal_vertices()
        angle_change = float(angle_str) * 360
        points = []
        # Dense interpolation: always at least 5 points per segment
        steps = max(5, int(length / 2))
        x_prev, y_prev = self.x, self.y
        dx = length * math.cos(math.radians(self.angle))
        dy = length * math.sin(math.radians(self.angle))
        for i in range(1, steps + 1):
            frac = i / steps
            xi = x_prev + frac * dx
            yi = y_prev + frac * dy
            # Skip ultra-short segments
            if math.hypot(xi - x_prev, yi - y_prev) < 5:
                continue
            points.append((xi, yi))
            x_prev, y_prev = xi, yi
        # Remove consecutive duplicate points
        dedup_points = [points[0]] if points else []
        for pt in points[1:]:
            if pt != dedup_points[-1]:
                dedup_points.append(pt)
        # Ensure at least 4 distinct points for Shapely
        if len(dedup_points) < 4:
            x0, y0 = dedup_points[-1] if dedup_points else (self.x, self.y)
            dedup_points.append((x0 + 1, y0))
            dedup_points.append((x0, y0 + 1))
        self.x, self.y = dedup_points[-1] if dedup_points else (self.x, self.y)
        self.angle += angle_change
        return dedup_points

    def _generate_minimal_vertices(self) -> List[Tuple[float, float]]:
        x, y = self.x, self.y
        return [(x, y), (x + 5, y), (x + 5, y + 5), (x, y)]

    def parse_arc_command(self, cmd: str, scale: float) -> List[Tuple[float, float]]:
        pattern = r'arc_(\w+)_([0-9.]+)_([0-9.]+)-([0-9.]+)'
        match = re.match(pattern, cmd)
        if not match:
            return self._generate_minimal_vertices()
        style, radius_str, arc_span_str, angle_str = match.groups()
        radius = float(radius_str) * scale
        arc_span = float(arc_span_str) * 360
        arc_span = max(0, min(arc_span, 360))
        angle_change = float(angle_str) * 360
        points = []
        # Always simulate arc movement for all styles
        arc_segments = max(10, int(abs(arc_span) // 18))
        for i in range(1, arc_segments + 1):
            seg_angle = self.angle + (arc_span * i / arc_segments)
            arc_x = self.x + radius * math.cos(math.radians(seg_angle))
            arc_y = self.y + radius * math.sin(math.radians(seg_angle))
            points.append((arc_x, arc_y))
        if len(points) < 4:
            # Fallback to minimal valid geometry
            x, y = self.x, self.y
            points = [(x, y), (x + 5, y), (x + 5, y + 5), (x, y)]
        if points:
            self.x, self.y = points[-1]
        self.angle += arc_span + angle_change
        return points

    def parse_logo_script(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        pattern = re.compile(r'(F|B|R|L)\s*(-?\d+(?:\.\d+)?)')
        x, y = 0, 0
        angle = 0  # Degrees, 0 points right
        vertices = [(x, y)]
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = pattern.match(line)
            if not match:
                continue
            cmd, val = match.group(1), float(match.group(2))
            if cmd == 'F':
                rad = radians(angle)
                x += val * cos(rad)
                y += val * sin(rad)
            elif cmd == 'B':
                rad = radians(angle)
                x -= val * cos(rad)
                y -= val * sin(rad)
            elif cmd == 'R':
                angle -= val
                angle %= 360
            elif cmd == 'L':
                angle += val
                angle %= 360
            vertices.append((x, y))
        return vertices