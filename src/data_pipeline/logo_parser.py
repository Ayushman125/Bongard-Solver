import re
from math import cos, sin, radians
from typing import List, Tuple
import math

class LogoParser:
    def parse_logo_script_from_lines(self, lines):
        """
        Accepts a list of LOGO command strings (as lines), returns ordered vertex list.
        This is used for Bongard-LOGO JSON action programs, where each image's program is a list of command strings.
        """
        import re
        from math import cos, sin, radians
        pattern = re.compile(r'(F|B|R|L)\s*(-?\d+(?:\.\d+)?)')
        x, y = 0, 0
        angle = 0
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
    
class BongardLogoParser:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.angle = 0.0  # degrees

    def reset(self):
        self.x = 0.0
        self.y = 0.0
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