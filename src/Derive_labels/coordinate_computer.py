import math
import numpy as np
# Import your action classes here
# from bongard import LineAction, ArcAction

class BongardCoordinateComputer:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0  # in degrees
        self.vertices = [(0.0, 0.0)]

    def reset(self, start_x=0, start_y=0, start_heading=0):
        self.x = start_x
        self.y = start_y
        self.heading = start_heading
        self.vertices = [(start_x, start_y)]

    def turn(self, angle):
        """Turn by angle degrees (positive = left, negative = right)"""
        self.heading = (self.heading + angle) % 360

    def move_forward(self, distance):
        """Move forward by distance units"""
        rad = math.radians(self.heading)
        self.x += distance * math.cos(rad)
        self.y += distance * math.sin(rad)
        self.vertices.append((self.x, self.y))

    def draw_arc(self, radius, angle):
        """Draw arc with given radius and angle"""
        start_angle = math.radians(self.heading - 90)  # Perpendicular to heading
        arc_angle = math.radians(angle)
        cx = self.x + radius * math.cos(start_angle)
        cy = self.y + radius * math.sin(start_angle)
        num_points = max(8, int(abs(angle) / 10))
        for i in range(1, num_points + 1):
            t = i / num_points
            current_angle = start_angle + arc_angle * t
            arc_x = cx + radius * math.cos(current_angle)
            arc_y = cy + radius * math.sin(current_angle)
            self.vertices.append((arc_x, arc_y))
        self.x = self.vertices[-1][0]
        self.y = self.vertices[-1][1]
        self.heading = (self.heading + angle) % 360

def compute_shape_vertices(actions, start_coords=(0,0), start_orientation=0, scaling_factor=1.0):
    """
    Compute precise vertices from Bongard actions mathematically
    """
    computer = BongardCoordinateComputer()
    computer.reset(start_coords[0], start_coords[1], start_orientation)
    for action in actions:
        # Handle turn
        if hasattr(action, 'turn_direction'):
            if action.turn_direction == "L":
                computer.turn(action.turn_angle)
            elif action.turn_direction == "R":
                computer.turn(-action.turn_angle)
        # Handle movement
        # Replace with your actual LineAction/ArcAction checks
        if hasattr(action, 'line_length'):
            distance = action.line_length * scaling_factor
            if getattr(action, 'line_type', 'normal') == "normal":
                computer.move_forward(distance)
            elif getattr(action, 'line_type', 'normal') == "zigzag":
                num_zigs = max(1, int(distance / 20))
                seg_length = distance / num_zigs
                for _ in range(num_zigs):
                    computer.turn(60)
                    computer.move_forward(seg_length / 2)
                    computer.turn(-120)
                    computer.move_forward(seg_length / 2)
                    computer.turn(60)
        elif hasattr(action, 'arc_radius'):
            radius = action.arc_radius * scaling_factor
            computer.draw_arc(radius, action.arc_angle)
    return computer.vertices

def extract_shape_features(vertices):
    """
    Extract geometric features from computed vertices
    """
    features = {}
    if not vertices or len(vertices) < 2:
        features['area'] = 0.0
        features['perimeter'] = 0.0
        features['compactness'] = 0.0
        features['centroid'] = (0, 0)
        return features
    # Area (shoelace formula)
    x = np.array([p[0] for p in vertices])
    y = np.array([p[1] for p in vertices])
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    features['area'] = area
    # Perimeter
    perimeter = sum(math.hypot(x2-x1, y2-y1) for (x1, y1), (x2, y2) in zip(vertices, vertices[1:]))
    features['perimeter'] = perimeter
    # Compactness
    if perimeter > 0:
        features['compactness'] = 4 * math.pi * area / (perimeter ** 2)
    else:
        features['compactness'] = 0.0
    # Centroid
    if area > 0:
        cx = sum(p[0] for p in vertices) / len(vertices)
        cy = sum(p[1] for p in vertices) / len(vertices)
        features['centroid'] = (cx, cy)
    else:
        features['centroid'] = (0, 0)
    return features
