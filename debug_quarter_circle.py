#!/usr/bin/env python3
"""Debug quarter circle detection functionality"""

import math
from typing import List, Tuple, Dict, Any

def extract_line_coordinates(action_program: List[str]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Extract line coordinates from action program"""
    lines = []
    current_pos = None
    
    for action in action_program:
        if action.startswith('start_'):
            coords = action[6:].split('_')
            current_pos = (float(coords[0]), float(coords[1]))
        elif action.startswith('line_') and current_pos is not None:
            parts = action.split('_')
            if len(parts) >= 3:
                coords = parts[2].split('-')
                if len(coords) == 2:
                    end_pos = (float(coords[0]), float(coords[1]))
                    lines.append((current_pos, end_pos))
                    current_pos = end_pos
    
    return lines

def are_lines_connected(line_coords: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> bool:
    """Check if lines form a connected path"""
    if len(line_coords) < 2:
        return False
    
    for i in range(len(line_coords) - 1):
        end_of_current = line_coords[i][1]
        start_of_next = line_coords[i + 1][0]
        
        # Check if end point of current line matches start point of next line
        distance = math.sqrt((end_of_current[0] - start_of_next[0])**2 + 
                           (end_of_current[1] - start_of_next[1])**2)
        if distance > 0.01:  # Small tolerance
            return False
    
    return True

def extract_path_points(line_coords: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """Extract all points from connected path"""
    if not line_coords:
        return []
    
    points = [line_coords[0][0]]  # Start with first point
    for line in line_coords:
        points.append(line[1])  # Add end point of each line
    
    return points

def angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
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

def analyze_path_curvature(path_points: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Analyze curvature characteristics of a path"""
    print(f"Analyzing path with {len(path_points)} points: {path_points}")
    
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
        
        print(f"  Segment {i}: {p1} -> {p2} -> {p3}")
        print(f"    v1: {v1}, v2: {v2}")
        
        # Angle between vectors
        angle = angle_between_vectors(v1, v2)
        angles.append(abs(angle))
        print(f"    Angle: {angle}° (abs: {abs(angle)}°)")
        
        # Curvature estimation
        segment_length = math.sqrt(v1[0]**2 + v1[1]**2)
        if segment_length > 0:
            curvature = abs(angle) / segment_length
            curvatures.append(curvature)
            print(f"    Segment length: {segment_length}, Curvature: {curvature}")
            
            # Radius estimation (rough approximation)
            if curvature > 0:
                radius = 1.0 / curvature
                radii.append(radius)
                print(f"    Estimated radius: {radius}")
    
    # Calculate consistency metrics
    total_angle = sum(angles)
    print(f"Total angle: {total_angle}° (individual angles: {angles})")
    
    curvature_consistency = 0.0
    if curvatures:
        avg_curvature = sum(curvatures) / len(curvatures)
        variance = sum((c - avg_curvature)**2 for c in curvatures) / len(curvatures)
        curvature_consistency = max(0, 1.0 - math.sqrt(variance) / (avg_curvature + 0.001))
        print(f"Curvature consistency: {curvature_consistency} (avg: {avg_curvature}, variance: {variance})")
    
    radius_consistency = 0.0
    estimated_radius = 0.0
    if radii:
        estimated_radius = sum(radii) / len(radii)
        variance = sum((r - estimated_radius)**2 for r in radii) / len(radii)
        radius_consistency = max(0, 1.0 - math.sqrt(variance) / (estimated_radius + 0.001))
        print(f"Radius consistency: {radius_consistency} (avg: {estimated_radius}, variance: {variance})")
    
    # Determine direction (clockwise/counterclockwise)
    direction = 'clockwise' if total_angle > 0 else 'counterclockwise'
    
    result = {
        'total_angle': abs(total_angle),
        'curvature_consistency': curvature_consistency,
        'radius_consistency': radius_consistency,
        'estimated_radius': estimated_radius,
        'direction': direction,
        'average_curvature': sum(curvatures) / len(curvatures) if curvatures else 0
    }
    
    print(f"Final analysis result: {result}")
    return result

def debug_quarter_circle_detection():
    """Debug quarter circle detection with actual data"""
    
    # Use the actual action program from the CSV data
    action_program = [
        'start_0.0_0.0',
        'line_normal_0.354-0.500',
        'line_normal_1.254-0.625', 
        'line_normal_1.608-0.750',
        'line_normal_1.9620000000000002-1.250',
        'line_normal_2.862-1.375',
        'line_normal_3.216-1.500'
    ]
    
    print("=== Quarter Circle Detection Debug ===")
    print(f"Action program: {action_program}")
    
    # Extract line coordinates
    line_coords = extract_line_coordinates(action_program)
    print(f"\nExtracted {len(line_coords)} line coordinates:")
    for i, (start, end) in enumerate(line_coords):
        print(f"  Line {i+1}: {start} -> {end}")
    
    # Check connectivity
    connected = are_lines_connected(line_coords)
    print(f"\nLines connected: {connected}")
    
    if not connected:
        print("❌ Lines are not connected - cannot form quarter circle")
        return
    
    # Extract path points
    path_points = extract_path_points(line_coords)
    print(f"\nPath points: {path_points}")
    
    if len(path_points) < 4:
        print("❌ Not enough path points for quarter circle")
        return
    
    # Analyze curvature
    print("\n=== Curvature Analysis ===")
    analysis = analyze_path_curvature(path_points)
    
    # Quarter circle specific checks
    print(f"\n=== Quarter Circle Evaluation ===")
    print(f"Total angle: {analysis['total_angle']}° (need 50-160°)")
    print(f"Curvature consistency: {analysis['curvature_consistency']:.3f} (need > 0.3)")
    print(f"Radius consistency: {analysis['radius_consistency']:.3f} (need > 0.15)")
    
    is_quarter_circle = False
    if 50 <= analysis['total_angle'] <= 160:
        print("✅ Total angle in permissive quarter circle range")
        if analysis['curvature_consistency'] > 0.3:
            print("✅ Curvature consistency sufficient")
            if analysis['radius_consistency'] > 0.15:
                print("✅ Radius consistency sufficient")
                is_quarter_circle = True
                
                # Calculate confidence based on how close to ideal quarter circle
                angle_score = 1.0 - abs(analysis['total_angle'] - 90) / 90.0  # Best at 90°
                angle_score = max(0.1, angle_score)
                
                base_confidence = analysis['curvature_consistency'] * analysis['radius_consistency']
                adjusted_confidence = min(0.85, base_confidence * angle_score)
                
                print(f"✅ QUARTER CIRCLE DETECTED with confidence {adjusted_confidence:.3f}")
                print(f"  - Angle score: {angle_score:.3f}")
                print(f"  - Base confidence: {base_confidence:.3f}")
            else:
                print("❌ Radius consistency too low")
        else:
            print("❌ Curvature consistency too low")
    else:
        print("❌ Total angle outside permissive quarter circle range")
    
    if not is_quarter_circle:
        print("❌ NOT A QUARTER CIRCLE")

if __name__ == "__main__":
    debug_quarter_circle_detection()
