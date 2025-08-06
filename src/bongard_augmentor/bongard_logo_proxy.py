"""
Proxy for Bongard-LOGO classes and functions.
Loads all required Bongard-LOGO symbols using importlib, regardless of Python path/package issues.
"""
import os
import sys
import importlib.util
import logging

BONGARD_LOGO_PATH = os.path.join(os.getcwd(), 'Bongard-LOGO')
BONGARD_MODULE_PATH = os.path.join(BONGARD_LOGO_PATH, 'bongard', '__init__.py')
BONGARD_PAINTER_PATH = os.path.join(BONGARD_LOGO_PATH, 'bongard', 'bongard_painter.py')

if BONGARD_LOGO_PATH not in sys.path:
    sys.path.insert(0, BONGARD_LOGO_PATH)

# Load bongard main module
spec = importlib.util.spec_from_file_location('bongard', BONGARD_MODULE_PATH)
bongard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bongard)

# Load bongard_painter
spec_painter = importlib.util.spec_from_file_location('bongard_painter', BONGARD_PAINTER_PATH)
bongard_painter = importlib.util.module_from_spec(spec_painter)
spec_painter.loader.exec_module(bongard_painter)

# Expose all required symbols
BongardImage = bongard.BongardImage
OneStrokeShape = bongard.OneStrokeShape
BasicAction = bongard.BasicAction
LineAction = bongard.LineAction
ArcAction = bongard.ArcAction
BongardImagePainter = bongard_painter.BongardImagePainter

# Define normalization functions with exact Bongard-LOGO logic
def normalize_line_length(line_length, line_length_max):
    """[0,line_length_max] -> [0,1]"""
    assert line_length >= 0, "line_length should not be negative!"
    normalized_line_length = line_length / line_length_max
    return normalized_line_length

def denormalize_line_length(normalized_line_length, line_length_max):
    """[0,1] -> [0,line_length_max]"""
    assert 0 <= normalized_line_length <= 1, "normalized_line_length should not be in [0, 1]!"
    line_length = normalized_line_length * line_length_max
    return line_length

def normalize_turn_angle(turn_direction, turn_angle):
    """
    L180 -> 180
    R180 -> -180
    [-180,180] -> [0,1]
    """
    assert turn_angle <= 180 and turn_angle >= 0, "angle should be in [0, 180]!"
    
    if turn_direction == "L":
        normalized_turn_angle = (turn_angle + 180) / 360
    elif turn_direction == "R":
        normalized_turn_angle = (180 - turn_angle) / 360
    else:
        raise Exception("Unsupported direction!")
    
    return normalized_turn_angle

def denormalize_turn_angle(normalized_turn_angle):
    """
    L180 -> 180
    R180 -> -180
    [0,1] -> [-180,180]
    """
    assert normalized_turn_angle >= 0 and normalized_turn_angle <= 1, "normalized_turn_angle should be in [0, 1]!"
    
    if normalized_turn_angle >= 0.5:
        direction = "L"
        angle = normalized_turn_angle * 360 - 180
    else:
        direction = "R"
        angle = 180 - normalized_turn_angle * 360
    
    return direction, angle

def normalize_arc_angle(arc_angle):
    """[-360,360] -> [0,1]"""
    assert -360 <= arc_angle <= 360, "arc_angle should not be in [-360, 360]!"
    normalized_arc_angle = (arc_angle + 360) / 720
    return normalized_arc_angle

def denormalize_arc_angle(normalized_arc_angle):
    """[0,1] -> [-360,360]"""
    assert 0 <= normalized_arc_angle <= 1, "normalized_arc_angle should not be in [0, 1]!"
    arc_angle = normalized_arc_angle * 720 - 360
    return arc_angle

def get_action_type(action_string):
    """Extract action type from action string"""
    return action_string.split("_")[0]
