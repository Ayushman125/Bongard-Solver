def test_degenerate_geometry_cases():
    from src.physics_inference import PhysicsInference
    # <3 vertices
    assert PhysicsInference.shoelace_area([(0,0),(1,1)]) == 0.0
    # collinear points
    assert PhysicsInference.shoelace_area([(0,0),(1,1),(2,2)]) == 0.0
    # duplicate points
    assert PhysicsInference.shoelace_area([(0,0),(0,0),(1,1),(1,1)]) == 0.0
    # zero-area shape
    assert PhysicsInference.shoelace_area([(0,0),(1,0),(2,0)]) == 0.0
import pytest
import numpy as np
from src.Derive_labels import shape_utils

def test_calculate_geometry_degenerate_cases():
    # 0 vertices
    g0 = shape_utils.calculate_geometry([])
    assert g0['area'] == 0.0
    assert g0['perimeter'] == 0.0
    assert g0['convexity_ratio'] == 0.0
    # 1 vertex
    g1 = shape_utils.calculate_geometry([(0, 0)])
    assert g1['area'] == 0.0
    assert g1['perimeter'] == 0.0
    assert g1['convexity_ratio'] == 0.0
    # 2 vertices (line)
    g2 = shape_utils.calculate_geometry([(0, 0), (1, 0)])
    assert g2['area'] == 0.0
    assert g2['perimeter'] == pytest.approx(1.0)
    assert g2['convexity_ratio'] == 0.0

def test_calculate_geometry_triangle():
    # Equilateral triangle
    tri = [(0, 0), (1, 0), (0.5, np.sqrt(3)/2)]
    g = shape_utils.calculate_geometry(tri)
    assert g['area'] > 0
    assert g['perimeter'] > 0
    assert 0.0 < g['convexity_ratio'] <= 1.0

def test_vertex_count_and_geometric_features():
    # Simulate a stroke with >=3 vertices (polygon)
    poly = [(0,0), (1,0), (1,1), (0,1)]
    g = shape_utils.calculate_geometry(poly)
    assert len(poly) >= 3
    assert g['area'] > 0
    assert g['perimeter'] > 0
    assert g['convexity_ratio'] > 0

def test_arc_geometric_features():
    # Simulate arc with interpolated points
    import numpy as np
    cx, cy, r = 0.5, 0.5, 0.3
    thetas = np.linspace(0, np.pi/2, 12)
    arc = [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in thetas]
    g = shape_utils.calculate_geometry(arc)
    assert len(arc) >= 3
    assert g['area'] >= 0
    assert g['perimeter'] > 0
    assert g['convexity_ratio'] >= 0

def test_calculate_geometry_nan_safety():
    # Pathological: all points same
    g = shape_utils.calculate_geometry([(0, 0), (0, 0), (0, 0)])
    assert g['area'] == 0.0
    assert g['convexity_ratio'] == 0.0
    # Pathological: colinear
    g = shape_utils.calculate_geometry([(0, 0), (1, 0), (2, 0)])
    assert g['area'] == 0.0
    assert g['convexity_ratio'] == 0.0
