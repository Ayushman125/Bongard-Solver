"""Unit tests for relation sampling."""

import pytest
import numpy as np

from src.bongard_generator.spatial_sampler import RelationSampler

class MockBBox:
    """Mock bounding box for testing."""
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def test_relation_sampler_init():
    """Test RelationSampler initialization."""
    sampler = RelationSampler(img_size=128)
    
    assert sampler.img_size == 128
    assert hasattr(sampler, 'margin')
    assert hasattr(sampler, 'min_separation')

def test_check_left_of():
    """Test left_of relation checking."""
    sampler = RelationSampler(img_size=100)
    
    # Clear left_of case
    bbox1 = MockBBox(10, 30, 20, 20)  # left object
    bbox2 = MockBBox(40, 30, 20, 20)  # right object
    
    assert sampler.check_left_of(bbox1, bbox2) == True
    assert sampler.check_left_of(bbox2, bbox1) == False

def test_check_above():
    """Test above relation checking."""
    sampler = RelationSampler(img_size=100)
    
    # Clear above case
    bbox1 = MockBBox(30, 10, 20, 20)  # top object
    bbox2 = MockBBox(30, 40, 20, 20)  # bottom object
    
    assert sampler.check_above(bbox1, bbox2) == True
    assert sampler.check_above(bbox2, bbox1) == False

def test_check_nested():
    """Test nested relation checking."""
    sampler = RelationSampler(img_size=100)
    
    # Clear nested case
    bbox_outer = MockBBox(10, 10, 50, 50)  # larger outer box
    bbox_inner = MockBBox(20, 20, 20, 20)  # smaller inner box
    
    assert sampler.check_nested(bbox_inner, bbox_outer) == True
    assert sampler.check_nested(bbox_outer, bbox_inner) == False

def test_check_overlap():
    """Test overlap relation checking."""
    sampler = RelationSampler(img_size=100)
    
    # Overlapping boxes
    bbox1 = MockBBox(10, 10, 30, 30)
    bbox2 = MockBBox(25, 25, 30, 30)
    
    assert sampler.check_overlap(bbox1, bbox2) == True
    assert sampler.check_overlap(bbox2, bbox1) == True
    
    # Non-overlapping boxes
    bbox3 = MockBBox(10, 10, 20, 20)
    bbox4 = MockBBox(50, 50, 20, 20)
    
    assert sampler.check_overlap(bbox3, bbox4) == False

def test_check_touching():
    """Test touching relation checking."""
    sampler = RelationSampler(img_size=100)
    
    # Adjacent boxes (sharing edge)
    bbox1 = MockBBox(10, 10, 20, 20)
    bbox2 = MockBBox(30, 10, 20, 20)  # Right adjacent
    
    assert sampler.check_touching(bbox1, bbox2) == True
    assert sampler.check_touching(bbox2, bbox1) == True
    
    # Separated boxes
    bbox3 = MockBBox(10, 10, 20, 20)
    bbox4 = MockBBox(50, 50, 20, 20)
    
    assert sampler.check_touching(bbox3, bbox4) == False

def test_check_surrounds():
    """Test surrounds relation checking."""
    sampler = RelationSampler(img_size=100)
    
    # Surrounding case (one object around another)
    bbox_inner = MockBBox(30, 30, 20, 20)
    bbox_outer = MockBBox(20, 20, 40, 40)
    
    assert sampler.check_surrounds(bbox_outer, bbox_inner) == True
    assert sampler.check_surrounds(bbox_inner, bbox_outer) == False

def test_compute_bbox_distance():
    """Test bounding box distance computation."""
    sampler = RelationSampler(img_size=100)
    
    # Test distance between centers
    bbox1 = MockBBox(0, 0, 10, 10)   # center at (5, 5)
    bbox2 = MockBBox(30, 40, 10, 10) # center at (35, 45)
    
    distance = sampler.compute_bbox_distance(bbox1, bbox2)
    
    # Distance should be sqrt((35-5)^2 + (45-5)^2) = sqrt(900 + 1600) = 50
    expected_distance = np.sqrt(30**2 + 40**2)
    assert abs(distance - expected_distance) < 1e-6

def test_compute_intersection_area():
    """Test intersection area computation."""
    sampler = RelationSampler(img_size=100)
    
    # Overlapping boxes
    bbox1 = MockBBox(10, 10, 30, 30)  # (10,10) to (40,40)
    bbox2 = MockBBox(25, 25, 30, 30)  # (25,25) to (55,55)
    
    area = sampler.compute_intersection_area(bbox1, bbox2)
    
    # Intersection should be (25,25) to (40,40) = 15x15 = 225
    expected_area = 15 * 15
    assert area == expected_area
    
    # Non-overlapping boxes
    bbox3 = MockBBox(0, 0, 10, 10)
    bbox4 = MockBBox(50, 50, 10, 10)
    
    area_no_overlap = sampler.compute_intersection_area(bbox3, bbox4)
    assert area_no_overlap == 0

def test_sample_positions_for_relation():
    """Test position sampling for specific relations."""
    sampler = RelationSampler(img_size=100)
    
    # Test left_of relation
    pos1, pos2 = sampler.sample_positions_for_relation("left_of", (20, 20), (20, 20))
    
    assert pos1 is not None and pos2 is not None
    assert len(pos1) == 4 and len(pos2) == 4  # (x, y, w, h)
    
    # Check that first object is to the left of second
    bbox1 = MockBBox(*pos1)
    bbox2 = MockBBox(*pos2)
    assert sampler.check_left_of(bbox1, bbox2) == True

def test_sample_positions_for_relation_invalid():
    """Test position sampling with invalid relation."""
    sampler = RelationSampler(img_size=100)
    
    # Invalid relation should return None
    result = sampler.sample_positions_for_relation("invalid_relation", (20, 20), (20, 20))
    
    assert result == (None, None)

def test_get_spatial_constraints():
    """Test spatial constraint retrieval."""
    sampler = RelationSampler(img_size=100)
    
    constraints = sampler.get_spatial_constraints("above")
    
    assert constraints is not None
    assert callable(constraints)
    
    # Test with positions
    pos1 = (20, 10, 20, 20)  # upper object
    pos2 = (20, 40, 20, 20)  # lower object
    
    assert constraints(pos1, pos2) == True  # pos1 above pos2
    assert constraints(pos2, pos1) == False  # pos2 not above pos1

def test_sample_relation_pair():
    """Test sampling of relation pair."""
    sampler = RelationSampler(img_size=100)
    
    # Sample positions for objects with relation
    result = sampler.sample_relation_pair(
        relation="overlap",
        size1=(25, 25),
        size2=(25, 25),
        max_attempts=10
    )
    
    pos1, pos2 = result
    
    if pos1 is not None and pos2 is not None:
        # Verify the relation holds
        bbox1 = MockBBox(*pos1)
        bbox2 = MockBBox(*pos2)
        assert sampler.check_overlap(bbox1, bbox2) == True

def test_available_relations():
    """Test that all expected relations are available."""
    sampler = RelationSampler(img_size=100)
    
    expected_relations = ["left_of", "above", "nested", "overlap", "touching", "surrounds"]
    
    for relation in expected_relations:
        constraints = sampler.get_spatial_constraints(relation)
        assert constraints is not None, f"Relation '{relation}' should be available"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
