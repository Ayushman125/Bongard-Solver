"""Comprehensive tests for the Bongard generator package."""

import pytest
import numpy as np
from typing import List, Dict, Any

from src.bongard_generator.cp_sampler import sample_scene_cp, CPSATSampler, SceneParameters
from src.bongard_generator.rule_loader import get_rule_by_description, get_all_rules, RULE_LOOKUP
from src.bongard_generator.spatial_sampler import RelationSampler
from src.bongard_generator.coverage import CoverageTracker
from src.bongard_generator.draw_utils import make_gradient_mask

class TestCPSATSampler:
    """Test CP-SAT constraint-based sampling."""
    
    def test_circle_positive_rule(self):
        """Test that circle rule generates scenes with circles."""
        rule = get_rule_by_description("SHAPE(CIRCLE)")
        if not rule:
            pytest.skip("SHAPE(CIRCLE) rule not found")
        
        scenes = sample_scene_cp(rule, 2, True, 64, 20, 40, 5)
        assert scenes is not None, "Scene generation should not fail"
        assert len(scenes) == 2, "Should generate exactly 2 objects"
        
        # At least one object should be a circle for positive examples
        shapes = [obj['shape'] for obj in scenes]
        assert 'circle' in shapes, "Positive circle rule should generate at least one circle"
    
    def test_circle_negative_rule(self):
        """Test that negative circle rule avoids circles."""
        rule = get_rule_by_description("SHAPE(CIRCLE)")
        if not rule:
            pytest.skip("SHAPE(CIRCLE) rule not found")
        
        scenes = sample_scene_cp(rule, 2, False, 64, 20, 40, 5)
        assert scenes is not None, "Scene generation should not fail"
        assert len(scenes) == 2, "Should generate exactly 2 objects"
        
        # No object should be a circle for negative examples
        shapes = [obj['shape'] for obj in scenes]
        assert 'circle' not in shapes, "Negative circle rule should avoid circles"
    
    def test_triangle_rule(self):
        """Test triangle rule compliance."""
        rule = get_rule_by_description("SHAPE(TRIANGLE)")
        if not rule:
            pytest.skip("SHAPE(TRIANGLE) rule not found")
        
        scenes = sample_scene_cp(rule, 3, True, 64, 15, 35, 10)
        assert scenes is not None
        shapes = [obj['shape'] for obj in scenes]
        assert 'triangle' in shapes
    
    def test_count_rule(self):
        """Test count rule compliance."""
        rule = get_rule_by_description("COUNT(3)")
        if not rule:
            pytest.skip("COUNT(3) rule not found")
        
        scenes = sample_scene_cp(rule, 3, True, 64, 20, 40, 5)
        assert scenes is not None
        assert len(scenes) == 3, "COUNT(3) rule should generate exactly 3 objects"
    
    def test_fill_rule(self):
        """Test fill rule compliance."""
        rule = get_rule_by_description("FILL(SOLID)")
        if not rule:
            pytest.skip("FILL(SOLID) rule not found")
        
        scenes = sample_scene_cp(rule, 2, True, 64, 20, 40, 5)
        assert scenes is not None
        fills = [obj['fill'] for obj in scenes]
        assert 'solid' in fills, "FILL(SOLID) rule should generate at least one solid object"

class TestSpatialSampler:
    """Test spatial relationship sampling."""
    
    def setup_method(self):
        """Setup test environment."""
        self.sampler = RelationSampler(64)
    
    def test_nested_relation(self):
        """Test nested spatial relationship."""
        scene = self.sampler.sample(2, 'nested')
        assert len(scene) == 2
        # Both objects should be at the same position for nesting
        pos1 = scene[0]['position']
        pos2 = scene[1]['position']
        assert pos1 == pos2, "Nested objects should be at the same position"
    
    def test_left_of_relation(self):
        """Test left_of spatial relationship."""
        scene = self.sampler.sample(3, 'left_of')
        assert len(scene) == 3
        
        # Objects should be arranged left to right
        x_positions = [obj['position'][0] for obj in scene]
        assert x_positions == sorted(x_positions), "Objects should be arranged left to right"
    
    def test_above_relation(self):
        """Test above spatial relationship."""
        scene = self.sampler.sample(3, 'above')
        assert len(scene) == 3
        
        # Objects should be arranged top to bottom
        y_positions = [obj['position'][1] for obj in scene]
        assert y_positions == sorted(y_positions), "Objects should be arranged top to bottom"
    
    def test_overlap_relation(self):
        """Test overlap spatial relationship."""
        scene = self.sampler.sample(2, 'overlap')
        assert len(scene) == 2
        
        # Objects should be close to each other (overlapping)
        pos1 = scene[0]['position']
        pos2 = scene[1]['position']
        distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
        assert distance < 10, "Overlapping objects should be very close"
    
    def test_near_relation(self):
        """Test near spatial relationship."""
        scene = self.sampler.sample(3, 'near')
        assert len(scene) == 3
        
        # Objects should be clustered together
        center_x = sum(obj['position'][0] for obj in scene) / len(scene)
        center_y = sum(obj['position'][1] for obj in scene) / len(scene)
        
        for obj in scene:
            distance = ((obj['position'][0] - center_x)**2 + (obj['position'][1] - center_y)**2)**0.5
            assert distance < 20, "Objects in 'near' relation should be clustered"
    
    def test_inside_relation(self):
        """Test inside spatial relationship."""
        scene = self.sampler.sample(3, 'inside')
        assert len(scene) == 3
        
        # First object should be larger (container)
        container = scene[0]
        assert 'size' in container
        assert container['size'] >= 60, "Container should be large"
        
        # Other objects should be smaller
        for obj in scene[1:]:
            if 'size' in obj:
                assert obj['size'] <= 30, "Objects inside should be smaller"
    
    def test_random_fallback(self):
        """Test fallback for unknown relations."""
        scene = self.sampler.sample(2, 'unknown_relation')
        assert len(scene) == 2
        
        # Should generate valid positions
        for obj in scene:
            x, y = obj['position']
            assert 0 <= x <= 64, "X position should be within bounds"
            assert 0 <= y <= 64, "Y position should be within bounds"

class TestCoverageTracking:
    """Test coverage tracking functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.tracker = CoverageTracker()
    
    def test_scene_recording(self):
        """Test scene recording for coverage."""
        # Mock scene data
        objs = [
            {'shape': 'circle', 'fill': 'solid'},
            {'shape': 'triangle', 'fill': 'outline'}
        ]
        scene_graph = {
            'relations': [{'type': 'left_of'}]
        }
        
        initial_count = self.tracker.total_scenes_generated
        self.tracker.record_scene(objs, scene_graph, "SHAPE(CIRCLE)", 1)
        
        assert self.tracker.total_scenes_generated == initial_count + 1
        
        # Check that coverage was recorded
        assert any(count > 0 for count in self.tracker.coverage.values())
    
    def test_coverage_quota_check(self):
        """Test coverage quota checking."""
        # Initially no cells should be covered
        assert not self.tracker.is_generation_complete(1)
        
        # Record some scenes to increase coverage
        for i in range(10):
            objs = [{'shape': 'circle', 'fill': 'solid'}]
            scene_graph = {'relations': [{'type': 'none'}]}
            self.tracker.record_scene(objs, scene_graph, "SHAPE(CIRCLE)", 1)
        
        # Check under-covered cells
        under_covered = self.tracker.get_under_covered_cells(5)
        assert isinstance(under_covered, list)
    
    def test_coverage_heatmap_data(self):
        """Test heatmap data generation."""
        # Record a scene
        objs = [{'shape': 'circle', 'fill': 'solid'}]
        scene_graph = {'relations': []}
        self.tracker.record_scene(objs, scene_graph, "SHAPE(CIRCLE)", 1)
        
        heatmap_data = self.tracker.get_coverage_heatmap_data()
        assert 'matrix' in heatmap_data
        assert 'total_cells' in heatmap_data
        assert 'covered_cells' in heatmap_data
        assert isinstance(heatmap_data['matrix'], dict)

class TestDrawingUtils:
    """Test drawing utilities."""
    
    def test_gradient_mask_generation(self):
        """Test gradient mask generation."""
        mask = make_gradient_mask(64, vertical=True)
        assert mask.size == (64, 64)
        assert mask.mode == "L"
        
        # Test vertical gradient
        pixels = list(mask.getdata())
        top_pixel = pixels[0]
        bottom_pixel = pixels[-64]  # Last row, first column
        assert top_pixel < bottom_pixel, "Vertical gradient should increase from top to bottom"
    
    def test_horizontal_gradient_mask(self):
        """Test horizontal gradient mask generation."""
        mask = make_gradient_mask(64, vertical=False)
        assert mask.size == (64, 64)
        
        # Test horizontal gradient
        pixels = list(mask.getdata())
        left_pixel = pixels[64]  # Second row, first column
        right_pixel = pixels[127]  # Second row, last column
        assert left_pixel < right_pixel, "Horizontal gradient should increase from left to right"

class TestRuleValidation:
    """Test rule validation and compliance."""
    
    def test_all_rules_loaded(self):
        """Test that all rules are properly loaded."""
        rules = get_all_rules()
        assert len(rules) > 0, "Should load at least some rules"
        
        # Check that default rule exists
        default_rule = get_rule_by_description("SHAPE(TRIANGLE)")
        assert default_rule is not None, "Default SHAPE(TRIANGLE) rule should exist"
    
    def test_rule_lookup_functionality(self):
        """Test rule lookup functionality."""
        # Test case insensitive lookup
        rule1 = get_rule_by_description("shape(circle)")
        rule2 = get_rule_by_description("SHAPE(CIRCLE)")
        
        if rule1 and rule2:
            assert rule1.description == rule2.description
    
    def test_rule_structure(self):
        """Test that rules have proper structure."""
        rules = get_all_rules()
        
        for rule in rules:
            assert hasattr(rule, 'description'), "Rule should have description"
            assert hasattr(rule, 'positive_features'), "Rule should have positive_features"
            assert hasattr(rule, 'negative_features'), "Rule should have negative_features"
            assert rule.description, "Rule description should not be empty"
            assert isinstance(rule.positive_features, dict), "positive_features should be dict"

class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_generation(self):
        """Test complete scene generation pipeline."""
        rule = get_rule_by_description("SHAPE(TRIANGLE)")
        if not rule:
            pytest.skip("SHAPE(TRIANGLE) rule not found")
        
        # Test both positive and negative examples
        positive_scene = sample_scene_cp(rule, 2, True, 64, 20, 40, 5)
        negative_scene = sample_scene_cp(rule, 2, False, 64, 20, 40, 5)
        
        assert positive_scene is not None, "Positive scene generation should succeed"
        assert negative_scene is not None, "Negative scene generation should succeed"
        
        # Validate scene structure
        for scene in [positive_scene, negative_scene]:
            for obj in scene:
                assert 'position' in obj, "Object should have position"
                assert 'shape' in obj, "Object should have shape"
                assert 'fill' in obj, "Object should have fill"
                assert 'color' in obj, "Object should have color"
    
    def test_coverage_integration(self):
        """Test integration between sampling and coverage tracking."""
        tracker = CoverageTracker()
        rule = get_rule_by_description("SHAPE(CIRCLE)")
        
        if not rule:
            pytest.skip("SHAPE(CIRCLE) rule not found")
        
        # Generate multiple scenes
        for i in range(5):
            scene = sample_scene_cp(rule, 2, True, 64, 20, 40, 5)
            if scene:
                scene_graph = {'relations': []}
                tracker.record_scene(scene, scene_graph, rule.description, 1)
        
        assert tracker.total_scenes_generated >= 5
        
        # Check coverage statistics
        stats = tracker.get_coverage_heatmap_data()
        assert stats['covered_cells'] > 0

# Smoke test for quick validation
def test_smoke_test():
    """Quick smoke test to ensure basic functionality works."""
    # Test rule loading
    rules = get_all_rules()
    assert len(rules) > 0
    
    # Test spatial sampling
    sampler = RelationSampler(64)
    scene = sampler.sample(2, 'left_of')
    assert len(scene) == 2
    
    # Test gradient generation
    gradient = make_gradient_mask(32)
    assert gradient.size == (32, 32)
    
    print(f"✓ Smoke test passed - {len(rules)} rules loaded")
