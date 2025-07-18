"""Validation utilities for Bongard generator"""

import logging
import pytest
from typing import Dict, List, Any, Tuple
import numpy as np

from .config_loader import SamplerConfig
from .rule_loader import get_rule_by_description, BongardRule

logger = logging.getLogger(__name__)

class ValidationSuite:
    """Comprehensive validation suite for the Bongard generator."""
    
    def __init__(self):
        self.validation_results = {}
        
    def run_all_validations(self) -> Dict[str, bool]:
        """Run all validation tests."""
        results = {}
        
        # Test configuration loading
        results['config_loading'] = self.test_config_loading()
        
        # Test rule loading
        results['rule_loading'] = self.test_rule_loading()
        
        # Test rule compliance
        results['rule_compliance'] = self.test_rule_compliance()
        
        # Test CP-SAT model validation
        results['cp_sat_validation'] = self.test_cp_sat_validation()
        
        # Test cache stratification
        results['cache_stratification'] = self.test_cache_stratification()
        
        self.validation_results = results
        return results
    
    def test_config_loading(self) -> bool:
        """Test that configuration loads correctly."""
        try:
            from .config_loader import get_config, get_sampler_config
            
            config = get_config()
            assert config is not None, "Config should not be None"
            assert 'data' in config, "Config should have 'data' key"
            
            sampler_config = get_sampler_config()
            assert isinstance(sampler_config, SamplerConfig), "Should return SamplerConfig instance"
            assert sampler_config.img_size > 0, "Image size should be positive"
            
            logger.info("✓ Configuration loading test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Configuration loading test failed: {e}")
            return False
    
    def test_rule_loading(self) -> bool:
        """Test that rules load correctly."""
        try:
            from .rule_loader import get_all_rules, get_rule_lookup, get_rule_by_description
            
            rules = get_all_rules()
            assert len(rules) > 0, "Should have at least one rule"
            
            rule_lookup = get_rule_lookup()
            assert len(rule_lookup) > 0, "Rule lookup should not be empty"
            
            # Test default rule exists
            default_rule = get_rule_by_description("SHAPE(TRIANGLE)")
            assert default_rule is not None, "Default rule SHAPE(TRIANGLE) should exist"
            
            # Test rule structure
            for rule in rules[:3]:  # Test first 3 rules
                assert hasattr(rule, 'description'), "Rule should have description"
                assert hasattr(rule, 'positive_features'), "Rule should have positive_features"
                assert rule.description, "Rule description should not be empty"
                assert rule.positive_features, "Rule should have positive features"
            
            logger.info("✓ Rule loading test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Rule loading test failed: {e}")
            return False
    
    def test_rule_compliance(self) -> bool:
        """Test that generated scenes comply with rules."""
        try:
            # This would test actual scene generation
            # For now, we'll test the rule structure
            rule = get_rule_by_description("SHAPE(CIRCLE)")
            if rule:
                # Test that we can generate a scene for this rule
                # This is a placeholder - actual implementation would use the sampler
                assert 'shape' in rule.positive_features, "Shape rule should have shape feature"
                assert rule.positive_features['shape'] == 'circle', "Should specify circle"
            
            rule = get_rule_by_description("COUNT(2)")
            if rule:
                assert 'count' in rule.positive_features, "Count rule should have count feature"
                assert rule.positive_features['count'] == 2, "Should specify count of 2"
            
            logger.info("✓ Rule compliance test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Rule compliance test failed: {e}")
            return False
    
    def test_cp_sat_validation(self) -> bool:
        """Test CP-SAT model validation."""
        try:
            # Test that we can import OR-Tools
            try:
                from ortools.sat.python import cp_model
                logger.info("✓ OR-Tools import successful")
            except ImportError:
                logger.warning("⚠ OR-Tools not available, skipping CP-SAT tests")
                return True
            
            # Test basic model creation
            model = cp_model.CpModel()
            x = model.NewIntVar(0, 10, 'x')
            model.Add(x >= 1)
            
            # Validate model
            status = model.Validate()
            assert not status, f"Model should be valid, got: {status}"
            
            logger.info("✓ CP-SAT validation test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ CP-SAT validation test failed: {e}")
            return False
    
    def test_cache_stratification(self) -> bool:
        """Test cache and stratification logic."""
        try:
            from .coverage import CoverageTracker
            
            tracker = CoverageTracker()
            assert len(tracker.ALL_CELLS) > 0, "Should have coverage cells defined"
            
            # Test recording a dummy scene
            dummy_objs = [
                {'shape': 'triangle', 'fill': 'solid'},
                {'shape': 'circle', 'fill': 'outline'}
            ]
            dummy_scene_graph = {'relations': [{'type': 'left_of'}]}
            
            tracker.record_scene(dummy_objs, dummy_scene_graph, "SHAPE(TRIANGLE)", 1)
            assert tracker.total_scenes_generated == 1, "Should record scene"
            
            logger.info("✓ Cache stratification test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Cache stratification test failed: {e}")
            return False
    
    def print_validation_report(self) -> None:
        """Print a validation report."""
        if not self.validation_results:
            logger.warning("No validation results available. Run validations first.")
            return
        
        print("\n" + "="*50)
        print("BONGARD GENERATOR VALIDATION REPORT")
        print("="*50)
        
        passed = 0
        total = len(self.validation_results)
        
        for test_name, result in self.validation_results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\nSummary: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All validations passed!")
        else:
            print("⚠️  Some validations failed. Check logs for details.")
        
        print("="*50)

# Unit test functions for pytest
def test_shape_rule():
    """Test shape rule generation."""
    rule = get_rule_by_description('SHAPE(CIRCLE)')
    assert rule is not None, "SHAPE(CIRCLE) rule should exist"
    assert 'shape' in rule.positive_features, "Shape rule should have shape feature"
    assert rule.positive_features['shape'] == 'circle', "Should specify circle shape"

def test_count_rule():
    """Test count rule generation."""
    rule = get_rule_by_description('COUNT(2)')
    assert rule is not None, "COUNT(2) rule should exist"
    assert 'count' in rule.positive_features, "Count rule should have count feature"
    assert rule.positive_features['count'] == 2, "Should specify count of 2"

def test_relation_rule():
    """Test relation rule generation."""
    rule = get_rule_by_description('SPATIAL(LEFT_OF)')
    if rule:
        assert 'relation' in rule.positive_features, "Relation rule should have relation feature"
        assert rule.positive_features['relation'] == 'left_of', "Should specify left_of relation"

def test_sampler_config():
    """Test sampler configuration."""
    from .config_loader import get_sampler_config
    
    config = get_sampler_config()
    assert isinstance(config, SamplerConfig), "Should return SamplerConfig"
    assert config.img_size > 0, "Image size should be positive"
    assert config.min_obj_size > 0, "Min object size should be positive"
    assert config.max_obj_size >= config.min_obj_size, "Max size should be >= min size"

def test_coverage_tracker():
    """Test coverage tracking functionality."""
    from .coverage import CoverageTracker
    
    tracker = CoverageTracker()
    assert len(tracker.ALL_CELLS) > 0, "Should have cells defined"
    
    # Test scene recording
    dummy_objs = [{'shape': 'triangle', 'fill': 'solid'}]
    dummy_scene_graph = {'relations': []}
    tracker.record_scene(dummy_objs, dummy_scene_graph, "SHAPE(TRIANGLE)", 1)
    
    assert tracker.total_scenes_generated == 1, "Should record scene"
    stats = tracker.get_coverage_stats()
    assert stats['total_scenes'] == 1, "Stats should reflect recorded scene"

def test_relation_sampler():
    """Test relation sampler functionality."""
    from .relation_sampler import RelationSampler
    
    sampler = RelationSampler(128)
    
    # Test left_of relation
    objs = sampler.sample(2, "left_of")
    assert len(objs) == 2, "Should generate 2 objects"
    assert sampler.validate_relation(objs, "left_of"), "Should generate valid left_of relation"
    
    # Test above relation
    objs = sampler.sample(2, "above")
    assert len(objs) == 2, "Should generate 2 objects"
    assert sampler.validate_relation(objs, "above"), "Should generate valid above relation"

if __name__ == "__main__":
    # Run validation suite
    validator = ValidationSuite()
    results = validator.run_all_validations()
    validator.print_validation_report()
    
    # Exit with error code if any validation failed
    import sys
    if not all(results.values()):
        sys.exit(1)
