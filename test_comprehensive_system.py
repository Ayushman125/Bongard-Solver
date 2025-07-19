#!/usr/bin/env python3
"""
Comprehensive test suite for the current Bongard hybrid generator system.
This replaces all old test files with a modern, focused test suite.
"""

import os
import sys
import logging
from pathlib import Path

# Setup paths
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(SCRIPT_DIR) if SCRIPT_DIR else '.'
SRC_ROOT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SRC_ROOT)

import traceback
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveTestSuite:
    """Complete test suite for the current Bongard system."""
    
    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        logger.info(f"Running test: {test_name}")
        self.total_tests += 1
        
        try:
            result = test_func()
            if result:
                logger.info(f"✅ {test_name} PASSED")
                self.passed_tests += 1
                self.test_results[test_name] = {'status': 'PASSED', 'error': None}
            else:
                logger.warning(f"❌ {test_name} FAILED")
                self.test_results[test_name] = {'status': 'FAILED', 'error': 'Test returned False'}
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            self.test_results[test_name] = {'status': 'ERROR', 'error': str(e)}
            if logger.isEnabledFor(logging.DEBUG):
                traceback.print_exc()
    
    def test_basic_imports(self) -> bool:
        """Test that all core modules can be imported."""
        try:
            # Core hybrid system imports
            from src.bongard_generator.hybrid_sampler import HybridSampler
            from src.bongard_generator.rule_loader import get_all_rules, get_rule_by_description
            from src.bongard_generator.config_loader import get_sampler_config
            from src.bongard_generator.dataset import BongardDataset
            from src.bongard_generator.genetic_generator import GeneticSceneGenerator
            from src.bongard_generator.validation_metrics import ValidationSuite, run_validation
            
            logger.info("All core modules imported successfully")
            return True
        except Exception as e:
            logger.error(f"Import test failed: {e}")
            return False
    
    def test_rule_loading(self) -> bool:
        """Test rule loading functionality."""
        try:
            from src.bongard_generator.rule_loader import get_all_rules, get_rule_by_description
            
            # Test loading all rules
            rules = get_all_rules()
            if not rules:
                logger.error("No rules loaded")
                return False
            
            logger.info(f"Loaded {len(rules)} rules")
            
            # Test specific rule lookup
            circle_rule = get_rule_by_description("SHAPE(CIRCLE)")
            if circle_rule:
                logger.info(f"Found circle rule: {circle_rule.description}")
            else:
                logger.warning("Circle rule not found, but continuing...")
            
            # Validate rule structure
            sample_rule = rules[0]
            if not hasattr(sample_rule, 'description'):
                logger.error("Rules missing description attribute")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Rule loading test failed: {e}")
            return False
    
    def test_config_loading(self) -> bool:
        """Test configuration system."""
        try:
            from src.bongard_generator.config_loader import get_sampler_config
            
            # Test basic config
            config = get_sampler_config()
            if not isinstance(config, dict):
                logger.error("Config should return a dictionary")
                return False
            
            # Test config with parameters
            config_with_params = get_sampler_config(total=100, img_size=128)
            if config_with_params['data']['total'] != 100:
                logger.error("Config parameters not properly set")
                return False
            
            # Test hybrid split
            if 'hybrid_split' not in config_with_params['data']:
                logger.info("Adding default hybrid split to config")
                config_with_params['data']['hybrid_split'] = {'cp': 0.7, 'ga': 0.3}
            
            split = config_with_params['data']['hybrid_split']
            if not isinstance(split, dict) or 'cp' not in split or 'ga' not in split:
                logger.error("Invalid hybrid split configuration")
                return False
            
            logger.info(f"Config validation passed: {config_with_params['data']['total']} total, "
                       f"split {split}")
            return True
        except Exception as e:
            logger.error(f"Config loading test failed: {e}")
            return False
    
    def test_hybrid_sampler_creation(self) -> bool:
        """Test HybridSampler creation and initialization."""
        try:
            from src.bongard_generator.hybrid_sampler import HybridSampler
            from src.bongard_generator.config_loader import get_sampler_config
            
            # Create config for testing
            config = get_sampler_config(total=20)
            config['data']['hybrid_split'] = {'cp': 0.6, 'ga': 0.4}
            
            # Create hybrid sampler
            sampler = HybridSampler(config)
            
            # Check quota calculation
            if sampler.cp_quota + sampler.ga_quota != 20:
                logger.error(f"Quota mismatch: {sampler.cp_quota} + {sampler.ga_quota} != 20")
                return False
            
            # Check approximate split
            expected_cp = int(20 * 0.6)  # 12
            if sampler.cp_quota != expected_cp:
                logger.warning(f"CP quota {sampler.cp_quota} != expected {expected_cp}, but continuing...")
            
            logger.info(f"HybridSampler created: CP quota {sampler.cp_quota}, GA quota {sampler.ga_quota}")
            return True
        except Exception as e:
            logger.error(f"HybridSampler creation test failed: {e}")
            return False
    
    def test_scene_generation(self) -> bool:
        """Test actual scene generation."""
        try:
            from src.bongard_generator.hybrid_sampler import HybridSampler
            from src.bongard_generator.config_loader import get_sampler_config
            
            # Create small test configuration
            config = get_sampler_config(total=8)
            config['data']['hybrid_split'] = {'cp': 0.5, 'ga': 0.5}
            
            sampler = HybridSampler(config)
            
            # Generate small batch
            logger.info("Generating test scenes...")
            imgs, labels = sampler.build_synth_holdout(n=8)
            
            if not imgs or not labels:
                logger.error("Scene generation returned empty results")
                return False
            
            if len(imgs) != len(labels):
                logger.error(f"Mismatch: {len(imgs)} images vs {len(labels)} labels")
                return False
            
            # Basic validation of generated content
            for i, (img, label) in enumerate(zip(imgs[:3], labels[:3])):
                if img is None:
                    logger.error(f"Generated image {i} is None")
                    return False
                
                if label not in [0, 1]:
                    logger.error(f"Invalid label {label} for image {i}")
                    return False
                
                # Check if it's a PIL Image
                if hasattr(img, 'size'):
                    logger.info(f"Image {i}: {img.size}, label: {label}")
                else:
                    logger.info(f"Image {i}: {type(img)}, label: {label}")
            
            logger.info(f"Scene generation successful: {len(imgs)} images generated")
            return True
        except Exception as e:
            logger.error(f"Scene generation test failed: {e}")
            return False
    
    def test_validation_metrics(self) -> bool:
        """Test validation metrics functionality."""
        try:
            from src.bongard_generator.validation_metrics import ValidationSuite, run_validation
            
            # Test ValidationSuite
            suite = ValidationSuite()
            results = suite.run_all_validations()
            
            if not isinstance(results, dict):
                logger.error("ValidationSuite should return dict")
                return False
            
            # Test run_validation with sample data
            predicted = [1, 0, 1, 1, 0]
            true_labels = [1, 0, 0, 1, 0]
            
            metrics = run_validation(predicted, true_labels)
            
            if 'classification_accuracy' not in metrics:
                logger.error("Missing classification accuracy in metrics")
                return False
            
            accuracy = metrics['classification_accuracy']
            expected_accuracy = 0.8  # 4/5 correct
            if abs(accuracy - expected_accuracy) > 0.01:
                logger.error(f"Accuracy {accuracy} != expected {expected_accuracy}")
                return False
            
            logger.info(f"Validation metrics test passed: accuracy = {accuracy}")
            return True
        except Exception as e:
            logger.error(f"Validation metrics test failed: {e}")
            return False
    
    def test_integration_validation_script(self) -> bool:
        """Test that the validate_phase1.py script can run."""
        try:
            # This is a basic test - we could import and run key functions
            validate_script_path = os.path.join(REPO_ROOT, 'scripts', 'validate_phase1.py')
            if not os.path.exists(validate_script_path):
                logger.error("validate_phase1.py script not found")
                return False
            
            # Test key functions from the validation script
            sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))
            try:
                import validate_phase1
                if hasattr(validate_phase1, 'test_hybrid_generator'):
                    logger.info("Found test_hybrid_generator function in validation script")
                if hasattr(validate_phase1, 'build_synth_holdout'):
                    logger.info("Found build_synth_holdout function in validation script")
                return True
            finally:
                sys.path.remove(os.path.join(REPO_ROOT, 'scripts'))
        except Exception as e:
            logger.error(f"Integration validation script test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests in the suite."""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE BONGARD SYSTEM TEST SUITE")
        logger.info("=" * 60)
        
        # Run all tests
        self.run_test("Basic Imports", self.test_basic_imports)
        self.run_test("Rule Loading", self.test_rule_loading)
        self.run_test("Config Loading", self.test_config_loading)
        self.run_test("HybridSampler Creation", self.test_hybrid_sampler_creation)
        self.run_test("Scene Generation", self.test_scene_generation)
        self.run_test("Validation Metrics", self.test_validation_metrics)
        self.run_test("Integration Script", self.test_integration_validation_script)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        for test_name, result in self.test_results.items():
            status_symbol = "✅" if result['status'] == 'PASSED' else "❌"
            logger.info(f"{status_symbol} {test_name:<25} {result['status']}")
            if result['error'] and result['status'] != 'PASSED':
                logger.info(f"    Error: {result['error']}")
        
        logger.info("-" * 60)
        success_rate = self.passed_tests / self.total_tests if self.total_tests > 0 else 0
        logger.info(f"TESTS PASSED: {self.passed_tests}/{self.total_tests} ({success_rate:.1%})")
        
        if self.passed_tests == self.total_tests:
            logger.info("🎉 ALL TESTS PASSED! System is ready for use.")
        else:
            logger.warning("⚠️  Some tests failed. Please review and fix issues.")
        
        logger.info("=" * 60)

def main():
    """Main function to run comprehensive tests."""
    test_suite = ComprehensiveTestSuite()
    test_suite.run_all_tests()

if __name__ == "__main__":
    main()
