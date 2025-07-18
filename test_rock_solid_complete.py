#!/usr/bin/env python3
"""
Complete test of the rock-solid Bongard generation pipeline.
Tests all components including CP-SAT, genetic algorithms, and neural feedback.
"""

import sys
import logging
from pathlib import Path
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core components
        from src.bongard_generator.scene_genome import SceneGenome, create_random_genome
        print("✓ SceneGenome imported successfully")
        
        from src.bongard_generator.tester_cnn import TesterCNN, MockTesterCNN
        print("✓ TesterCNN imported successfully")
        
        from src.bongard_generator.drawing import BongardRenderer
        print("✓ BongardRenderer imported successfully")
        
        from src.bongard_generator.constraints import PlacementOptimizer
        print("✓ PlacementOptimizer imported successfully")
        
        from src.bongard_rules import ALL_BONGARD_RULES
        print(f"✓ ALL_BONGARD_RULES imported successfully ({len(ALL_BONGARD_RULES)} rules)")
        
        from src.bongard_generator.genetic_pipeline import RockSolidPipeline, PipelineConfig
        print("✓ RockSolidPipeline imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_scene_genome():
    """Test SceneGenome functionality."""
    print("\nTesting SceneGenome...")
    
    try:
        from src.bongard_generator.scene_genome import create_random_genome
        
        # Create a random genome
        genome = create_random_genome("SHAPE(triangle)", label=1)
        print(f"✓ Created genome with rule: {genome.rule_desc}")
        print(f"  - Label: {genome.label}")
        print(f"  - Parameters keys: {list(genome.params.keys())}")
        
        # Test mutation
        mutated = genome.mutate()
        print(f"✓ Mutation successful, generation: {mutated.generation}")
        
        # Test crossover
        genome2 = create_random_genome("FILL(solid)", label=0)
        child = genome.crossover(genome2)
        print(f"✓ Crossover successful, child rule: {child.rule_desc}")
        
        return True
        
    except Exception as e:
        print(f"✗ SceneGenome test failed: {e}")
        traceback.print_exc()
        return False

def test_tester_cnn():
    """Test TesterCNN functionality."""
    print("\nTesting TesterCNN...")
    
    try:
        from src.bongard_generator.tester_cnn import MockTesterCNN
        import numpy as np
        
        # Create mock tester
        tester = MockTesterCNN()
        print("✓ MockTesterCNN created")
        
        # Test confidence prediction
        dummy_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        confidence = tester.predict_confidence(dummy_image, "SHAPE(triangle)")
        print(f"✓ Confidence prediction: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ TesterCNN test failed: {e}")
        traceback.print_exc()
        return False

def test_renderer():
    """Test BongardRenderer functionality."""
    print("\nTesting BongardRenderer...")
    
    try:
        from src.bongard_generator.drawing import BongardRenderer
        
        # Create renderer
        renderer = BongardRenderer()
        print("✓ BongardRenderer created")
        
        # Test scene rendering
        test_objects = [
            {
                'shape': 'circle',
                'position': (64, 64),
                'size': 30,
                'fill': 'solid',
                'color': 'black'
            }
        ]
        
        img = renderer.render_scene(test_objects, output_format='numpy')
        print(f"✓ Scene rendered, shape: {img.shape}, dtype: {img.dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ Renderer test failed: {e}")
        traceback.print_exc()
        return False

def test_pipeline_config():
    """Test PipelineConfig."""
    print("\nTesting PipelineConfig...")
    
    try:
        from src.bongard_generator.genetic_pipeline import PipelineConfig
        
        config = PipelineConfig(
            population_size=10,
            generations=5,
            min_samples_per_cell=2
        )
        print("✓ PipelineConfig created")
        print(f"  - Population size: {config.population_size}")
        print(f"  - Generations: {config.generations}")
        print(f"  - Min samples: {config.min_samples_per_cell}")
        
        return True
        
    except Exception as e:
        print(f"✗ PipelineConfig test failed: {e}")
        traceback.print_exc()
        return False

def test_pipeline_initialization():
    """Test RockSolidPipeline initialization."""
    print("\nTesting RockSolidPipeline initialization...")
    
    try:
        from src.bongard_generator.genetic_pipeline import RockSolidPipeline, PipelineConfig
        
        # Create test config
        config = PipelineConfig(
            population_size=5,
            generations=2,
            min_samples_per_cell=1
        )
        
        # Initialize pipeline
        pipeline = RockSolidPipeline(config)
        print("✓ RockSolidPipeline initialized")
        print(f"  - Coverage cells: {len(pipeline.all_cells)}")
        print(f"  - Population dict: {len(pipeline.populations)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        traceback.print_exc()
        return False

def test_mini_evolution():
    """Test a mini evolution cycle."""
    print("\nTesting mini evolution cycle...")
    
    try:
        from src.bongard_generator.genetic_pipeline import RockSolidPipeline, PipelineConfig
        
        # Create minimal config for testing
        config = PipelineConfig(
            population_size=3,
            generations=1,
            min_samples_per_cell=1,
            elite_size=1
        )
        
        # Initialize pipeline
        pipeline = RockSolidPipeline(config)
        print("✓ Pipeline initialized for mini evolution")
        
        # Initialize populations
        pipeline._initialize_populations()
        print(f"✓ Populations initialized: {len(pipeline.populations)} rules")
        
        # Test single generation evolution (this might take a moment)
        print("Running single generation evolution...")
        gen_stats = pipeline._evolve_generation(0)
        print(f"✓ Generation evolved successfully")
        print(f"  - Total evaluations: {gen_stats['total_evaluations']}")
        print(f"  - Average fitness: {gen_stats['average_fitness']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Mini evolution test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Rock-Solid Pipeline Complete Test ===\n")
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    tests = [
        ("Import Test", test_imports),
        ("SceneGenome Test", test_scene_genome),
        ("TesterCNN Test", test_tester_cnn),
        ("Renderer Test", test_renderer),
        ("Config Test", test_pipeline_config),
        ("Pipeline Init Test", test_pipeline_initialization),
        ("Mini Evolution Test", test_mini_evolution),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("The rock-solid pipeline is ready for full execution!")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please fix before running full pipeline.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
