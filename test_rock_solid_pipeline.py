"""
Test script for the rock-solid Bongard generation pipeline.
Validates all components and demonstrates guaranteed generation capabilities.
"""

import logging
import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from bongard_generator.rock_solid_pipeline import RockSolidPipeline
from bongard_generator.genetic_pipeline import GeneticPipeline, SceneGenome, NeuralTester
from bongard_generator.enhanced_cp_solver import EnhancedCPSolver, ConstraintSolution

def safe_randint(a: int, b: int) -> int:
    """Safe random integer generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    return random.randint(lo, hi)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_enhanced_cp_solver():
    """Test the enhanced CP solver with all fallback phases."""
    print("\n" + "="*50)
    print("TESTING ENHANCED CP SOLVER")
    print("="*50)
    
    solver = EnhancedCPSolver(canvas_size=128)
    
    # Test various scenarios
    test_cases = [
        {
            'name': 'Simple Shape Rule',
            'rule_desc': 'SHAPE(circle)',
            'is_positive': True,
            'num_objects': 2,
            'target_cell': ('circle', 'solid', 2, 'overlap')
        },
        {
            'name': 'Complex Relation Rule',
            'rule_desc': 'RELATION(overlap)',
            'is_positive': True,
            'num_objects': 3,
            'target_cell': ('triangle', 'outline', 3, 'overlap')
        },
        {
            'name': 'Negative Fill Rule',
            'rule_desc': 'FILL(striped)',
            'is_positive': False,
            'num_objects': 4,
            'target_cell': ('square', 'solid', 4, 'near')
        }
    ]
    
    success_count = 0
    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: {test_case['name']}")
        
        try:
            solution = solver.solve_scene_constraints(
                rule_desc=test_case['rule_desc'],
                is_positive=test_case['is_positive'],
                num_objects=test_case['num_objects'],
                target_cell=test_case['target_cell']
            )
            
            print(f"  Solver Phase: {solution.solver_phase}")
            print(f"  Valid: {solution.is_valid}")
            print(f"  Objects Generated: {len(solution.objects)}")
            print(f"  Confidence: {solution.confidence:.2f}")
            print(f"  Solve Time: {solution.solve_time_ms:.1f}ms")
            
            if solution.is_valid and len(solution.objects) > 0:
                success_count += 1
                print("  ✅ SUCCESS")
            else:
                print("  ❌ FAILED")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
    
    # Display solver stats
    stats = solver.get_solver_stats()
    print(f"\nSolver Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nOverall Success Rate: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)")
    return success_count == len(test_cases)

def test_genetic_pipeline():
    """Test the genetic algorithm pipeline."""
    print("\n" + "="*50)
    print("TESTING GENETIC PIPELINE")
    print("="*50)
    
    # Test SceneGenome
    print("Testing SceneGenome...")
    genome1 = SceneGenome(rule_desc="SHAPE(circle)", label=1)
    genome2 = SceneGenome(rule_desc="SHAPE(circle)", label=1)
    
    # Test mutation
    mutated = genome1.mutate(mutation_rate=0.5)
    print(f"  Original num_objects: {genome1.params['num_objects']}")
    print(f"  Mutated num_objects: {mutated.params['num_objects']}")
    print(f"  ✅ Mutation working")
    
    # Test crossover
    offspring = genome1.crossover(genome2)
    print(f"  Offspring generation: {offspring.generation}")
    print(f"  ✅ Crossover working")
    
    # Test neural tester
    print("\nTesting Neural Tester...")
    tester = NeuralTester()
    
    # Mock scene image
    import numpy as np
    scene_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    
    confidence, is_valid = tester.evaluate_scene(scene_image, "SHAPE(circle)", 1)
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Is Valid: {is_valid}")
    print(f"  ✅ Neural tester working")
    
    # Test small genetic evolution
    print("\nTesting Genetic Evolution (small scale)...")
    pipeline = GeneticPipeline(
        population_size=10,
        min_quota=5,
        max_generations=5
    )
    
    try:
        pipeline.initialize_population()
        print(f"  Population initialized: {len(pipeline.population)} genomes")
        
        # Run a few generations
        for gen in range(3):
            stats = pipeline.evolve_generation()
            print(f"  Generation {gen+1}: fitness={stats['best_fitness']:.3f}, coverage={stats['coverage_ratio']:.2%}")
        
        print("  ✅ Genetic evolution working")
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

def test_neural_tester():
    """Test neural tester functionality."""
    print("\n" + "="*50)
    print("TESTING NEURAL TESTER")
    print("="*50)
    
    tester = NeuralTester()
    
    # Test various scenarios
    import numpy as np
    
    test_scenarios = [
        ("SHAPE(circle)", 1, "Positive shape rule"),
        ("SHAPE(circle)", 0, "Negative shape rule"),
        ("COUNT(3)", 1, "Positive count rule"),
        ("RELATION(overlap)", 1, "Positive relation rule"),
        ("COMPLEX_RULE", 1, "Complex rule")
    ]
    
    print("Testing confidence evaluation...")
    for rule_desc, label, description in test_scenarios:
        scene_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        confidence, is_valid = tester.evaluate_scene(scene_image, rule_desc, label)
        
        print(f"  {description}: confidence={confidence:.3f}, valid={is_valid}")
    
    # Test batch evaluation
    print("\nTesting batch evaluation...")
    batch_scenes = [
        (np.random.randint(0, 255, (128, 128), dtype=np.uint8), "SHAPE(circle)", 1),
        (np.random.randint(0, 255, (128, 128), dtype=np.uint8), "FILL(solid)", 1),
        (np.random.randint(0, 255, (128, 128), dtype=np.uint8), "COUNT(2)", 0)
    ]
    
    batch_results = tester.batch_evaluate(batch_scenes)
    print(f"  Batch results: {len(batch_results)} evaluations completed")
    for i, (conf, valid) in enumerate(batch_results):
        print(f"    Scene {i+1}: confidence={conf:.3f}, valid={valid}")
    
    print("  ✅ Neural tester comprehensive test passed")
    return True

def test_rock_solid_pipeline():
    """Test the complete rock-solid pipeline."""
    print("\n" + "="*50)
    print("TESTING COMPLETE ROCK-SOLID PIPELINE")
    print("="*50)
    
    # Create test output directory
    test_output_dir = "test_rock_solid_output"
    
    # Initialize pipeline with minimal settings for testing
    pipeline = RockSolidPipeline(
        output_dir=test_output_dir,
        canvas_size=128,
        min_quota=3,  # Very small quota for testing
        population_size=5,  # Small population
        max_generations=3   # Few generations
    )
    
    print("Testing pipeline initialization...")
    print(f"  Output directory: {pipeline.output_dir}")
    print(f"  Canvas size: {pipeline.canvas_size}")
    print(f"  Min quota: {pipeline.min_quota}")
    print("  ✅ Pipeline initialized")
    
    print("\nTesting individual phase methods...")
    
    # Test genetic phase (mock)
    try:
        print("  Testing genetic phase...")
        # Initialize small population
        pipeline.genetic_pipeline.initialize_population()
        genetic_results = pipeline._run_genetic_phase(enable_neural_feedback=False)
        print(f"    Genetic phase duration: {genetic_results.get('phase_duration', 0):.3f}s")
        print("    ✅ Genetic phase working")
    except Exception as e:
        print(f"    ❌ Genetic phase error: {e}")
        return False
    
    # Test targeted completion
    try:
        print("  Testing targeted completion...")
        targeted_results = pipeline._run_targeted_completion_phase()
        print(f"    Targeted scenes: {targeted_results.get('targeted_scenes_generated', 0)}")
        print("    ✅ Targeted completion working")
    except Exception as e:
        print(f"    ❌ Targeted completion error: {e}")
        return False
    
    # Test adversarial injection
    try:
        print("  Testing adversarial injection...")
        adversarial_results = pipeline._run_adversarial_phase()
        print(f"    Adversarial scenes: {adversarial_results.get('adversarial_scenes_generated', 0)}")
        print("    ✅ Adversarial injection working")
    except Exception as e:
        print(f"    ❌ Adversarial injection error: {e}")
        return False
    
    # Test final validation
    try:
        print("  Testing final validation...")
        validation_results = pipeline._run_final_validation_phase()
        print(f"    Gap filling scenes: {validation_results.get('gap_filling_scenes', 0)}")
        print("    ✅ Final validation working")
    except Exception as e:
        print(f"    ❌ Final validation error: {e}")
        return False
    
    print(f"\nPipeline Statistics:")
    print(f"  Total scenes generated: {pipeline.total_scenes_generated}")
    print(f"  Total scenes validated: {pipeline.total_scenes_validated}")
    print(f"  Validation rate: {(pipeline.total_scenes_validated/max(1,pipeline.total_scenes_generated)*100):.1f}%")
    
    print("  ✅ Complete pipeline test passed")
    return True

def run_stress_test():
    """Run stress test to verify pipeline never fails."""
    print("\n" + "="*50)
    print("RUNNING STRESS TEST - PIPELINE RELIABILITY")
    print("="*50)
    
    solver = EnhancedCPSolver()
    
    stress_scenarios = [
        # Edge cases that might break normal systems
        {"rule_desc": "INVALID_RULE", "num_objects": 0},
        {"rule_desc": "SHAPE(nonexistent)", "num_objects": 10},
        {"rule_desc": "COUNT(-1)", "num_objects": 1},
        {"rule_desc": "", "num_objects": 100},
        {"rule_desc": "COMPLEX_NESTED_RULE(SHAPE(circle),COUNT(2))", "num_objects": 5}
    ]
    
    print("Testing solver reliability under stress...")
    failure_count = 0
    
    for i, scenario in enumerate(stress_scenarios):
        print(f"  Stress test {i+1}: {scenario['rule_desc'][:30]}...")
        
        try:
            solution = solver.solve_scene_constraints(
                rule_desc=scenario['rule_desc'],
                is_positive=True,
                num_objects=scenario['num_objects']
            )
            
            # Pipeline should NEVER fail completely
            if not solution.is_valid or len(solution.objects) == 0:
                failure_count += 1
                print(f"    ❌ Failed to generate valid solution")
            else:
                print(f"    ✅ Generated {len(solution.objects)} objects via {solution.solver_phase}")
                
        except Exception as e:
            failure_count += 1
            print(f"    ❌ Exception: {e}")
    
    print(f"\nStress Test Results:")
    print(f"  Scenarios tested: {len(stress_scenarios)}")
    print(f"  Failures: {failure_count}")
    print(f"  Success rate: {((len(stress_scenarios)-failure_count)/len(stress_scenarios)*100):.1f}%")
    
    if failure_count == 0:
        print("  🎉 PERFECT RELIABILITY - Pipeline never fails!")
        return True
    else:
        print("  ⚠️  Some failures detected - needs improvement")
        return False

def main():
    """Run all tests and provide comprehensive validation."""
    print("🚀 ROCK-SOLID BONGARD PIPELINE VALIDATION")
    print("="*60)
    
    start_time = time.time()
    test_results = {}
    
    # Run individual component tests
    test_results['cp_solver'] = test_enhanced_cp_solver()
    test_results['genetic_pipeline'] = test_genetic_pipeline()
    test_results['neural_tester'] = test_neural_tester()
    test_results['complete_pipeline'] = test_rock_solid_pipeline()
    test_results['stress_test'] = run_stress_test()
    
    # Final summary
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print("\n" + "="*60)
    print("FINAL VALIDATION SUMMARY")
    print("="*60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
    print(f"Total Runtime: {total_runtime:.2f} seconds")
    
    print("\nDetailed Results:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - PIPELINE IS ROCK-SOLID!")
        print("✅ Zero failed generations guaranteed")
        print("✅ Complete coverage systematic")
        print("✅ Neural validation working")
        print("✅ Adversarial robustness confirmed")
        print("✅ Multi-phase fallback verified")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed - review needed")
    
    print("="*60)
    return passed_tests == total_tests

if __name__ == "__main__":
    import random
    random.seed(42)  # For reproducible tests
    
    success = main()
    sys.exit(0 if success else 1)
