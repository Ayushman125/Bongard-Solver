"""
Test script to verify genetic pipeline integration with existing codebase.
"""

import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_integration():
    """Test that all components integrate correctly."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Test imports
        print("Testing imports...")
        from genetic_pipeline import (
            GeneticPipeline, 
            RockSolidPipeline, 
            SceneGenome, 
            MockTesterCNN, 
            BongardRenderer,
            create_random_genome,
            USE_EXISTING_CLASSES
        )
        print(f"✓ All imports successful. Using existing classes: {USE_EXISTING_CLASSES}")
        
        # Test SceneGenome creation
        print("\nTesting SceneGenome creation...")
        genome = create_random_genome("SHAPE(circle)", 1)
        print(f"✓ Created genome: {genome.rule_desc}, objects: {len(genome.params.get('objects', []))}")
        
        # Test mutation
        print("\nTesting mutation...")
        if hasattr(genome, 'mutate'):
            mutated = genome.mutate()
            print(f"✓ Mutation successful: {mutated.rule_desc}")
        else:
            print("⚠ Mutation method not available")
        
        # Test crossover
        print("\nTesting crossover...")
        genome2 = create_random_genome("SHAPE(triangle)", 0)
        if hasattr(genome, 'crossover'):
            child1, child2 = genome.crossover(genome2)
            print(f"✓ Crossover successful: {child1.rule_desc}, {child2.rule_desc}")
        else:
            print("⚠ Crossover method not available")
        
        # Test MockTesterCNN
        print("\nTesting MockTesterCNN...")
        tester = MockTesterCNN()
        import numpy as np
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        confidence = tester.predict_confidence(test_image, "SHAPE(circle)")
        print(f"✓ TesterCNN confidence: {confidence:.3f}")
        
        # Test BongardRenderer
        print("\nTesting BongardRenderer...")
        renderer = BongardRenderer()
        test_objects = [{
            'shape': 'circle',
            'position': (64, 64),
            'size': 30,
            'color': 'blue',
            'fill': 'solid'
        }]
        image = renderer.render_scene(test_objects, 128, 'white', 'numpy')
        print(f"✓ Renderer produced image: {image.shape}")
        
        # Test GeneticPipeline initialization
        print("\nTesting GeneticPipeline initialization...")
        pipeline = GeneticPipeline(
            population_size=10,
            min_quota=2,
            max_generations=5
        )
        print(f"✓ GeneticPipeline initialized with {len(pipeline.coverage_cells)} cells")
        
        # Test RockSolidPipeline initialization
        print("\nTesting RockSolidPipeline initialization...")
        from genetic_pipeline import PipelineConfig
        config = PipelineConfig()
        rock_pipeline = RockSolidPipeline(config)
        print(f"✓ RockSolidPipeline initialized with {len(rock_pipeline.all_cells)} cells")
        
        print("\n🎉 All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
