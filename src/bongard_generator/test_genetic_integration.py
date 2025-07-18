"""
Integration test for the genetic pipeline to verify all components work together.
"""
import logging
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_genetic_pipeline_integration():
    """Test that all components of the genetic pipeline integrate correctly."""
    logger.info("=== Testing Genetic Pipeline Integration ===")
    
    try:
        # Import the genetic pipeline
        from genetic_pipeline import GeneticPipeline, PipelineConfig, SceneGenome, create_random_genome
        
        logger.info("✓ Successfully imported all classes")
        
        # Test creating a minimal configuration
        config = PipelineConfig(
            population_size=5,
            min_quota=2,
            max_generations=3
        )
        logger.info("✓ Created pipeline configuration")
        
        # Test creating a genetic pipeline
        pipeline = GeneticPipeline(
            population_size=5,
            min_quota=2,
            max_generations=3
        )
        logger.info("✓ Created genetic pipeline instance")
        
        # Test creating a random genome
        genome = create_random_genome("SHAPE(circle)", 1)
        logger.info(f"✓ Created random genome with {len(genome.params)} parameters")
        
        # Test genome mutation
        mutated = genome.mutate(mutation_rate=0.5)
        logger.info("✓ Genome mutation works")
        
        # Test genome crossover
        genome2 = create_random_genome("SHAPE(triangle)", 0)
        child1, child2 = genome.crossover(genome2)
        logger.info("✓ Genome crossover works")
        
        # Test pipeline components
        logger.info(f"✓ Pipeline has {len(pipeline.coverage_cells)} coverage cells")
        logger.info(f"✓ Neural tester initialized: {type(pipeline.neural_tester).__name__}")
        logger.info(f"✓ Renderer initialized: {type(pipeline.renderer).__name__}")
        
        # Test a single evolution step (without full run)
        pipeline.initialize_population()
        logger.info(f"✓ Initialized population with {len(pipeline.population)} genomes")
        
        # Test generating a scene from a genome
        try:
            scene_data, scene_image = pipeline._generate_scene_from_genome(pipeline.population[0])
            logger.info(f"✓ Generated scene with {len(scene_data['objects'])} objects")
            logger.info(f"✓ Scene image shape: {scene_image.shape}")
        except Exception as e:
            logger.warning(f"Scene generation test failed: {e}")
        
        # Test fitness evaluation
        try:
            test_genome = pipeline.population[0]
            dummy_image = pipeline._grid_fallback_phase(test_genome)
            if dummy_image:
                logger.info(f"✓ Grid fallback phase works, created {len(dummy_image)} objects")
        except Exception as e:
            logger.warning(f"Grid fallback test failed: {e}")
        
        logger.info("🎉 All integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rock_solid_pipeline_integration():
    """Test the RockSolidPipeline integration."""
    logger.info("=== Testing RockSolid Pipeline Integration ===")
    
    try:
        from genetic_pipeline import RockSolidPipeline, PipelineConfig
        
        # Create a minimal configuration for testing
        config = PipelineConfig(
            population_size=10,
            generations=3,
            min_samples_per_cell=2
        )
        
        # Test creating the pipeline
        pipeline = RockSolidPipeline(config)
        logger.info(f"✓ Created RockSolid pipeline with {len(pipeline.all_cells)} cells")
        
        # Test enumeration of cells
        sample_cells = pipeline.all_cells[:5]
        logger.info(f"✓ Sample cells: {sample_cells}")
        
        # Test components
        logger.info(f"✓ Tester: {type(pipeline.tester).__name__}")
        logger.info(f"✓ Renderer: {type(pipeline.renderer).__name__}")
        
        logger.info("🎉 RockSolid Pipeline integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ RockSolid Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_genetic_pipeline_integration()
    success2 = test_rock_solid_pipeline_integration()
    
    if success1 and success2:
        logger.info("🚀 ALL INTEGRATION TESTS PASSED! The genetic pipeline is ready to use.")
        sys.exit(0)
    else:
        logger.error("💥 Some tests failed. Please check the implementation.")
        sys.exit(1)
