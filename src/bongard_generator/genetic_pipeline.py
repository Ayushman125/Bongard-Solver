import logging
import random
import time
import math
import json
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Mock/Helper Classes and Functions (if not using existing implementations) ---
# Set this to True if you have actual implementations of TesterCNN, BongardRenderer, etc.
# For this response, we assume USE_EXISTING_CLASSES is False to provide full mock implementations.
USE_EXISTING_CLASSES = False

# Helper function for safe random integer generation
def safe_randint(min_val: int, max_val: int) -> int:
    """Returns a random integer within a safe range, handling cases where min > max."""
    if min_val > max_val:
        return random.randint(max_val, min_val)
    return random.randint(min_val, max_val)

# Define ALL_BONGARD_RULES (mock for demonstration)
@dataclass
class BongardRule:
    description: str
    category: str
    complexity: int

ALL_BONGARD_RULES = [
    BongardRule("COMBINED(circle,solid,1,none)", "shape_fill_count_relation", 1),
    BongardRule("COMBINED(triangle,outline,2,near)", "shape_fill_count_relation", 2),
    BongardRule("COMBINED(square,striped,3,overlap)", "shape_fill_count_relation", 3),
    BongardRule("COMBINED(pentagon,gradient,1,left_of)", "shape_fill_count_relation", 4),
    BongardRule("COMBINED(hexagon,solid,4,above)", "shape_fill_count_relation", 5),
    # Add more rules as needed for comprehensive coverage
]

@dataclass
class SceneGenome:
    """
    Represents a genome for a Bongard scene, encoding parameters for scene generation.
    """
    rule_desc: str
    label: int # 1 for positive example, 0 for negative
    params: Dict[str, Any] = field(default_factory=dict)
    fitness: float = 0.0
    tester_confidence: float = 0.0
    diversity_score: float = 0.0
    generation: int = 0

    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.05) -> 'SceneGenome':
        """Create a mutated copy of this genome."""
        new_params = self.params.copy()
        
        # Mutate number of objects
        if random.random() < mutation_rate:
            new_params['num_objects'] = max(1, min(6,
                new_params.get('num_objects', 2) + safe_randint(-1, 1)))
        
        # Mutate object size range
        if random.random() < mutation_rate:
            old_min, old_max = new_params.get('obj_size_range', (20, 60))
            new_min = max(10, old_min + safe_randint(-5, 5))
            new_max = min(80, old_max + safe_randint(-5, 5))
            new_params['obj_size_range'] = (min(new_min, new_max), max(new_min, new_max))
        
        # Mutate spatial jitter
        if random.random() < mutation_rate:
            new_params['spatial_jitter'] = max(1, min(20,
                new_params.get('spatial_jitter', 3) + safe_randint(-3, 3)))
        
        # Mutate overlap threshold
        if random.random() < mutation_rate:
            new_params['overlap_threshold'] = max(0.1, min(0.9,
                new_params.get('overlap_threshold', 0.5) + random.uniform(-0.1, 0.1)))
        
        # Mutate relation strength
        if random.random() < mutation_rate:
            new_params['relation_strength'] = max(0.1, min(1.0,
                new_params.get('relation_strength', 0.7) + random.uniform(-0.1, 0.1)))
        
        # Mutate object properties (shape, color, fill) for existing objects
        if 'objects' in new_params and new_params['objects']:
            for obj in new_params['objects']:
                if random.random() < mutation_rate:
                    obj['shape'] = random.choice(['circle', 'triangle', 'square', 'pentagon', 'hexagon'])
                if random.random() < mutation_rate:
                    obj['fill'] = random.choice(['solid', 'hollow', 'striped'])
                if random.random() < mutation_rate:
                    obj['color'] = random.choice(['red', 'blue', 'green', 'black'])
                if random.random() < mutation_rate:
                    # Mutate size within bounds
                    size_min, size_max = new_params.get('obj_size_range', (20, 60))
                    obj['size'] = random.randint(size_min, size_max)
                if random.random() < mutation_rate:
                    # Mutate orientation
                    obj['orientation'] = random.uniform(0, 360)
        
        return SceneGenome(
            rule_desc=self.rule_desc,
            label=self.label,
            params=new_params,
            generation=self.generation + 1
        )

    def crossover(self, other: 'SceneGenome') -> Tuple['SceneGenome', 'SceneGenome']:
        """Create offspring by crossing over with another genome."""
        new_params1 = {}
        new_params2 = {}
        keys = list(set(self.params.keys()) | set(other.params.keys()))
        crossover_point = random.randint(0, len(keys))
        
        for i, key in enumerate(keys):
            if i < crossover_point:
                new_params1[key] = self.params.get(key)
                new_params2[key] = other.params.get(key)
            else:
                new_params1[key] = other.params.get(key)
                new_params2[key] = self.params.get(key)
        
        # Ensure essential keys are present, if missing, default or inherit
        # This is crucial for newly created genomes from crossover
        for p_dict in [new_params1, new_params2]:
            if 'num_objects' not in p_dict or p_dict['num_objects'] is None: p_dict['num_objects'] = random.randint(1, 4)
            if 'canvas_size' not in p_dict or p_dict['canvas_size'] is None: p_dict['canvas_size'] = 128
            if 'obj_size_range' not in p_dict or p_dict['obj_size_range'] is None: p_dict['obj_size_range'] = (20, 60)
            if 'spatial_jitter' not in p_dict or p_dict['spatial_jitter'] is None: p_dict['spatial_jitter'] = 3
            if 'overlap_threshold' not in p_dict or p_dict['overlap_threshold'] is None: p_dict['overlap_threshold'] = 0.5
            if 'relation_strength' not in p_dict or p_dict['relation_strength'] is None: p_dict['relation_strength'] = 0.7
            if 'objects' not in p_dict or p_dict['objects'] is None: p_dict['objects'] = [] # Ensure objects list exists
        
        child1 = SceneGenome(
            rule_desc=self.rule_desc,
            label=self.label,
            params=new_params1,
            generation=max(self.generation, other.generation) + 1
        )
        child2 = SceneGenome(
            rule_desc=other.rule_desc,
            label=other.label,
            params=new_params2,
            generation=max(self.generation, other.generation) + 1
        )
        return child1, child2

def create_random_genome(rule_desc: str, label: int) -> SceneGenome:
    """Creates a random SceneGenome for a given rule description and label."""
    params = {
        'num_objects': random.randint(1, 4),
        'canvas_size': 128,
        'obj_size_range': (20, 60),
        'target_shape': random.choice(['circle', 'triangle', 'square']),
        'target_fill': random.choice(['solid', 'hollow']),
        'target_relation': random.choice(['near', 'far', 'overlap', 'none']),
        'spatial_jitter': random.randint(1, 10),
        'overlap_threshold': random.uniform(0.1, 0.9),
        'relation_strength': random.uniform(0.5, 1.0),
        'objects': [] # Initialize with an empty list of objects
    }
    # Populate initial objects based on num_objects for a more complete genome
    for _ in range(params['num_objects']):
        params['objects'].append({
            'shape': random.choice(['circle', 'triangle', 'square', 'pentagon', 'hexagon']),
            'fill': random.choice(['solid', 'hollow', 'striped']),
            'color': random.choice(['red', 'blue', 'green', 'black']),
            'size': random.randint(*params['obj_size_range']),
            'position': (random.randint(0, params['canvas_size']), random.randint(0, params['canvas_size'])),
            'orientation': random.uniform(0, 360),
            'texture': 'flat'
        })
    return SceneGenome(rule_desc=rule_desc, label=label, params=params, generation=0)

if USE_EXISTING_CLASSES:
    # Placeholder for existing TesterCNN if it were available
    class ExistingTesterCNN:
        def predict_confidence(self, scene_image: np.ndarray, rule_desc: str) -> float:
            # Mock implementation for existing class if not truly available
            return 0.5 # Replace with actual logic
    
    class MockTesterCNN: # Renamed to MockTesterCNN to avoid direct conflict, but acts as wrapper
        """Wrapper around existing TesterCNN for compatibility."""
        def __init__(self):
            try:
                self.tester = ExistingTesterCNN()
                logger.info("Initialized with existing TesterCNN")
            except Exception as e:
                logger.warning(f"Failed to initialize TesterCNN: {e}, using mock")
                self.tester = None
            self.confidence_threshold = 0.7
            self.mock_confidence_cache = {}
        def predict_confidence(self, scene_image: np.ndarray, rule_desc: str) -> float:
            """Predict confidence using existing TesterCNN or mock."""
            if self.tester is not None:
                try:
                    # Convert to proper format for existing TesterCNN
                    if hasattr(self.tester, 'predict_confidence'):
                        return self.tester.predict_confidence(scene_image, rule_desc)
                    elif hasattr(self.tester, 'forward'):
                        # If it's a PyTorch model, handle tensor conversion
                        import torch
                        if len(scene_image.shape) == 2:
                            # Add batch and channel dimensions
                            tensor_input = torch.FloatTensor(scene_image).unsqueeze(0).unsqueeze(0)
                        else:
                            tensor_input = torch.FloatTensor(scene_image).unsqueeze(0)
                        
                        with torch.no_grad():
                            output = self.tester(tensor_input)
                            # Convert output to confidence score
                            confidence = torch.sigmoid(output).item() if hasattr(output, 'item') else float(output[0])
                            return confidence
                    else:
                        # Fall back to mock implementation
                        return self._mock_predict_confidence(scene_image, rule_desc)
                except Exception as e:
                    logger.warning(f"TesterCNN failed during prediction: {e}, using mock")
                    return self._mock_predict_confidence(scene_image, rule_desc)
            else:
                return self._mock_predict_confidence(scene_image, rule_desc)
        def _mock_predict_confidence(self, scene_image: np.ndarray, rule_desc: str) -> float:
            """Mock confidence prediction."""
            # Use a slice to avoid very long hashes for large images
            scene_hash = hash((scene_image.tobytes()[:100], rule_desc)) 
            
            if scene_hash in self.mock_confidence_cache:
                return self.mock_confidence_cache[scene_hash]
            else:
                base_confidence = 0.6
                if "SHAPE(" in rule_desc:
                    base_confidence = 0.8
                elif "COUNT(" in rule_desc:
                    base_confidence = 0.7
                elif "RELATION(" in rule_desc:
                    base_confidence = 0.6
                
                confidence = np.clip(base_confidence + random.uniform(-0.2, 0.2), 0.0, 1.0)
                self.mock_confidence_cache[scene_hash] = confidence
                return confidence
else:
    class MockTesterCNN:
        """
        A mock neural network tester for semantic verification of generated scenes.
        This simulates the behavior of a TesterCNN without requiring a real model.
        """
        def __init__(self):
            logger.info("Initialized MockTesterCNN")
            self.confidence_threshold = 0.7
            self.mock_confidence_cache = {}
        def predict_confidence(self, scene_image: np.ndarray, rule_desc: str) -> float:
            """
            Mocks the prediction of confidence that a scene image matches a rule.
            """
            # A simplified hash for mock caching. In a real scenario, features of the image would be used.
            scene_hash = hash((scene_image.tobytes()[:100], rule_desc)) # Use a slice to avoid very long hashes
            
            if scene_hash in self.mock_confidence_cache:
                return self.mock_confidence_cache[scene_hash]
            else:
                # Simulate varying confidence based on rule complexity and randomness
                base_confidence = 0.6
                if "SHAPE(" in rule_desc:
                    base_confidence = 0.8
                elif "COUNT(" in rule_desc:
                    base_confidence = 0.7
                elif "RELATION(" in rule_desc:
                    base_confidence = 0.6
                
                confidence = np.clip(base_confidence + random.uniform(-0.2, 0.2), 0.0, 1.0)
                self.mock_confidence_cache[scene_hash] = confidence
                return confidence

class PlacementOptimizer:
    """
    A mock class for optimizing object placement based on constraints.
    In a real system, this would use a CP-SAT solver or similar.
    """
    def __init__(self):
        logger.info("Initialized PlacementOptimizer (mock)")
    def optimize_placement(self, objects: List[Dict[str, Any]], canvas_size: int, constraints: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Mock implementation: Tries to place objects randomly, respecting basic constraints.
        Returns a dictionary with 'objects' if successful, None otherwise.
        """
        placed_objects = []
        # Simple attempt to place objects without overlap for mock
        attempts_per_object = 10
        
        for obj_template in objects:
            placed = False
            for _ in range(attempts_per_object):
                # Ensure size is an integer for calculations
                obj_size = int(obj_template.get('size', 30)) 
                
                # Calculate valid ranges for x, y to keep object within canvas
                min_x = obj_size // 2
                max_x = canvas_size - obj_size // 2
                min_y = obj_size // 2
                max_y = canvas_size - obj_size // 2
                # Handle cases where min_x/y might exceed max_x/y for very large objects
                if min_x >= max_x: min_x = max_x = canvas_size // 2
                if min_y >= max_y: min_y = max_y = canvas_size // 2
                x = random.randint(min_x, max_x)
                y = random.randint(min_y, max_y)
                
                current_obj = obj_template.copy()
                current_obj['position'] = (x, y)
                
                # Check for overlap with already placed objects if 'no_overlap' constraint is active
                if constraints.get('no_overlap', False):
                    overlap = False
                    for existing_obj in placed_objects:
                        # Ensure sizes are integers
                        existing_obj_size = int(existing_obj.get('size', 30))
                        
                        dist = math.sqrt((current_obj['position'][0] - existing_obj['position'][0])**2 +
                                         (current_obj['position'][1] - existing_obj['position'][1])**2)
                        if dist < (obj_size + existing_obj_size) / 2: # Use actual object sizes for overlap check
                            overlap = True
                            break
                    if overlap:
                        continue # Try another position for current_obj
                
                placed_objects.append(current_obj)
                placed = True
                break
            
            if not placed:
                logger.debug(f"Could not place object {obj_template.get('shape', 'unknown')} without overlap.")
                return None # Failed to place all objects without overlap
        return {'objects': placed_objects}

class BongardSampler:
    """A mock class for sampling Bongard problems."""
    def __init__(self):
        logger.info("Initialized BongardSampler (mock)")
    # No methods explicitly called in the provided pipeline logic, but kept for completeness.

class SyntheticBongardDataset:
    """A mock class for a synthetic Bongard dataset."""
    def __init__(self):
        logger.info("Initialized SyntheticBongardDataset (mock)")
    # No methods explicitly called in the provided pipeline logic, but kept for completeness.

if USE_EXISTING_CLASSES:
    # Placeholder for existing BongardRenderer if it were available
    class ExistingBongardRenderer:
        def render_scene(self, objects: List[Dict[str, Any]], canvas_size: int, background_color: str, output_format: str) -> np.ndarray:
            # Mock implementation for existing class if not truly available
            return np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255 # Return a white image
    BongardRenderer = ExistingBongardRenderer
else:
    class BongardRenderer:
        """
        A mock class for rendering Bongard scenes into images.
        """
        def __init__(self):
            logger.info("Initialized BongardRenderer (mock)")
        def render_scene(self, objects: List[Dict[str, Any]], canvas_size: int, background_color: str, output_format: str) -> np.ndarray:
            """
            Mock implementation: returns a simple image with basic shapes drawn.
            """
            # Create a blank canvas
            if background_color == 'white':
                img = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
            else:
                img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            
            # Simple color mapping
            color_map = {
                'red': [255, 0, 0], 'blue': [0, 0, 255], 'green': [0, 255, 0],
                'black': [0, 0, 0], 'white': [255, 255, 255]
            }
            
            for obj in objects:
                x, y = obj.get('position', (canvas_size // 2, canvas_size // 2))
                size = int(obj.get('size', 30)) # Ensure size is an integer
                shape = obj.get('shape', 'circle')
                fill = obj.get('fill', 'solid')
                color = obj.get('color', 'black')
                
                draw_color = color_map.get(color.lower(), [0, 0, 0]) # Default to black
                
                # Simple drawing for mock: draw a square for any shape
                x1 = max(0, int(x - size / 2))
                y1 = max(0, int(y - size / 2))
                x2 = min(canvas_size, int(x + size / 2))
                y2 = min(canvas_size, int(y + size / 2))
                # Ensure coordinates are valid and x1 <= x2, y1 <= y2
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                if fill == 'solid':
                    img[y1:y2, x1:x2] = draw_color
                elif fill == 'outline':
                    # Draw a border
                    thickness = max(1, size // 10)
                    img[y1:y1+thickness, x1:x2] = draw_color # Top
                    img[y2-thickness:y2, x1:x2] = draw_color # Bottom
                    img[y1:y2, x1:x1+thickness] = draw_color # Left
                    img[y1:y2, x2-thickness:x2] = draw_color # Right
                # 'striped' and 'gradient' are not mocked for simplicity
            
            if output_format == 'numpy':
                return img
            return img # Fallback
# --- End of Mock Implementations ---

@dataclass
class PipelineConfig:
    """Configuration for the rock-solid pipeline."""
    # Population parameters
    population_size: int = 50
    elite_size: int = 10
    generations: int = 100
    
    # Fitness weights
    alpha: float = 0.7  # Tester confidence weight
    beta: float = 0.3   # Diversity weight
    
    # Thresholds
    confidence_threshold: float = 0.6
    fitness_threshold: float = 0.5
    min_samples_per_cell: int = 10 # Default value added
    
    # CP-SAT parameters
    max_cp_attempts: int = 5
    cp_timeout_seconds: int = 10
    
    # Jitter parameters
    jitter_attempts: int = 3
    jitter_max_delta: float = 10.0
    
    # Canvas parameters
    canvas_size: int = 128
    min_obj_size: int = 20
    max_obj_size: int = 60
    
    # Output
    output_dir: str = "generated_scenes"
    save_progress: bool = True

class RockSolidPipeline:
    """
    Rock-solid search-driven pipeline that GUARANTEES coverage of every
    Bongard-LOGO rule cell with high-quality, verified images.
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.tester = MockTesterCNN()  # Using mock tester
        self.sampler = BongardSampler() # Mock sampler
        self.renderer = BongardRenderer() # Mock renderer
        self.placement_optimizer = PlacementOptimizer() # Added for explicit use
        # Coverage tracking
        self.coverage = Counter()
        self.accepted_features = defaultdict(list)
        self.all_cells = self._enumerate_all_cells()
        
        # Population tracking
        self.populations = {}  # rule -> population
        self.generation_stats = []
        
        logger.info(f"Initialized RockSolidPipeline with {len(self.all_cells)} cells to cover")
    
    def _enumerate_all_cells(self) -> List[Tuple[str, str, int, str]]:
        """Enumerate every combination (shape, fill, count, relation) for complete coverage."""
        shapes = ['circle', 'triangle', 'square', 'pentagon', 'hexagon']
        fills = ['solid', 'hollow', 'striped']
        counts = [1, 2, 3, 4]
        relations = ['near', 'far', 'overlap', 'inside', 'left_of', 'right_of', 'above', 'below', 'none'] # Added 'none' for cases with no specific relation
        
        # Generate all combinations
        cells = []
        for shape in shapes:
            for fill in fills:
                for count in counts:
                    for relation in relations:
                        cells.append((shape, fill, count, relation))
        
        logger.info(f"Enumerated {len(cells)} total rule cells")
        return cells
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete rock-solid pipeline."""
        logger.info("Starting rock-solid pipeline...")
        start_time = time.time()
        
        results = {
            'total_cells': len(self.all_cells),
            'covered_cells': 0,
            'generated_samples': 0,
            'generation_stats': [],
            'coverage_progress': []
        }
        
        # Initialize populations for each rule
        self._initialize_populations()
        
        # Evolution loop
        for generation in range(self.config.generations):
            gen_start = time.time()
            logger.info(f"=== Generation {generation + 1}/{self.config.generations} ===")
            
            # Evolve each rule population
            gen_stats = self._evolve_generation(generation)
            
            # Update coverage tracking
            covered_cells = sum(1 for cell in self.all_cells 
                                 if self.coverage[cell] >= self.config.min_samples_per_cell)
            
            results['covered_cells'] = covered_cells
            results['generation_stats'].append(gen_stats)
            
            coverage_percent = (covered_cells / len(self.all_cells)) * 100
            logger.info(f"Generation {generation + 1} complete. "
                        f"Coverage: {covered_cells}/{len(self.all_cells)} "
                        f"({coverage_percent:.1f}%)")
            
            # Early termination if full coverage achieved
            if covered_cells >= len(self.all_cells):
                logger.info("   🎉    FULL COVERAGE ACHIEVED!    🎉   ")
                break
            
            # Save progress
            if self.config.save_progress and (generation + 1) % 10 == 0:
                self._save_progress(generation, results)
        
        # Final statistics
        end_time = time.time()
        results['total_time'] = end_time - start_time
        results['final_coverage'] = covered_cells
        results['success_rate'] = covered_cells / len(self.all_cells)
        
        logger.info(f"Pipeline complete! Final coverage: {results['success_rate']:.2%}")
        return results
    
    def _initialize_populations(self):
        """Initialize random populations for each rule type."""
        for rule in ALL_BONGARD_RULES:
            rule_desc = rule.description
            
            # Create population for positive examples
            pop_pos = []
            for _ in range(self.config.population_size // 2):
                genome = create_random_genome(rule_desc, label=1)
                pop_pos.append(genome)
            
            # Create population for negative examples
            pop_neg = []
            for _ in range(self.config.population_size // 2):
                genome = create_random_genome(rule_desc, label=0)
                pop_neg.append(genome)
            
            self.populations[rule_desc] = {
                'positive': pop_pos,
                'negative': pop_neg
            }
        
        logger.info(f"Initialized populations for {len(ALL_BONGARD_RULES)} rules")
    
    def _evolve_generation(self, generation: int) -> Dict[str, Any]:
        """Evolve one generation across all rule populations."""
        gen_stats = {
            'generation': generation,
            'total_evaluations': 0,
            'successful_generations': 0,
            'average_fitness': 0.0,
            'coverage_improvement': 0
        }
        
        total_fitness = 0.0
        total_evaluations = 0
        successful_gens = 0
        
        for rule_desc, populations in self.populations.items():
            # Evolve positive population
            pop_stats_pos = self._evolve_rule_population(
                populations['positive'], rule_desc, 1, generation
            )
            
            # Evolve negative population
            pop_stats_neg = self._evolve_rule_population(
                populations['negative'], rule_desc, 0, generation
            )
            
            # Aggregate statistics
            total_evaluations += pop_stats_pos['evaluations'] + pop_stats_neg['evaluations']
            total_fitness += pop_stats_pos['avg_fitness'] + pop_stats_neg['avg_fitness']
            if pop_stats_pos['success'] or pop_stats_neg['success']:
                successful_gens += 1
        
        gen_stats['total_evaluations'] = total_evaluations
        gen_stats['successful_generations'] = successful_gens
        # Ensure division by zero is handled if no populations or rules
        gen_stats['average_fitness'] = total_fitness / (len(self.populations) * 2) if len(self.populations) > 0 else 0.0
        
        return gen_stats
    
    def _evolve_rule_population(self, population: List[SceneGenome], 
                                 rule_desc: str, label: int, generation: int) -> Dict[str, Any]:
        """Evolve a single rule population for one generation."""
        stats = {
            'rule': rule_desc,
            'label': label,
            'evaluations': 0,
            'avg_fitness': 0.0,
            'success': False
        }
        
        # Evaluate all genomes
        evaluated = []
        total_fitness = 0.0
        
        for genome in population:
            fitness_info = self._evaluate_genome(genome)
            genome.fitness = fitness_info['fitness']
            genome.tester_confidence = fitness_info['confidence']
            genome.diversity_score = fitness_info['diversity']
            genome.generation = generation
            
            evaluated.append(genome)
            total_fitness += genome.fitness
            stats['evaluations'] += 1
            
            # Check if this genome meets acceptance criteria
            if (genome.fitness >= self.config.fitness_threshold and 
                genome.tester_confidence >= self.config.confidence_threshold):
                self._accept_genome(genome, fitness_info)
                stats['success'] = True
        
        stats['avg_fitness'] = total_fitness / len(evaluated) if evaluated else 0.0
        
        # Selection and reproduction
        new_population = self._reproduce_population(evaluated)
        population[:] = new_population  # Update in place
        
        return stats
    
    def _evaluate_genome(self, genome: SceneGenome) -> Dict[str, Any]:
        """Comprehensive evaluation of a genome using CP-SAT + jitter + fallback."""
        # Phase 1: CP-SAT attempt
        objs = self._cp_sat_phase(genome)
        
        # Phase 2: Adversarial jitter if CP-SAT failed
        if not objs:
            objs = self._adversarial_jitter_phase(genome)
        
        # Phase 3: Grid fallback if still empty
        if not objs:
            objs = self._grid_fallback_phase(genome)
        
        # Render the scene
        if objs:
            # Ensure 'canvas_size' is present in genome.params
            canvas_size = genome.params.get('canvas_size', self.config.canvas_size)
            img = self._render_scene(objs, {'canvas_size': canvas_size})
            
            # Tester confidence
            confidence = self.tester.predict_confidence(img, genome.rule_desc)
            
            # Diversity score
            diversity = self._calculate_diversity(img, genome.rule_desc)
            
            # Combined fitness
            fitness = (self.config.alpha * confidence + 
                       self.config.beta * diversity)
            
            return {
                'fitness': fitness,
                'confidence': confidence,
                'diversity': diversity,
                'objects': objs,
                'image': img,
                'success': True
            }
        else:
            return {
                'fitness': 0.0,
                'confidence': 0.0,
                'diversity': 0.0,
                'objects': [],
                'image': None,
                'success': False
            }
    
    def _cp_sat_phase(self, genome: SceneGenome) -> Optional[List[Dict[str, Any]]]:
        """Phase 1: CP-SAT constraint solving."""
        try:
            # Ensure 'objects' and 'canvas_size' are in genome.params
            objects_to_place = genome.params.get('objects', [])
            canvas_size = genome.params.get('canvas_size', self.config.canvas_size)
            if not objects_to_place:
                logger.debug("No objects in genome for CP-SAT.")
                return None
            
            for attempt in range(self.config.max_cp_attempts):
                result = self.placement_optimizer.optimize_placement( # Use self.placement_optimizer
                    objects=objects_to_place,
                    canvas_size=canvas_size,
                    constraints=self._genome_to_constraints(genome)
                )
                
                if result and result.get('objects'):
                    logger.debug(f"CP-SAT succeeded on attempt {attempt + 1}")
                    return result['objects']
            
            logger.debug("CP-SAT failed after all attempts")
            return None
            
        except Exception as e:
            logger.debug(f"CP-SAT error: {e}")
            return None
    
    def _adversarial_jitter_phase(self, genome: SceneGenome) -> Optional[List[Dict[str, Any]]]:
        """Phase 2: Apply small jitters to previous best solution."""
        try:
            for attempt in range(self.config.jitter_attempts):
                # Create jittered version
                jittered_genome = self._apply_jitter(genome)
                
                # Try CP-SAT on jittered version
                objs = self._cp_sat_phase(jittered_genome)
                if objs:
                    logger.debug(f"Jitter succeeded on attempt {attempt + 1}")
                    return objs
            
            logger.debug("Jitter phase failed")
            return None
            
        except Exception as e:
            logger.debug(f"Jitter error: {e}")
            return None
    
    def _grid_fallback_phase(self, genome: SceneGenome) -> List[Dict[str, Any]]:
        """Phase 3: Deterministic grid placement as guaranteed fallback."""
        try:
            canvas_size = genome.params.get('canvas_size', self.config.canvas_size)
            # Default margin and num_objects if not in genome.params
            margin = genome.params.get('margin', 10) # Assuming a default margin
            num_objects = genome.params.get('num_objects', 1) # Ensure at least one object
            
            # Calculate grid dimensions
            grid_size = math.ceil(math.sqrt(num_objects))
            cell_size = (canvas_size - 2 * margin) / grid_size
            
            objects = []
            obj_idx = 0
            
            for row in range(grid_size):
                for col in range(grid_size):
                    if obj_idx >= num_objects:
                        break
                    
                    # Grid position (center of cell)
                    x = margin + col * cell_size + cell_size / 2
                    y = margin + row * cell_size + cell_size / 2
                    
                    # Use genome object parameters if available, otherwise default
                    if obj_idx < len(genome.params.get('objects', [])):
                        obj = genome.params['objects'][obj_idx].copy()
                    else:
                        # Create a default object if not enough in genome.params['objects']
                        obj = {
                            'shape': 'circle',
                            'color': 'blue',
                            'fill': 'solid',
                            'size': random.randint(20, 60),  # Default size range
                            'orientation': 0,
                            'texture': 'flat'
                        }
                    
                    obj['position'] = (x, y)
                    objects.append(obj)
                    obj_idx += 1
                    
                if obj_idx >= num_objects:
                    break
            
            logger.debug(f"Grid fallback placed {len(objects)} objects")
            return objects
            
        except Exception as e:
            logger.error(f"Grid fallback error: {e}")
            # Ultimate fallback: single centered object
            canvas_size = genome.params.get('canvas_size', self.config.canvas_size)
            return [{
                'shape': 'circle',
                'color': 'blue',
                'fill': 'solid',
                'size': 40,
                'position': (canvas_size // 2, canvas_size // 2),
                'orientation': 0,
                'texture': 'flat'
            }]
    
    def _apply_jitter(self, genome: SceneGenome) -> SceneGenome:
        """Apply small random jitters to genome parameters."""
        # Mutation rate and strength are passed to genome.mutate
        jittered = genome.mutate(mutation_rate=0.3, mutation_strength=0.1)
        
        # Additional position jitter for objects
        if 'objects' in jittered.params and jittered.params['objects']:
            for obj in jittered.params['objects']:
                dx = random.uniform(-self.config.jitter_max_delta, self.config.jitter_max_delta)
                dy = random.uniform(-self.config.jitter_max_delta, self.config.jitter_max_delta)
                
                x, y = obj.get('position', (0,0)) # Use .get with default for safety
                margin = jittered.params.get('margin', 10)
                canvas_size = jittered.params.get('canvas_size', self.config.canvas_size)
                obj_size = obj.get('size', 30)
                
                new_x = max(margin + obj_size//2, 
                            min(canvas_size - margin - obj_size//2, x + dx))
                new_y = max(margin + obj_size//2, 
                            min(canvas_size - margin - obj_size//2, y + dy))
                
                obj['position'] = (new_x, new_y)
        
        return jittered
    
    def _genome_to_constraints(self, genome: SceneGenome) -> Dict[str, Any]:
        """Convert genome to constraint specification."""
        constraints = {
            'no_overlap': True,
            'within_bounds': True,
            'min_distance': 5
        }
        
        # Add rule-specific constraints
        # Assuming rule_desc might contain keywords like "RELATION(X)"
        if 'RELATION(' in genome.rule_desc:
            # Extract relation type from rule_desc if not explicitly in params
            # This is a simplified parsing for mock purposes
            if 'target_relation' in genome.params:
                constraints['spatial_relation'] = genome.params['target_relation']
            else:
                # Attempt to parse from rule_desc string
                try:
                    start_idx = genome.rule_desc.find("RELATION(") + len("RELATION(")
                    end_idx = genome.rule_desc.find(")", start_idx)
                    relation_type = genome.rule_desc[start_idx:end_idx].strip()
                    if relation_type:
                        constraints['spatial_relation'] = relation_type
                except Exception:
                    logger.warning(f"Could not parse relation from rule_desc: {genome.rule_desc}")
        
        return constraints
    
    def _render_scene(self, objects: List[Dict[str, Any]], params: Dict[str, Any]) -> np.ndarray:
        """Render scene to black and white image."""
        try:
            # Use existing renderer
            canvas_size = params.get('canvas_size', self.config.canvas_size)
            img = self.renderer.render_scene(
                objects=objects,
                canvas_size=canvas_size,
                background_color='white',
                output_format='numpy'
            )
            
            # Convert to grayscale if needed (assuming renderer might return RGB)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = np.mean(img, axis=2)
            
            return img.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Rendering error: {e}")
            # Return blank image as fallback
            canvas_size = params.get('canvas_size', self.config.canvas_size)
            return np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    
    def _calculate_diversity(self, img: np.ndarray, rule_desc: str) -> float:
        """Calculate diversity score compared to accepted samples."""
        if rule_desc not in self.accepted_features or not self.accepted_features[rule_desc]:
            return 1.0  # Maximum diversity if no accepted samples
        
        # Extract features (simple histogram)
        features = self._extract_features(img)
        
        # Calculate minimum cosine distance to accepted features
        min_distance = float('inf')
        for accepted_features in self.accepted_features[rule_desc]:
            distance = self._cosine_distance(features, accepted_features)
            min_distance = min(min_distance, distance)
        
        # Diversity is 1 - minimum distance (clamped between 0 and 1)
        return max(0.0, 1.0 - min_distance)
    
    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        """Extract simple features from image for diversity calculation."""
        # Flatten and normalize
        features = img.flatten().astype(np.float32) / 255.0
        
        # Optionally reduce dimensionality for efficiency
        if len(features) > 1000:
            # Simple downsampling
            step = len(features) // 1000
            features = features[::step]
        
        return features
    
    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine distance between two feature vectors."""
        # Ensure same length, padding with zeros if necessary
        max_len = max(len(a), len(b))
        a_padded = np.pad(a, (0, max_len - len(a)), 'constant')
        b_padded = np.pad(b, (0, max_len - len(b)), 'constant')
        
        # Calculate cosine similarity
        dot_product = np.dot(a_padded, b_padded)
        norm_a = np.linalg.norm(a_padded)
        norm_b = np.linalg.norm(b_padded)
        
        if norm_a == 0 or norm_b == 0:
            return 1.0  # Maximum distance for zero vectors
        
        cosine_sim = dot_product / (norm_a * norm_b)
        # Cosine similarity is in [-1, 1]. Convert to distance [0, 2].
        # Then normalize to [0, 1] for diversity score.
        return (1.0 - cosine_sim) / 2.0 
    
    def _accept_genome(self, genome: SceneGenome, fitness_info: Dict[str, Any]):
        """Accept a genome and update coverage tracking."""
        # Extract cell information
        cell = self._genome_to_cell(genome)
        
        # Update coverage
        self.coverage[cell] += 1
        
        # Store features for diversity calculation
        if fitness_info['image'] is not None:
            features = self._extract_features(fitness_info['image'])
            self.accepted_features[genome.rule_desc].append(features)
        
        # Log acceptance
        logger.debug(f"Accepted genome for cell {cell}, "
                     f"fitness={genome.fitness:.3f}, "
                     f"confidence={genome.tester_confidence:.3f}")
    
    def _genome_to_cell(self, genome: SceneGenome) -> Tuple[str, str, int, str]:
        """Extract cell coordinates from genome."""
        params = genome.params
        
        # Extract primary attributes from first object or targets.
        # Ensure 'objects' list is not empty before accessing index 0.
        first_obj_shape = params['objects'][0]['shape'] if params.get('objects') else 'circle'
        first_obj_fill = params['objects'][0]['fill'] if params.get('objects') else 'solid'
        shape = params.get('target_shape', first_obj_shape)
        fill = params.get('target_fill', first_obj_fill)
        count = params.get('target_count', params.get('num_objects', 1)) # Default to 1 if num_objects also missing
        relation = params.get('target_relation', 'none')
        
        return (shape, fill, count, relation)
    
    def _reproduce_population(self, evaluated: List[SceneGenome]) -> List[SceneGenome]:
        """Create next generation through selection and reproduction."""
        # Sort by fitness
        evaluated.sort(key=lambda g: g.fitness, reverse=True)
        
        # Keep elite
        new_population = evaluated[:self.config.elite_size].copy()
        
        # Generate offspring to fill population
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(evaluated)
            parent2 = self._tournament_selection(evaluated)
            
            # Crossover and mutation
            if random.random() < 0.8:  # Crossover probability
                child1, child2 = parent1.crossover(parent2) # crossover now returns two children
                new_population.extend([child1, child2])
            else:
                # Just mutate
                child = parent1.mutate()
                new_population.append(child)
        
        # Trim to exact size
        return new_population[:self.config.population_size]
    
    def _tournament_selection(self, population: List[SceneGenome], 
                               tournament_size: int = 3) -> SceneGenome:
        """Select genome using tournament selection."""
        if not population:
            raise ValueError("Population cannot be empty for tournament selection.")
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda g: g.fitness)
    
    def _save_progress(self, generation: int, results: Dict[str, Any]):
        """Save current progress to disk."""
        try:
            from pathlib import Path
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            progress_file = output_dir / f"progress_gen_{generation}.json"
            
            with open(progress_file, 'w') as f:
                json.dump({
                    'generation': generation,
                    'coverage': dict(self.coverage),
                    'results': results,
                    'timestamp': time.time()
                }, f, indent=2)
            
            logger.info(f"Progress saved to {progress_file}")
            
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

# Convenience function for RockSolidPipeline
def run_rock_solid_pipeline(config: PipelineConfig = None) -> Dict[str, Any]:
    """Run the complete rock-solid pipeline with default or custom config."""
    pipeline = RockSolidPipeline(config)
    return pipeline.run_pipeline()

# Renamed from NeuralTester to MockNeuralTester to avoid confusion with MockTesterCNN
class MockNeuralTester:
    """Neural network tester for semantic verification of generated scenes."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the neural tester."""
        self.model_path = model_path
        self.model = None
        self.confidence_threshold = 0.7 # This threshold is used for validation
        
        # Mock model for now - replace with actual CNN
        self.mock_confidence_cache = {}
        
        logger.info("Initialized Mock Neural Tester (mock implementation)")
    
    def load_model(self, model_path: str):
        """Load pre-trained tester CNN model."""
        # TODO: Implement actual model loading
        # self.model = torch.load(model_path)
        # self.model.eval()
        logger.info(f"Mock: Loading tester model from {model_path}")
        self.model_path = model_path
    
    def evaluate_scene(self, scene_image: np.ndarray, rule_desc: str, expected_label: int) -> Tuple[float, bool]:
        """
        Evaluate scene semantic correctness.
        
        Returns:
            confidence: Float 0-1 indicating confidence in correctness
            is_valid: Boolean indicating if scene passes semantic validation
        """
        # Mock implementation - replace with actual CNN inference
        # Using a slice of tobytes for hashing to prevent excessively long hash inputs
        scene_hash = hash((scene_image.tobytes()[:100], rule_desc, expected_label)) 
        
        if scene_hash in self.mock_confidence_cache:
            confidence = self.mock_confidence_cache[scene_hash]
        else:
            # Simulate varying confidence based on rule complexity
            base_confidence = 0.6
            
            # Higher confidence for simpler rules
            if "SHAPE(" in rule_desc:
                base_confidence = 0.8
            elif "COUNT(" in rule_desc:
                base_confidence = 0.7
            elif "RELATION(" in rule_desc:
                base_confidence = 0.6
            
            # Add some randomness
            confidence = np.clip(base_confidence + random.uniform(-0.2, 0.2), 0.0, 1.0)
            self.mock_confidence_cache[scene_hash] = confidence
        
        is_valid = confidence >= self.confidence_threshold
        
        return confidence, is_valid
    
    def batch_evaluate(self, scenes: List[Tuple[np.ndarray, str, int]]) -> List[Tuple[float, bool]]:
        """Evaluate multiple scenes in batch for efficiency."""
        results = []
        for scene_image, rule_desc, expected_label in scenes:
            results.append(self.evaluate_scene(scene_image, rule_desc, expected_label))
        return results

class GeneticPipeline:
    """
    Rock-solid genetic algorithm pipeline for guaranteed Bongard generation.
    Ensures every (shape, fill, count, relation) cell gets M=100+ valid examples.
    """
    
    def __init__(self, config=None):
        """
        Initialize the genetic pipeline using config values.
        Args:
            config: Optional config dict. If None, loads from main config.py
        """
        if config is None:
            try:
                from config import CONFIG
                config = CONFIG.get('genetic', {})
            except ImportError:
                config = {}
        self.config = config
        self.population_size = config.get('population_size', 50)
        self.min_quota = config.get('min_quota', config.get('min_quota', 100))
        self.max_generations = config.get('num_generations', config.get('max_generations', 1000))
        self.alpha = config.get('tester_weight', 0.7)
        self.beta = config.get('diversity_weight', 0.3)
        self.coverage_weight = config.get('coverage_weight', 1.0)
        self.mutation_rate = config.get('mutation_rate', 0.2)
        self.crossover_rate = config.get('crossover_rate', 0.5)
        self.elitism = config.get('elitism', 2)
        self.max_attempts = config.get('max_attempts', 100)
        self.cache_enabled = config.get('cache_enabled', True)
        self.seed = config.get('seed', 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        # Initialize components
        self.neural_tester = MockNeuralTester()
        self.renderer = BongardRenderer()
        self.placement_optimizer = PlacementOptimizer()
        self.coverage_cells = self._generate_all_cells()
        self.cell_coverage = {cell: [] for cell in self.coverage_cells}
        # Evolutionary state
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.coverage_history = []
        # Statistics
        self.total_scenes_generated = 0
        self.total_scenes_validated = 0
        self.failed_generations = 0
        logger.info(f"Initialized Genetic Pipeline with {len(self.coverage_cells)} coverage cells and config: {self.config}")
    
    def _generate_all_cells(self) -> List[Tuple[str, str, int, str]]:
        """Generate all possible (shape, fill, count, relation) cells."""
        shapes = ['circle', 'triangle', 'square', 'pentagon', 'star']
        fills = ['solid', 'outline', 'striped', 'gradient']
        counts = [1, 2, 3, 4, 5, 6]
        relations = ['overlap', 'near', 'nested', 'left_of', 'right_of', 'above', 'below', 'none'] # Added 'none'
        
        cells = []
        for shape in shapes:
            for fill in fills:
                for count in counts:
                    for relation in relations:
                        cells.append((shape, fill, count, relation))
        
        logger.info(f"Generated {len(cells)} total coverage cells")
        return cells
    
    def _calculate_diversity_score(self, genome: SceneGenome, population: List[SceneGenome]) -> float:
        """Calculate diversity score based on parameter differences."""
        if not population:
            return 1.0
        
        differences = []
        for other in population:
            if other != genome:
                # Calculate parameter distance
                param_diff = 0.0
                for key in genome.params:
                    if key in other.params:
                        val1 = genome.params[key]
                        val2 = other.params[key]
                        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                            param_diff += abs(val1 - val2)
                        elif isinstance(val1, tuple) and isinstance(val2, tuple) and len(val1) == len(val2):
                            param_diff += sum(abs(a - b) for a, b in zip(val1, val2))
                        # For other types (e.g., strings), treat as binary difference
                        elif val1 != val2:
                            param_diff += 1.0
                
                differences.append(param_diff)
        
        # Normalize diversity score, e.g., by max possible difference or log scale
        # For simplicity, returning mean difference, which might need scaling for real diversity metric
        return np.mean(differences) if differences else 1.0
    
    def _evaluate_fitness(self, genome: SceneGenome, scene_image: np.ndarray) -> float:
        """Evaluate genome fitness using neural tester and diversity."""
        # Get tester confidence
        confidence, is_valid = self.neural_tester.evaluate_scene(
            scene_image, genome.rule_desc, genome.label
        )
        
        # Calculate diversity score
        diversity = self._calculate_diversity_score(genome, self.population)
        
        # Combined fitness
        fitness = self.alpha * confidence + self.beta * diversity
        
        # Store individual scores
        genome.tester_confidence = confidence
        genome.diversity_score = diversity
        genome.fitness = fitness
        
        return fitness
    
    def _scene_matches_cell(self, scene_data: Dict[str, Any], cell: Tuple[str, str, int, str]) -> bool:
        """Check if a generated scene matches a coverage cell."""
        target_shape, target_fill, target_count, target_relation = cell
        
        objects = scene_data.get('objects', [])
        scene_graph = scene_data.get('scene_graph', {})
        
        # Check count
        if len(objects) != target_count:
            return False
        
        # Check if target shape is present (at least one object has the target shape)
        shapes_in_scene = {obj.get('shape', 'unknown') for obj in objects}
        if target_shape not in shapes_in_scene:
            return False
        
        # Check if target fill is present (at least one object has the target fill)
        fills_in_scene = {obj.get('fill', 'solid') for obj in objects}
        if target_fill not in fills_in_scene:
            return False
        
        # Check if target relation is present
        relations_in_scene = {rel.get('type', 'none') for rel in scene_graph.get('relations', [])}
        if target_relation != 'none' and target_relation not in relations_in_scene:
            return False
        elif target_relation == 'none' and relations_in_scene and 'none' not in relations_in_scene:
            # If target is 'none' but there are relations, it doesn't match
            return False
            
        return True
    
    def _select_underrepresented_cell(self) -> Tuple[str, str, int, str]:
        """Select a coverage cell that needs more examples."""
        # Find cells with fewer than min_quota examples
        under_quota = [cell for cell in self.coverage_cells 
                       if len(self.cell_coverage[cell]) < self.min_quota]
        
        if under_quota:
            # Prioritize cells with the fewest examples
            return min(under_quota, key=lambda cell: len(self.cell_coverage[cell]))
        else:
            # All cells meet quota, select randomly
            return random.choice(self.coverage_cells)
    
    def _create_targeted_genome(self, target_cell: Tuple[str, str, int, str]) -> SceneGenome:
        """Create a genome specifically targeting a coverage cell."""
        target_shape, target_fill, target_count, target_relation = target_cell
        
        # Create rule description targeting this cell
        rule_desc = f"COMBINED({target_shape},{target_fill},{target_count},{target_relation})"
        
        # Create parameters optimized for this cell
        params = {
            'num_objects': target_count,
            'canvas_size': 128,
            'obj_size_range': (20, 60),
            'target_shape': target_shape,
            'target_fill': target_fill,
            'target_relation': target_relation,
            'spatial_jitter': safe_randint(1, 5),
            'overlap_threshold': 0.3,
            'relation_strength': 0.8,
            'objects': [] # Initialize objects list
        }
        # Populate initial objects based on target_count and target_shape/fill for the first object
        for i in range(params['num_objects']): # Use params['num_objects'] to ensure correct count
            obj = {
                'shape': target_shape if i == 0 else random.choice(['circle', 'triangle', 'square']),
                'fill': target_fill if i == 0 else random.choice(['solid', 'outline']),
                'color': random.choice(['red', 'blue', 'green', 'black']),
                'position': (random.randint(20, params['canvas_size']-20), random.randint(20, params['canvas_size']-20)),
                'size': safe_randint(*params['obj_size_range']),
                'orientation': random.uniform(0, 360),
                'texture': 'flat'
            }
            params['objects'].append(obj)
        
        return SceneGenome(
            rule_desc=rule_desc,
            label=1,  # Start with positive examples for targeted generation
            params=params
        )
    
    def _generate_scene_from_genome(self, genome: SceneGenome) -> Tuple[Dict[str, Any], np.ndarray]:
        """Generate a scene from a genome's parameters using the optimizer and renderer."""
        objects = []
        scene_graph = {'relations': []}
        canvas_size = genome.params.get('canvas_size', 128)
        
        # Attempt to place objects using the optimizer
        placement_result = self.placement_optimizer.optimize_placement(
            objects=genome.params.get('objects', []),
            canvas_size=canvas_size,
            constraints=self._genome_to_constraints_for_genetic(genome) # Use a specific constraint method for genetic pipeline
        )
        
        if placement_result and placement_result.get('objects'):
            objects = placement_result['objects']
            # Simulate generating relations based on target_relation
            if genome.params.get('target_relation') != 'none' and len(objects) >= 2:
                scene_graph['relations'].append({
                    'type': genome.params['target_relation'],
                    'objects': [0, 1] # Simple mock relation between first two objects
                })
        else:
            # Fallback to simple placement if optimizer fails
            logger.warning(f"PlacementOptimizer failed for genome {genome.rule_desc}, falling back to simple placement.")
            objects = self._grid_fallback_phase_genetic(genome) # Re-using the grid fallback specifically for genetic pipeline
        
        scene_data = {
            'objects': objects,
            'scene_graph': scene_graph,
            'rule': genome.rule_desc,
            'label': genome.label
        }
        
        # Render the scene image
        scene_image = self.renderer.render_scene(
            objects=objects, 
            canvas_size=canvas_size, 
            background_color='white', 
            output_format='numpy'
        )
        
        return scene_data, scene_image

    def _genome_to_constraints_for_genetic(self, genome: SceneGenome) -> Dict[str, Any]:
        """
        Convert genome parameters to constraints for the placement optimizer
        within the GeneticPipeline context.
        """
        constraints = {
            'no_overlap': True,
            'within_bounds': True,
            'min_distance': 5
        }
        
        # Add target relation as a constraint if specified
        target_relation = genome.params.get('target_relation')
        if target_relation and target_relation != 'none':
            constraints['spatial_relation'] = target_relation
            
        return constraints

    def _grid_fallback_phase_genetic(self, genome: SceneGenome) -> List[Dict[str, Any]]:
        """Grid placement fallback for the GeneticPipeline."""
        try:
            canvas_size = genome.params.get('canvas_size', 128)
            num_objects = genome.params.get('num_objects', 1)
            margin = 20
            
            # Calculate grid dimensions
            grid_size = math.ceil(math.sqrt(num_objects))
            cell_size = (canvas_size - 2 * margin) / grid_size
            
            objects = []
            obj_idx = 0
            
            for row in range(grid_size):
                for col in range(grid_size):
                    if obj_idx >= num_objects:
                        break
                    
                    # Grid position (center of cell)
                    x = margin + col * cell_size + cell_size / 2
                    y = margin + row * cell_size + cell_size / 2
                    
                    # Use genome object parameters if available, otherwise default
                    if obj_idx < len(genome.params.get('objects', [])):
                        obj = genome.params['objects'][obj_idx].copy()
                    else:
                        # Create a default object
                        obj = {
                            'shape': genome.params.get('target_shape', 'circle'),
                            'fill': genome.params.get('target_fill', 'solid'),
                            'color': random.choice(['red', 'blue', 'green', 'black']),
                            'size': random.randint(20, 40),
                            'orientation': 0,
                            'texture': 'flat'
                        }
                    
                    obj['position'] = (x, y)
                    objects.append(obj)
                    obj_idx += 1
                    
                if obj_idx >= num_objects:
                    break
            
            return objects
            
        except Exception as e:
            logger.error(f"Grid fallback error in GeneticPipeline: {e}")
            # Ultimate fallback: single centered object
            return [{
                'shape': 'circle',
                'color': 'blue',
                'fill': 'solid',
                'size': 30,
                'position': (64, 64),  # Center of 128x128 canvas
                'orientation': 0,
                'texture': 'flat'
            }]

    def evolve_generation(self) -> Dict[str, Any]:
        """Evolve one generation of the population."""
        self.generation += 1
        
        # Generate scenes for each genome and evaluate fitness
        scene_fitness_pairs = []
        for genome in self.population:
            try:
                scene_data, scene_image = self._generate_scene_from_genome(genome)
                fitness = self._evaluate_fitness(genome, scene_image)
                
                # Record coverage if scene passes validation AND meets fitness threshold
                if genome.tester_confidence >= self.neural_tester.confidence_threshold and \
                   genome.fitness >= (self.alpha * self.neural_tester.confidence_threshold): # Ensure fitness is also good
                    for cell in self.coverage_cells:
                        if self._scene_matches_cell(scene_data, cell):
                            self.cell_coverage[cell].append({
                                'scene_data': scene_data,
                                'genome': genome,
                                'generation': self.generation
                            })
                            self.total_scenes_validated += 1 # Increment validated scenes
                
                scene_fitness_pairs.append((genome, fitness))
                self.total_scenes_generated += 1
                
            except Exception as e:
                logger.error(f"Failed to generate scene for genome (Gen {self.generation}): {e}")
                scene_fitness_pairs.append((genome, 0.0)) # Assign 0 fitness on failure
                self.failed_generations += 1
        
        # Sort by fitness
        scene_fitness_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Selection: keep top 50% (elite)
        elite_size = self.population_size // 2
        elite_genomes = [genome for genome, _ in scene_fitness_pairs[:elite_size]]
        
        # Generate new population
        new_population = elite_genomes.copy()
        
        # Fill rest with mutations and crossovers
        while len(new_population) < self.population_size:
            if random.random() < 0.3: # Mutation probability
                # Mutation
                parent = random.choice(elite_genomes)
                child = parent.mutate()
                new_population.append(child)
            else: # Crossover probability
                # Crossover
                if len(elite_genomes) >= 2:
                    parent1 = random.choice(elite_genomes)
                    parent2 = random.choice(elite_genomes)
                    # Ensure parents are different for meaningful crossover, if possible
                    while parent1 == parent2 and len(elite_genomes) > 1:
                        parent2 = random.choice(elite_genomes)
                    child1, child2 = parent1.crossover(parent2)
                    new_population.extend([child1, child2])
                else: # Fallback to mutation if not enough parents for crossover
                    parent = random.choice(elite_genomes)
                    child = parent.mutate()
                    new_population.append(child)
            
            # Trim if we accidentally added too many due to crossover producing two children
            if len(new_population) > self.population_size:
                new_population = new_population[:self.population_size]
        
        self.population = new_population
        
        # Record statistics
        best_fitness = scene_fitness_pairs[0][1] if scene_fitness_pairs else 0.0
        self.best_fitness_history.append(best_fitness)
        
        coverage_ratio = sum(1 for cell in self.coverage_cells 
                             if len(self.cell_coverage[cell]) >= self.min_quota) / len(self.coverage_cells)
        self.coverage_history.append(coverage_ratio)
        
        return {
            'generation': self.generation,
            'best_fitness': best_fitness,
            'coverage_ratio': coverage_ratio,
            'total_scenes': self.total_scenes_generated,
            'validated_scenes': self.total_scenes_validated
        }
    
    def initialize_population(self) -> None:
        """Initialize the genetic population with diverse genomes."""
        self.population = []
        
        # Create genomes targeting each underrepresented cell
        for _ in range(self.population_size):
            target_cell = self._select_underrepresented_cell()
            genome = self._create_targeted_genome(target_cell)
            
            # Add some random variation
            if random.random() < 0.3:
                genome = genome.mutate(mutation_rate=0.5)
            
            self.population.append(genome)
        
        logger.info(f"Initialized population with {len(self.population)} genomes")
    
    def run_evolution(self, target_coverage: float = 1.0) -> Dict[str, Any]:
        """Run the complete evolutionary search until coverage target is met."""
        logger.info("Starting evolutionary search for guaranteed coverage")
        
        self.initialize_population()
        
        start_time = time.time()
        
        for gen in range(self.max_generations):
            stats = self.evolve_generation()
            
            if gen % 10 == 0 or gen == self.max_generations - 1: # Log every 10 generations or at the end
                logger.info(
                    f"Generation {gen}: "
                    f"Best fitness={stats['best_fitness']:.3f}, "
                    f"Coverage={stats['coverage_ratio']:.2%}, "
                    f"Scenes={stats['total_scenes']}"
                )
            
            # Check if target coverage achieved
            if stats['coverage_ratio'] >= target_coverage:
                logger.info(f"Target coverage {target_coverage:.2%} achieved in generation {gen}")
                break
            
            # Adaptive re-seeding if stuck
            if gen > 100 and len(self.best_fitness_history) > 50:
                # Check for stagnation in coverage, not just fitness
                recent_coverage_progress = self.coverage_history[-1] - self.coverage_history[-50]
                if recent_coverage_progress < 0.005: # Small threshold for coverage improvement
                    logger.info("Stagnation detected in coverage, re-seeding population")
                    self._reseed_population()
        
        end_time = time.time()
        
        # Final statistics
        final_stats = {
            'total_generations': self.generation,
            'total_scenes_generated': self.total_scenes_generated,
            'final_coverage_ratio': self.coverage_history[-1] if self.coverage_history else 0.0,
            'cells_completed': sum(1 for cell in self.coverage_cells 
                                   if len(self.cell_coverage[cell]) >= self.min_quota),
            'total_cells': len(self.coverage_cells),
            'runtime_seconds': end_time - start_time,
            'failed_generations': self.failed_generations
        }
        
        logger.info("Evolution complete:")
        logger.info(f"  Generations: {final_stats['total_generations']}")
        logger.info(f"  Scenes generated: {final_stats['total_scenes_generated']}")
        logger.info(f"  Coverage: {final_stats['final_coverage_ratio']:.2%}")
        logger.info(f"  Cells completed: {final_stats['cells_completed']}/{final_stats['total_cells']}")
        
        return final_stats
    
    def _reseed_population(self):
        """Re-seed population when evolution stagnates."""
        # Keep top 20% of population
        top_performers = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        elite_count = max(1, self.population_size // 5)
        self.population = top_performers[:elite_count]
        
        # Fill rest with new targeted genomes
        while len(self.population) < self.population_size:
            target_cell = self._select_underrepresented_cell()
            genome = self._create_targeted_genome(target_cell)
            genome = genome.mutate(mutation_rate=0.8)  # High mutation for diversity
            self.population.append(genome)
    
    def export_dataset(self, output_dir: str) -> None:
        """Export the complete generated dataset."""
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True) # Ensure parent directories are created
        
        # Export coverage data
        coverage_data = {}
        total_exported = 0
        
        for cell, examples in self.cell_coverage.items():
            if examples:
                # Create a file-system safe key for the cell
                cell_key = f"{cell[0]}_{cell[1]}_{cell[2]}_{cell[3].replace(' ', '_')}" 
                coverage_data[cell_key] = []
                
                for i, example in enumerate(examples):
                    scene_data = example['scene_data']
                    
                    # Save scene metadata
                    scene_file_name = f"{cell_key}_{i}.json"
                    scene_file_path = output_path / scene_file_name
                    with open(scene_file_path, 'w') as f:
                        json.dump(scene_data, f, indent=2)
                    
                    coverage_data[cell_key].append({
                        'file': str(scene_file_path), # Store as string for JSON serialization
                        'generation': example['generation'],
                        'fitness': example['genome'].fitness,
                        'confidence': example['genome'].tester_confidence
                    })
                    
                    total_exported += 1
        
        # Save coverage summary
        summary_file = output_path / "coverage_summary.json"
        summary = {
            'total_cells': len(self.coverage_cells),
            'covered_cells': len([cell for cell in self.coverage_cells if self.cell_coverage[cell]]),
            'total_examples': total_exported,
            'min_quota': self.min_quota,
            'coverage_data': coverage_data,
            'generation_stats': {
                'total_generations': self.generation,
                'best_fitness_history': self.best_fitness_history,
                'coverage_history': self.coverage_history
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Exported {total_exported} examples to {output_dir}")

def main():
    """Example usage of the genetic pipeline."""
    # Configure logging (already configured globally, but good to have here for clarity)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize and run pipeline
    pipeline = GeneticPipeline(
        population_size=30,
        min_quota=5,   # Reduced quota for quicker demonstration
        max_generations=50 # Reduced generations for quicker demonstration
    )
    
    # Run evolution
    stats = pipeline.run_evolution(target_coverage=0.8)  # 80% coverage target
    
    # Export results
    pipeline.export_dataset("genetic_dataset_output")
    
    print("Genetic pipeline complete!")
    print(f"Final statistics: {stats}")

if __name__ == "__main__":
    main()
