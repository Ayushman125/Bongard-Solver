"""
Hybrid sampler combining CP-SAT constraint solving with genetic algorithms
for coverage-driven diverse dataset generation.
"""

import logging
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import copy
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

from .cp_sampler import CPSATSampler
from .coverage import CoverageTracker, CoverageDimensions, CoverageCell
from .models import build_scene_graph, create_scene_gnn


@dataclass
class Individual:
    """Represents an individual in the genetic algorithm."""
    objects: List[Dict]
    fitness: float = 0.0
    coverage_score: float = 0.0
    constraint_score: float = 0.0
    diversity_score: float = 0.0
    metadata: Dict = field(default_factory=dict)


class HybridSampler:
    """
    Hybrid sampler that combines:
    1. CP-SAT for constraint satisfaction and valid object placement
    2. Genetic algorithms for coverage-driven optimization
    3. Domain randomization for realistic variation
    """
    
    def __init__(self, 
                 canvas_size: Tuple[int, int] = (200, 200),
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elite_ratio: float = 0.2,
                 coverage_weight: float = 0.4,
                 constraint_weight: float = 0.4,
                 diversity_weight: float = 0.2,
                 use_gnn: bool = False,
                 gnn_checkpoint: str = None,
                 gnn_threshold: float = 0.5,
                 **kwargs):
        
        self.canvas_size = canvas_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.use_gnn = use_gnn
        self.gnn_threshold = gnn_threshold
        
        # Fitness weights
        self.coverage_weight = coverage_weight
        self.constraint_weight = constraint_weight
        self.diversity_weight = diversity_weight
        
        # Initialize GNN if enabled
        self._gnn_model = None
        self._gnn_device = 'cpu'
        if use_gnn:
            self._initialize_gnn(gnn_checkpoint)
        
        # Initialize CP-SAT sampler for constraint handling
        try:
            from .cp_sampler import SceneParameters
            scene_params = SceneParameters(
                canvas_size=canvas_size,
                min_obj_size=kwargs.get('min_obj_size', 20),
                max_obj_size=kwargs.get('max_obj_size', 60),
                max_objects=kwargs.get('max_objects', 6),
                colors=['red', 'blue', 'green', 'yellow', 'black'],
                shapes=['circle', 'square', 'triangle'],
                fills=['solid', 'outline', 'striped']
            )
            self.cp_sampler = CPSATSampler(scene_params)
        except Exception as e:
            logger.warning(f"Could not initialize CP-SAT sampler: {e}")
            self.cp_sampler = None
        
        # Coverage tracker for diversity metrics
        self.coverage_tracker = CoverageTracker()
        
        # Population and evolution state
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        
        # Statistics
        self.fitness_history = []
        self.coverage_history = []
        self.gnn_filtered_count = 0
    
    def _initialize_gnn(self, checkpoint_path: str = None):
        """Initialize GNN model for scene filtering."""
        try:
            import torch
            self._gnn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Create a simple config object for GNN
            class Config:
                def __init__(self, canvas_size):
                    self.canvas_size = max(canvas_size) if isinstance(canvas_size, tuple) else canvas_size
                    self.img_size = self.canvas_size
                    self.gnn_radius = 0.3
                    self.gnn_hidden = 64
                    self.gnn_layers = 2
                    self.gnn_dropout = 0.1
                    self.gnn_attention = False
            
            config = Config(self.canvas_size)
            self._gnn_model = create_scene_gnn(config)
            
            if self._gnn_model is None:
                logger.warning("Could not create GNN model, disabling GNN filtering")
                self.use_gnn = False
                return
            
            # Load checkpoint if provided
            if checkpoint_path and Path(checkpoint_path).exists():
                self._gnn_model.load_state_dict(torch.load(checkpoint_path, map_location=self._gnn_device))
                logger.info(f"Loaded GNN checkpoint from {checkpoint_path}")
            else:
                logger.warning("No GNN checkpoint found, using randomly initialized model")
            
            self._gnn_model.to(self._gnn_device)
            self._gnn_model.eval()
            
        except ImportError:
            logger.warning("PyTorch not available, disabling GNN filtering")
            self.use_gnn = False
        except Exception as e:
            logger.error(f"Failed to initialize GNN: {e}")
            self.use_gnn = False
    
    def _gnn_filter_objects(self, objects: List[Dict]) -> bool:
        """Filter scene using GNN quality assessment."""
        if not self.use_gnn or self._gnn_model is None:
            return True
        
        try:
            import torch
            # Create simple config for graph building
            class Config:
                def __init__(self, canvas_size):
                    self.canvas_size = max(canvas_size) if isinstance(canvas_size, tuple) else canvas_size
                    self.gnn_radius = 0.3
            
            config = Config(self.canvas_size)
            scene_graph = build_scene_graph(objects, config)
            
            if scene_graph is None:
                return True
            
            scene_graph = scene_graph.to(self._gnn_device)
            
            with torch.no_grad():
                quality_score = self._gnn_model(scene_graph).item()
            
            if quality_score < self.gnn_threshold:
                self.gnn_filtered_count += 1
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"GNN filtering failed: {e}")
            return True
        
    def initialize_population(self) -> None:
        """Initialize population with diverse individuals using CP-SAT."""
        self.population.clear()
        
        attempts = 0
        max_attempts = self.population_size * 3  # Allow more attempts with GNN filtering
        
        while len(self.population) < self.population_size and attempts < max_attempts:
            attempts += 1
            
            # Use CP-SAT to generate valid object configurations
            try:
                objects = self.cp_sampler.sample(
                    num_objects=random.randint(3, 8),
                    object_types=['circle', 'rectangle', 'triangle', 'ellipse', 'polygon']
                )
                
                if not objects:
                    # Fallback to random generation if CP-SAT fails
                    objects = self._generate_random_objects()
                
                # Apply GNN filtering if enabled
                if self.use_gnn and not self._gnn_filter_objects(objects):
                    continue  # Skip this configuration, GNN rejected it
                
                individual = Individual(objects=objects)
                self.population.append(individual)
                    
            except Exception as e:
                logger.warning(f"CP-SAT failed, using random generation: {e}")
                objects = self._generate_random_objects()
                
                # Apply GNN filtering to random objects too
                if self.use_gnn and not self._gnn_filter_objects(objects):
                    continue
                
                individual = Individual(objects=objects)
                self.population.append(individual)
        
        # If we couldn't generate enough individuals due to GNN filtering, fill with random ones
        while len(self.population) < self.population_size:
            logger.warning("GNN filtering too restrictive, adding random individuals")
            objects = self._generate_random_objects()
            individual = Individual(objects=objects)
            self.population.append(individual)
        
        # Evaluate initial population
        self._evaluate_population()
        
        if self.use_gnn:
            logger.info(f"GNN filtered {self.gnn_filtered_count} configurations during initialization")
        
    def _generate_random_objects(self, num_objects: Optional[int] = None) -> List[Dict]:
        """Generate random objects as fallback."""
        if num_objects is None:
            num_objects = random.randint(3, 8)
            
        objects = []
        object_types = ['circle', 'rectangle', 'triangle', 'ellipse']
        
        for _ in range(num_objects):
            obj = {
                'type': random.choice(object_types),
                'x': random.randint(20, self.canvas_size[0] - 20),
                'y': random.randint(20, self.canvas_size[1] - 20),
                'size': random.randint(15, 40),
                'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                'stroke_width': random.randint(1, 4),
                'fill_pattern': random.choice(['solid', 'striped', 'dotted']),
                'rotation': random.uniform(0, 360)
            }
            objects.append(obj)
            
        return objects
    
    def _evaluate_population(self) -> None:
        """Evaluate fitness for all individuals in population."""
        for individual in self.population:
            individual.fitness = self._calculate_fitness(individual)
            
        # Update best individual
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(self.population[0])
    
    def _calculate_fitness(self, individual: Individual) -> float:
        """Calculate comprehensive fitness score for an individual."""
        # Coverage score: How well does this fill coverage gaps?
        coverage_score = self._calculate_coverage_score(individual)
        
        # Constraint score: How well does this satisfy spatial constraints?
        constraint_score = self._calculate_constraint_score(individual)
        
        # Diversity score: How different is this from existing population?
        diversity_score = self._calculate_diversity_score(individual)
        
        # Store component scores
        individual.coverage_score = coverage_score
        individual.constraint_score = constraint_score
        individual.diversity_score = diversity_score
        
        # Weighted combination
        fitness = (self.coverage_weight * coverage_score +
                  self.constraint_weight * constraint_score +
                  self.diversity_weight * diversity_score)
        
        return fitness
    
    def _calculate_coverage_score(self, individual: Individual) -> float:
        """Calculate how well individual fills coverage gaps."""
        try:
            # Convert individual to coverage dimensions
            dims = self._extract_coverage_dimensions(individual)
            
            # Get priority cells (underrepresented combinations)
            priority_cells = self.coverage_tracker.get_priority_cells(threshold=0.1)
            
            if not priority_cells:
                return 0.5  # Neutral score if no priority gaps
            
            # Score based on matching priority patterns
            matches = 0
            for cell in priority_cells:
                if self._matches_coverage_cell(dims, cell):
                    matches += 1
            
            return min(matches / len(priority_cells), 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_constraint_score(self, individual: Individual) -> float:
        """Calculate constraint satisfaction score."""
        score = 1.0
        
        # Check object overlaps
        objects = individual.objects
        overlap_penalty = 0
        
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                if self._objects_overlap(objects[i], objects[j]):
                    overlap_penalty += 0.1
        
        score -= min(overlap_penalty, 0.8)
        
        # Check boundary violations
        boundary_penalty = 0
        for obj in objects:
            if (obj['x'] - obj['size']//2 < 0 or 
                obj['x'] + obj['size']//2 > self.canvas_size[0] or
                obj['y'] - obj['size']//2 < 0 or 
                obj['y'] + obj['size']//2 > self.canvas_size[1]):
                boundary_penalty += 0.1
        
        score -= min(boundary_penalty, 0.5)
        
        return max(score, 0.0)
    
    def _calculate_diversity_score(self, individual: Individual) -> float:
        """Calculate diversity relative to population."""
        if len(self.population) <= 1:
            return 1.0
        
        # Calculate average distance to other individuals
        total_distance = 0
        count = 0
        
        for other in self.population:
            if other is individual:
                continue
            distance = self._individual_distance(individual, other)
            total_distance += distance
            count += 1
        
        if count == 0:
            return 1.0
        
        avg_distance = total_distance / count
        return min(avg_distance, 1.0)
    
    def _individual_distance(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate distance between two individuals."""
        # Simple distance based on object differences
        distance = 0.0
        
        # Object count difference
        distance += abs(len(ind1.objects) - len(ind2.objects)) * 0.1
        
        # Object type distribution difference
        types1 = defaultdict(int)
        types2 = defaultdict(int)
        
        for obj in ind1.objects:
            types1[obj['type']] += 1
        for obj in ind2.objects:
            types2[obj['type']] += 1
        
        all_types = set(types1.keys()) | set(types2.keys())
        for obj_type in all_types:
            distance += abs(types1[obj_type] - types2[obj_type]) * 0.05
        
        return min(distance, 1.0)
    
    def _objects_overlap(self, obj1: Dict, obj2: Dict) -> bool:
        """Check if two objects overlap."""
        dx = abs(obj1['x'] - obj2['x'])
        dy = abs(obj1['y'] - obj2['y'])
        min_distance = (obj1['size'] + obj2['size']) // 2 + 5  # 5px separation
        
        return (dx * dx + dy * dy) < (min_distance * min_distance)
    
    def evolve(self) -> Individual:
        """Run genetic algorithm evolution."""
        self.initialize_population()
        
        for gen in range(self.generations):
            self.generation = gen
            
            # Selection
            parents = self._selection()
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self._crossover(parents[i], parents[i + 1])
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                offspring.extend([child1, child2])
            
            # Combine population with offspring
            combined = self.population + offspring
            
            # Evaluate all individuals
            for individual in combined:
                if individual.fitness == 0.0:  # Not evaluated yet
                    individual.fitness = self._calculate_fitness(individual)
            
            # Select next generation
            combined.sort(key=lambda x: x.fitness, reverse=True)
            self.population = combined[:self.population_size]
            
            # Update statistics
            best_fitness = self.population[0].fitness
            avg_coverage = np.mean([ind.coverage_score for ind in self.population])
            
            self.fitness_history.append(best_fitness)
            self.coverage_history.append(avg_coverage)
            
            # Update best individual
            if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
                self.best_individual = copy.deepcopy(self.population[0])
            
            if gen % 10 == 0:
                print(f"Generation {gen}: Best fitness = {best_fitness:.3f}, "
                      f"Avg coverage = {avg_coverage:.3f}")
        
        return self.best_individual
    
    def _selection(self) -> List[Individual]:
        """Tournament selection."""
        tournament_size = 3
        parents = []
        
        for _ in range(self.population_size):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Crossover between two individuals."""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # Object-level crossover
        objects1 = parent1.objects[:]
        objects2 = parent2.objects[:]
        
        # Mix objects from both parents
        min_len = min(len(objects1), len(objects2))
        crossover_point = random.randint(1, min_len - 1)
        
        new_objects1 = objects1[:crossover_point] + objects2[crossover_point:]
        new_objects2 = objects2[:crossover_point] + objects1[crossover_point:]
        
        child1 = Individual(objects=new_objects1)
        child2 = Individual(objects=new_objects2)
        
        return child1, child2
    
    def _mutate(self, individual: Individual) -> Individual:
        """Mutate an individual."""
        if random.random() > self.mutation_rate:
            return individual
        
        individual = copy.deepcopy(individual)
        objects = individual.objects
        
        mutation_type = random.choice(['add', 'remove', 'modify', 'swap'])
        
        if mutation_type == 'add' and len(objects) < 10:
            # Add new object
            new_obj = self._generate_random_objects(1)[0]
            objects.append(new_obj)
            
        elif mutation_type == 'remove' and len(objects) > 2:
            # Remove random object
            objects.pop(random.randint(0, len(objects) - 1))
            
        elif mutation_type == 'modify' and objects:
            # Modify random object
            obj_idx = random.randint(0, len(objects) - 1)
            obj = objects[obj_idx]
            
            # Randomly modify one attribute
            attr = random.choice(['x', 'y', 'size', 'color', 'rotation'])
            if attr in ['x', 'y']:
                obj[attr] += random.randint(-20, 20)
                obj[attr] = max(20, min(obj[attr], self.canvas_size[0] - 20))
            elif attr == 'size':
                obj[attr] += random.randint(-10, 10)
                obj[attr] = max(10, min(obj[attr], 60))
            elif attr == 'color':
                obj[attr] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            elif attr == 'rotation':
                obj[attr] += random.uniform(-30, 30)
                
        elif mutation_type == 'swap' and len(objects) >= 2:
            # Swap two objects
            idx1, idx2 = random.sample(range(len(objects)), 2)
            objects[idx1], objects[idx2] = objects[idx2], objects[idx1]
        
        individual.fitness = 0.0  # Mark for re-evaluation
        return individual
    
    def sample(self, **kwargs) -> List[Dict]:
        """Main sampling interface - returns best evolved solution."""
        best_individual = self.evolve()
        return best_individual.objects if best_individual else []
    
    def _extract_coverage_dimensions(self, individual: Individual) -> CoverageDimensions:
        """Extract coverage dimensions from individual for evaluation."""
        objects = individual.objects
        
        if not objects:
            return CoverageDimensions()
        
        # Extract features from objects
        shape_types = [obj['type'] for obj in objects]
        fill_patterns = [obj.get('fill_pattern', 'solid') for obj in objects]
        
        return CoverageDimensions(
            shape_count=len(objects),
            dominant_shape=max(set(shape_types), key=shape_types.count),
            fill_pattern=max(set(fill_patterns), key=fill_patterns.count),
            spatial_relation='scattered',  # Simplified for now
            stroke_pattern='normal'  # Simplified for now
        )
    
    def _matches_coverage_cell(self, dims: CoverageDimensions, cell: CoverageCell) -> bool:
        """Check if dimensions match a coverage cell."""
        return (dims.shape_count == cell.shape_count and
                dims.dominant_shape == cell.dominant_shape and
                dims.fill_pattern == cell.fill_pattern)
