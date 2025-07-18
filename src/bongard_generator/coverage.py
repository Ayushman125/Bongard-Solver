"""Coverage tracking and validation for comprehensive Bongard problem generation"""

import logging
import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Set
import numpy as np

from .config_loader import CONFIG, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_FILL_MAP, RELATION_MAP

logger = logging.getLogger(__name__)

class CoverageTracker:
    """Tracks coverage of different Bongard problem categories and ensures comprehensive sampling."""
    
    def __init__(self):
        self.all_shapes = list(ATTRIBUTE_SHAPE_MAP.keys())
        self.all_fills = list(ATTRIBUTE_FILL_MAP.keys())
        self.all_relations = list(RELATION_MAP.keys())
        self.max_objects = CONFIG['data']['synthetic_data_config']['max_objects_per_image']
        
        # Define all possible cells for coverage
        self.ALL_CELLS = list(itertools.product(
            self.all_shapes,
            self.all_fills,
            list(range(1, self.max_objects + 1)),
            self.all_relations
        ))
        
        # Coverage counters
        self.coverage = Counter()
        self.rule_coverage = Counter()
        self.scene_coverage = Counter()
        
        # Statistics
        self.total_scenes_generated = 0
        self.validation_failures = 0
        
        logger.info(f"Initialized coverage tracker with {len(self.ALL_CELLS)} possible cells")

    def record_scene(self, objs: List[Dict[str, Any]], scene_graph: Dict[str, Any], rule_description: str, label: int) -> None:
        """Record a generated scene for coverage tracking."""
        self.total_scenes_generated += 1
        
        # Extract scene properties
        shapes = [obj.get('shape', 'unknown') for obj in objs]
        fills = [obj.get('fill', 'solid') for obj in objs]
        count = len(objs)
        relations = [rel.get('type', 'none') for rel in scene_graph.get('relations', [])]
        
        # Record rule coverage
        rule_key = (rule_description, label)
        self.rule_coverage[rule_key] += 1
        
        # Record cell coverage for each applicable cell
        for cell in self.ALL_CELLS:
            shape, fill, cell_count, relation = cell
            
            # Check if this scene matches this cell
            matches_cell = self._scene_matches_cell(shapes, fills, count, relations, cell)
            if matches_cell:
                self.coverage[cell] += 1
                
        # Record scene-level statistics
        scene_signature = (tuple(sorted(shapes)), tuple(sorted(fills)), count, tuple(sorted(relations)))
        self.scene_coverage[scene_signature] += 1

    def _scene_matches_cell(self, shapes: List[str], fills: List[str], count: int, relations: List[str], cell: Tuple) -> bool:
        """Check if a scene matches a specific coverage cell."""
        cell_shape, cell_fill, cell_count, cell_relation = cell
        
        # Check count match
        if count != cell_count:
            return False
            
        # Check if target shape is present (for shape rules)
        if cell_shape != 'none' and cell_shape not in shapes:
            return False
            
        # Check if target fill is present (for fill rules)
        if cell_fill != 'none' and cell_fill not in fills:
            return False
            
        # Check if target relation is present (for relation rules)
        if cell_relation != 'none' and cell_relation not in relations:
            return False
            
        return True

    def is_covered(self, cell: Tuple, min_quota: int = 1) -> bool:
        """Check if a cell has met the minimum coverage quota."""
        return self.coverage[cell] >= min_quota

    def is_generation_complete(self, min_quota: int = 10) -> bool:
        """Check if generation is complete (all cells meet quota)."""
        return all(self.coverage[cell] >= min_quota for cell in self.ALL_CELLS)
    
    def get_under_covered_cells(self, min_quota: int = 10) -> List[Tuple]:
        """Get cells that haven't met the minimum quota."""
        return [cell for cell in self.ALL_CELLS if self.coverage[cell] < min_quota]
    
    def get_coverage_heatmap_data(self) -> Dict[str, Any]:
        """Get data for plotting coverage heatmap."""
        # Create a matrix for heatmap visualization
        shape_fill_matrix = {}
        for cell in self.ALL_CELLS:
            shape, fill, count, relation = cell
            key = f"{shape}_{fill}"
            if key not in shape_fill_matrix:
                shape_fill_matrix[key] = 0
            shape_fill_matrix[key] += self.coverage[cell]
        
        return {
            'matrix': shape_fill_matrix,
            'total_cells': len(self.ALL_CELLS),
            'covered_cells': len([cell for cell in self.ALL_CELLS if self.coverage[cell] > 0])
        }
    
    def should_halt_generation(self, target_quota: int = 50) -> bool:
        """Determine if generation should halt based on coverage."""
        under_covered = self.get_under_covered_cells(target_quota)
        if not under_covered:
            logger.info(f"All {len(self.ALL_CELLS)} cells have met quota of {target_quota}")
            return True
        
        # Check if we're making progress
        progress_ratio = len([cell for cell in self.ALL_CELLS if self.coverage[cell] > 0]) / len(self.ALL_CELLS)
        if progress_ratio < 0.1 and self.total_scenes_generated > 1000:
            logger.warning(f"Low coverage progress: {progress_ratio:.2%} after {self.total_scenes_generated} scenes")
            return True
            
        return False
        min_quota = 1
        under_covered = [cell for cell in self.ALL_CELLS if not self.is_covered(cell, min_quota)]
        
        # Get rule distribution
        rule_stats = dict(self.rule_coverage.most_common(10))
        
        # Get most/least covered cells
        most_covered = self.coverage.most_common(5)
        least_covered = [(cell, count) for cell, count in self.coverage.items() if count > 0]
        least_covered = sorted(least_covered, key=lambda x: x[1])[:5]
        
        return {
            'total_scenes': self.total_scenes_generated,
            'total_cells': total_cells,
            'covered_cells': covered_cells,
            'coverage_percentage': coverage_percentage,
            'under_covered_count': len(under_covered),
            'under_covered_cells': under_covered[:10],  # Show first 10
            'validation_failures': self.validation_failures,
            'rule_distribution': rule_stats,
            'most_covered_cells': most_covered,
            'least_covered_cells': least_covered,
            'unique_scene_signatures': len(self.scene_coverage)
        }

    def get_priority_cells(self, n: int = 10) -> List[Tuple]:
        """Get the n cells that need the most attention (least covered)."""
        # Sort cells by coverage count (ascending)
        sorted_cells = sorted(self.ALL_CELLS, key=lambda cell: self.coverage[cell])
        return sorted_cells[:n]

    def force_inject_cell(self, cell: Tuple) -> Dict[str, Any]:
        """Generate parameters to force injection of a specific cell."""
        shape, fill, count, relation = cell
        
        return {
            'target_shape': shape if shape != 'none' else None,
            'target_fill': fill if fill != 'none' else None,
            'target_count': count,
            'target_relation': relation if relation != 'none' else None,
            'force_mode': True
        }

    def validate_scene_semantics(self, objs: List[Dict[str, Any]], rule_description: str, expected_label: int) -> bool:
        """Validate that a scene semantically matches its rule and label."""
        try:
            # Extract rule type and value
            if "SHAPE(" in rule_description:
                target_shape = rule_description.split("(")[1].split(")")[0].lower()
                shapes = [obj.get('shape', 'unknown') for obj in objs]
                
                if expected_label == 1:  # Positive example
                    if target_shape not in shapes:
                        self.validation_failures += 1
                        logger.warning(f"Validation failed: Expected shape '{target_shape}' not found in positive example")
                        return False
                else:  # Negative example
                    if all(shape == target_shape for shape in shapes):
                        self.validation_failures += 1
                        logger.warning(f"Validation failed: All shapes are '{target_shape}' in negative example")
                        return False
                        
            elif "COUNT(" in rule_description:
                target_count = int(rule_description.split("(")[1].split(")")[0])
                actual_count = len(objs)
                
                if expected_label == 1:  # Positive example
                    if actual_count != target_count:
                        self.validation_failures += 1
                        logger.warning(f"Validation failed: Expected count {target_count}, got {actual_count} in positive example")
                        return False
                else:  # Negative example
                    if actual_count == target_count:
                        self.validation_failures += 1
                        logger.warning(f"Validation failed: Count is {target_count} in negative example")
                        return False
                        
            elif "FILL(" in rule_description:
                target_fill = rule_description.split("(")[1].split(")")[0].lower()
                fills = [obj.get('fill', 'solid') for obj in objs]
                
                if expected_label == 1:  # Positive example
                    if target_fill not in fills:
                        self.validation_failures += 1
                        logger.warning(f"Validation failed: Expected fill '{target_fill}' not found in positive example")
                        return False
                else:  # Negative example
                    if all(fill == target_fill for fill in fills):
                        self.validation_failures += 1
                        logger.warning(f"Validation failed: All fills are '{target_fill}' in negative example")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in semantic validation: {e}")
            self.validation_failures += 1
            return False

    def print_coverage_report(self) -> None:
        """Print a detailed coverage report."""
        stats = self.get_coverage_stats()
        
        print("\n" + "="*60)
        print("BONGARD GENERATOR COVERAGE REPORT")
        print("="*60)
        
        print(f"Total scenes generated: {stats['total_scenes']}")
        print(f"Total coverage cells: {stats['total_cells']}")
        print(f"Covered cells: {stats['covered_cells']}")
        print(f"Coverage percentage: {stats['coverage_percentage']:.2f}%")
        print(f"Under-covered cells: {stats['under_covered_count']}")
        print(f"Validation failures: {stats['validation_failures']}")
        print(f"Unique scene signatures: {stats['unique_scene_signatures']}")
        
        print(f"\nTop rules generated:")
        for rule, count in list(stats['rule_distribution'].items())[:5]:
            print(f"  {rule}: {count}")
            
        print(f"\nMost covered cells:")
        for cell, count in stats['most_covered_cells']:
            print(f"  {cell}: {count}")
            
        print(f"\nLeast covered cells:")
        for cell, count in stats['least_covered_cells']:
            print(f"  {cell}: {count}")
            
        if stats['under_covered_count'] > 0:
            print(f"\nSample under-covered cells:")
            for cell in stats['under_covered_cells']:
                print(f"  {cell}")
        
        print("="*60)

class AdversarialSampler:
    """Generates adversarial and edge-case examples for robust training."""
    
    def __init__(self, img_size: int = 128):
        self.img_size = img_size
        
    def adversarial_scene(self, target_relation: str, n_objs: int = 2) -> List[Dict[str, Any]]:
        """Generate adversarial scenes that sit on category boundaries."""
        if target_relation == "overlap":
            return self._near_overlap_boundary(n_objs)
        elif target_relation == "nested":
            return self._near_nested_boundary(n_objs)
        elif target_relation == "left_of":
            return self._near_spatial_boundary(n_objs, "left_of")
        elif target_relation == "above":
            return self._near_spatial_boundary(n_objs, "above")
        else:
            return self._random_adversarial(n_objs)
    
    def _near_overlap_boundary(self, n_objs: int) -> List[Dict[str, Any]]:
        """Generate objects that are just barely overlapping or not overlapping."""
        center = self.img_size / 2
        size = 30
        
        # 49% vs 51% overlap
        overlap_type = np.random.choice(["barely_overlap", "barely_not_overlap"])
        
        if overlap_type == "barely_overlap":
            offset = size * 0.49  # Just barely overlapping
        else:
            offset = size * 1.01  # Just barely not overlapping
            
        objs = [
            {
                "position": (center - offset/2, center),
                "width_pixels": size,
                "height_pixels": size
            },
            {
                "position": (center + offset/2, center),
                "width_pixels": size,
                "height_pixels": size
            }
        ]
        
        # Add random objects for the rest
        for _ in range(n_objs - 2):
            objs.append({
                "position": (np.random.uniform(20, self.img_size-20), 
                           np.random.uniform(20, self.img_size-20)),
                "width_pixels": np.random.randint(20, 40),
                "height_pixels": np.random.randint(20, 40)
            })
            
        return objs
    
    def _near_nested_boundary(self, n_objs: int) -> List[Dict[str, Any]]:
        """Generate objects with tiny margins for nested vs not nested."""
        center = self.img_size / 2
        
        outer_size = 60
        # Inner object either just fits or just doesn't fit
        nested_type = np.random.choice(["just_nested", "just_not_nested"])
        
        if nested_type == "just_nested":
            inner_size = outer_size - 4  # Just fits with tiny margin
        else:
            inner_size = outer_size + 2  # Just doesn't fit
            
        objs = [
            {
                "position": (center, center),
                "width_pixels": outer_size,
                "height_pixels": outer_size
            },
            {
                "position": (center + np.random.uniform(-2, 2), center + np.random.uniform(-2, 2)),
                "width_pixels": inner_size,
                "height_pixels": inner_size
            }
        ]
        
        # Add random objects for the rest
        for _ in range(n_objs - 2):
            objs.append({
                "position": (np.random.uniform(20, self.img_size-20), 
                           np.random.uniform(20, self.img_size-20)),
                "width_pixels": np.random.randint(20, 40),
                "height_pixels": np.random.randint(20, 40)
            })
            
        return objs
    
    def _near_spatial_boundary(self, n_objs: int, relation: str) -> List[Dict[str, Any]]:
        """Generate objects near spatial relationship boundaries."""
        size = 30
        
        if relation == "left_of":
            # Objects with exactly half-pixel offsets
            x1 = self.img_size * 0.4
            x2 = x1 + size + np.random.uniform(-1, 1)  # Just touching or just separated
            y = self.img_size / 2
            
            objs = [
                {"position": (x1, y), "width_pixels": size, "height_pixels": size},
                {"position": (x2, y), "width_pixels": size, "height_pixels": size}
            ]
        else:  # above
            # Similar for vertical relationship
            y1 = self.img_size * 0.4
            y2 = y1 + size + np.random.uniform(-1, 1)
            x = self.img_size / 2
            
            objs = [
                {"position": (x, y1), "width_pixels": size, "height_pixels": size},
                {"position": (x, y2), "width_pixels": size, "height_pixels": size}
            ]
        
        # Add random objects for the rest
        for _ in range(n_objs - 2):
            objs.append({
                "position": (np.random.uniform(20, self.img_size-20), 
                           np.random.uniform(20, self.img_size-20)),
                "width_pixels": np.random.randint(20, 40),
                "height_pixels": np.random.randint(20, 40)
            })
            
        return objs
    
    def _random_adversarial(self, n_objs: int) -> List[Dict[str, Any]]:
        """Generate random adversarial configuration."""
        objs = []
        for _ in range(n_objs):
            # Extreme jitter and unusual positions
            objs.append({
                "position": (np.random.uniform(5, self.img_size-5), 
                           np.random.uniform(5, self.img_size-5)),
                "width_pixels": np.random.choice([10, 15, 70, 80]),  # Very small or very large
                "height_pixels": np.random.choice([10, 15, 70, 80])
            })
        return objs
