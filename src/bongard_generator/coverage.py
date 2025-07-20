"""
Unified coverage tracking for comprehensive Bongard-LOGO style dataset generation.
This module provides a single, backward-compatible EnhancedCoverageTracker,
driven by a central configuration.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

from .config import GeneratorConfig

logger = logging.getLogger(__name__)

class EnhancedCoverageTracker:
    """
    Unified and enhanced coverage tracker. This is the single source of truth for coverage.
    It is backward-compatible with older systems that might expect a simpler interface.
    """
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.coverage_goals = self.config.coverage_goals
        
        # Define cells based on config or defaults
        self.cells = self._generate_coverage_cells()
        self.coverage = defaultdict(int)
        
        # For legacy ALL_CELLS compatibility
        self.ALL_CELLS = [cell for cell in self.cells]

        self.priority_cells = set(self.coverage_goals.get("priority_cells", []))
        
        self.total_recorded = 0
        logger.info(f"Unified Coverage tracker initialized with {len(self.cells)} cells.")
        logger.info(f"High priority cells: {len(self.priority_cells)}")

    def _generate_coverage_cells(self) -> List[Tuple]:
        """Generate comprehensive coverage cells for dataset validation from config."""
        # This would be dynamically built based on self.config in a real implementation
        # For now, returning a fixed set for simplicity.
        return [
            ('shape', 'circle'), ('shape', 'square'), ('shape', 'triangle'),
            ('count', 1), ('count', 2), ('count', 3),
            ('relation', 'left_of'), ('relation', 'above'),
            ('color', 'red'), ('color', 'blue')
        ]

    def record(self, features):
        """Record the features of a generated example."""
        # Handle list of objects by extracting features from each object
        if isinstance(features, list):
            for obj in features:
                if isinstance(obj, dict):
                    # Extract key features from each object
                    obj_features = {}
                    if 'shape' in obj:
                        obj_features['shape'] = obj['shape']
                    if 'color' in obj:
                        obj_features['color'] = obj['color']
                    if 'size' in obj:
                        obj_features['size'] = obj['size']
                    if 'fill' in obj:
                        obj_features['fill'] = obj['fill']
                    # Record features for this object
                    if obj_features:
                        self.record(obj_features)
            return
        
        # Original logic: features is a dict
        if isinstance(features, dict):
            for key, value in features.items():
                cell = (key, value)
                if cell in self.cells:
                    self.coverage[cell] += 1
            self.total_recorded += 1

    def get_under_covered_cells(self, min_quota: Optional[int] = None) -> List[Tuple]:
        """
        Return under-covered cells.
        The logic is now driven by quotas defined in the main config.
        """
        default_quota = self.coverage_goals.get("default_quota", 10)
        priority_quota = self.coverage_goals.get("priority_quota", 20)

        # Support legacy calls that might pass min_quota
        if min_quota is not None:
            default_quota = min_quota

        under_covered = []
        for cell in self.cells:
            quota = priority_quota if cell in self.priority_cells else default_quota
            if self.coverage.get(cell, 0) < quota:
                under_covered.append(cell)
        return under_covered

    def get_coverage_state(self) -> Dict[str, Any]:
        """Return the current coverage state."""
        return {
            "total_recorded": self.total_recorded,
            "total_cells": len(self.cells),
            "covered_cells": sum(1 for count in self.coverage.values() if count > 0),
            "coverage_counts": {str(k): v for k, v in self.coverage.items()} # Convert tuple keys to strings for serialization
        }

# For backward compatibility, legacy systems can still import CoverageTracker
CoverageTracker = EnhancedCoverageTracker
