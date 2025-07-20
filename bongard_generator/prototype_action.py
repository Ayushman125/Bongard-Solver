"""
Prototype action handler that can inject real prototype shapes into scenes.
"""
import os

class PrototypeAction:
    def __init__(self, prototypes_dir):
        self.prototypes_dir = prototypes_dir
        self.available_prototypes = []
        self.shapes = []  # Match the actual implementation
        
        # Check if prototypes directory exists
        if os.path.exists(prototypes_dir):
            self.available_prototypes = [f for f in os.listdir(prototypes_dir) 
                                       if f.endswith('.png') or f.endswith('.jpg')]
            # Create mock shapes list
            self.shapes = [{'name': f, 'path': os.path.join(prototypes_dir, f)} 
                          for f in self.available_prototypes]
        
        # If no shapes found, create some dummy shapes for testing
        if not self.shapes:
            self.shapes = [
                {'name': 'circle', 'shape_type': 'circle'},
                {'name': 'square', 'shape_type': 'square'},
                {'name': 'triangle', 'shape_type': 'triangle'}
            ]
            
        print(f"PrototypeAction initialized with {len(self.shapes)} prototypes.")

    def inject_random(self, rule):
        """
        Injects prototype shapes into a scene based on the rule.
        This is a placeholder implementation.
        """
        # In a real implementation, this would:
        # 1. Load actual prototype shapes from files
        # 2. Apply transformations based on the rule
        # 3. Return positioned objects with prototype shapes
        
        # Placeholder: return a simple scene
        return [
            {'shape': 'prototype_circle', 'color': 'black', 'size': 'medium', 'position': (75, 75)},
            {'shape': 'prototype_square', 'color': 'black', 'size': 'small', 'position': (175, 175)}
        ]
