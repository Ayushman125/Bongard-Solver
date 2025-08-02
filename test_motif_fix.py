from src.scene_graphs_building.feature_extraction import compute_physics_attributes
import logging
logging.basicConfig(level=logging.INFO)

# Test case: motif node with vertices (simulating what MotifMiner creates)
test_motif = {
    'id': 'motif_0',
    'object_id': 'motif_0', 
    'object_type': 'motif', 
    'vertices': [[0.5, 0.5], [1.0, 1.0], [1.5, 0.5], [1.0, 0.0]],
    'is_closed': False,
    'is_motif': True
}

print('=== Testing motif centroid computation fix ===')
print('Before compute_physics_attributes:')
print(f'  object_type: {test_motif.get("object_type")}')
print(f'  vertices: {len(test_motif.get("vertices", []))} points')
print(f'  is_closed: {test_motif.get("is_closed")}')

compute_physics_attributes(test_motif)

print('\nAfter compute_physics_attributes:')
print(f'  geometry_valid: {test_motif.get("geometry_valid")}')
print(f'  fallback_geometry: {test_motif.get("fallback_geometry")}')
print(f'  centroid: {test_motif.get("centroid")}')
print(f'  area: {test_motif.get("area")}')
print(f'  perimeter: {test_motif.get("perimeter")}')

# Expected result: geometry_valid=True, fallback_geometry=False, 
# centroid computed properly from vertices, no "fallback centroid" log message
