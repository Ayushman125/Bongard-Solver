import numpy as np
from src.physics_inference import PhysicsInference

def test_physics_inference_centroid():
    from src.data_pipeline.physics_infer import PhysicsInference
    vertices = [(0,0), (1,0), (1,1), (0,1)]
    poly = PhysicsInference.polygon_from_vertices(vertices)
    centroid = PhysicsInference.centroid(poly)
    assert isinstance(centroid, tuple)

def test_extract_center_of_mass_batch():
    pi = PhysicsInference()
    masks = [np.ones((10,10)), np.zeros((10,10))]
    coms = pi.extract_center_of_mass_batch(masks)
    assert coms.shape == (2,2)

def test_compute_stability_batch():
    pi = PhysicsInference()
    masks = [np.ones((10,10)), np.zeros((10,10))]
    stabilities = pi.compute_stability_batch(masks)
    assert isinstance(stabilities, list)
    assert 'is_stable' in stabilities[0]
