import numpy as np
from src.utils.system1_abstraction import extract_com, extract_inertia_tensor

def test_extract_com_square():
    mask = np.zeros((10,10),bool)
    mask[2:5,2:5] = True
    x,y = extract_com(mask)
    assert 3.4 < x < 3.6
    assert 3.4 < y < 3.6

def test_inertia_tensor_symmetry():
    mask = np.zeros((10,10),bool)
    mask[2:5,2:5] = True
    I = extract_inertia_tensor(mask)
    assert I.shape == (2,2)
    assert abs(I[0,1] - I[1,0]) < 1e-6
