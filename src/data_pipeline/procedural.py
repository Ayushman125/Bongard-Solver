# Small procedural/affine operators for hard negative mining
def rotate_15(cmds):
    return [("rotate", 15)] + cmds

def scale_90(cmds):
    return [("scale", (0.9, 1.1))] + cmds

def shear_0_2(cmds):
    return [("shear", 0.2)] + cmds

PROC_OPS = [rotate_15, scale_90, shear_0_2]
from perlin_noise import PerlinNoise
import random

def perlin_jitter(cmds):
    out = []
    for item in cmds:
        if isinstance(item, tuple) and len(item) == 2:
            cmd, param = item
            if isinstance(param, (int, float)):
                out.append((cmd, param + random.uniform(-0.05, 0.05)))
            else:
                out.append((cmd, param))
        else:
            out.append(item)
    return out

def subdiv_jitter(cmds):
    out = []
    for item in cmds:
        if isinstance(item, tuple) and len(item) == 2:
            cmd, param = item
            out.append((cmd, param))
            out.append((cmd, param))
        else:
            out.append(item)
    return out

def wave_distort(cmds):
    out = []
    for item in cmds:
        if isinstance(item, tuple) and len(item) == 2:
            cmd, param = item
            if isinstance(param, (int, float)):
                out.append((cmd, param + random.uniform(-0.05, 0.05)))
            else:
                out.append((cmd, param))
        else:
            out.append(item)
    return out

def radial_perturb(cmds):
    out = []
    for item in cmds:
        if isinstance(item, tuple) and len(item) == 2:
            cmd, param = item
            if isinstance(param, tuple) and len(param) == 2:
                out.append((cmd, (param[0]*1.1, param[1]*0.9)))
            else:
                out.append((cmd, param))
        else:
            out.append(item)
    return out

_noise = PerlinNoise()
def noise_scale(cmds):
    factor = 1 + _noise([random.random(), random.random()]) * 0.1
    return [("scale", factor)] + cmds
