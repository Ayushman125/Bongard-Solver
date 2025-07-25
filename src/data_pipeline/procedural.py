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
