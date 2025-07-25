from noise import pnoise2
import random

def perlin_jitter(cmds):
    return [(cmd, param + random.uniform(-0.05, 0.05) if isinstance(param, (int, float)) else param) for cmd, param in cmds]

def subdiv_jitter(cmds):
    out = []
    for cmd, param in cmds:
        out.append((cmd, param))
        out.append((cmd, param))
    return out

def wave_distort(cmds):
    return [(cmd, param + random.uniform(-0.05,0.05) if isinstance(param, (int, float)) else param) for cmd, param in cmds]

def radial_perturb(cmds):
    return [(cmd, (param[0]*1.1, param[1]*0.9)) if isinstance(param, tuple) and len(param)==2 else (cmd,param) for cmd,param in cmds]

def noise_scale(cmds):
    factor = 1 + pnoise2(random.random(), random.random()) * 0.1
    return [("scale", factor)] + cmds
