import random

def rotate(cmds):
    angle = random.choice([15, 30, 45, 60])
    return [("rotate", angle)] + cmds

def scale(cmds):
    factor = random.uniform(0.8, 1.2)
    return [("scale", factor)] + cmds

def shear(cmds):
    shear_val = random.uniform(-0.3, 0.3)
    return [("shear", shear_val)] + cmds

def translate(cmds):
    dx, dy = random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)
    return [("translate", (dx, dy))] + cmds

affine_transforms = {
    "rotate": rotate,
    "scale": scale,
    "shear": shear,
    "translate": translate,
}
