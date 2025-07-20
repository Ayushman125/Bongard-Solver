# src/bongard_generator/shape_renderer.py
import math, random
from PIL import ImageDraw

def draw_shape(draw: ImageDraw.Draw, obj: dict, cfg):
    x, y, s = obj['x'], obj['y'], obj['size']
    shape = obj['shape']
    fill  = obj.get('fill', 'solid')
    stroke = obj.get('stroke_width', getattr(cfg, 'stroke_min', 1))
    angle = obj.get('rotation', random.uniform(0,360))
    pts = []

    # Reg polygons
    if shape == 'circle':
        bbox = [x-s/2, y-s/2, x+s/2, y+s/2]
        if fill == 'solid':
            draw.ellipse(bbox, fill='black')
        else:
            draw.ellipse(bbox, outline='black', width=stroke)
        return

    elif shape in ('square','triangle','pentagon','star'):
        n = {'square':4,'triangle':3,'pentagon':5,'star':10}[shape]
        base = []
        for i in range(n):
            r = (s/2) if (shape!='star' or i%2==0) else (s/4)
            theta = math.radians(90 + i*(360/n))
            base.append((x + r*math.cos(theta), y + r*math.sin(theta)))
        # rotate & jitter
        pts = []
        for px,py in base:
            dx,dy = px-x, py-y
            rdx = dx*math.cos(math.radians(angle)) - dy*math.sin(math.radians(angle))
            rdy = dx*math.sin(math.radians(angle)) + dy*math.cos(math.radians(angle))
            jittered = (x + rdx + random.uniform(-cfg.jitter_px,cfg.jitter_px),
                        y + rdy + random.uniform(-cfg.jitter_px,cfg.jitter_px))
            pts.append(jittered)
        # fill
        if fill=='solid':
            draw.polygon(pts, fill='black')
        else:
            draw.polygon(pts, outline='black', width=stroke)
        return

    # fallback to freeform actions
    if 'actions' in obj:
        for act in obj['actions']:
            act.draw(draw, (x,y))
        return

    # ultimate fallback: tiny circle
    draw.ellipse((x-3,y-3,x+3,y+3), fill='black')
