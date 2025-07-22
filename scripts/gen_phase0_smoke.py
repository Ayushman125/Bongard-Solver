import numpy as np
import cv2           # pip install opencv-python
import os
from pathlib import Path

OUT = Path("data/phase0_smoke")
OUT.mkdir(parents=True, exist_ok=True)
size = 128  # or 64

def blank():
    return np.zeros((size, size), dtype=np.uint8)

def save(mask, idx):
    np.save(OUT / f"mask_{idx:02d}.npy", mask)

def draw_square(mask):
    x1, y1 = np.random.randint(5, size//2, 2)
    x2, y2 = np.random.randint(size//2, size-5, 2)
    cv2.rectangle(mask, (x1,y1), (x2,y2), 1, -1)

def draw_circle(mask):
    center = tuple(np.random.randint(20, size-20, 2))
    r = np.random.randint(10, size//4)
    cv2.circle(mask, center, r, 1, -1)

def draw_triangle(mask):
    pts = np.random.randint(0, size, (3,2))
    cv2.drawContours(mask, [pts], 0, 1, -1)

def draw_ellipse(mask):
    center = tuple(np.random.randint(20, size-20, 2))
    axes = tuple(np.random.randint(10, size//4, 2))
    angle = np.random.randint(0,360)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)

def draw_polygon(mask):
    pts = np.random.randint(0, size, (np.random.randint(4,7),2))
    cv2.fillPoly(mask, [pts], 1)

generators = [draw_square, draw_circle, draw_triangle, draw_ellipse, draw_polygon]

for i in range(1,21):
    m = blank()
    # 1–3 shapes per mask for variety
    for _ in range(np.random.randint(1,4)):
        np.random.choice(generators)(m)
    save(m, i)

print(f"Generated 20 masks in {OUT}")
