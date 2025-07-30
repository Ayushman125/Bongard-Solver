"""
Module: synthetic_shapes.py
Purpose: Generate synthetic datasets for all geometric shape categories (ellipse, circle, rectangle, polygon, triangle, point cloud, etc.)
Each function generates labeled samples for a specific shape type, with options for noise, occlusion, and transformations.
"""
import numpy as np
import cv2
import random
from typing import List, Dict, Tuple

# Utility for random color
COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]

def random_ellipse(image_size=(128,128), noise=0.0, partial=False) -> Tuple[np.ndarray, Dict]:
    """Generate a synthetic ellipse (optionally partial) and return (image, label_dict)"""
    img = np.zeros(image_size, dtype=np.uint8)
    center = (random.randint(32,96), random.randint(32,96))
    axes = (random.randint(20,40), random.randint(10,30))
    angle = random.uniform(0,180)
    startAngle = 0
    endAngle = 360 if not partial else random.randint(180, 300)
    color = 255
    thickness = random.randint(2,4)
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)
    # Optionally add noise
    if noise > 0:
        img = img + (np.random.randn(*img.shape) * noise * 255).astype(np.uint8)
    label = {
        'type': 'ellipse',
        'center': center,
        'axes': axes,
        'angle': angle,
        'partial': partial
    }
    return img, label

def random_circle(image_size=(128,128), noise=0.0, partial=False) -> Tuple[np.ndarray, Dict]:
    """Generate a synthetic circle (optionally partial) and return (image, label_dict)"""
    img = np.zeros(image_size, dtype=np.uint8)
    center = (random.randint(32,96), random.randint(32,96))
    radius = random.randint(15,40)
    startAngle = 0
    endAngle = 360 if not partial else random.randint(180, 300)
    color = 255
    thickness = random.randint(2,4)
    cv2.ellipse(img, center, (radius, radius), 0, startAngle, endAngle, color, thickness)
    if noise > 0:
        img = img + (np.random.randn(*img.shape) * noise * 255).astype(np.uint8)
    label = {
        'type': 'circle',
        'center': center,
        'radius': radius,
        'partial': partial
    }
    return img, label

def random_rectangle(image_size=(128,128), noise=0.0, rotated=False) -> Tuple[np.ndarray, Dict]:
    """Generate a synthetic rectangle (optionally rotated) and return (image, label_dict)"""
    img = np.zeros(image_size, dtype=np.uint8)
    w, h = random.randint(20,60), random.randint(20,60)
    x, y = random.randint(20, 100), random.randint(20, 100)
    angle = random.uniform(0, 180) if rotated else 0
    rect = ((x, y), (w, h), angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(img, [box], 0, 255, random.randint(2,4))
    if noise > 0:
        img = img + (np.random.randn(*img.shape) * noise * 255).astype(np.uint8)
    label = {
        'type': 'rectangle',
        'box': box.tolist(),
        'angle': angle
    }
    return img, label

def random_polygon(image_size=(128,128), sides=5, noise=0.0) -> Tuple[np.ndarray, Dict]:
    """Generate a synthetic polygon with N sides and return (image, label_dict)"""
    img = np.zeros(image_size, dtype=np.uint8)
    center = (random.randint(40,88), random.randint(40,88))
    radius = random.randint(20,40)
    angle_offset = random.uniform(0, 2*np.pi)
    pts = []
    for i in range(sides):
        theta = angle_offset + 2*np.pi*i/sides
        x = int(center[0] + radius * np.cos(theta))
        y = int(center[1] + radius * np.sin(theta))
        pts.append([x, y])
    pts = np.array(pts, np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=255, thickness=random.randint(2,4))
    if noise > 0:
        img = img + (np.random.randn(*img.shape) * noise * 255).astype(np.uint8)
    label = {
        'type': 'polygon',
        'points': pts.tolist(),
        'sides': sides
    }
    return img, label

def random_triangle(image_size=(128,128), noise=0.0) -> Tuple[np.ndarray, Dict]:
    return random_polygon(image_size, sides=3, noise=noise)

def random_point_cloud(image_size=(128,128), n_points=30, spread=40, noise=0.0) -> Tuple[np.ndarray, Dict]:
    """Generate a synthetic point cloud and return (image, label_dict)"""
    img = np.zeros(image_size, dtype=np.uint8)
    center = (random.randint(40,88), random.randint(40,88))
    pts = []
    for _ in range(n_points):
        x = int(center[0] + random.gauss(0, spread))
        y = int(center[1] + random.gauss(0, spread))
        pts.append([x, y])
        cv2.circle(img, (x, y), 1, 255, -1)
    if noise > 0:
        img = img + (np.random.randn(*img.shape) * noise * 255).astype(np.uint8)
    label = {
        'type': 'point_cloud',
        'points': pts
    }
    return img, label

def generate_dataset(shape_types: List[str], n_per_type=100, noise=0.05, out_dir=None) -> List[Dict]:
    """Generate a dataset with all shape types. Optionally save images and return label dicts."""
    all_labels = []
    for shape in shape_types:
        for i in range(n_per_type):
            if shape == 'ellipse':
                img, label = random_ellipse(noise=noise)
            elif shape == 'circle':
                img, label = random_circle(noise=noise)
            elif shape == 'rectangle':
                img, label = random_rectangle(noise=noise)
            elif shape == 'polygon':
                img, label = random_polygon(noise=noise)
            elif shape == 'triangle':
                img, label = random_triangle(noise=noise)
            elif shape == 'point_cloud':
                img, label = random_point_cloud(noise=noise)
            else:
                continue
            label['image'] = img
            all_labels.append(label)
            if out_dir is not None:
                cv2.imwrite(f"{out_dir}/{shape}_{i}.png", img)
    return all_labels

# Example usage:
if __name__ == "__main__":
    shapes = ['ellipse', 'circle', 'rectangle', 'polygon', 'triangle', 'point_cloud']
    labels = generate_dataset(shapes, n_per_type=10, noise=0.03, out_dir=None)
    # Visualize a few
    import matplotlib.pyplot as plt
    for i, label in enumerate(labels[:6]):
        plt.subplot(2,3,i+1)
        plt.imshow(label['image'], cmap='gray')
        plt.title(label['type'])
        plt.axis('off')
    plt.show()
