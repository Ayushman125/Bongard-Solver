"""
Create example BONGARD-LOGO problem visualizations
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path


def create_simple_shapes():
    """Generate simple geometric shapes for examples"""
    
    # Free-form shapes (strokes)
    fig, axes = plt.subplots(2, 6, figsize=(14, 6))
    fig.suptitle('Set A: Free-Form Shapes (Positive - "Ice Cream Cone" Pattern)', fontsize=12, fontweight='bold')
    
    for idx, ax in enumerate(axes[0]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw ice cream cone pattern
        # Cone (triangle)
        triangle = patches.Polygon([[5, 3], [3, 0], [7, 0]], closed=True, 
                                  edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(triangle)
        
        # Add slight variation to each
        offset = idx * 0.3
        circle = patches.Circle((5, 6 + offset), 1.5, edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(circle)
        
        ax.text(5, -1, f'A{idx+1}', ha='center', fontsize=9)
    
    for idx, ax in enumerate(axes[1]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw DIFFERENT pattern (negative)
        # Simple line
        ax.plot([3, 7], [5, 5], 'k-', linewidth=2)
        ax.plot([5, 5], [5, 8], 'k-', linewidth=2)
        
        ax.text(5, -1, f'B{idx+1}', ha='center', fontsize=9)
    
    return fig


def create_basic_shapes():
    """Generate basic shape combinations"""
    
    fig, axes = plt.subplots(2, 6, figsize=(14, 6))
    fig.suptitle('Set A: Basic Shapes (Positive - "Fan" + "Trapezoid")', fontsize=12, fontweight='bold')
    
    for idx, ax in enumerate(axes[0]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw fan shape (multiple lines from point)
        center_x, center_y = 3, 5
        for i in range(5):
            angle = i * 30 / 180 * np.pi
            x = center_x + 2 * np.cos(angle)
            y = center_y + 2 * np.sin(angle)
            ax.plot([center_x, x], [center_y, y], 'k-', linewidth=2)
        
        # Draw trapezoid
        trap = patches.Polygon([[6, 2], [8, 2], [7.5, 5], [6.5, 5]], 
                              closed=True, edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(trap)
        
        ax.text(5, -1, f'A{idx+1}', ha='center', fontsize=9)
    
    for idx, ax in enumerate(axes[1]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Different shapes (negative)
        # Circle
        circle = patches.Circle((3, 5), 1.5, edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(circle)
        
        # Rectangle
        rect = patches.Rectangle((6, 3), 2, 3, edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(rect)
        
        ax.text(5, -1, f'B{idx+1}', ha='center', fontsize=9)
    
    return fig


def create_abstract_shapes():
    """Generate abstract concept shapes (convex vs concave)"""
    
    fig, axes = plt.subplots(2, 6, figsize=(14, 6))
    fig.suptitle('Set A: Abstract Shapes (Positive - "Convex")', fontsize=12, fontweight='bold')
    
    for idx, ax in enumerate(axes[0]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw convex polygon
        angle_offset = idx * 20
        angles = np.linspace(0, 2*np.pi, 5) + angle_offset/180*np.pi
        radius = 2
        x = 5 + radius * np.cos(angles)
        y = 5 + radius * np.sin(angles)
        
        polygon = patches.Polygon(list(zip(x, y)), closed=True, 
                                 edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(polygon)
        
        ax.text(5, -1, f'A{idx+1}', ha='center', fontsize=9)
    
    for idx, ax in enumerate(axes[1]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw concave shape (star-like)
        angles = []
        radii = []
        for i in range(10):
            if i % 2 == 0:
                angles.append(i * 36 / 180 * np.pi)
                radii.append(2.5)
            else:
                angles.append(i * 36 / 180 * np.pi)
                radii.append(1.0)
        
        x = 5 + np.array(radii) * np.cos(angles)
        y = 5 + np.array(radii) * np.sin(angles)
        
        polygon = patches.Polygon(list(zip(x, y)), closed=True, 
                                 edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(polygon)
        
        ax.text(5, -1, f'B{idx+1}', ha='center', fontsize=9)
    
    return fig


def create_test_examples():
    """Create test example visualizations"""
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Test Images with Predictions', fontsize=12, fontweight='bold')
    
    # Test 1: Convex (positive)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    angles = np.linspace(0, 2*np.pi, 6)
    x = 5 + 2 * np.cos(angles)
    y = 5 + 2 * np.sin(angles)
    polygon = patches.Polygon(list(zip(x, y)), closed=True, 
                             edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(polygon)
    
    ax.text(5, -1.5, '✓ CORRECT', ha='center', fontsize=11, fontweight='bold', color='green')
    ax.text(5, -2.5, 'S1: 78% | S2: 92%', ha='center', fontsize=9)
    ax.set_title('Test Positive', fontsize=10)
    
    # Test 2: Concave (negative)
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    angles = []
    radii = []
    for i in range(10):
        if i % 2 == 0:
            angles.append(i * 36 / 180 * np.pi)
            radii.append(2.5)
        else:
            angles.append(i * 36 / 180 * np.pi)
            radii.append(1.0)
    
    x = 5 + np.array(radii) * np.cos(angles)
    y = 5 + np.array(radii) * np.sin(angles)
    polygon = patches.Polygon(list(zip(x, y)), closed=True, 
                             edgecolor='black', facecolor='none', linewidth=2)
    ax.add_patch(polygon)
    
    ax.text(5, -1.5, '✓ CORRECT', ha='center', fontsize=11, fontweight='bold', color='green')
    ax.text(5, -2.5, 'S1: 82% | S2: 88%', ha='center', fontsize=9)
    ax.set_title('Test Negative', fontsize=10)
    
    # Performance summary
    ax = axes[2]
    ax.axis('off')
    summary_text = """
    Hybrid Dual-System
    Prediction: CORRECT ✓
    
    System 1 (Neural)
    Confidence: 80%
    
    System 2 (Bayesian)
    Confidence: 90%
    
    Final Decision: ✓
    Arbitration Weight:
    S1: 47% | S2: 53%
    """
    ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace')
    ax.set_title('Analysis', fontsize=10)
    
    return fig


def main():
    # Create output directories
    output_dir = Path('examples/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating example visualizations...")
    
    # Generate free-form examples
    print("Creating free-form shape examples...")
    fig = create_simple_shapes()
    fig.tight_layout()
    fig.savefig(output_dir / '01-freeform-example.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: examples/visualizations/01-freeform-example.png")
    
    # Generate basic shape examples
    print("Creating basic shape examples...")
    fig = create_basic_shapes()
    fig.tight_layout()
    fig.savefig(output_dir / '02-basic-example.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: examples/visualizations/02-basic-example.png")
    
    # Generate abstract shape examples
    print("Creating abstract shape examples...")
    fig = create_abstract_shapes()
    fig.tight_layout()
    fig.savefig(output_dir / '03-abstract-example.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: examples/visualizations/03-abstract-example.png")
    
    # Generate test examples
    print("Creating test prediction examples...")
    fig = create_test_examples()
    fig.tight_layout()
    fig.savefig(output_dir / '04-test-predictions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: examples/visualizations/04-test-predictions.png")
    
    print("\n✅ All example visualizations created successfully!")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
