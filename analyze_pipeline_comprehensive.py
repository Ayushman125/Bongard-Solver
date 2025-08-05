#!/usr/bin/env python3
"""
Comprehensive analysis of the Bongard-LOGO mask generation pipeline.
Analyzes the entire pipeline to identify mapping issues between real, parsed, and mask images.
"""

import sys
import numpy as np
import cv2
import logging
from pathlib import Path
import json
from typing import List, Dict, Any
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from src.bongard_augmentor.hybrid import ActionMaskGenerator
from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser

# Set up comprehensive logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_pipeline_images():
    """Analyze the generated real, parsed, and mask images to identify mapping issues."""
    
    print("=== COMPREHENSIVE PIPELINE ANALYSIS ===")
    
    data_dir = Path("data")
    
    # Find all image sets
    image_sets = {}
    for img_file in data_dir.glob("*.png"):
        if "_real.png" in img_file.name:
            base_name = img_file.name.replace("_real.png", "")
            image_sets[base_name] = {
                "real": img_file,
                "parsed": data_dir / f"{base_name}_parsed.png",
                "mask": data_dir / f"{base_name}_mask.png"
            }
    
    print(f"Found {len(image_sets)} image sets for analysis")
    
    # Analyze first few sets in detail
    analysis_results = []
    for i, (base_name, paths) in enumerate(list(image_sets.items())[:3]):
        print(f"\n--- ANALYZING {base_name} ---")
        
        # Load images
        real_img = cv2.imread(str(paths["real"]), cv2.IMREAD_GRAYSCALE)
        parsed_img = cv2.imread(str(paths["parsed"]), cv2.IMREAD_GRAYSCALE) 
        mask_img = cv2.imread(str(paths["mask"]), cv2.IMREAD_GRAYSCALE)
        
        if real_img is None or parsed_img is None or mask_img is None:
            print(f"❌ Could not load all images for {base_name}")
            continue
        
        # Analyze each image
        analysis = {
            "name": base_name,
            "real": analyze_single_image(real_img, "Real"),
            "parsed": analyze_single_image(parsed_img, "Parsed"),
            "mask": analyze_single_image(mask_img, "Mask")
        }
        
        analysis_results.append(analysis)
        
        # Create comparison visualization
        create_comparison_visualization(base_name, real_img, parsed_img, mask_img)
    
    # Summary analysis
    print_analysis_summary(analysis_results)
    
    return analysis_results

def analyze_single_image(img: np.ndarray, img_type: str) -> Dict[str, Any]:
    """Analyze a single image and return statistics."""
    
    analysis = {
        "type": img_type,
        "shape": img.shape,
        "dtype": str(img.dtype),
        "min_val": int(np.min(img)),
        "max_val": int(np.max(img)),
        "mean_val": float(np.mean(img)),
        "std_val": float(np.std(img)),
        "unique_vals": len(np.unique(img)),
        "nonzero_pixels": int(np.count_nonzero(img)),
        "zero_pixels": int(np.sum(img == 0)),
        "white_pixels": int(np.sum(img == 255))
    }
    
    total_pixels = img.size
    analysis["nonzero_percent"] = 100 * analysis["nonzero_pixels"] / total_pixels
    analysis["white_percent"] = 100 * analysis["white_pixels"] / total_pixels
    
    # Detect image characteristics
    if analysis["unique_vals"] == 2 and 0 in np.unique(img) and 255 in np.unique(img):
        analysis["is_binary"] = True
    else:
        analysis["is_binary"] = False
    
    if analysis["white_percent"] > 95:
        analysis["mostly_white"] = True
        analysis["mostly_black"] = False
    elif analysis["nonzero_percent"] < 5:
        analysis["mostly_black"] = True
        analysis["mostly_white"] = False
    else:
        analysis["mostly_white"] = False
        analysis["mostly_black"] = False
    
    print(f"{img_type:>6}: {analysis['shape']} | "
          f"Range: [{analysis['min_val']}-{analysis['max_val']}] | "
          f"Mean: {analysis['mean_val']:.1f} | "
          f"Nonzero: {analysis['nonzero_percent']:.1f}% | "
          f"Binary: {analysis['is_binary']}")
    
    return analysis

def create_comparison_visualization(base_name: str, real_img: np.ndarray, 
                                  parsed_img: np.ndarray, mask_img: np.ndarray):
    """Create a side-by-side comparison of real, parsed, and mask images."""
    
    # Ensure all images are same size for comparison
    h, w = real_img.shape
    parsed_resized = cv2.resize(parsed_img, (w, h))
    mask_resized = cv2.resize(mask_img, (w, h))
    
    # Create comparison
    comparison = np.hstack([real_img, parsed_resized, mask_resized])
    
    # Convert to BGR for text overlay
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
    
    # Add labels
    cv2.putText(comparison_bgr, "REAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(comparison_bgr, "PARSED", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(comparison_bgr, "MASK", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Save comparison
    output_path = f"analysis_{base_name}_comparison.png"
    cv2.imwrite(output_path, comparison_bgr)
    print(f"   Saved comparison: {output_path}")

def analyze_action_commands():
    """Analyze the action commands and their parsing."""
    
    print("\n=== ACTION COMMAND ANALYSIS ===")
    
    # Sample action commands from the logs
    sample_commands = [
        'line_normal_1.000-0.500',
        'line_normal_0.608-0.121', 
        'line_normal_0.224-0.600',
        'line_normal_1.000-0.074',
        'line_normal_0.224-0.926',
        'line_normal_0.224-0.352'
    ]
    
    print(f"Sample commands: {sample_commands}")
    
    # Initialize parser and generator
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    try:
        # Parse commands
        parsed_data = generator.action_parser.parse_action_commands(sample_commands, "test_analysis")
        
        print(f"Parsed vertices count: {len(parsed_data.vertices)}")
        print(f"Vertices sample: {parsed_data.vertices[:3]}...")
        
        # Analyze coordinate ranges
        if parsed_data.vertices:
            verts = np.array(parsed_data.vertices)
            min_coords = np.min(verts, axis=0)
            max_coords = np.max(verts, axis=0)
            range_coords = max_coords - min_coords
            
            print(f"Coordinate analysis:")
            print(f"  Min: ({min_coords[0]:.1f}, {min_coords[1]:.1f})")
            print(f"  Max: ({max_coords[0]:.1f}, {max_coords[1]:.1f})")
            print(f"  Range: ({range_coords[0]:.1f}, {range_coords[1]:.1f})")
            
            # Test mask generation
            mask = generator._render_vertices_to_mask(parsed_data.vertices)
            mask_stats = analyze_single_image(mask, "Generated")
            
            cv2.imwrite("analysis_generated_mask.png", mask)
            print(f"   Saved generated mask: analysis_generated_mask.png")
            
    except Exception as e:
        print(f"❌ Command analysis failed: {e}")
        import traceback
        traceback.print_exc()

def analyze_coordinate_transformation():
    """Analyze the coordinate transformation pipeline in detail."""
    
    print("\n=== COORDINATE TRANSFORMATION ANALYSIS ===")
    
    # Test the normalization logic step by step
    sample_vertices = [
        (0.0, 0.0), 
        (1440.0, 0.0), 
        (805.55, -603.33), 
        (747.10, -920.55),
        (345.35, 462.27),
        (286.91, 145.05),
        (-2.32, 2.23)
    ]
    
    print(f"Input vertices: {len(sample_vertices)} points")
    print(f"Sample: {sample_vertices[:3]}...")
    
    # Simulate the normalization from hybrid.py
    verts = np.array(sample_vertices)
    min_x, min_y = np.min(verts, axis=0)
    max_x, max_y = np.max(verts, axis=0)
    range_x = max_x - min_x
    range_y = max_y - min_y
    
    print(f"Original bounds: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
    print(f"Ranges: X={range_x:.1f}, Y={range_y:.1f}")
    
    # Canvas settings
    image_size = (512, 512)
    margin = 20
    available_w = image_size[1] - 2 * margin
    available_h = image_size[0] - 2 * margin
    
    # Calculate scaling
    scale_x = available_w / range_x if range_x > 0 else 1.0
    scale_y = available_h / range_y if range_y > 0 else 1.0
    scale = min(scale_x, scale_y)
    
    print(f"Scale calculation: min({scale_x:.4f}, {scale_y:.4f}) = {scale:.4f}")
    
    # Apply normalization
    verts_norm = np.array(verts, dtype=np.float64)
    verts_norm[:,0] = (verts[:,0] - min_x) * scale
    verts_norm[:,1] = (verts[:,1] - min_y) * scale
    
    # Center in canvas
    scaled_w = np.max(verts_norm[:,0]) - np.min(verts_norm[:,0])
    scaled_h = np.max(verts_norm[:,1]) - np.min(verts_norm[:,1])
    
    center_offset_x = (image_size[1] - scaled_w) / 2 - np.min(verts_norm[:,0])
    center_offset_y = (image_size[0] - scaled_h) / 2 - np.min(verts_norm[:,1])
    
    verts_norm[:,0] += center_offset_x
    verts_norm[:,1] += center_offset_y
    
    # Clamp to bounds
    verts_norm[:,0] = np.clip(verts_norm[:,0], 0, image_size[1] - 1)
    verts_norm[:,1] = np.clip(verts_norm[:,1], 0, image_size[0] - 1)
    
    print(f"Final bounds: X[{np.min(verts_norm[:,0]):.1f}, {np.max(verts_norm[:,0]):.1f}], Y[{np.min(verts_norm[:,1]):.1f}, {np.max(verts_norm[:,1]):.1f}]")
    print(f"Final vertices sample: {verts_norm[:3].tolist()}")

def print_analysis_summary(analysis_results: List[Dict]):
    """Print a summary of the analysis results."""
    
    print("\n=== ANALYSIS SUMMARY ===")
    
    for result in analysis_results:
        print(f"\n{result['name']}:")
        
        real = result['real']
        parsed = result['parsed'] 
        mask = result['mask']
        
        # Check for issues
        issues = []
        
        if mask['mostly_white']:
            issues.append("Mask is mostly white (coordinate issue)")
        elif mask['mostly_black']:
            issues.append("Mask is mostly black (no content)")
        
        if not mask['is_binary']:
            issues.append("Mask is not binary")
        
        if real['nonzero_percent'] > 50 and mask['nonzero_percent'] < 5:
            issues.append("Real image has content but mask is empty")
        
        if abs(real['nonzero_percent'] - parsed['nonzero_percent']) > 20:
            issues.append("Real and parsed images differ significantly")
        
        if issues:
            print(f"  ❌ Issues found: {', '.join(issues)}")
        else:
            print(f"  ✅ No major issues detected")
        
        print(f"  Real: {real['nonzero_percent']:.1f}% content")
        print(f"  Parsed: {parsed['nonzero_percent']:.1f}% content") 
        print(f"  Mask: {mask['nonzero_percent']:.1f}% content")

if __name__ == "__main__":
    # Run comprehensive analysis
    analysis_results = analyze_pipeline_images()
    analyze_action_commands()
    analyze_coordinate_transformation()
