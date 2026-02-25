"""
Generate Visual Example Gallery for BONGARD-LOGO Solver

This script creates visual demonstrations of the hybrid dual-system solver
in action, similar to the figures in the original BONGARD-LOGO paper.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np


def load_evaluation_results(results_file: str) -> Dict:
    """Load evaluation results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_bongard_visualization(
    problem_data: Dict,
    output_path: str,
    title: str = None,
    show_predictions: bool = True
):
    """
    Create a BONGARD-LOGO style visualization
    
    Args:
        problem_data: Dictionary containing:
            - pos_images: List of 6 positive image paths
            - neg_images: List of 6 negative image paths
            - test_pos: Positive test image path
            - test_neg: Negative test image path
            - concept: Concept description (optional)
            - s1_confidence: System 1 confidence (optional)
            - s2_confidence: System 2 confidence (optional)
            - prediction_correct: Bool (optional)
        output_path: Where to save the visualization
        title: Title for the figure
        show_predictions: Whether to show model predictions
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(4, 7, figure=fig, hspace=0.3, wspace=0.2)
    
    # Title
    if title:
        fig.suptitle(title, fontsize=20, fontweight='bold')
    
    # Set A (Positive Examples) - Top 3
    ax_a_label = fig.add_subplot(gs[0, 0])
    ax_a_label.text(0.5, 0.5, 'Set A\n(Positive)', 
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax_a_label.axis('off')
    
    for i in range(3):
        ax = fig.add_subplot(gs[0, i+1])
        if i < len(problem_data.get('pos_images', [])):
            img = Image.open(problem_data['pos_images'][i])
            ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'A{i+1}', fontsize=10)
    
    # Set A (Positive Examples) - Bottom 3
    for i in range(3, 6):
        ax = fig.add_subplot(gs[1, i-2])
        if i < len(problem_data.get('pos_images', [])):
            img = Image.open(problem_data['pos_images'][i])
            ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'A{i+1}', fontsize=10)
    
    # Separator
    ax_sep = fig.add_subplot(gs[1, 5:7])
    ax_sep.text(0.5, 0.5, 'VS', ha='center', va='center', 
                fontsize=24, fontweight='bold')
    ax_sep.axis('off')
    
    # Set B (Negative Examples) - Top 3
    ax_b_label = fig.add_subplot(gs[2, 0])
    ax_b_label.text(0.5, 0.5, 'Set B\n(Negative)', 
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax_b_label.axis('off')
    
    for i in range(3):
        ax = fig.add_subplot(gs[2, i+1])
        if i < len(problem_data.get('neg_images', [])):
            img = Image.open(problem_data['neg_images'][i])
            ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'B{i+1}', fontsize=10)
    
    # Set B (Negative Examples) - Bottom 3
    for i in range(3, 6):
        ax = fig.add_subplot(gs[3, i-2])
        if i < len(problem_data.get('neg_images', [])):
            img = Image.open(problem_data['neg_images'][i])
            ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'B{i+1}', fontsize=10)
    
    # Test Images
    ax_test_label = fig.add_subplot(gs[3, 5])
    ax_test_label.text(0.5, 0.5, 'Test', ha='center', va='center', 
                       fontsize=14, fontweight='bold')
    ax_test_label.axis('off')
    
    # Positive test
    ax_test_pos = fig.add_subplot(gs[3, 6])
    if 'test_pos' in problem_data:
        img = Image.open(problem_data['test_pos'])
        ax_test_pos.imshow(img, cmap='gray')
        ax_test_pos.set_title('✓', fontsize=14, color='green', fontweight='bold')
    ax_test_pos.axis('off')
    
    # Concept and Predictions Info
    info_text = []
    if 'concept' in problem_data:
        info_text.append(f"Concept: {problem_data['concept']}")
    
    if show_predictions and 's1_confidence' in problem_data:
        info_text.append(f"\nSystem 1 (Neural): {problem_data['s1_confidence']:.2%}")
        info_text.append(f"System 2 (Bayesian): {problem_data['s2_confidence']:.2%}")
        
        if 'prediction_correct' in problem_data:
            result = "✓ CORRECT" if problem_data['prediction_correct'] else "✗ INCORRECT"
            color = 'green' if problem_data['prediction_correct'] else 'red'
            info_text.append(f"\nPrediction: {result}")
    
    if info_text:
        ax_info = fig.add_subplot(gs[0:2, 5:7])
        ax_info.text(0.05, 0.95, '\n'.join(info_text), 
                     va='top', ha='left', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax_info.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def generate_example_gallery(
    results_dir: str,
    output_dir: str,
    num_examples: int = 5,
    splits: List[str] = ['test_ff', 'test_bd', 'test_hd_comb', 'test_hd_novel']
):
    """
    Generate example gallery from evaluation results
    
    Args:
        results_dir: Directory containing evaluation results
        output_dir: Directory to save generated examples
        num_examples: Number of examples per split
        splits: Which splits to generate examples for
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split in splits:
        split_dir = output_path / split
        split_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating examples for {split}...")
        
        # Example usage - you would load real data here
        # This is a template showing the expected structure
        example_problem = {
            'pos_images': [],  # List of 6 positive image paths
            'neg_images': [],  # List of 6 negative image paths
            'test_pos': '',    # Positive test image path
            'test_neg': '',    # Negative test image path
            'concept': f'Example concept from {split}',
            's1_confidence': 0.85,
            's2_confidence': 0.92,
            'prediction_correct': True
        }
        
        # Generate visualization
        # create_bongard_visualization(
        #     example_problem,
        #     str(split_dir / 'example_001.png'),
        #     title=f'{split.upper()}: Example Problem'
        # )
    
    print(f"\nExample gallery generation complete!")
    print(f"Output directory: {output_dir}")
    print("\nNote: This is a template. To generate real examples,")
    print("add image paths from your evaluation results.")


def main():
    parser = argparse.ArgumentParser(
        description='Generate visual example gallery for BONGARD-LOGO solver'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='logs',
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='examples/gallery',
        help='Directory to save generated examples'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=5,
        help='Number of examples to generate per split'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['test_ff', 'test_bd', 'test_hd_comb', 'test_hd_novel'],
        help='Which splits to generate examples for'
    )
    
    args = parser.parse_args()
    
    generate_example_gallery(
        args.results_dir,
        args.output_dir,
        args.num_examples,
        args.splits
    )


if __name__ == '__main__':
    main()
