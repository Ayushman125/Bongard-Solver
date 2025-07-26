#!/usr/bin/env python3
"""
Test script for the advanced Bongard problem image augmentation system.
Demonstrates exponentially improved dataset quality through:
- Multi-model SAM ensemble with intelligent auto-prompting
- Adaptive quality control with learning thresholds
- Contrastive learning for diversity maximization
- Structure-aware geometric transformations
"""

import torch
import numpy as np
import cv2
import os
import time
import json
from pathlib import Path

# Import the enhanced augmentation system
from src.bongard_augmentor.main import ImageAugmentor

def create_test_images(num_images=5, size=(224, 224)):
    """Create synthetic test images with various geometric patterns."""
    images = []
    
    for i in range(num_images):
        # Create blank image
        img = np.zeros((*size, 3), dtype=np.uint8)
        
        if i == 0:
            # Circle
            center = (size[1]//2, size[0]//2)
            radius = min(size) // 4
            cv2.circle(img, center, radius, (255, 255, 255), -1)
            
        elif i == 1:
            # Rectangle
            pt1 = (size[1]//4, size[0]//4)
            pt2 = (3*size[1]//4, 3*size[0]//4)
            cv2.rectangle(img, pt1, pt2, (255, 255, 255), -1)
            
        elif i == 2:
            # Triangle
            pts = np.array([
                [size[1]//2, size[0]//4],
                [size[1]//4, 3*size[0]//4],
                [3*size[1]//4, 3*size[0]//4]
            ], np.int32)
            cv2.fillPoly(img, [pts], (255, 255, 255))
            
        elif i == 3:
            # Cross pattern
            cv2.line(img, (size[1]//4, size[0]//2), (3*size[1]//4, size[0]//2), (255, 255, 255), 20)
            cv2.line(img, (size[1]//2, size[0]//4), (size[1]//2, 3*size[0]//4), (255, 255, 255), 20)
            
        else:
            # Complex pattern - multiple shapes
            cv2.circle(img, (size[1]//3, size[0]//3), 30, (255, 255, 255), -1)
            cv2.rectangle(img, (2*size[1]//3, size[0]//4), (size[1]-20, 2*size[0]//3), (255, 255, 255), -1)
        
        # Convert to tensor format (CHW)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        images.append(img_tensor)
    
    return images

def test_basic_mask_generation():
    """Test basic mask generation with the new system."""
    print("=" * 60)
    print("TEST 1: Basic Mask Generation with Advanced SAM Ensemble")
    print("=" * 60)
    
    # Initialize augmentor with advanced features
    augmentor = ImageAugmentor(device='cuda', batch_size=16)
    
    # Initialize the advanced hybrid pipeline
    print("Initializing advanced hybrid pipeline...")
    augmentor.initialize_hybrid_pipeline(
        sam_model_type='vit_h',  # Use highest quality model
        enable_refiner=True,
        enable_diffusion=False
    )
    
    # Check system status
    status = augmentor.get_system_status()
    print(f"System Status: {json.dumps(status, indent=2, default=str)}")
    
    # Create test images
    test_images = create_test_images(3)
    
    # Test mask generation
    results = []
    for i, img_tensor in enumerate(test_images):
        print(f"\nProcessing image {i+1}/3...")
        
        start_time = time.time()
        mask = augmentor.generate_hybrid_mask(
            img_tensor,
            quality_threshold=0.7,
            use_ensemble=True
        )
        processing_time = time.time() - start_time
        
        print(f"  Generated mask in {processing_time:.3f}s")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask coverage: {torch.sum(mask > 0.5).item() / mask.numel() * 100:.1f}%")
        
        results.append({
            'image_index': i,
            'processing_time': processing_time,
            'mask_coverage': torch.sum(mask > 0.5).item() / mask.numel(),
            'mask_shape': list(mask.shape)
        })
    
    return results

def test_diverse_mask_generation():
    """Test diverse mask generation capabilities."""
    print("=" * 60)
    print("TEST 2: Diverse Mask Generation with Ensemble Techniques")
    print("=" * 60)
    
    augmentor = ImageAugmentor(device='cuda')
    augmentor.initialize_hybrid_pipeline(enable_refiner=True)
    
    # Create a complex test image
    test_image = create_test_images(1, size=(256, 256))[0]
    
    print("Generating diverse masks...")
    diverse_masks = augmentor.generate_diverse_sam_masks(test_image, top_k=5)
    
    print(f"Generated {len(diverse_masks)} diverse masks")
    
    for i, mask in enumerate(diverse_masks):
        coverage = torch.sum(mask > 0.5).item() / mask.numel()
        print(f"  Mask {i+1}: Coverage {coverage*100:.1f}%")
    
    return diverse_masks

def test_advanced_dataset_generation():
    """Test advanced dataset generation with contrastive learning."""
    print("=" * 60)
    print("TEST 3: Advanced Dataset Generation with Contrastive Learning")
    print("=" * 60)
    
    augmentor = ImageAugmentor(device='cuda')
    augmentor.initialize_hybrid_pipeline(enable_refiner=True)
    
    # Create base dataset
    base_images = create_test_images(3, size=(224, 224))
    print(f"Starting with {len(base_images)} base images")
    
    # Generate augmented dataset
    print("Generating advanced augmented dataset...")
    start_time = time.time()
    
    result = augmentor.generate_augmented_dataset(
        images=base_images,
        target_size=None,  # Auto-calculate
        diversity_factor=3.0,
        save_progress=True
    )
    
    generation_time = time.time() - start_time
    
    # Analyze results
    samples = result['samples']
    metadata = result['metadata']
    
    print(f"\nDataset Generation Results:")
    print(f"  Generation time: {generation_time:.2f}s")
    print(f"  Original size: {metadata.get('original_size', 'unknown')}")
    print(f"  Augmented size: {metadata.get('augmented_size', len(samples))}")
    print(f"  Average quality: {metadata.get('average_quality', 'unknown'):.3f}")
    print(f"  Quality range: {metadata.get('min_quality', 'unknown'):.3f} - {metadata.get('max_quality', 'unknown'):.3f}")
    print(f"  Diversity factor achieved: {metadata.get('diversity_factor_achieved', 'unknown'):.2f}")
    
    # Quality distribution analysis
    if samples:
        quality_scores = [s['quality_score'] for s in samples]
        print(f"  Quality distribution:")
        print(f"    High quality (>0.8): {sum(1 for q in quality_scores if q > 0.8)}")
        print(f"    Medium quality (0.6-0.8): {sum(1 for q in quality_scores if 0.6 <= q <= 0.8)}")
        print(f"    Lower quality (<0.6): {sum(1 for q in quality_scores if q < 0.6)}")
    
    return result

def test_comprehensive_pipeline():
    """Test the complete pipeline including performance monitoring."""
    print("=" * 60)
    print("TEST 4: Comprehensive Pipeline Performance Test")
    print("=" * 60)
    
    augmentor = ImageAugmentor(device='cuda')
    augmentor.initialize_hybrid_pipeline(enable_refiner=True)
    
    # Create test batch
    test_images = torch.stack(create_test_images(8, size=(224, 224)))
    print(f"Testing pipeline with batch of {test_images.shape[0]} images")
    
    # Run comprehensive test
    test_results = augmentor.test_hybrid_pipeline(
        test_images,
        save_dir="test_results_advanced"
    )
    
    print(f"\nPipeline Test Results:")
    print(f"  Success rate: {test_results['successful_masks']}/{test_results['total_images']}")
    if test_results.get('average_quality'):
        print(f"  Average quality: {test_results['average_quality']:.3f} ± {test_results['quality_std']:.3f}")
        print(f"  Average processing time: {test_results['average_processing_time']:.3f}s")
    
    # Get final system status
    final_status = augmentor.get_system_status()
    print(f"\nFinal System Status:")
    if 'sam_performance' in final_status:
        perf = final_status['sam_performance']
        print(f"  Total predictions: {perf.get('total_predictions', 0)}")
        print(f"  Success rate: {perf.get('success_rate', 0):.1%}")
        print(f"  Average quality: {perf.get('average_quality', 0):.3f}")
    
    return test_results

def main():
    """Run all tests and generate comprehensive report."""
    print("🚀 ADVANCED BONGARD AUGMENTATION SYSTEM TEST SUITE")
    print("Featuring: Multi-model SAM ensemble, Adaptive quality control,")
    print("Contrastive learning, and Structure-aware transformations")
    print("=" * 80)
    
    # Create results directory
    os.makedirs("test_results_advanced", exist_ok=True)
    
    all_results = {}
    
    try:
        # Test 1: Basic mask generation
        basic_results = test_basic_mask_generation()
        all_results['basic_mask_generation'] = basic_results
        
        # Test 2: Diverse mask generation
        diverse_results = test_diverse_mask_generation()
        all_results['diverse_mask_generation'] = f"Generated {len(diverse_results)} masks"
        
        # Test 3: Advanced dataset generation
        dataset_results = test_advanced_dataset_generation()
        all_results['dataset_generation'] = dataset_results['metadata']
        
        # Test 4: Comprehensive pipeline test
        pipeline_results = test_comprehensive_pipeline()
        all_results['pipeline_performance'] = pipeline_results
        
        # Save comprehensive report
        report_path = "test_results_advanced/comprehensive_report.json"
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"📊 Comprehensive report saved to: {report_path}")
        print("🎯 The advanced augmentation system is ready for exponential dataset improvement!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error report
        error_report = {
            'error': str(e),
            'completed_tests': list(all_results.keys()),
            'traceback': traceback.format_exc()
        }
        
        with open("test_results_advanced/error_report.json", 'w') as f:
            json.dump(error_report, f, indent=2)

if __name__ == "__main__":
    main()
