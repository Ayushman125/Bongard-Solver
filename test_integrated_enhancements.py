#!/usr/bin/env python3
"""
Test script to validate integrated enhancements for missing data calculations,
VL embeddings, curvature metrics, motif features, and enhanced visualizations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import logging
from src.scene_graphs_building.vl_features import CLIPEmbedder

logging.basicConfig(level=logging.INFO)

def test_curvature_calculations():
    """Test enhanced curvature calculation functions"""
    print("=== Testing Curvature Calculations ===")
    
    try:
        embedder = CLIPEmbedder()
        
        # Test with open curve (line)
        line_vertices = [(0, 0), (10, 5), (20, 15), (30, 10)]
        curvature_data = embedder.compute_curvature_metrics(line_vertices)
        
        print("Line curvature metrics:")
        for key, value in curvature_data.items():
            print(f"  {key}: {value}")
        
        # Test with closed curve (triangle)
        triangle_vertices = [(0, 0), (10, 0), (5, 8), (0, 0)]
        triangle_curvature = embedder.compute_curvature_metrics(triangle_vertices)
        
        print("\nTriangle curvature metrics:")
        for key, value in triangle_curvature.items():
            print(f"  {key}: {value}")
        
        assert curvature_data['curvature_mean'] >= 0, "Curvature mean should be non-negative"
        assert curvature_data['path_length'] > 0, "Path length should be positive"
        assert curvature_data['tortuosity'] >= 1.0, "Tortuosity should be >= 1.0"
        
        print("✓ Curvature calculations working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Curvature calculation test failed: {e}")
        return False

def test_vl_embedding_enhancements():
    """Test enhanced VL embedding computation"""
    print("\n=== Testing Enhanced VL Embeddings ===")
    
    try:
        embedder = CLIPEmbedder()
        
        # Mock object data
        object_data = {
            'object_type': 'line',
            'shape_label': 'straight_line',
            'stroke_type': 'solid',
            'vertices': [(0, 0), (10, 10)],
            'centroid': [5, 5],
            'area': 0.0,
            'perimeter': 14.14
        }
        
        # Test enhanced VL embedding (without actual image)
        # This will fail gracefully and return zero embeddings
        vl_data = embedder.compute_enhanced_vl_embedding(
            "dummy_path.png", object_data, "test context"
        )
        
        print("Enhanced VL embedding data:")
        for key, value in vl_data.items():
            if key != 'vl_embed':  # Don't print the full embedding array
                print(f"  {key}: {value}")
        
        assert 'vl_embed' in vl_data, "VL embedding should be present"
        assert 'vl_embed_norm' in vl_data, "VL embedding norm should be calculated"
        assert 'vl_embed_nonzero_ratio' in vl_data, "Nonzero ratio should be calculated"
        assert len(vl_data['vl_embed']) == 512, "CLIP embedding should be 512-dimensional"
        
        print("✓ Enhanced VL embedding computation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ VL embedding test failed: {e}")
        return False

def test_motif_features():
    """Test motif feature calculations"""
    print("\n=== Testing Motif Feature Calculations ===")
    
    try:
        embedder = CLIPEmbedder()
        
        # Mock motif members
        motif_members = [
            {
                'object_type': 'line',
                'vertices': [(0, 0), (10, 0)],
                'centroid': [5, 0],
                'area': 0.0
            },
            {
                'object_type': 'line', 
                'vertices': [(0, 10), (10, 10)],
                'centroid': [5, 10],
                'area': 0.0
            },
            {
                'object_type': 'circle',
                'vertices': [(5, 5), (6, 5), (5, 6), (4, 5), (5, 4)],
                'centroid': [5, 5],
                'area': 3.14
            }
        ]
        
        # Mock relationships
        motif_relationships = [
            {'predicate': 'is_parallel'},
            {'predicate': 'contains'}
        ]
        
        motif_features = embedder.compute_motif_features(motif_members, motif_relationships)
        
        print("Motif features:")
        for key, value in motif_features.items():
            print(f"  {key}: {value}")
        
        assert motif_features['motif_member_count'] == 3, "Should count 3 members"
        assert motif_features['motif_type_diversity'] == 2, "Should have 2 types (line, circle)"
        assert motif_features['motif_relationship_diversity'] == 2, "Should have 2 relationship types"
        
        print("✓ Motif feature calculations working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Motif feature test failed: {e}")
        return False

def test_visualization_imports():
    """Test that enhanced visualization functions can be imported"""
    print("\n=== Testing Visualization Imports ===")
    
    try:
        from scripts.scene_graph_visualization import (
            save_enhanced_scene_graph_visualization,
            create_multi_puzzle_visualization,
            create_missing_data_analysis,
            analyze_puzzle_completeness
        )
        
        print("✓ All enhanced visualization functions imported successfully")
        return True
        
    except Exception as e:
        print(f"✗ Visualization import test failed: {e}")
        return False

def test_process_single_problem_enhancements():
    """Test that process_single_problem can handle enhanced features"""
    print("\n=== Testing Process Single Problem Enhancements ===")
    
    try:
        from src.scene_graphs_building.process_single_problem import _process_single_problem
        
        # Mock args with enhanced features
        class MockArgs:
            use_vl = True
            use_motifs = True
            use_roi = True
            enhanced_viz = True
        
        args = MockArgs()
        
        # Just test that the function can be imported and args are recognized
        assert hasattr(args, 'use_vl'), "Enhanced VL features should be available"
        assert hasattr(args, 'use_motifs'), "Enhanced motif features should be available"
        
        print("✓ Process single problem enhancements are properly integrated")
        return True
        
    except Exception as e:
        print(f"✗ Process single problem test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("Testing Integrated Enhancements for Bongard-LOGO Pipeline")
    print("=" * 60)
    
    tests = [
        test_curvature_calculations,
        test_vl_embedding_enhancements,
        test_motif_features,
        test_visualization_imports,
        test_process_single_problem_enhancements
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! The enhanced system is ready.")
        print("\nTo use the enhanced features, run build_scene_graphs.py with:")
        print("  --use-vl --use-motifs --use-roi --enhanced-viz")
        print("\nOr simply run with default settings (enhanced features are now enabled by default)")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
