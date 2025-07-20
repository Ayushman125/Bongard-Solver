#!/usr/bin/env python3
"""Debug version to capture exact error location"""
import traceback
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))

# Patch the builder to get more detailed error info
def patch_builder_for_debug():
    from src.bongard_generator import builder
    original_filter_render = builder.BongardGenerator._filter_and_render_scenes
    
    def debug_filter_render(self, scenes, rule):
        filtered_scenes = []
        
        for objects, metadata in scenes:
            try:
                # GNN filtering if enabled
                if hasattr(self.cfg, 'use_gnn') and self.cfg.use_gnn and hasattr(self, '_gnn'):
                    from src.bongard_generator.scene_graph import build_scene_graph
                    scene_graph = build_scene_graph(objects, self.cfg)
                    scene_graph = scene_graph.to(self.cfg.device)
                    
                    import torch
                    with torch.no_grad():
                        quality_score = self._gnn(scene_graph).item()
                    
                    # Skip low-quality scenes
                    gnn_threshold = getattr(self.cfg, 'gnn_thresh', 0.5)
                    if quality_score < gnn_threshold:
                        continue
                    
                    metadata['gnn_quality_score'] = quality_score
                
                # Render the scene to an image
                from src.bongard_generator.dataset import create_composite_scene
                print(f"DEBUG: cfg.canvas_size = {self.cfg.canvas_size} (type: {type(self.cfg.canvas_size)})")
                scene_image = create_composite_scene(objects, self.cfg)
                
                # Record coverage information
                if hasattr(self, 'coverage'):
                    self.coverage.record(rule, objects)
                
                # Add final metadata
                metadata['object_count'] = len(objects)
                metadata['render_success'] = True
                
                filtered_scenes.append((scene_image, objects, metadata))
                
            except Exception as e:
                print(f"DEBUG: Error in scene processing: {e}")
                traceback.print_exc()
                continue
        
        return filtered_scenes
    
    builder.BongardGenerator._filter_and_render_scenes = debug_filter_render

try:
    patch_builder_for_debug()
    
    # Now run the main script
    from final_validation import main
    main()
    
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
