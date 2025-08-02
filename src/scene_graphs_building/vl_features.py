import clip
import torch
import logging
from PIL import Image
import numpy as np

class CLIPEmbedder:

    def __init__(self, device='cpu'):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.device = device
        self._feature_cache = {}  # (path, bbox tuple) -> feature

    def embed_image(self, image_or_path, bounding_box=None, mask=None, fallback_global=True, logo_object_data=None):
        """
        Embed an image using LOGO mode object data for precise ROI extraction.
        Integrated with LOGO action program parsing for semantic-aware visual features.
        
        Args:
            image_or_path: PIL Image object or path string
            bounding_box: Optional bounding box for cropping (derived from LOGO vertices)
            mask: Optional mask for background removal (generated from LOGO stroke paths)
            fallback_global: Whether to fallback to global image if crop fails
            logo_object_data: Dictionary containing LOGO-specific object data (vertices, stroke_type, etc.)
        """
        try:
            # Handle both PIL Image objects and file paths with robust path mapping
            if isinstance(image_or_path, str):
                from src.scene_graphs_building.data_loading import remap_path, robust_image_open
                path = remap_path(image_or_path)
                cache_key = (path, tuple(bounding_box) if bounding_box is not None else None)
                if cache_key in self._feature_cache:
                    return self._feature_cache[cache_key]
                img = robust_image_open(path).convert('RGB')
            else:
                img = image_or_path.convert('RGB') if image_or_path.mode != 'RGB' else image_or_path
                cache_key = None
            
            # LOGO MODE: Use precise object geometry for ROI extraction
            if logo_object_data and logo_object_data.get('vertices'):
                vertices = logo_object_data['vertices']
                object_type = logo_object_data.get('object_type', 'unknown')
                
                # Generate precise bounding box from LOGO vertices
                if not bounding_box and len(vertices) >= 2:
                    x_coords = [v[0] for v in vertices]
                    y_coords = [v[1] for v in vertices]
                    # Add padding based on stroke type
                    padding = 10 if object_type in ['line', 'curve'] else 5
                    bounding_box = (
                        max(0, min(x_coords) - padding),
                        max(0, min(y_coords) - padding),
                        min(img.width, max(x_coords) + padding),
                        min(img.height, max(y_coords) + padding)
                    )
                
                # Generate stroke-aware mask for lines and curves
                if not mask and object_type in ['line', 'curve'] and len(vertices) >= 2:
                    mask = self._generate_stroke_mask(vertices, img.size, stroke_width=8)
            
            # Apply ROI cropping with LOGO-derived bounds
            if bounding_box is not None:
                x1, y1, x2, y2 = [int(round(x)) for x in bounding_box]
                # Validate crop bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 > x1 and y2 > y1:
                    img = img.crop((x1, y1, x2, y2))
            
            # Apply LOGO-generated mask for background suppression
            if mask is not None:
                mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(img.size, Image.NEAREST)
                img_np = np.array(img)
                if img_np.shape[-1] == 3:
                    img_np = np.concatenate([img_np, 255 * np.ones((*img_np.shape[:2], 1), dtype=np.uint8)], axis=-1)
                img_np[..., :3][mask_img == 0] = 255  # white background
                img_np[..., 3][mask_img == 0] = 0     # transparent alpha
                img = Image.fromarray(img_np)
            
            # Quality check: ensure minimum viable image size
            min_size = 16  # Increased from 8 for better feature quality
            if img.width < min_size or img.height < min_size:
                if fallback_global and isinstance(image_or_path, str):
                    from src.scene_graphs_building.data_loading import robust_image_open
                    img = robust_image_open(image_or_path).convert('RGB')
                else:
                    # Create minimal valid image from LOGO data
                    img = img.resize((min_size, min_size), Image.LANCZOS)
            
            # Extract CLIP features
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(image_input)
            feat_np = feat.cpu().numpy().flatten()
            
            # Cache results for efficiency
            if cache_key is not None:
                self._feature_cache[cache_key] = feat_np
                
            return feat_np
            
        except Exception as e:
            logging.warning(f"CLIP embedding failed for LOGO object: {e}")
            return np.zeros(512)
    
    def _generate_stroke_mask(self, vertices, image_size, stroke_width=8):
        """Generate a binary mask for stroke objects using LOGO vertex data"""
        try:
            from PIL import Image, ImageDraw
            import numpy as np
            
            mask = Image.new('L', image_size, 0)
            draw = ImageDraw.Draw(mask)
            
            if len(vertices) >= 2:
                # Convert vertices to PIL format
                coords = [(int(v[0]), int(v[1])) for v in vertices]
                
                # Draw stroke path with specified width
                if len(coords) == 2:
                    draw.line(coords, fill=255, width=stroke_width)
                else:
                    # For multi-point paths, draw connected segments
                    for i in range(len(coords) - 1):
                        draw.line([coords[i], coords[i + 1]], fill=255, width=stroke_width)
            
            return np.array(mask) / 255.0
            
        except Exception as e:
            logging.warning(f"Failed to generate stroke mask: {e}")
            return None

    def contrastive_edges(self, objects, threshold=0.15, use_roi=False, logo_mode=True):
        """
        Generate contrastive edges using LOGO mode object data for precise visual similarity.
        Completely integrated with LOGO action program parsing and stroke geometry.
        
        Args:
            objects: List of objects with LOGO-derived geometry and metadata
            threshold: Similarity threshold for edge creation
            use_roi: Use region-of-interest cropping based on LOGO vertices
            logo_mode: Enable LOGO-specific processing (always True in new integration)
        """
        if not objects:
            return []
            
        try:
            feats = []
            valid_objects = []
            
            for o in objects:
                try:
                    # Prioritize pre-computed VL embeddings
                    if o.get('vl_embed') is not None:
                        feat = o['vl_embed']
                    else:
                        # Extract features using LOGO object data
                        logo_data = {
                            'vertices': o.get('vertices', []),
                            'object_type': o.get('object_type', 'unknown'),
                            'action_command': o.get('action_command', ''),
                            'stroke_type': o.get('stroke_type', ''),
                            'is_closed': o.get('is_closed', False)
                        }
                        
                        # Use LOGO-derived ROI and masking
                        if use_roi and logo_mode:
                            feat = self.embed_image(
                                o.get('image_path', ''),
                                bounding_box=o.get('bounding_box'),
                                mask=None,  # Will be generated from LOGO vertices
                                fallback_global=True,
                                logo_object_data=logo_data
                            )
                        else:
                            # Global image embedding with LOGO context
                            feat = self.embed_image(
                                o.get('image_path', ''),
                                bounding_box=None,
                                mask=None,
                                fallback_global=True,
                                logo_object_data=logo_data
                            )
                    
                    if feat is not None and len(feat) > 0 and not np.allclose(feat, 0):
                        feats.append(feat)
                        valid_objects.append(o)
                    
                except Exception as e:
                    logging.warning(f"Failed to compute LOGO VL features for object {o.get('object_id', 'unknown')}: {e}")
                    continue
            
            if len(valid_objects) < 2:
                return []
            
            # Compute similarity matrix with LOGO-aware normalization
            feats = np.stack(feats)
            normed = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
            sim = np.dot(normed, normed.T)
            
            edges = []
            n = len(valid_objects)
            
            for i in range(n):
                for j in range(i+1, n):
                    similarity = float(sim[i, j])
                    
                    # LOGO MODE: Enhanced similarity thresholding based on object types
                    obj_i = valid_objects[i]
                    obj_j = valid_objects[j]
                    
                    # Adjust threshold based on LOGO object characteristics
                    adjusted_threshold = threshold
                    if obj_i.get('object_type') == obj_j.get('object_type'):
                        adjusted_threshold *= 0.9  # Lower threshold for same types
                    
                    if obj_i.get('stroke_type') == obj_j.get('stroke_type') and obj_i.get('stroke_type'):
                        adjusted_threshold *= 0.85  # Even lower for same stroke types
                    
                    if similarity > (1 - adjusted_threshold):
                        id_i = obj_i.get('object_id', obj_i.get('id'))
                        id_j = obj_j.get('object_id', obj_j.get('id'))
                        
                        # Enhanced edge metadata with LOGO context
                        edge_data = {
                            'source': id_i,
                            'target': id_j,
                            'predicate': 'visual_similarity',
                            'similarity': similarity,
                            'source_type': 'vision_language_logo',
                            'logo_context': {
                                'same_object_type': obj_i.get('object_type') == obj_j.get('object_type'),
                                'same_stroke_type': obj_i.get('stroke_type') == obj_j.get('stroke_type'),
                                'both_closed': obj_i.get('is_closed', False) and obj_j.get('is_closed', False),
                                'geometric_similarity': self._calculate_geometric_similarity(obj_i, obj_j)
                            }
                        }
                        
                        edges.append(edge_data)
            
            return edges
            
        except Exception as e:
            logging.warning(f"LOGO contrastive edge computation failed: {e}")
            return []
    
    def _calculate_geometric_similarity(self, obj_a, obj_b):
        """Calculate geometric similarity between LOGO objects"""
        try:
            # Compare key geometric properties
            features_a = {
                'area': obj_a.get('area', 0),
                'perimeter': obj_a.get('perimeter', 0),
                'aspect_ratio': obj_a.get('aspect_ratio', 1),
                'compactness': obj_a.get('compactness', 0),
                'vertex_count': len(obj_a.get('vertices', []))
            }
            
            features_b = {
                'area': obj_b.get('area', 0),
                'perimeter': obj_b.get('perimeter', 0),
                'aspect_ratio': obj_b.get('aspect_ratio', 1),
                'compactness': obj_b.get('compactness', 0),
                'vertex_count': len(obj_b.get('vertices', []))
            }
            
            # Calculate normalized differences
            similarities = []
            for key in features_a:
                val_a = features_a[key]
                val_b = features_b[key]
                if val_a == 0 and val_b == 0:
                    similarities.append(1.0)
                elif max(val_a, val_b) == 0:
                    similarities.append(0.0)
                else:
                    similarities.append(1.0 - abs(val_a - val_b) / max(val_a, val_b))
            
            return float(np.mean(similarities))
            
        except Exception as e:
            logging.warning(f"Failed to calculate geometric similarity: {e}")
            return 0.0
