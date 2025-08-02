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
            
            # Extract CLIP features with enhanced context
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(image_input)
                # Normalize for better similarity computations
                feat = feat / feat.norm(dim=-1, keepdim=True)
            
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

    def compute_enhanced_vl_embedding(self, image_path, object_data, context_description=""):
        """
        Compute enhanced VL embedding with context and better error handling.
        Handles missing data gracefully and provides quality metrics.
        """
        try:
            # Generate context-aware description
            object_type = object_data.get('object_type', 'unknown')
            shape_class = object_data.get('shape_label', 'unknown')
            stroke_type = object_data.get('stroke_type', 'unknown')
            
            # Create semantic description
            description = f"{object_type} {shape_class} with {stroke_type} stroke"
            if context_description:
                description += f" in context of {context_description}"
            
            # Compute image embedding with object-specific cropping
            vertices = object_data.get('vertices', [])
            bounding_box = None
            if len(vertices) >= 2:
                x_coords = [v[0] for v in vertices]
                y_coords = [v[1] for v in vertices]
                padding = 15  # Larger padding for better context
                bounding_box = (
                    max(0, min(x_coords) - padding),
                    max(0, min(y_coords) - padding),
                    max(x_coords) + padding,
                    max(y_coords) + padding
                )
            
            # Get image embedding
            image_embedding = self.embed_image(
                image_path, 
                bounding_box=bounding_box, 
                logo_object_data=object_data
            )
            
            # Compute text embedding for semantic context
            if hasattr(self, 'model'):
                import clip
                text_tokens = clip.tokenize([description]).to(self.device)
                with torch.no_grad():
                    text_features = self.model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_embedding = text_features.cpu().numpy().flatten()
                
                # Combine image and text features (weighted average)
                combined_embedding = 0.7 * image_embedding + 0.3 * text_embedding
            else:
                combined_embedding = image_embedding
            
            # Compute quality metrics
            embedding_norm = np.linalg.norm(combined_embedding)
            nonzero_ratio = np.count_nonzero(combined_embedding) / len(combined_embedding)
            
            return {
                'vl_embed': combined_embedding.tolist(),
                'vl_embed_norm': float(embedding_norm),
                'vl_embed_nonzero_ratio': float(nonzero_ratio),
                'vl_context_description': description
            }
            
        except Exception as e:
            logging.error(f"Error computing enhanced VL embedding for {image_path}: {e}")
            return {
                'vl_embed': [0.0] * 512,
                'vl_embed_norm': 0.0,
                'vl_embed_nonzero_ratio': 0.0,
                'vl_context_description': 'error_in_computation'
            }

    def compute_curvature_metrics(self, vertices):
        """
        Compute comprehensive curvature metrics for open curves.
        Addresses the missing curvature calculations in the CSV data.
        """
        try:
            if not vertices or len(vertices) < 3:
                return {
                    'curvature_mean': 0.0,
                    'curvature_max': 0.0,
                    'curvature_std': 0.0,
                    'path_length': 0.0,
                    'tortuosity': 1.0,
                    'turning_angle_total': 0.0,
                    'inflection_points': 0
                }
            
            # Convert vertices to numpy array
            vertices_array = np.array(vertices)
            
            # Calculate local curvatures using discrete approximation
            curvatures = []
            for i in range(1, len(vertices_array) - 1):
                p1, p2, p3 = vertices_array[i-1], vertices_array[i], vertices_array[i+1]
                
                # Calculate vectors
                v1 = p2 - p1
                v2 = p3 - p2
                
                # Calculate curvature using cross product method
                cross_prod = np.cross(v1, v2)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                    # Curvature = |cross_product| / (|v1| * |v2|)
                    curvature = abs(cross_prod) / (norm_v1 * norm_v2)
                    curvatures.append(curvature)
            
            curvatures = np.array(curvatures) if curvatures else np.array([0.0])
            
            # Calculate path length
            path_length = 0.0
            for i in range(1, len(vertices_array)):
                path_length += np.linalg.norm(vertices_array[i] - vertices_array[i-1])
            
            # Calculate tortuosity (path_length / euclidean_distance)
            if len(vertices_array) >= 2:
                euclidean_distance = np.linalg.norm(vertices_array[-1] - vertices_array[0])
                tortuosity = path_length / euclidean_distance if euclidean_distance > 1e-6 else 1.0
            else:
                tortuosity = 1.0
            
            # Calculate turning angles
            turning_angles = []
            for i in range(1, len(vertices_array) - 1):
                v1 = vertices_array[i] - vertices_array[i-1]
                v2 = vertices_array[i+1] - vertices_array[i]
                
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical issues
                    angle = np.arccos(cos_angle)
                    turning_angles.append(angle)
            
            turning_angle_total = np.sum(turning_angles) if turning_angles else 0.0
            
            # Count inflection points (sign changes in curvature)
            inflection_points = 0
            if len(curvatures) > 1:
                curvature_diffs = np.diff(curvatures)
                sign_changes = np.diff(np.sign(curvature_diffs))
                inflection_points = np.count_nonzero(sign_changes)
            
            return {
                'curvature_mean': float(np.mean(curvatures)),
                'curvature_max': float(np.max(curvatures)),
                'curvature_std': float(np.std(curvatures)),
                'path_length': float(path_length),
                'tortuosity': float(tortuosity),
                'turning_angle_total': float(turning_angle_total),
                'inflection_points': int(inflection_points)
            }
            
        except Exception as e:
            logging.error(f"Error computing curvature metrics: {e}")
            return {
                'curvature_mean': 0.0,
                'curvature_max': 0.0,
                'curvature_std': 0.0,
                'path_length': 0.0,
                'tortuosity': 1.0,
                'turning_angle_total': 0.0,
                'inflection_points': 0
            }

    def compute_motif_features(self, motif_members, motif_relationships=None):
        """
        Compute comprehensive motif features for groups of objects.
        Addresses missing motif data in CSV files.
        """
        try:
            if not motif_members:
                return self._empty_motif_features()
            
            # Basic motif statistics
            member_count = len(motif_members)
            member_types = [obj.get('object_type', 'unknown') for obj in motif_members]
            member_type_diversity = len(set(member_types))
            
            # Geometric features
            all_vertices = []
            centroids = []
            areas = []
            
            for member in motif_members:
                vertices = member.get('vertices', [])
                if vertices:
                    all_vertices.extend(vertices)
                
                centroid = member.get('centroid')
                if centroid:
                    centroids.append(centroid)
                
                area = member.get('area', 0.0)
                areas.append(area)
            
            # Calculate motif bounding box and spatial distribution
            if all_vertices:
                all_vertices = np.array(all_vertices)
                motif_bbox = [
                    float(np.min(all_vertices[:, 0])),
                    float(np.min(all_vertices[:, 1])),
                    float(np.max(all_vertices[:, 0])),
                    float(np.max(all_vertices[:, 1]))
                ]
                motif_span_x = motif_bbox[2] - motif_bbox[0]
                motif_span_y = motif_bbox[3] - motif_bbox[1]
            else:
                motif_bbox = [0.0, 0.0, 0.0, 0.0]
                motif_span_x = motif_span_y = 0.0
            
            # Calculate centroid spread (spatial regularity)
            centroid_spread = 0.0
            if len(centroids) >= 2:
                centroids_array = np.array(centroids)
                pairwise_distances = []
                for i in range(len(centroids_array)):
                    for j in range(i+1, len(centroids_array)):
                        dist = np.linalg.norm(centroids_array[i] - centroids_array[j])
                        pairwise_distances.append(dist)
                centroid_spread = np.std(pairwise_distances) if pairwise_distances else 0.0
            
            # Structural features from relationships
            internal_connectivity = 0.0
            relationship_diversity = 0
            if motif_relationships:
                relationship_types = [rel.get('predicate', 'unknown') for rel in motif_relationships]
                relationship_diversity = len(set(relationship_types))
                internal_connectivity = len(motif_relationships) / max(1, member_count * (member_count - 1) / 2)
            
            # Size distribution metrics
            area_mean = np.mean(areas) if areas else 0.0
            area_std = np.std(areas) if areas else 0.0
            size_uniformity = 1.0 / (1.0 + area_std) if area_std > 0 else 1.0
            
            return {
                'motif_member_count': member_count,
                'motif_type_diversity': member_type_diversity,
                'motif_spatial_span_x': motif_span_x,
                'motif_spatial_span_y': motif_span_y,
                'motif_centroid_spread': float(centroid_spread),
                'motif_internal_connectivity': float(internal_connectivity),
                'motif_relationship_diversity': relationship_diversity,
                'motif_size_uniformity': float(size_uniformity),
                'motif_total_area': float(np.sum(areas)),
                'motif_bbox': motif_bbox
            }
            
        except Exception as e:
            logging.error(f"Error computing motif features: {e}")
            return self._empty_motif_features()
    
    def _empty_motif_features(self):
        """Return empty motif features dict for error cases"""
        return {
            'motif_member_count': 0,
            'motif_type_diversity': 0,
            'motif_spatial_span_x': 0.0,
            'motif_spatial_span_y': 0.0,
            'motif_centroid_spread': 0.0,
            'motif_internal_connectivity': 0.0,
            'motif_relationship_diversity': 0,
            'motif_size_uniformity': 0.0,
            'motif_total_area': 0.0,
            'motif_bbox': [0.0, 0.0, 0.0, 0.0]
        }
