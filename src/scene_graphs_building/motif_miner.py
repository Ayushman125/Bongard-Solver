import numpy as np
from sklearn.cluster import DBSCAN
from itertools import combinations
import logging
from scipy.spatial import ConvexHull


from src.scene_graphs_building.config import SHAPE_MAP, COMMONSENSE_LABEL_MAP, MOTIF_CATEGORIES, CONCEPTNET_KEEP_RELS

from sklearn.cluster import SpectralClustering
from shapely.geometry import LineString
from src.scene_graphs_building.feature_extraction import compute_physics_attributes

class MotifMiner:
    @staticmethod
    def aggregate_motif_vertices(members):
        # Only aggregate from valid members
        valid_members = [m for m in members if m.get('geometry_valid', False) and m.get('vertices') is not None and len(m.get('vertices')) >= 3]
        if not valid_members:
            return None
        all_vertices = []
        for m in valid_members:
            all_vertices.extend(m['vertices'])
        return all_vertices if all_vertices else None

    @staticmethod
    def propagate_motif_validity(motif_node, member_nodes):
        # Motif is valid if any member is valid
        motif_node['geometry_valid'] = any(m.get('geometry_valid', False) for m in member_nodes)
        motif_node['feature_valid'] = {}
        for key in ['centroid_valid','area_valid','orientation_valid','aspect_ratio_valid','perimeter_valid','compactness_valid','convexity_valid','inertia_valid','num_segments_valid','num_junctions_valid','curvature_valid','skeleton_length_valid','symmetry_axis_valid']:
            motif_node['feature_valid'][key] = any(m.get('feature_valid', {}).get(key, False) for m in member_nodes)
    def decompose(self, vertices):
        # Dummy: return input as one motif
        return [vertices]

    # Use all shape/categorical concepts from config
    from src.scene_graphs_building.config import SHAPE_MAP, COMMONSENSE_LABEL_MAP, MOTIF_CATEGORIES, CONCEPTNET_KEEP_RELS
    MOTIF_LABELS = {i: k for i, k in enumerate(SHAPE_MAP.keys())}

    def cluster_motifs(self, objects, method='logo_aware'):
        """
        LOGO MODE: Cluster motifs using LOGO action program data and stroke connectivity.
        Completely rewritten to use LOGO-derived geometry and action sequence data.
        """
        # Log LOGO object structure for debugging
        for i, o in enumerate(objects[:3]):  # Log first 3 for brevity
            logging.info(f"MotifMiner.cluster_motifs: LOGO object[{i}] type={type(o)}, keys={list(o.keys())}")
        
        # LOGO MODE: Extract objects with valid LOGO-derived geometry
        valid_objects = []
        logo_features = []
        
        for o in objects:
            # Prioritize LOGO-specific attributes
            vertices = o.get('vertices', [])
            object_type = o.get('object_type', 'unknown')
            action_index = o.get('action_index', -1)
            parent_shape_id = o.get('parent_shape_id', '')
            action_command = o.get('action_command', '')
            
            # Only process objects with valid LOGO geometry
            if vertices and len(vertices) >= 2 and object_type in ['line', 'arc', 'polygon']:
                # Extract LOGO-specific features for clustering
                features = [
                    len(vertices),                                    # Vertex count
                    1.0 if object_type == 'line' else 0.0,          # Type indicators
                    1.0 if object_type == 'arc' else 0.0,
                    1.0 if object_type == 'polygon' else 0.0,
                    action_index if action_index >= 0 else -1,       # Sequence order
                    o.get('area', 0.0),                              # Geometric properties
                    o.get('perimeter', 0.0),
                    o.get('aspect_ratio', 1.0),
                    o.get('horizontal_asymmetry', 0.0),              # LOGO-specific asymmetry
                    o.get('vertical_asymmetry', 0.0)
                ]
                
                # Add action command encoding
                if 'line_' in action_command:
                    features.extend([1.0, 0.0, 0.0])  # line type
                elif 'arc_' in action_command:
                    features.extend([0.0, 1.0, 0.0])  # arc type
                else:
                    features.extend([0.0, 0.0, 1.0])  # other/composite
                
                valid_objects.append(o)
                logo_features.append(features)
        
        if len(valid_objects) < 2:
            logging.info("MotifMiner.cluster_motifs: Insufficient valid LOGO objects for clustering")
            return {}, []
        
        # LOGO MODE: Cluster based on action sequence and connectivity
        features_array = np.array(logo_features)
        
        # Group by parent shape first (LOGO action sequences)
        shape_groups = {}
        for i, obj in enumerate(valid_objects):
            shape_id = obj.get('parent_shape_id', 'unknown')
            if shape_id not in shape_groups:
                shape_groups[shape_id] = []
            shape_groups[shape_id].append((i, obj))
        
        # Create motifs based on LOGO connectivity and action sequences
        motif_dict = {}
        motif_nodes = []
        motif_id = 0
        
        for shape_id, members in shape_groups.items():
            if len(members) >= 2:  # Only create motifs for multi-stroke shapes
                member_objects = [m[1] for m in members]
                member_indices = [m[0] for m in members]
                
                # Aggregate LOGO geometry from all members
                all_vertices = []
                for obj in member_objects:
                    all_vertices.extend(obj.get('vertices', []))
                
                if all_vertices:
                    # Compute motif properties from LOGO data
                    motif_node = {
                        'object_id': f'motif_{motif_id}',
                        'motif_id': motif_id,
                        'is_motif': True,
                        'node_type': 'motif',
                        'object_type': 'motif',
                        'source': 'logo_motif',
                        'vertices': all_vertices,
                        'member_count': len(member_objects),
                        'parent_shape_id': shape_id,
                        'member_types': [obj.get('object_type', 'unknown') for obj in member_objects],
                        'action_sequence': [obj.get('action_command', '') for obj in member_objects],
                        'stroke_count': len(member_objects)
                    }
                    
                    # Compute physics attributes using LOGO vertices
                    from src.scene_graphs_building.feature_extraction import compute_physics_attributes
                    compute_physics_attributes(motif_node)
                    
                    # Add LOGO-specific motif features
                    motif_node.update({
                        'motif_type': self._classify_logo_motif(member_objects),
                        'connectivity_pattern': self._analyze_logo_connectivity(member_objects),
                        'symmetry_type': self._detect_logo_symmetry(member_objects),
                        'action_complexity': len(set(obj.get('action_command', '') for obj in member_objects))
                    })
                    
                    # LOGO MODE: Add VL similarity computation using actual image data
                    try:
                        from src.scene_graphs_building.vl_features import CLIPEmbedder
                        clip_embedder = CLIPEmbedder()
                        
                        # Use first member's image path for motif embedding
                        image_path = next((obj.get('image_path') for obj in member_objects if obj.get('image_path')), None)
                        if image_path:
                            # Create precise ROI from aggregated vertices
                            x_coords = [v[0] for v in all_vertices]
                            y_coords = [v[1] for v in all_vertices]
                            bounding_box = (
                                min(x_coords) - 10, min(y_coords) - 10,
                                max(x_coords) + 10, max(y_coords) + 10
                            )
                            
                            logo_data = {
                                'vertices': all_vertices,
                                'object_type': 'motif',
                                'stroke_count': len(member_objects)
                            }
                            
                            vl_embed = clip_embedder.embed_image(
                                image_path, 
                                bounding_box=bounding_box,
                                logo_object_data=logo_data
                            )
                            
                            motif_node['vl_embed'] = vl_embed
                            motif_node['clip_sim'] = float(np.linalg.norm(vl_embed))
                            motif_node['vl_sim'] = motif_node['clip_sim']
                        else:
                            motif_node['vl_embed'] = None
                            motif_node['clip_sim'] = None
                            motif_node['vl_sim'] = None
                            
                    except Exception as e:
                        logging.warning(f"MotifMiner: VL embedding failed for motif {motif_id}: {e}")
                        motif_node['vl_embed'] = None
                        motif_node['clip_sim'] = None
                        motif_node['vl_sim'] = None
                    
                    # Propagate validity from member objects
                    self.propagate_motif_validity(motif_node, member_objects)
                    
                    # Store motif
                    motif_dict[motif_id] = [m.get('object_id') for m in member_objects]
                    motif_nodes.append(motif_node)
                    motif_id += 1
        
        logging.info(f"MotifMiner.cluster_motifs: motif_dict keys={list(motif_dict.keys())}")
        return motif_dict, motif_nodes
    
    def _classify_logo_motif(self, member_objects):
        """Classify motif type based on LOGO action patterns"""
        object_types = [obj.get('object_type', 'unknown') for obj in member_objects]
        action_commands = [obj.get('action_command', '') for obj in member_objects]
        
        # Analyze action pattern
        if all('line_' in cmd for cmd in action_commands):
            return 'line_sequence'
        elif all('arc_' in cmd for cmd in action_commands):
            return 'arc_sequence'
        elif any('line_' in cmd for cmd in action_commands) and any('arc_' in cmd for cmd in action_commands):
            return 'mixed_sequence'
        elif len(set(object_types)) == 1:
            return f"{object_types[0]}_cluster"
        else:
            return 'composite_shape'
    
    def _analyze_logo_connectivity(self, member_objects):
        """Analyze connectivity pattern from LOGO action sequence"""
        connections = 0
        for i in range(len(member_objects) - 1):
            curr_obj = member_objects[i]
            next_obj = member_objects[i + 1]
            
            # Check if endpoints are connected (from LOGO action sequence)
            curr_action_idx = curr_obj.get('action_index', -1)
            next_action_idx = next_obj.get('action_index', -1)
            
            if abs(curr_action_idx - next_action_idx) == 1:
                connections += 1
        
        # Return connectivity ratio
        max_connections = len(member_objects) - 1
        return connections / max_connections if max_connections > 0 else 0.0
    
    def _detect_logo_symmetry(self, member_objects):
        """Detect symmetry patterns in LOGO action sequence"""
        if len(member_objects) < 2:
            return 'none'
        
        # Check for action command symmetry
        action_commands = [obj.get('action_command', '') for obj in member_objects]
        
        # Check if sequence is palindromic (symmetric)
        if action_commands == action_commands[::-1]:
            return 'palindromic'
        
        # Check for repeated patterns
        if len(set(action_commands)) < len(action_commands):
            return 'repetitive'
        
        # Check for geometric symmetry using vertices
        centroids = []
        for obj in member_objects:
            vertices = obj.get('vertices', [])
            if vertices:
                centroid = [sum(v[0] for v in vertices) / len(vertices), 
                           sum(v[1] for v in vertices) / len(vertices)]
                centroids.append(centroid)
        
        if len(centroids) >= 2:
            # Simple check: if centroids are roughly collinear, might be symmetric
            if len(centroids) == 2:
                return 'bilateral'
            # For more complex symmetry detection, could add more analysis
        
        return 'asymmetric'
        valid_objects = []
        logo_features = []
        
        for o in objects:
            # Prioritize LOGO-specific attributes
            vertices = o.get('vertices', [])
            object_type = o.get('object_type', 'unknown')
            action_index = o.get('action_index', -1)
            parent_shape_id = o.get('parent_shape_id', '')
            
            if not vertices or len(vertices) < 2:
                continue
            
            # Calculate LOGO-aware features for clustering
            logo_feature_vector = self._extract_logo_features(o)
            if logo_feature_vector is not None:
                valid_objects.append(o)
                logo_features.append(logo_feature_vector)
        
        if len(valid_objects) < 2:
            logging.warning("MotifMiner: Insufficient valid LOGO objects for clustering.")
            return {}, []
        
        logo_features = np.array(logo_features)
        
        # LOGO MODE: Use action sequence and connectivity for clustering
        if method == 'logo_aware':
            clustering_labels = self._logo_aware_clustering(valid_objects, logo_features)
        elif method == 'action_sequence':
            clustering_labels = self._action_sequence_clustering(valid_objects)
        else:
            # Fallback to spatial clustering with LOGO features
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=25, min_samples=1).fit(logo_features)
            clustering_labels = clustering.labels_
        
        # Build motif dictionary and enhance objects with LOGO context
        motif_dict = {}
        clusters = {}
        
        for obj, label in zip(valid_objects, clustering_labels):
            obj['motif_label'] = int(label)
            
            # LOGO MODE: Enhanced semantic labeling using action commands
            action_cmd = obj.get('action_command', '')
            stroke_type = obj.get('stroke_type', '')
            object_type = obj.get('object_type', 'unknown')
            
            # Determine semantic label from LOGO action context
            if 'line' in action_cmd or object_type == 'line':
                semantic_label = 'line_stroke'
            elif 'arc' in action_cmd or object_type == 'curve':
                semantic_label = 'arc_stroke'
            elif object_type == 'polygon':
                semantic_label = 'polygon_shape'
            else:
                semantic_label = f"logo_{object_type}"
            
            obj['shape_label'] = semantic_label
            obj['semantic_label'] = COMMONSENSE_LABEL_MAP.get(semantic_label.split('_')[0], semantic_label)
            obj['kb_concept'] = obj['semantic_label']
            
            clusters.setdefault(int(label), []).append(obj)
            motif_dict.setdefault(int(label), []).append(obj.get('object_id', obj.get('id')))
        
        # Create LOGO-aware motif super-nodes
        motif_nodes = []
        for label, members in clusters.items():
            if len(members) < 2:
                continue
            motif_id = f"motif_{label}"
            member_ids = [m.get('object_id', m.get('id')) for m in members]
            # LOGO MODE: Aggregate geometry using precise LOGO vertices
            motif_vertices = self._aggregate_logo_vertices(members)
            # Analyze LOGO action sequence for motif classification
            motif_type = self._classify_logo_motif(members)
            # Create enhanced motif node with LOGO context
            motif_node = {
                'id': motif_id,  # Ensure 'id' key is present for downstream compatibility
                'object_id': motif_id,  # Also add 'object_id' for compatibility
                'object_type': 'motif',
                'motif_type': motif_type,
                'member_nodes': member_ids,
                'vertices': motif_vertices,
                'source': 'logo_motif_mining',
                'logo_context': self._extract_motif_logo_context(members)
            }
            # Calculate aggregate properties from LOGO data
            motif_node.update(self._calculate_motif_properties(members, motif_vertices))
            # LOGO MODE: Compute CLIP/VL features for motifs using precise geometry
            if motif_vertices is not None and len(motif_vertices) >= 3:
                try:
                    from src.scene_graphs_building.vl_features import CLIPEmbedder
                    clip_embedder = CLIPEmbedder()
                    # Use first member's image path for motif embedding
                    sample_member = members[0]
                    image_path = sample_member.get('image_path')
                    if image_path:
                        # Create LOGO-aware bounding box for motif ROI
                        motif_bbox = self._calculate_motif_bbox(motif_vertices, padding=15)
                        logo_motif_data = {
                            'vertices': motif_vertices,
                            'object_type': 'motif',
                            'action_command': f"motif_{motif_type}",
                            'stroke_type': motif_type,
                            'is_closed': len(motif_vertices) > 2 and np.allclose(motif_vertices[0], motif_vertices[-1], atol=1e-5)
                        }
                        vl_embed = clip_embedder.embed_image(
                            image_path,
                            bounding_box=motif_bbox,
                            logo_object_data=logo_motif_data
                        )
                        motif_node['vl_embed'] = vl_embed
                        motif_node['clip_sim'] = float(np.linalg.norm(vl_embed))
                        motif_node['attribute_validity'] = {'clip_sim': 'valid'}
                        logging.info(f"MotifMiner: Computed LOGO VL features for motif {motif_id}")
                except Exception as e:
                    motif_node['vl_embed'] = None
                    motif_node['clip_sim'] = None
                    motif_node['attribute_validity'] = {'clip_sim': 'not_applicable'}
                    logging.warning(f"MotifMiner: LOGO VL computation failed for motif {motif_id}: {e}")
            else:
                motif_node['vl_embed'] = None
                motif_node['clip_sim'] = None
                motif_node['attribute_validity'] = {'clip_sim': 'not_applicable'}
            # Ensure motif validity based on member validity
            self.propagate_motif_validity(motif_node, members)
            motif_nodes.append(motif_node)
        
        logging.info(f"MotifMiner: Created {len(motif_nodes)} LOGO-aware motifs from {len(valid_objects)} objects")
        return motif_dict, motif_nodes

    def create_motif_supernode(self, member_nodes, motif_label, motif_id):
        """
        Create a motif super-node with valid geometry and semantic labels.
        Assigns 'vertices', 'shape_label', 'motif_label', and aggregates features.
        Merges all computed physics attributes for full predicate support.
        """
        # 1. Aggregate geometry
        vertices = self.aggregate_motif_vertices(member_nodes)
        # 2. Compute physics attributes on aggregated vertices
        physics_attrs = {}
        vertices_arr = np.array(vertices)
        is_valid_polygon = False
        if vertices_arr.shape[0] >= 3:
            v = vertices_arr
            if np.linalg.matrix_rank(v - v[0]) >= 2:
                is_valid_polygon = True
        if is_valid_polygon:
            physics_attrs['vertices'] = vertices
            try:
                from src.scene_graphs_building.feature_extraction import compute_physics_attributes
                compute_physics_attributes(physics_attrs)
            except Exception as e:
                logging.warning(f"MotifMiner.create_motif_supernode: compute_physics_attributes failed for motif {motif_id}: {e}")
        else:
            logging.warning(f"MotifMiner.create_motif_supernode: Skipping physics attribute computation for motif {motif_id} due to degenerate or invalid geometry. Vertices: {vertices}")
        # 3. Aggregate semantic cues
        area = sum([n.get('area', 0) for n in member_nodes if n.get('feature_valid', {}).get('area_valid', False)])
        motif_score = sum([n.get('motif_score', 0) for n in member_nodes if n.get('motif_score', None) is not None]) / max(1, len([n for n in member_nodes if n.get('motif_score', None) is not None]))
        vl_embed = np.mean([n.get('vl_embed', np.zeros(512)) for n in member_nodes if n.get('vl_embed', None) is not None], axis=0) if any(n.get('vl_embed', None) is not None for n in member_nodes) else np.zeros(512)
        # 4. Compose supernode, merging all physics attributes
        supernode = {
            'id': motif_id,
            'is_motif': True,
            'shape_label': self.MOTIF_LABELS.get(motif_label, f"motif_{motif_label}"),
            'motif_label': motif_label,
            'members': [n['id'] for n in member_nodes],
            'area': area,
            'motif_score': motif_score,
            'vl_embed': vl_embed,
        }
        # Merge all computed physics attributes into supernode
        supernode.update(physics_attrs)
        # Ensure centroid is present for motif supernode
        if 'cx' in supernode and 'cy' in supernode:
            supernode['centroid'] = [supernode['cx'], supernode['cy']]
        # Validate and fill missing required features, using None for degenerate/inapplicable
        REQUIRED_FEATURES = [
            'curvature', 'skeleton_length', 'symmetry_axis', 'gnn_score',
            'clip_sim', 'motif_score', 'vl_sim'
        ]
        supernode['attribute_validity'] = {}
        for feat in REQUIRED_FEATURES:
            if feat not in supernode:
                if feat in ['curvature', 'skeleton_length', 'symmetry_axis'] and not is_valid_polygon:
                    supernode[feat] = None
                    supernode['attribute_validity'][feat] = 'not_applicable'
                    logging.info(f"MotifMiner.create_motif_supernode: supernode {motif_id} missing '{feat}', set to None (not_applicable, degenerate geometry).")
                else:
                    supernode[feat] = 0.0
                    supernode['attribute_validity'][feat] = 'default_zero'
                    logging.info(f"MotifMiner.create_motif_supernode: supernode {motif_id} missing '{feat}', set to default 0.0.")
        logging.debug(f"MotifMiner.create_motif_supernode: Supernode {motif_id} keys: {list(supernode.keys())}")
        logging.info(f"MotifMiner.create_motif_supernode: Created motif supernode {motif_id} with shape_label={supernode['shape_label']} and {len(vertices)} vertices. Physics keys: {list(physics_attrs.keys())}")
        return supernode

    @staticmethod
    def add_motif_edges(G, motif_supernode, member_nodes):
        """
        Add motif-aware edges to the graph:
        - part_of_motif: from member to motif supernode
        - motif_similarity: between motifs of same type
        """
        motif_id = motif_supernode['id']
        # part_of_motif edges
        for n in member_nodes:
            G.add_edge(n['id'], motif_id, predicate='part_of_motif', source='motif')
        # motif_similarity edges (between motifs of same type)
        for other_id, other_data in G.nodes(data=True):
            if other_id == motif_id:
                continue
            if other_data.get('is_motif') and other_data.get('shape_label') == motif_supernode['shape_label']:
                G.add_edge(motif_id, other_id, predicate='motif_similarity', source='motif')

    @staticmethod
    def normalize_features(nodes, feature_list=None):
        """
        Robust normalization for motif and regular node features. Avoids nan by using epsilon and fallback. Skips nodes with missing/None features.
        """
        eps = 1e-6
        if feature_list is None:
            feature_list = ['area', 'motif_score']
        motif_nodes = [n for n in nodes if n.get('is_motif')]
        regular_nodes = [n for n in nodes if not n.get('is_motif')]
        for feat in feature_list:
            # Motif normalization (skip None values)
            motif_vals = np.array([n.get(feat, 0) for n in motif_nodes if n.get(feat) is not None])
            motif_mean = motif_vals.mean() if len(motif_vals) else 0.0
            motif_std = motif_vals.std() if len(motif_vals) else 0.0
            for n in motif_nodes:
                val = n.get(feat, 0)
                if val is None:
                    n[feat+'_norm'] = 0.0
                    continue
                if motif_std > eps:
                    n[feat+'_norm'] = (val - motif_mean) / (motif_std + eps)
                elif motif_mean > eps:
                    n[feat+'_norm'] = val / (motif_mean + eps)
                else:
                    n[feat+'_norm'] = 0.0
            # Regular normalization (skip None values)
            reg_vals = np.array([n.get(feat, 0) for n in regular_nodes if n.get(feat) is not None])
            reg_mean = reg_vals.mean() if len(reg_vals) else 0.0
            reg_std = reg_vals.std() if len(reg_vals) else 0.0
            for n in regular_nodes:
                val = n.get(feat, 0)
                if val is None:
                    n[feat+'_norm'] = 0.0
                    continue
                if reg_std > eps:
                    n[feat+'_norm'] = (val - reg_mean) / (reg_std + eps)
                elif reg_mean > eps:
                    n[feat+'_norm'] = val / (reg_mean + eps)
                else:
                    n[feat+'_norm'] = 0.0

    def aggregate_motif_vertices(self, member_nodes):
        """
        For Bongard-LOGO/NVLabs: aggregate motif geometry directly from member vertices using ConvexHull.
        Skip mask/contour/skeleton logic unless vertices are missing.
        """
        import numpy as np
        import logging
        from scipy.spatial import ConvexHull

        valid_members = [n for n in member_nodes if 'vertices' in n and n['vertices'] is not None and len(n['vertices']) > 0]
        if valid_members:
            all_vertices = np.concatenate([n['vertices'] for n in valid_members])
            if len(all_vertices) >= 3:
                try:
                    hull = ConvexHull(all_vertices)
                    logging.info(f"MotifMiner.aggregate_motif_vertices: Using convex hull of member vertices. Output shape={all_vertices[hull.vertices].shape}")
                    return all_vertices[hull.vertices]
                except Exception as e:
                    logging.warning(f"MotifMiner.aggregate_motif_vertices: Convex hull of member vertices failed: {e}")
            # Optionally, use alpha-shape for highly non-convex motifs
            if len(all_vertices) >= 4:
                try:
                    from shapely.geometry import Polygon
                    import alphashape
                    alpha_poly = alphashape.alphashape(all_vertices, 0.1)
                    if isinstance(alpha_poly, Polygon):
                        coords = np.array(alpha_poly.exterior.coords)
                        logging.info(f"MotifMiner.aggregate_motif_vertices: Using alpha shape of member vertices. Output shape={coords.shape}")
                        if len(coords) >= 3:
                            return coords
                except Exception as e:
                    logging.warning(f"MotifMiner.aggregate_motif_vertices: Alpha shape failed: {e}")
            # Fallback: if all points are collinear or degenerate, return all points as motif geometry
            if len(all_vertices) >= 2:
                logging.warning("MotifMiner.aggregate_motif_vertices: All points are collinear or degenerate, returning all points as motif geometry.")
                return all_vertices
        logging.warning("MotifMiner.aggregate_motif_vertices: No valid member vertices found. Returning empty vertices list.")
        return np.zeros((0,2))
    
    def find_bridge_patterns(self, graph):
        """Find bridge-like patterns in the scene graph"""
        patterns = []
        
        try:
            nodes = list(graph.nodes(data=True))
            
            # Look for bridge patterns: parallel lines with connecting elements
            parallel_pairs = []
            for i, (id_a, data_a) in enumerate(nodes):
                for j, (id_b, data_b) in enumerate(nodes[i+1:], i+1):
                    # Check if nodes are parallel
                    if graph.has_edge(id_a, id_b):
                        edge_data = graph.get_edge_data(id_a, id_b)
                        if any(d.get('predicate') == 'is_parallel' for d in edge_data.values()):
                            parallel_pairs.append((id_a, id_b))
            
            # For each parallel pair, look for connecting elements
            for id_a, id_b in parallel_pairs:
                connecting_nodes = []
                for node_id, node_data in nodes:
                    if node_id != id_a and node_id != id_b:
                        # Check if this node connects the parallel lines
                        connects_a = graph.has_edge(node_id, id_a) or graph.has_edge(id_a, node_id)
                        connects_b = graph.has_edge(node_id, id_b) or graph.has_edge(id_b, node_id)
                        if connects_a and connects_b:
                            connecting_nodes.append(node_id)
                
                if connecting_nodes:
                    patterns.append({
                        'type': 'bridge',
                        'nodes': [id_a, id_b] + connecting_nodes,
                        'parallel_base': [id_a, id_b],
                        'connectors': connecting_nodes,
                        'confidence': min(1.0, len(connecting_nodes) / 2.0)
                    })
        
        except Exception as e:
            logging.warning(f"Bridge pattern detection failed: {e}")
        
        return patterns
    
    def find_apex_patterns(self, graph):
        """Find apex patterns (pointed/triangular structures)"""
        patterns = []
        
        try:
            nodes = list(graph.nodes(data=True))
            
            # Look for nodes that could be apex points
            for node_id, node_data in nodes:
                if node_data.get('has_prominent_apex', False) or node_data.get('apex_relative_to_center'):
                    # Find nodes connected to this potential apex
                    connected_nodes = []
                    for neighbor in graph.neighbors(node_id):
                        connected_nodes.append(neighbor)
                    
                    # Also check predecessors (for directed edges)
                    for pred in graph.predecessors(node_id):
                        if pred not in connected_nodes:
                            connected_nodes.append(pred)
                    
                    if len(connected_nodes) >= 2:
                        # Check if this forms an apex pattern
                        asymmetry_score = node_data.get('horizontal_asymmetry', 0.0)
                        apex_confidence = 0.5 + asymmetry_score * 0.5
                        
                        patterns.append({
                            'type': 'apex',
                            'nodes': [node_id] + connected_nodes,
                            'apex_node': node_id,
                            'base_nodes': connected_nodes,
                            'confidence': apex_confidence,
                            'asymmetry_score': asymmetry_score
                        })
        
        except Exception as e:
            logging.warning(f"Apex pattern detection failed: {e}")
        
        return patterns
    
    def find_symmetry_patterns(self, graph):
        """Find symmetry patterns in the scene graph"""
        patterns = []
        
        try:
            nodes = list(graph.nodes(data=True))
            
            # Look for symmetrical arrangements
            symmetric_pairs = []
            for i, (id_a, data_a) in enumerate(nodes):
                for j, (id_b, data_b) in enumerate(nodes[i+1:], i+1):
                    # Check for symmetry relationships
                    if graph.has_edge(id_a, id_b):
                        edge_data = graph.get_edge_data(id_a, id_b)
                        if any(d.get('predicate') == 'forms_symmetry' for d in edge_data.values()):
                            symmetric_pairs.append((id_a, id_b))
                    
                    # Also check based on geometric similarity
                    if data_a.get('object_type') == data_b.get('object_type'):
                        area_a = data_a.get('area', 0)
                        area_b = data_b.get('area', 0)
                        if area_a > 0 and area_b > 0:
                            size_ratio = min(area_a, area_b) / max(area_a, area_b)
                            if size_ratio > 0.8:  # Similar size
                                # Check orientation similarity
                                orient_a = data_a.get('orientation', 0)
                                orient_b = data_b.get('orientation', 0)
                                angle_diff = abs(orient_a - orient_b)
                                angle_diff = min(angle_diff, 360 - angle_diff)
                                
                                if angle_diff < 15:  # Similar orientation
                                    symmetric_pairs.append((id_a, id_b))
            
            # Group symmetric pairs into larger patterns
            if symmetric_pairs:
                patterns.append({
                    'type': 'symmetry',
                    'nodes': list(set([node for pair in symmetric_pairs for node in pair])),
                    'symmetric_pairs': symmetric_pairs,
                    'confidence': len(symmetric_pairs) / max(1, len(nodes) // 2)
                })
        
        except Exception as e:
            logging.warning(f"Symmetry pattern detection failed: {e}")
        
        return patterns