import numpy as np
from sklearn.cluster import DBSCAN
from itertools import combinations
import logging
from scipy.spatial import ConvexHull


class MotifMiner:
    def decompose(self, vertices):
        # Dummy: return input as one motif
        return [vertices]

    MOTIF_LABELS = {
        0: "cluster",
        1: "ring",
        2: "chain",
        3: "grid"
    }

    def cluster_motifs(self, objects):
        # Log type and keys of each object for debugging
        for i, o in enumerate(objects):
            logging.info(f"MotifMiner.cluster_motifs: object[{i}] type={type(o)}, keys={list(o.keys())}")
        # Cluster by centroid proximity, return dict: motif_label -> [node_ids]
        try:
            centroids = np.array([o['centroid'] for o in objects])
        except Exception as e:
            logging.error(f"MotifMiner.cluster_motifs: Error extracting centroids: {e}")
            return {}
        clustering = DBSCAN(eps=20, min_samples=1).fit(centroids)
        motif_dict = {}
        clusters = {}
        # Record cluster membership and assign string labels
        for obj, label in zip(objects, clustering.labels_):
            obj['motif_label'] = int(label)
            obj['shape_label'] = self.MOTIF_LABELS.get(int(label), f"motif_{label}")
            clusters.setdefault(int(label), []).append(obj)
            # Use 'object_id' for membership
            motif_dict.setdefault(int(label), []).append(obj.get('object_id', obj.get('id')))
        # For each motif, create motif node with member list and geometry
        motif_nodes = []
        for label, members in clusters.items():
            if len(members) < 2:
                continue
            motif_id = f"motif_{label}"
            member_ids = [m.get('object_id', m.get('id')) for m in members]
            # Aggregate geometry from member vertices
            all_vertices = [v for m in members for v in (m.get('vertices') or [])]
            if len(all_vertices) >= 3:
                try:
                    arr = np.array(all_vertices)
                    hull = ConvexHull(arr)
                    motif_vertices = arr[hull.vertices].tolist()
                except Exception as e:
                    logging.warning(f"MotifMiner.cluster_motifs: Convex hull failed for motif {label}: {e}")
                    motif_vertices = all_vertices
            else:
                motif_vertices = all_vertices
            # Compute physics attributes and merge all into motif_node
            physics_attrs = {}
            try:
                from src.scene_graphs_building.feature_extraction import compute_physics_attributes
                compute_physics_attributes(physics_attrs)
                # Use motif geometry for attribute computation
                physics_attrs['vertices'] = motif_vertices
                compute_physics_attributes(physics_attrs)
            except Exception as e:
                logging.warning(f"MotifMiner.cluster_motifs: compute_physics_attributes failed for motif {motif_id}: {e}")
            motif_node = {
                'id': motif_id,
                'is_motif': True,
                'member_nodes': member_ids,
                'shape_label': self.MOTIF_LABELS.get(label, f"motif_{label}"),
                'motif_label': label,
                **physics_attrs
            }
            # Assign placeholder for gnn_score
            motif_node['gnn_score'] = 0.0
            # Aggregate motif_score from member nodes (mean)
            motif_score = sum([m.get('motif_score', 0.0) for m in members]) / max(1, len(members))
            motif_node['motif_score'] = motif_score
            # Ensure centroid is present for motif node
            if 'cx' in motif_node and 'cy' in motif_node:
                motif_node['centroid'] = [motif_node['cx'], motif_node['cy']]
            # --- Compute and assign clip_sim and vl_sim for motif node ---
            try:
                from src.scene_graphs_building.clip_embedder import CLIPEmbedder
                # Use mean vl_embed from member nodes
                vl_embed = np.mean([n.get('vl_embed', np.zeros(512)) for n in members], axis=0)
                motif_node['vl_embed'] = vl_embed
                # Compute similarity to all other motif nodes (or a reference set)
                # For demonstration, self-similarity (should be extended to compare with other motifs)
                clip_embedder = CLIPEmbedder()
                # If motif_node has an image or features, compute CLIP similarity
                # Here, we use vl_embed for both clip_sim and vl_sim for demonstration
                motif_node['clip_sim'] = float(np.linalg.norm(vl_embed))
                motif_node['vl_sim'] = float(np.linalg.norm(vl_embed))
                # For real use, replace with actual similarity computation between motif_node and other nodes
            except Exception as e:
                logging.warning(f"MotifMiner.cluster_motifs: CLIP/VL similarity computation failed for motif {motif_id}: {e}")
            # Validate and fill missing required features
            REQUIRED_FEATURES = [
                'curvature', 'skeleton_length', 'symmetry_axis', 'gnn_score',
                'clip_sim', 'motif_score', 'vl_sim'
            ]
            for feat in REQUIRED_FEATURES:
                if feat not in motif_node:
                    motif_node[feat] = 0.0
                    logging.info(f"MotifMiner.cluster_motifs: motif node {motif_id} missing '{feat}', set to default 0.0.")
            motif_nodes.append(motif_node)
        logging.info(f"MotifMiner.cluster_motifs: motif_dict keys={list(motif_dict.keys())}")
        # Always return both motif_dict and motif_nodes for downstream motif construction
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
        physics_attrs['vertices'] = vertices
        try:
            from src.scene_graphs_building.feature_extraction import compute_physics_attributes
            compute_physics_attributes(physics_attrs)
        except Exception as e:
            logging.warning(f"MotifMiner.create_motif_supernode: compute_physics_attributes failed for motif {motif_id}: {e}")
        # 3. Aggregate semantic cues
        area = sum([n.get('area', 0) for n in member_nodes])
        motif_score = sum([n.get('motif_score', 0) for n in member_nodes]) / max(1, len(member_nodes))
        vl_embed = np.mean([n.get('vl_embed', np.zeros(512)) for n in member_nodes], axis=0)
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
        # Validate and fill missing required features
        REQUIRED_FEATURES = [
            'curvature', 'skeleton_length', 'symmetry_axis', 'gnn_score',
            'clip_sim', 'motif_score', 'vl_sim'
        ]
        for feat in REQUIRED_FEATURES:
            if feat not in supernode:
                supernode[feat] = 0.0
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
        Robust normalization for motif and regular node features. Avoids nan by using epsilon and fallback.
        """
        eps = 1e-6
        if feature_list is None:
            feature_list = ['area', 'motif_score']
        motif_nodes = [n for n in nodes if n.get('is_motif')]
        regular_nodes = [n for n in nodes if not n.get('is_motif')]
        for feat in feature_list:
            # Motif normalization
            motif_vals = np.array([n.get(feat, 0) for n in motif_nodes])
            motif_mean = motif_vals.mean() if len(motif_vals) else 0.0
            motif_std = motif_vals.std() if len(motif_vals) else 0.0
            for n in motif_nodes:
                val = n.get(feat, 0)
                if motif_std > eps:
                    n[feat+'_norm'] = (val - motif_mean) / (motif_std + eps)
                elif motif_mean > eps:
                    n[feat+'_norm'] = val / (motif_mean + eps)
                else:
                    n[feat+'_norm'] = 0.0
            # Regular normalization
            reg_vals = np.array([n.get(feat, 0) for n in regular_nodes])
            reg_mean = reg_vals.mean() if len(reg_vals) else 0.0
            reg_std = reg_vals.std() if len(reg_vals) else 0.0
            for n in regular_nodes:
                val = n.get(feat, 0)
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