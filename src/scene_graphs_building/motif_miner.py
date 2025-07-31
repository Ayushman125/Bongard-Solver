
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
        # Group objects by cluster label
        clusters = {}
        for obj, label in zip(objects, clustering.labels_):
            obj['motif_label'] = int(label)
            # Always assign string shape_label for KB lookups and semantic logic
            obj['shape_label'] = self.MOTIF_LABELS.get(int(label), f"motif_{label}")
            clusters.setdefault(int(label), []).append(obj)
            motif_dict.setdefault(int(label), []).append(obj['id'])
        # For each motif, compute and assign geometry
        for label, members in clusters.items():
            # Only create motif node if more than one member
            if len(members) < 2:
                continue
            # Compute convex hull over all member vertices
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
            # Assign geometry to each motif node (or create a supernode if needed)
            for m in members:
                m['motif_vertices'] = motif_vertices
        logging.info(f"MotifMiner.cluster_motifs: motif_dict keys={list(motif_dict.keys())}")
        return motif_dict

    def create_motif_supernode(self, member_nodes, motif_label, motif_id):
        """
        Create a motif super-node with valid geometry and semantic labels.
        Assigns 'vertices', 'shape_label', 'motif_label', and aggregates features.
        """
        # Aggregate geometry
        vertices = self.aggregate_motif_vertices(member_nodes)
        shape_label = self.MOTIF_LABELS.get(motif_label, f"motif_{motif_label}")
        # Aggregate area, score, and embeddings if available
        area = sum([n.get('area', 0) for n in member_nodes])
        motif_score = sum([n.get('motif_score', 0) for n in member_nodes]) / max(1, len(member_nodes))
        vl_embed = np.mean([n.get('vl_embed', np.zeros(512)) for n in member_nodes], axis=0)
        # Compose super-node
        supernode = {
            'id': motif_id,
            'is_motif': True,
            'vertices': vertices,
            'shape_label': shape_label,
            'motif_label': motif_label,
            'area': area,
            'motif_score': motif_score,
            'vl_embed': vl_embed,
            'members': [n['id'] for n in member_nodes],
        }
        logging.info(f"MotifMiner.create_motif_supernode: Created motif supernode {motif_id} with shape_label={shape_label} and {len(vertices)} vertices.")
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
    def normalize_features(nodes):
        """
        Normalize motif and regular node features separately to avoid scale mismatch.
        """
        motif_nodes = [n for n in nodes if n.get('is_motif')]
        regular_nodes = [n for n in nodes if not n.get('is_motif')]
        # Example normalization: area
        if motif_nodes:
            motif_areas = np.array([n.get('area', 0) for n in motif_nodes])
            motif_area_mean = motif_areas.mean() if len(motif_areas) else 1.0
            for n in motif_nodes:
                n['area_norm'] = n.get('area', 0) / motif_area_mean
        if regular_nodes:
            reg_areas = np.array([n.get('area', 0) for n in regular_nodes])
            reg_area_mean = reg_areas.mean() if len(reg_areas) else 1.0
            for n in regular_nodes:
                n['area_norm'] = n.get('area', 0) / reg_area_mean
        # Add more normalization as needed (motif_score, vl_embed, etc.)

    def aggregate_motif_vertices(self, member_nodes):
        """
        Aggregate motif super-node geometry using only real, data-driven techniques:
        1. Multi-method mask aggregation (union, extract_clean_contours)
        2. Skeleton-based reconstruction from union mask
        3. Multi-scale contour extraction
        4. Convex hull and alpha shape of member vertices
        5. PCA envelope (rectangle/ellipse)
        If all fail, log and return empty array.
        """

        import numpy as np
        import logging
        from scipy.spatial import ConvexHull

        # Log full input and member node geometry for diagnosis
        logging.info(f"MotifMiner.aggregate_motif_vertices: Called with {len(member_nodes)} member_nodes.")
        for i, n in enumerate(member_nodes):
            v = n.get('vertices')
            m = n.get('mask')
            c = n.get('centroid')
            logging.info(f"MotifMiner.aggregate_motif_vertices: member[{i}] id={n.get('id', None)} shape_label={n.get('shape_label', None)} centroid={c} vertices_shape={np.array(v).shape if v is not None else None} mask_shape={getattr(m, 'shape', None)}")
            if v is not None:
                logging.info(f"MotifMiner.aggregate_motif_vertices: member[{i}] vertices sample={v[:5] if len(v) > 5 else v}")
            if m is not None:
                logging.info(f"MotifMiner.aggregate_motif_vertices: member[{i}] mask sample={m[:5,:5] if hasattr(m, 'shape') and m.shape[0] > 5 and m.shape[1] > 5 else m}")

        # 1. Multi-method mask aggregation
        member_masks = [n.get('mask') for n in member_nodes if 'mask' in n and n['mask'] is not None]
        logging.info(f"MotifMiner.aggregate_motif_vertices: member_masks count={len(member_masks)}")
        if member_masks:
            try:
                union_mask = np.logical_or.reduce(member_masks)
                logging.info(f"MotifMiner.aggregate_motif_vertices: union_mask shape={union_mask.shape}")
                from src.scene_graphs_building.feature_extraction import extract_clean_contours
                polygons = extract_clean_contours(union_mask)
                logging.info(f"MotifMiner.aggregate_motif_vertices: extract_clean_contours polygons count={len(polygons)}")
                if polygons and len(polygons[0]) >= 3:
                    logging.info(f"MotifMiner.aggregate_motif_vertices: Using union mask and extract_clean_contours. Output shape={np.array(polygons[0]).shape}")
                    return np.array(polygons[0])
            except Exception as e:
                logging.warning(f"MotifMiner.aggregate_motif_vertices: Mask union/contour extraction failed: {e}")
        # 2. Skeleton-based reconstruction
        if member_masks:
            try:
                import cv2
                from skimage.morphology import skeletonize
                mask = np.logical_or.reduce(member_masks)
                skel = skeletonize(mask > 0)
                skel = skel.astype(np.uint8)
                contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                logging.info(f"MotifMiner.aggregate_motif_vertices: skeleton contours count={len(contours)}")
                if contours:
                    coords = contours[0].reshape(-1,2)
                    logging.info(f"MotifMiner.aggregate_motif_vertices: skeleton coords shape={coords.shape}")
                    if len(coords) >= 3:
                        logging.info("MotifMiner.aggregate_motif_vertices: Using skeleton-based reconstruction.")
                        return coords
            except Exception as e:
                logging.warning(f"MotifMiner.aggregate_motif_vertices: Skeleton-based reconstruction failed: {e}")
        # 3. Multi-scale contour extraction
        if member_masks:
            try:
                import cv2
                from src.scene_graphs_building.feature_extraction import extract_clean_contours
                mask = np.logical_or.reduce(member_masks)
                scales = [1.0, 0.5, 1.5]
                polygons = []
                for scale in scales:
                    mask_scaled = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    polys = extract_clean_contours(mask_scaled)
                    logging.info(f"MotifMiner.aggregate_motif_vertices: scale={scale}, polys count={len(polys)}")
                    polygons.extend(polys)
                polygons = [poly for poly in polygons if len(poly) >= 3]
                logging.info(f"MotifMiner.aggregate_motif_vertices: multi-scale polygons count={len(polygons)}")
                if polygons:
                    logging.info(f"MotifMiner.aggregate_motif_vertices: Using multi-scale contour extraction. Output shape={np.array(polygons[0]).shape}")
                    return np.array(polygons[0])
            except Exception as e:
                logging.warning(f"MotifMiner.aggregate_motif_vertices: Multi-scale contour extraction failed: {e}")
        # 4. Convex hull and alpha shape of member vertices
        valid_members = [n for n in member_nodes if 'vertices' in n and n['vertices'] is not None and np.any(n['vertices'])]
        logging.info(f"MotifMiner.aggregate_motif_vertices: valid_members count={len(valid_members)}")
        if valid_members:
            all_vertices = np.concatenate([n['vertices'] for n in valid_members])
            logging.info(f"MotifMiner.aggregate_motif_vertices: all_vertices shape={all_vertices.shape}")
            if len(all_vertices) >= 3:
                try:
                    hull = ConvexHull(all_vertices)
                    logging.info(f"MotifMiner.aggregate_motif_vertices: Using convex hull of member vertices. Output shape={all_vertices[hull.vertices].shape}")
                    return all_vertices[hull.vertices]
                except Exception as e:
                    logging.warning(f"MotifMiner.aggregate_motif_vertices: Convex hull of member vertices failed: {e}")
            # Alpha shape (concave hull)
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
        # 5. PCA envelope (rectangle/ellipse)
        if valid_members and len(all_vertices) >= 3:
            try:
                coords = all_vertices
                mean = np.mean(coords, axis=0)
                cov = np.cov(coords.T)
                eigvals, eigvecs = np.linalg.eig(cov)
                order = np.argsort(eigvals)[::-1]
                eigvecs = eigvecs[:, order]
                width, height = np.sqrt(eigvals[order]) * 5
                rect = np.array([
                    mean + width * eigvecs[:,0] + height * eigvecs[:,1],
                    mean + width * eigvecs[:,0] - height * eigvecs[:,1],
                    mean - width * eigvecs[:,0] - height * eigvecs[:,1],
                    mean - width * eigvecs[:,0] + height * eigvecs[:,1]
                ])
                logging.info(f"MotifMiner.aggregate_motif_vertices: Using PCA envelope. Output shape={rect.shape}")
                return rect
            except Exception as e:
                logging.warning(f"MotifMiner.aggregate_motif_vertices: PCA envelope failed: {e}")
        # If all real geometry methods fail, log and return empty array
        logging.warning("MotifMiner.aggregate_motif_vertices: All real geometry aggregation methods failed. Returning empty vertices list.")
        return np.zeros((0,2))
