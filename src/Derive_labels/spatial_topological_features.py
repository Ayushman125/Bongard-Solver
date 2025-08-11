"""
Spatial-Relational & Topological Feature Extraction Module
Implements graph metrics, adjacency, intersection, planarity, and related features for Bongard-Solver.
"""
import numpy as np
import networkx as nx
from shapely.geometry import Polygon, LineString

def compute_spatial_topological_features(image_dict):
    """
    Compute spatial-relational and topological features for a Bongard image.
    Args:
        image_dict (dict): Dict with 'vertices', 'strokes', etc.
    Returns:
        dict: Features (keys: see 2.1–2.20)
    """
    features = {}
    strokes = image_dict.get('strokes', [])
    vertices = image_dict.get('vertices', [])
    # 2.1 Stroke Adjacency Graph Metrics
    G = nx.Graph()
    for i, stroke in enumerate(strokes):
        G.add_node(i)
        for j, other in enumerate(strokes):
            if i != j:
                v1 = np.array(stroke.get('vertices', []))
                v2 = np.array(other.get('vertices', []))
                if v1.size and v2.size:
                    min_dist = np.min(np.linalg.norm(v1[:,None]-v2[None,:], axis=2))
                    if min_dist < 0.05:
                        G.add_edge(i, j)
    features['adjacency_degree'] = dict(G.degree())
    features['adjacency_clustering'] = nx.clustering(G)
    # 2.2 Intersection Network Motifs
    features['intersection_count'] = sum(1 for i in G.edges)
    # 2.3 Topological Betweenness Centrality
    features['betweenness_centrality'] = nx.betweenness_centrality(G)
    # 2.4 Stroke Containment Graph Score
    features['containment_score'] = sum(1 for i, stroke in enumerate(strokes)
        for j, other in enumerate(strokes) if i != j and Polygon(stroke.get('vertices', [])).contains(Polygon(other.get('vertices', []))))
    # 2.5 Stroke Overlap Heatmap
    features['overlap_heatmap'] = [Polygon(stroke.get('vertices', [])).intersection(Polygon(other.get('vertices', []))).area
        for i, stroke in enumerate(strokes) for j, other in enumerate(strokes) if i != j]
    adjacency_count = 0
    intersection_count = 0
    containment_count = 0
    overlap_area = 0.0
    geometries = []
    for stroke in strokes:
        verts = stroke.get('vertices', [])
        # Professional handling: treat as polygon if valid, else as line if valid
        geom = None
        if len(verts) >= 3:
            try:
                poly = Polygon(verts)
                if poly.is_valid and poly.area > 1e-8:
                    geom = ('polygon', poly)
                else:
                    line = LineString(verts)
                    if line.is_valid and line.length > 1e-8:
                        geom = ('line', line)
            except Exception:
                line = LineString(verts)
                if line.is_valid and line.length > 1e-8:
                    geom = ('line', line)
        elif len(verts) >= 2:
            line = LineString(verts)
            if line.is_valid and line.length > 1e-8:
                geom = ('line', line)
        geometries.append(geom if geom else (None, None))
    # 2.19 Geodesic Distance on Convex Hull
    features['geodesic_dist_convex_hull'] = float(Polygon(vertices).convex_hull.length) if vertices else 0
    # 2.20 Skeleton Graph Features
    features['skeleton_graph_features'] = {'num_nodes': len(G.nodes), 'num_edges': len(G.edges)}
    return features
