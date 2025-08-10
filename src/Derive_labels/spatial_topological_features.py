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
    # 2.6 Geo-Graph Duality Features
    features['geo_graph_duality'] = len(G.nodes) / max(len(vertices), 1)
    # 2.7 Euler Characteristic
    features['euler_characteristic'] = len(G.nodes) - len(G.edges)
    # 2.8 Homology-Based Hole Persistence
    features['hole_persistence'] = sum(1 for stroke in strokes if Polygon(stroke.get('vertices', [])).is_empty == False and Polygon(stroke.get('vertices', [])).area > 0)
    # 2.9 Planarity Score
    features['planarity_score'] = nx.check_planarity(G)[0]
    # 2.10 Connected Component Count
    features['connected_components'] = nx.number_connected_components(G)
    # 2.11 Stroke-to-Centroid Distance Distribution
    centroid = np.mean(np.array(vertices), axis=0) if vertices else np.zeros(2)
    features['stroke_to_centroid_dist'] = [float(np.linalg.norm(np.mean(np.array(stroke.get('vertices', [])), axis=0) - centroid)) for stroke in strokes if stroke.get('vertices')]
    # 2.12 Relative Stroke Orientation Histogram
    features['stroke_orientation_hist'] = [np.arctan2(stroke.get('vertices', [])[1][1] - stroke.get('vertices', [])[0][1], stroke.get('vertices', [])[1][0] - stroke.get('vertices', [])[0][0]) if len(stroke.get('vertices', [])) > 1 else 0 for stroke in strokes]
    # 2.13 Stroke Ringness (cycle detection)
    features['stroke_ringness'] = nx.cycle_basis(G)
    # 2.14 Stroke-Stroke Minimum Distance
    features['stroke_min_dist'] = [min(np.linalg.norm(np.array(stroke.get('vertices', []))[0] - np.array(other.get('vertices', []))[0])) for stroke in strokes for other in strokes if stroke != other]
    # 2.15 Stroke Convex-Hull Graph Ratio
    features['convex_hull_graph_ratio'] = Polygon(vertices).convex_hull.area / max(Polygon(vertices).area, 1e-6) if vertices else 0
    # 2.16 Stroke Pairwise Angle Difference
    features['pairwise_angle_diff'] = [abs(features['stroke_orientation_hist'][i] - features['stroke_orientation_hist'][j]) for i in range(len(strokes)) for j in range(i+1, len(strokes))]
    # 2.17 Stroke-to-Stroke Spatial Entropy
    features['spatial_entropy'] = float(np.std(features['stroke_to_centroid_dist'])) if features['stroke_to_centroid_dist'] else 0
    # 2.18 Polygon Adjacency Matrix Features
    features['adjacency_matrix'] = nx.to_numpy_array(G).tolist()
    # 2.19 Geodesic Distance on Convex Hull
    features['geodesic_dist_convex_hull'] = float(Polygon(vertices).convex_hull.length) if vertices else 0
    # 2.20 Skeleton Graph Features
    features['skeleton_graph_features'] = {'num_nodes': len(G.nodes), 'num_edges': len(G.edges)}
    return features
