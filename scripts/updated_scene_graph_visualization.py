"""
Updated Scene Graph Visualization Script for Bongard-LOGO Puzzles
Handles full puzzle relationships with 12+ images, improved node/edge visualization,
and comprehensive data display including missing calculations.
"""

import os
import logging
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from collections import defaultdict
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Tuple
import colorcet as cc
from matplotlib.colors import ListedColormap

# Configure logging
logging.basicConfig(level=logging.INFO)

__all__ = [
    'save_enhanced_scene_graph_visualization',
    'save_comprehensive_scene_graph_csv',
    'create_puzzle_overview_visualization',
    'analyze_missing_data_report'
]

class BongardVisualizationEngine:
    """Enhanced visualization engine for Bongard puzzles with comprehensive relationship mapping"""
    
    def __init__(self):
        self.color_palette = self._setup_color_palette()
        self.node_styles = self._setup_node_styles()
        self.edge_styles = self._setup_edge_styles()
        
    def _setup_color_palette(self):
        """Setup comprehensive color palette for different node/edge types"""
        return {
            # Node colors by type
            'line': '#FF6B6B',           # Red
            'polygon': '#4ECDC4',        # Teal  
            'curve': '#45B7D1',          # Blue
            'arc': '#96CEB4',            # Light green
            'circle': '#FECA57',         # Yellow
            'open_curve': '#A8E6CF',     # Mint green
            'motif': '#FF8C42',          # Orange
            'grouped_object': '#DDA0DD', # Plum
            
            # Edge colors by predicate type
            'spatial': '#2C3E50',        # Dark blue-gray
            'geometric': '#8E44AD',      # Purple
            'semantic': '#E74C3C',       # Red
            'topological': '#27AE60',    # Green
            'structural': '#F39C12',     # Orange
            'motif_relation': '#E67E22', # Dark orange
            'part_of': '#3498DB',        # Blue
            'unknown': '#95A5A6',        # Gray
            
            # Special highlighting
            'missing_data': '#E8E8E8',   # Light gray
            'invalid_geometry': '#FFCCCB', # Light red
            'high_confidence': '#90EE90'  # Light green
        }
    
    def _setup_node_styles(self):
        """Define node styling parameters"""
        return {
            'line': {'shape': 's', 'size': 300},        # Square
            'polygon': {'shape': 'o', 'size': 400},     # Circle
            'curve': {'shape': '^', 'size': 350},       # Triangle up
            'arc': {'shape': 'v', 'size': 350},         # Triangle down
            'circle': {'shape': 'o', 'size': 450},      # Large circle
            'open_curve': {'shape': 'd', 'size': 375},  # Diamond
            'motif': {'shape': 'h', 'size': 600},       # Hexagon
            'grouped_object': {'shape': 'p', 'size': 500} # Pentagon
        }
    
    def _setup_edge_styles(self):
        """Define edge styling parameters"""
        return {
            'spatial': {'style': '-', 'width': 1.5, 'alpha': 0.7},
            'geometric': {'style': '--', 'width': 2.0, 'alpha': 0.8},
            'semantic': {'style': '-.', 'width': 1.8, 'alpha': 0.9},
            'topological': {'style': ':', 'width': 2.2, 'alpha': 0.8},
            'structural': {'style': '-', 'width': 2.5, 'alpha': 0.9},
            'motif_relation': {'style': '-', 'width': 3.0, 'alpha': 1.0},
            'part_of': {'style': '-', 'width': 2.0, 'alpha': 0.8},
            'unknown': {'style': '-', 'width': 1.0, 'alpha': 0.5}
        }

def _categorize_predicate(predicate: str) -> str:
    """Categorize predicates into broader relationship types"""
    spatial_predicates = {
        'is_above', 'is_below', 'is_left_of', 'is_right_of', 'near_objects', 
        'adjacent_endpoints', 'shares_endpoint', 'contains', 'inside'
    }
    geometric_predicates = {
        'is_parallel', 'is_perpendicular', 'same_orientation', 'intersects',
        'has_length_ratio_imbalance', 'has_tilted_orientation', 'forms_symmetry'
    }
    semantic_predicates = {
        'same_shape_class', 'visual_similarity', 'same_category', 'RelatedTo', 'Synonym'
    }
    topological_predicates = {
        'forms_bridge_pattern', 'forms_apex_pattern', 'forms_symmetry_pattern',
        'forms_x_junction', 'connectivity_pattern'
    }
    structural_predicates = {
        'has_geometric_complexity_difference', 'has_dominant_direction',
        'structural_similarity', 'motif_pattern'
    }
    motif_predicates = {
        'part_of_motif', 'motif_similarity', 'repetition_pattern'
    }
    
    if predicate in spatial_predicates:
        return 'spatial'
    elif predicate in geometric_predicates:
        return 'geometric'
    elif predicate in semantic_predicates:
        return 'semantic'
    elif predicate in topological_predicates:
        return 'topological'
    elif predicate in structural_predicates:
        return 'structural'
    elif predicate in motif_predicates:
        return 'motif_relation'
    elif predicate == 'part_of':
        return 'part_of'
    else:
        return 'unknown'

def save_enhanced_scene_graph_visualization(G, image_path, out_dir, problem_id, 
                                          show_all_relationships=True, 
                                          highlight_missing_data=True):
    """
    Enhanced visualization supporting full puzzle relationships with improved layout and styling
    """
    os.makedirs(out_dir, exist_ok=True)
    engine = BongardVisualizationEngine()
    
    # Convert MultiGraph to simplified graph for visualization
    if hasattr(G, 'is_multigraph') and G.is_multigraph():
        vis_graph = _simplify_multigraph(G)
    else:
        vis_graph = G.copy()
    
    # Analyze graph structure for layout optimization
    node_info = _analyze_node_structure(vis_graph)
    layout_info = _compute_optimal_layout(vis_graph, node_info)
    
    # Create enhanced visualizations
    _create_full_puzzle_visualization(
        vis_graph, image_path, out_dir, problem_id, engine, layout_info, 
        highlight_missing_data
    )
    
    _create_relationship_network_visualization(
        vis_graph, out_dir, problem_id, engine, layout_info,
        show_all_relationships
    )
    
    _create_motif_hierarchy_visualization(
        vis_graph, out_dir, problem_id, engine, layout_info
    )
    
    _create_missing_data_visualization(
        vis_graph, out_dir, problem_id, engine, layout_info
    )

def _simplify_multigraph(G):
    """Convert MultiGraph to simple graph by aggregating edge predicates"""
    simple_G = nx.Graph() if not G.is_directed() else nx.DiGraph()
    simple_G.add_nodes_from(G.nodes(data=True))
    
    edge_aggregates = defaultdict(list)
    for u, v, data in G.edges(data=True):
        predicate = data.get('predicate', 'unknown')
        edge_aggregates[(u, v)].append(predicate)
    
    for (u, v), predicates in edge_aggregates.items():
        # Aggregate predicates and calculate relationship strength
        predicate_counts = defaultdict(int)
        for pred in predicates:
            predicate_counts[pred] += 1
        
        # Primary predicate is most frequent
        primary_predicate = max(predicate_counts.keys(), key=predicate_counts.get)
        
        # Edge attributes
        edge_data = {
            'predicate': primary_predicate,
            'predicate_list': list(set(predicates)),
            'predicate_count': len(predicates),
            'relationship_strength': len(predicates) / 10.0,  # Normalize
            'predicate_category': _categorize_predicate(primary_predicate)
        }
        
        simple_G.add_edge(u, v, **edge_data)
    
    return simple_G

def _analyze_node_structure(G):
    """Analyze node structure for optimal visualization"""
    node_info = {}
    
    for node, data in G.nodes(data=True):
        object_type = data.get('object_type', 'unknown')
        is_motif = data.get('is_motif', False)
        source = data.get('source', 'unknown')
        
        # Check for missing data
        missing_fields = []
        if not data.get('vl_embed') or all(x == 0 for x in data.get('vl_embed', [])):
            missing_fields.append('vl_embed')
        if not data.get('centroid'):
            missing_fields.append('centroid')
        if data.get('curvature_score', 0) == 0 and object_type in ['curve', 'open_curve']:
            missing_fields.append('curvature_score')
        if not data.get('motif_id') and is_motif:
            missing_fields.append('motif_id')
        
        node_info[node] = {
            'object_type': object_type,
            'is_motif': is_motif,
            'source': source,
            'missing_fields': missing_fields,
            'importance': _calculate_node_importance(G, node),
            'position_hint': data.get('centroid', (0, 0))
        }
    
    return node_info

def _calculate_node_importance(G, node):
    """Calculate node importance based on connectivity and attributes"""
    degree = G.degree(node)
    
    # Factor in node attributes
    data = G.nodes[node]
    importance = degree
    
    if data.get('is_motif'):
        importance += 5  # Motifs are more important
    if data.get('source') == 'geometric_grouping':
        importance += 2  # Grouped objects are important
    if data.get('gnn_score', 0) > 0.5:
        importance += 1  # High GNN score
        
    return importance

def _compute_optimal_layout(G, node_info):
    """Compute optimal layout considering node importance and relationships"""
    # Use spring layout with custom positioning
    pos = nx.spring_layout(G, k=3, iterations=100)
    
    # Adjust positions based on node importance and missing data
    for node, info in node_info.items():
        if info['position_hint'] != (0, 0):
            # Use actual centroid for positioning hint
            centroid = np.array(info['position_hint'])
            centroid_norm = centroid / (np.max(centroid) + 1e-6)  # Normalize
            pos[node] = centroid_norm
    
    return {
        'pos': pos,
        'node_info': node_info
    }

def _create_full_puzzle_visualization(G, image_path, out_dir, problem_id, engine, layout_info, highlight_missing):
    """Create comprehensive puzzle visualization with all relationships"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    
    pos = layout_info['pos']
    node_info = layout_info['node_info']
    
    # Load and display background image if available
    if image_path and os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            ax.imshow(img, alpha=0.3, extent=[0, 1, 0, 1])
        except Exception as e:
            logging.warning(f"Could not load background image: {e}")
    
    # Draw nodes with enhanced styling
    _draw_enhanced_nodes(G, ax, pos, node_info, engine, highlight_missing)
    
    # Draw edges with category-based styling
    _draw_enhanced_edges(G, ax, pos, engine)
    
    # Add comprehensive legend
    _add_comprehensive_legend(ax, engine)
    
    # Add title and metadata
    ax.set_title(f"Full Scene Graph - {problem_id}\n"
                f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}", 
                fontsize=16, fontweight='bold')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    
    # Save with high DPI
    out_path = os.path.join(out_dir, f"{problem_id}_enhanced_full_graph.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    logging.info(f"Enhanced full visualization saved to {out_path}")

def _draw_enhanced_nodes(G, ax, pos, node_info, engine, highlight_missing):
    """Draw nodes with enhanced visual styling and missing data highlighting"""
    for node, data in G.nodes(data=True):
        if node not in pos:
            continue
            
        x, y = pos[node]
        info = node_info[node]
        object_type = info['object_type']
        
        # Determine node color
        if highlight_missing and info['missing_fields']:
            color = engine.color_palette['missing_data']
            edge_color = 'red'
            edge_width = 2
        elif not data.get('geometry_valid', True):
            color = engine.color_palette['invalid_geometry']
            edge_color = 'orange'
            edge_width = 2
        else:
            color = engine.color_palette.get(object_type, engine.color_palette['unknown'])
            edge_color = 'black'
            edge_width = 1
        
        # Get node style
        style = engine.node_styles.get(object_type, engine.node_styles['line'])
        size = style['size'] * (1 + info['importance'] * 0.1)  # Scale by importance
        
        # Draw node
        ax.scatter(x, y, c=color, s=size, marker=style['shape'],
                  edgecolors=edge_color, linewidths=edge_width, alpha=0.8)
        
        # Add node label
        label = _create_node_label(node, data, info)
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold' if info['is_motif'] else 'normal',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

def _create_node_label(node, data, info):
    """Create informative node label"""
    object_type = info['object_type']
    
    # Basic label
    if info['is_motif']:
        label = f"M:{node.split('_')[-1] if '_' in node else node}"
    else:
        label = f"{object_type[0].upper()}:{node.split('_')[-1] if '_' in node else node}"
    
    # Add important attributes
    if data.get('gnn_score', 0) > 0.5:
        label += f"\nGNN:{data['gnn_score']:.2f}"
    
    if info['missing_fields']:
        label += f"\n❌{len(info['missing_fields'])}"
    
    return label

def _draw_enhanced_edges(G, ax, pos, engine):
    """Draw edges with enhanced styling based on predicate categories"""
    edge_categories = defaultdict(list)
    
    # Group edges by category
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        category = data.get('predicate_category', 'unknown')
        edge_categories[category].append((u, v, data))
    
    # Draw edges by category for proper layering
    for category, edges in edge_categories.items():
        color = engine.color_palette.get(category, engine.color_palette['unknown'])
        style_info = engine.edge_styles.get(category, engine.edge_styles['unknown'])
        
        for u, v, data in edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Edge thickness based on relationship strength
            strength = data.get('relationship_strength', 1.0)
            width = style_info['width'] * strength
            
            ax.plot([x1, x2], [y1, y2], 
                   color=color, 
                   linestyle=style_info['style'],
                   linewidth=width,
                   alpha=style_info['alpha'],
                   zorder=1)
            
            # Add edge label for important relationships
            if strength > 0.5 or data.get('predicate_count', 1) > 3:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                predicate = data.get('predicate', 'unknown')
                ax.annotate(predicate, (mid_x, mid_y), 
                           fontsize=6, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))

def _add_comprehensive_legend(ax, engine):
    """Add comprehensive legend for node types, edge types, and missing data indicators"""
    # Create legend elements
    legend_elements = []
    
    # Node type legend
    for obj_type, color in engine.color_palette.items():
        if obj_type in engine.node_styles:
            style = engine.node_styles[obj_type]
            legend_elements.append(
                plt.scatter([], [], c=color, s=100, marker=style['shape'], 
                           label=f"{obj_type.replace('_', ' ').title()}")
            )
    
    # Add special indicators
    legend_elements.extend([
        plt.scatter([], [], c=engine.color_palette['missing_data'], s=100, 
                   marker='o', edgecolors='red', linewidths=2, label="Missing Data"),
        plt.scatter([], [], c=engine.color_palette['invalid_geometry'], s=100,
                   marker='o', edgecolors='orange', linewidths=2, label="Invalid Geometry")
    ])
    
    # Position legend outside plot area
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
             fontsize=10, title="Node Types & Status")

def _create_relationship_network_visualization(G, out_dir, problem_id, engine, layout_info, show_all):
    """Create focused relationship network visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    pos = layout_info['pos']
    
    # Filter graph for relationship visualization
    if not show_all:
        # Show only high-importance relationships
        filtered_edges = [(u, v, d) for u, v, d in G.edges(data=True) 
                         if d.get('relationship_strength', 1.0) > 0.3]
        rel_graph = nx.Graph()
        rel_graph.add_nodes_from(G.nodes(data=True))
        rel_graph.add_edges_from(filtered_edges)
    else:
        rel_graph = G
    
    # Draw relationship network
    _draw_enhanced_nodes(rel_graph, ax, pos, layout_info['node_info'], engine, False)
    _draw_enhanced_edges(rel_graph, ax, pos, engine)
    
    ax.set_title(f"Relationship Network - {problem_id}", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    out_path = os.path.join(out_dir, f"{problem_id}_relationship_network.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Relationship network saved to {out_path}")

def _create_motif_hierarchy_visualization(G, out_dir, problem_id, engine, layout_info):
    """Create motif hierarchy visualization"""
    # Extract motif nodes and their relationships
    motif_nodes = [n for n, d in G.nodes(data=True) if d.get('is_motif', False)]
    
    if not motif_nodes:
        logging.info(f"No motifs found for {problem_id}, skipping motif visualization")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Create motif subgraph
    motif_graph = G.subgraph(motif_nodes + 
                            [n for n, d in G.nodes(data=True) 
                             if any(G.has_edge(n, m) for m in motif_nodes)])
    
    # Use hierarchical layout for motifs
    motif_pos = nx.spring_layout(motif_graph, k=2, iterations=50)
    
    _draw_enhanced_nodes(motif_graph, ax, motif_pos, layout_info['node_info'], engine, True)
    _draw_enhanced_edges(motif_graph, ax, motif_pos, engine)
    
    ax.set_title(f"Motif Hierarchy - {problem_id}", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    out_path = os.path.join(out_dir, f"{problem_id}_motif_hierarchy.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Motif hierarchy saved to {out_path}")

def _create_missing_data_visualization(G, out_dir, problem_id, engine, layout_info):
    """Create visualization highlighting missing and incomplete data"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    pos = layout_info['pos']
    node_info = layout_info['node_info']
    
    # Missing data heatmap
    missing_counts = defaultdict(int)
    for node, info in node_info.items():
        for field in info['missing_fields']:
            missing_counts[field] += 1
    
    if missing_counts:
        fields = list(missing_counts.keys())
        counts = list(missing_counts.values())
        
        ax1.bar(fields, counts, color=engine.color_palette['missing_data'])
        ax1.set_title("Missing Data Fields", fontweight='bold')
        ax1.set_ylabel("Number of Nodes")
        ax1.tick_params(axis='x', rotation=45)
    
    # Missing data node map
    for node, data in G.nodes(data=True):
        if node not in pos:
            continue
        
        info = node_info[node]
        x, y = pos[node]
        
        # Color intensity based on missing data count
        alpha = min(1.0, len(info['missing_fields']) / 5.0)
        color = engine.color_palette['missing_data'] if info['missing_fields'] else 'lightgreen'
        
        ax2.scatter(x, y, c=color, s=300, alpha=alpha, edgecolors='black')
        
        if info['missing_fields']:
            ax2.annotate(f"{len(info['missing_fields'])}", (x, y), 
                        ha='center', va='center', fontweight='bold')
    
    ax2.set_title("Missing Data Distribution", fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{problem_id}_missing_data_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Missing data analysis saved to {out_path}")

def save_comprehensive_scene_graph_csv(G, out_dir, problem_id):
    """
    Save comprehensive CSV files with enhanced data analysis and missing field identification
    """
    logging.info(f"Saving comprehensive CSV data for problem_id={problem_id}")
    
    # Enhanced node data processing
    nodes_data = []
    missing_data_summary = defaultdict(int)
    
    for n, data in G.nodes(data=True):
        d = _sanitize_data(data.copy())
        d['id'] = n
        
        # Add computed fields and missing data analysis
        d.update(_compute_missing_fields_analysis(d))
        d.update(_compute_enhanced_geometric_features(d))
        d.update(_compute_semantic_features(d))
        
        # Track missing data
        for field, is_missing in d.get('missing_fields_analysis', {}).items():
            if is_missing:
                missing_data_summary[field] += 1
        
        d['all_data'] = json.dumps(d, ensure_ascii=False)
        nodes_data.append(d)
    
    # Enhanced edge data processing
    edges_data = []
    edge_categories = defaultdict(int)
    
    try:
        if hasattr(G, 'is_multigraph') and G.is_multigraph():
            for u, v, k, data in G.edges(keys=True, data=True):
                d = _sanitize_data(data.copy())
                d.update({
                    'source': u,
                    'target': v,
                    'key': k,
                    'predicate_category': _categorize_predicate(d.get('predicate', 'unknown')),
                    'relationship_strength': _calculate_edge_strength(d)
                })
                edge_categories[d['predicate_category']] += 1
                d['all_data'] = json.dumps(d, ensure_ascii=False)
                edges_data.append(d)
        else:
            for u, v, data in G.edges(data=True):
                d = _sanitize_data(data.copy())
                d.update({
                    'source': u,
                    'target': v,
                    'predicate_category': _categorize_predicate(d.get('predicate', 'unknown')),
                    'relationship_strength': _calculate_edge_strength(d)
                })
                edge_categories[d['predicate_category']] += 1
                d['all_data'] = json.dumps(d, ensure_ascii=False)
                edges_data.append(d)
    except Exception as e:
        logging.error(f"Error processing edges for CSV: {e}")
        raise
    
    # Save enhanced CSV files
    os.makedirs(out_dir, exist_ok=True)
    
    # Nodes CSV with comprehensive analysis
    nodes_df = pd.DataFrame(nodes_data)
    nodes_csv_path = os.path.join(out_dir, f"{problem_id}_enhanced_nodes.csv")
    nodes_df.to_csv(nodes_csv_path, index=False)
    
    # Edges CSV with relationship analysis
    if edges_data:
        edges_df = pd.DataFrame(edges_data)
        edges_csv_path = os.path.join(out_dir, f"{problem_id}_enhanced_edges.csv")
        edges_df.to_csv(edges_csv_path, index=False)
    else:
        logging.warning(f"No edges found for problem {problem_id}")
    
    # Generate analysis report
    _generate_data_analysis_report(nodes_df, edges_data, missing_data_summary, 
                                  edge_categories, out_dir, problem_id)
    
    logging.info(f"Comprehensive CSV files saved for {problem_id}")

def _sanitize_data(obj):
    """Enhanced data sanitization"""
    if isinstance(obj, dict):
        return {k: _sanitize_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_data(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None:
        return None
    else:
        return obj

def _compute_missing_fields_analysis(data):
    """Analyze which critical fields are missing or invalid"""
    analysis = {}
    
    # Critical fields to check
    critical_fields = {
        'vl_embed': lambda x: x is None or (isinstance(x, list) and all(v == 0 for v in x)),
        'centroid': lambda x: x is None or x == [None, None],
        'curvature_score': lambda x: x == 0 and data.get('object_type') in ['curve', 'open_curve'],
        'motif_id': lambda x: x is None and data.get('is_motif', False),
        'area': lambda x: x == 0 and data.get('object_type') == 'polygon',
        'orientation': lambda x: x is None,
        'vertices': lambda x: x is None or len(x) == 0,
        'gnn_score': lambda x: x is None or x == 0,
        'kb_concept': lambda x: x is None or x == '',
        'geometry_valid': lambda x: x is False,
    }
    
    missing_fields = {}
    for field, check_func in critical_fields.items():
        value = data.get(field)
        is_missing = check_func(value)
        missing_fields[f'{field}_missing'] = is_missing
        analysis[f'{field}_status'] = 'missing' if is_missing else 'valid'
    
    analysis['missing_fields_analysis'] = missing_fields
    analysis['missing_field_count'] = sum(missing_fields.values())
    analysis['data_completeness_score'] = 1.0 - (analysis['missing_field_count'] / len(critical_fields))
    
    return analysis

def _compute_enhanced_geometric_features(data):
    """Compute enhanced geometric features for open curves and complex shapes"""
    enhanced_features = {}
    
    vertices = data.get('vertices', [])
    object_type = data.get('object_type', 'unknown')
    
    if not vertices or len(vertices) < 2:
        return enhanced_features
    
    vertices_array = np.array(vertices)
    
    # Enhanced curvature calculation for open curves
    if object_type in ['curve', 'open_curve', 'arc'] and len(vertices) >= 3:
        enhanced_features.update(_calculate_enhanced_curvature(vertices_array))
    
    # Path complexity analysis
    enhanced_features.update(_calculate_path_complexity(vertices_array))
    
    # Geometric descriptors
    enhanced_features.update(_calculate_geometric_descriptors(vertices_array, object_type))
    
    return enhanced_features

def _calculate_enhanced_curvature(vertices_array):
    """Calculate comprehensive curvature metrics"""
    features = {}
    
    try:
        # Local curvature at each point
        curvatures = []
        for i in range(1, len(vertices_array) - 1):
            p1, p2, p3 = vertices_array[i-1], vertices_array[i], vertices_array[i+1]
            
            # Calculate curvature using discrete derivative approximation
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Avoid division by zero
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                curvatures.append(0.0)
                continue
            
            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # Curvature is angle change per unit length
            avg_length = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2
            curvature = angle / (avg_length + 1e-6)
            curvatures.append(curvature)
        
        if curvatures:
            features.update({
                'enhanced_curvature_mean': float(np.mean(curvatures)),
                'enhanced_curvature_max': float(np.max(curvatures)),
                'enhanced_curvature_std': float(np.std(curvatures)),
                'enhanced_curvature_total': float(np.sum(curvatures)),
                'curvature_points': len(curvatures)
            })
        
    except Exception as e:
        logging.warning(f"Enhanced curvature calculation failed: {e}")
        features.update({
            'enhanced_curvature_mean': 0.0,
            'enhanced_curvature_max': 0.0,
            'enhanced_curvature_std': 0.0,
            'enhanced_curvature_total': 0.0,
            'curvature_points': 0
        })
    
    return features

def _calculate_path_complexity(vertices_array):
    """Calculate path complexity metrics"""
    features = {}
    
    if len(vertices_array) < 2:
        return features
    
    try:
        # Path length
        path_length = np.sum([np.linalg.norm(vertices_array[i+1] - vertices_array[i]) 
                             for i in range(len(vertices_array)-1)])
        
        # Straight line distance (start to end)
        straight_distance = np.linalg.norm(vertices_array[-1] - vertices_array[0])
        
        # Tortuosity (path length / straight distance)
        tortuosity = path_length / (straight_distance + 1e-6)
        
        # Direction changes
        direction_changes = 0
        if len(vertices_array) >= 3:
            for i in range(1, len(vertices_array) - 1):
                v1 = vertices_array[i] - vertices_array[i-1]
                v2 = vertices_array[i+1] - vertices_array[i]
                
                if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    
                    # Count significant direction changes (> 15 degrees)
                    if angle > np.pi / 12:
                        direction_changes += 1
        
        features.update({
            'path_length': float(path_length),
            'straight_distance': float(straight_distance),
            'tortuosity': float(tortuosity),
            'direction_changes': int(direction_changes),
            'path_complexity_score': float(tortuosity * (1 + direction_changes / 10.0))
        })
        
    except Exception as e:
        logging.warning(f"Path complexity calculation failed: {e}")
        
    return features

def _calculate_geometric_descriptors(vertices_array, object_type):
    """Calculate additional geometric descriptors"""
    features = {}
    
    try:
        # Bounding box analysis
        if len(vertices_array) > 0:
            min_coords = np.min(vertices_array, axis=0)
            max_coords = np.max(vertices_array, axis=0)
            bbox_size = max_coords - min_coords
            
            features.update({
                'bbox_width': float(bbox_size[0]),
                'bbox_height': float(bbox_size[1]),
                'bbox_aspect_ratio': float(bbox_size[0] / (bbox_size[1] + 1e-6)),
                'bbox_area': float(bbox_size[0] * bbox_size[1])
            })
        
        # Convex hull properties (if applicable)
        if len(vertices_array) >= 3:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(vertices_array)
                
                features.update({
                    'convex_hull_area': float(hull.volume),  # 'volume' is area in 2D
                    'convex_hull_vertices': int(len(hull.vertices)),
                    'convexity_ratio': float(len(hull.vertices) / len(vertices_array))
                })
            except:
                pass
        
        # Symmetry analysis
        if len(vertices_array) >= 4:
            features.update(_analyze_symmetry(vertices_array))
        
    except Exception as e:
        logging.warning(f"Geometric descriptors calculation failed: {e}")
    
    return features

def _analyze_symmetry(vertices_array):
    """Analyze symmetry properties"""
    features = {}
    
    try:
        center = np.mean(vertices_array, axis=0)
        
        # Reflection symmetry (horizontal and vertical)
        reflected_h = 2 * center[1] - vertices_array[:, 1]
        reflected_v = 2 * center[0] - vertices_array[:, 0]
        
        # Simple symmetry score based on point matching
        h_symmetry = _calculate_reflection_score(vertices_array[:, 1], reflected_h)
        v_symmetry = _calculate_reflection_score(vertices_array[:, 0], reflected_v)
        
        features.update({
            'horizontal_symmetry_score': float(h_symmetry),
            'vertical_symmetry_score': float(v_symmetry),
            'overall_symmetry_score': float((h_symmetry + v_symmetry) / 2)
        })
        
    except Exception as e:
        logging.warning(f"Symmetry analysis failed: {e}")
    
    return features

def _calculate_reflection_score(original, reflected):
    """Calculate how well points match their reflections"""
    try:
        # Find closest matches
        distances = []
        for i, orig_point in enumerate(original):
            min_dist = min(abs(orig_point - refl_point) for refl_point in reflected)
            distances.append(min_dist)
        
        # Score based on average distance (lower is more symmetric)
        avg_distance = np.mean(distances)
        max_distance = np.max(np.abs(original - np.mean(original)))
        
        # Normalize to 0-1 scale (1 = perfect symmetry)
        symmetry_score = max(0, 1 - avg_distance / (max_distance + 1e-6))
        return symmetry_score
        
    except:
        return 0.0

def _compute_semantic_features(data):
    """Compute semantic features and embeddings analysis"""
    features = {}
    
    # VL embedding analysis
    vl_embed = data.get('vl_embed', [])
    if vl_embed and not all(x == 0 for x in vl_embed):
        vl_array = np.array(vl_embed)
        features.update({
            'vl_embed_norm': float(np.linalg.norm(vl_array)),
            'vl_embed_mean': float(np.mean(vl_array)),
            'vl_embed_std': float(np.std(vl_array)),
            'vl_embed_nonzero_ratio': float(np.count_nonzero(vl_array) / len(vl_array))
        })
    else:
        features.update({
            'vl_embed_norm': 0.0,
            'vl_embed_mean': 0.0,
            'vl_embed_std': 0.0,
            'vl_embed_nonzero_ratio': 0.0
        })
    
    # Motif analysis
    if data.get('is_motif', False):
        features.update({
            'motif_member_count': data.get('member_count', 0),
            'motif_complexity': len(data.get('member_types', [])),
            'motif_stroke_count': data.get('stroke_count', 0)
        })
    
    return features

def _calculate_edge_strength(edge_data):
    """Calculate relationship strength for edges"""
    strength = 1.0
    
    # Factor in confidence scores
    if 'pattern_confidence' in edge_data:
        strength *= edge_data['pattern_confidence']
    
    if 'similarity_score' in edge_data:
        strength *= edge_data['similarity_score']
    
    # Factor in predicate frequency (if multi-edge)
    if 'predicate_count' in edge_data:
        strength *= min(2.0, edge_data['predicate_count'] / 5.0)
    
    return min(1.0, strength)

def _generate_data_analysis_report(nodes_df, edges_data, missing_data_summary, 
                                  edge_categories, out_dir, problem_id):
    """Generate comprehensive data analysis report"""
    report_path = os.path.join(out_dir, f"{problem_id}_data_analysis_report.json")
    
    # Calculate statistics
    total_nodes = len(nodes_df)
    nodes_with_missing_data = len(nodes_df[nodes_df['missing_field_count'] > 0])
    avg_completeness = nodes_df['data_completeness_score'].mean()
    
    # Object type distribution
    object_type_dist = nodes_df['object_type'].value_counts().to_dict()
    
    # Missing data analysis
    missing_data_analysis = {
        'total_missing_fields': dict(missing_data_summary),
        'nodes_with_missing_data': int(nodes_with_missing_data),
        'missing_data_percentage': float(nodes_with_missing_data / total_nodes * 100),
        'average_completeness_score': float(avg_completeness)
    }
    
    # Edge analysis
    edge_analysis = {
        'total_edges': len(edges_data),
        'edge_categories': dict(edge_categories),
        'average_relationship_strength': float(np.mean([e.get('relationship_strength', 1.0) for e in edges_data])) if edges_data else 0.0
    }
    
    # Generate report
    report = {
        'problem_id': problem_id,
        'timestamp': pd.Timestamp.now().isoformat(),
        'summary': {
            'total_nodes': total_nodes,
            'total_edges': len(edges_data),
            'object_type_distribution': object_type_dist,
            'data_quality_score': float(avg_completeness)
        },
        'missing_data_analysis': missing_data_analysis,
        'edge_analysis': edge_analysis,
        'recommendations': _generate_recommendations(missing_data_summary, avg_completeness)
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logging.info(f"Data analysis report saved to {report_path}")

def _generate_recommendations(missing_data_summary, avg_completeness):
    """Generate recommendations for improving data quality"""
    recommendations = []
    
    if 'vl_embed' in missing_data_summary and missing_data_summary['vl_embed'] > 0:
        recommendations.append("Implement VL embedding computation using CLIP or similar models")
    
    if 'curvature_score' in missing_data_summary and missing_data_summary['curvature_score'] > 0:
        recommendations.append("Implement enhanced curvature calculation for open curves")
    
    if 'motif_id' in missing_data_summary and missing_data_summary['motif_id'] > 0:
        recommendations.append("Improve motif detection and ID assignment")
    
    if avg_completeness < 0.7:
        recommendations.append("Overall data completeness is low - review feature extraction pipeline")
    
    if avg_completeness < 0.5:
        recommendations.append("Critical data quality issues detected - immediate attention required")
    
    return recommendations

def create_puzzle_overview_visualization(all_graphs, out_dir, puzzle_name):
    """Create overview visualization showing all 12+ images in a puzzle"""
    if not all_graphs:
        logging.warning(f"No graphs provided for puzzle overview: {puzzle_name}")
        return
    
    # Determine grid layout
    n_graphs = len(all_graphs)
    cols = 4 if n_graphs <= 12 else 5
    rows = (n_graphs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    engine = BongardVisualizationEngine()
    
    for i, (graph_data, image_path) in enumerate(all_graphs):
        if i >= len(axes):
            break
            
        ax = axes[i]
        G = graph_data
        
        # Compute layout for this graph
        pos = nx.spring_layout(G, k=1, iterations=50)
        node_info = _analyze_node_structure(G)
        
        # Draw simplified version
        _draw_simplified_nodes(G, ax, pos, node_info, engine)
        _draw_simplified_edges(G, ax, pos, engine)
        
        # Add image background if available
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                ax.imshow(img, alpha=0.2, extent=[0, 1, 0, 1])
            except:
                pass
        
        ax.set_title(f"Image {i+1}\n{G.number_of_nodes()}N/{G.number_of_edges()}E", 
                    fontsize=10)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_graphs, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Puzzle Overview: {puzzle_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f"{puzzle_name}_puzzle_overview.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Puzzle overview saved to {out_path}")

def _draw_simplified_nodes(G, ax, pos, node_info, engine):
    """Draw simplified nodes for overview visualization"""
    for node, data in G.nodes(data=True):
        if node not in pos:
            continue
        
        x, y = pos[node]
        object_type = data.get('object_type', 'unknown')
        is_motif = data.get('is_motif', False)
        
        color = engine.color_palette.get(object_type, engine.color_palette['unknown'])
        size = 150 if is_motif else 100
        
        ax.scatter(x, y, c=color, s=size, alpha=0.8, edgecolors='black', linewidths=0.5)

def _draw_simplified_edges(G, ax, pos, engine):
    """Draw simplified edges for overview visualization"""
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        category = data.get('predicate_category', 'unknown')
        color = engine.color_palette.get(category, engine.color_palette['unknown'])
        
        ax.plot([x1, x2], [y1, y2], color=color, alpha=0.3, linewidth=0.5)

def analyze_missing_data_report(csv_files_dir, output_file):
    """Analyze all CSV files to generate comprehensive missing data report"""
    missing_data_analysis = {
        'files_analyzed': [],
        'global_missing_fields': defaultdict(int),
        'problem_specific_issues': {},
        'recommendations': [],
        'summary_statistics': {}
    }
    
    # Process all node CSV files
    for filename in os.listdir(csv_files_dir):
        if filename.endswith('_nodes.csv') or filename.endswith('_enhanced_nodes.csv'):
            filepath = os.path.join(csv_files_dir, filename)
            problem_id = filename.replace('_nodes.csv', '').replace('_enhanced_nodes.csv', '')
            
            try:
                df = pd.read_csv(filepath)
                missing_data_analysis['files_analyzed'].append(filename)
                
                # Analyze missing data in this file
                problem_issues = []
                
                # Check VL embeddings
                vl_embed_issues = 0
                if 'vl_embed' in df.columns:
                    for _, row in df.iterrows():
                        vl_embed = row['vl_embed']
                        if pd.isna(vl_embed) or str(vl_embed).startswith('[0.0, 0.0'):
                            vl_embed_issues += 1
                            missing_data_analysis['global_missing_fields']['vl_embed'] += 1
                
                if vl_embed_issues > 0:
                    problem_issues.append(f"VL embeddings missing/zero: {vl_embed_issues}")
                
                # Check other critical fields
                critical_fields = ['centroid', 'curvature_score', 'motif_id', 'area', 'perimeter']
                for field in critical_fields:
                    if field in df.columns:
                        missing_count = df[field].isna().sum()
                        zero_count = (df[field] == 0).sum() if field in ['curvature_score', 'area'] else 0
                        
                        if missing_count > 0:
                            problem_issues.append(f"{field} missing: {missing_count}")
                            missing_data_analysis['global_missing_fields'][field] += missing_count
                        
                        if zero_count > missing_count:  # More zeros than NAs suggests calculation issue
                            problem_issues.append(f"{field} zero values: {zero_count}")
                
                if problem_issues:
                    missing_data_analysis['problem_specific_issues'][problem_id] = problem_issues
                
            except Exception as e:
                logging.error(f"Error analyzing {filename}: {e}")
    
    # Generate summary statistics
    total_files = len(missing_data_analysis['files_analyzed'])
    missing_data_analysis['summary_statistics'] = {
        'total_files_analyzed': total_files,
        'files_with_issues': len(missing_data_analysis['problem_specific_issues']),
        'most_common_missing_fields': dict(sorted(missing_data_analysis['global_missing_fields'].items(), 
                                                 key=lambda x: x[1], reverse=True)[:10])
    }
    
    # Generate recommendations
    recommendations = []
    if missing_data_analysis['global_missing_fields']['vl_embed'] > 0:
        recommendations.append("Implement VL embedding computation pipeline using CLIP or similar models")
    
    if missing_data_analysis['global_missing_fields']['curvature_score'] > 0:
        recommendations.append("Implement curvature calculation for open curves and complex shapes")
    
    if missing_data_analysis['global_missing_fields']['motif_id'] > 0:
        recommendations.append("Improve motif detection and assignment algorithms")
    
    recommendations.append("Review feature extraction pipeline for geometric calculations")
    recommendations.append("Implement validation checks in data processing pipeline")
    
    missing_data_analysis['recommendations'] = recommendations
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(missing_data_analysis, f, indent=2, default=str)
    
    logging.info(f"Missing data analysis report saved to {output_file}")
    return missing_data_analysis

# Example usage function
def process_puzzle_visualization(problem_data, output_dir):
    """
    Process a complete puzzle with multiple images
    
    Args:
        problem_data: List of (graph, image_path) tuples for each image in puzzle
        output_dir: Output directory for visualizations
    """
    if not problem_data:
        return
    
    # Extract puzzle name from first graph
    first_graph = problem_data[0][0]
    puzzle_name = "unknown_puzzle"
    for node, data in first_graph.nodes(data=True):
        if 'image_path' in data:
            puzzle_name = os.path.basename(data['image_path']).split('_')[0]
            break
    
    # Create puzzle overview
    create_puzzle_overview_visualization(problem_data, output_dir, puzzle_name)
    
    # Process each individual graph
    for i, (graph, image_path) in enumerate(problem_data):
        problem_id = f"{puzzle_name}_image_{i}"
        
        # Enhanced individual visualization
        save_enhanced_scene_graph_visualization(
            graph, image_path, output_dir, problem_id,
            show_all_relationships=True,
            highlight_missing_data=True
        )
        
        # Comprehensive CSV export
        save_comprehensive_scene_graph_csv(graph, output_dir, problem_id)
    
    # Generate missing data analysis for the puzzle
    report_file = os.path.join(output_dir, f"{puzzle_name}_missing_data_report.json")
    analyze_missing_data_report(output_dir, report_file)
    
    logging.info(f"Complete puzzle visualization processing finished for {puzzle_name}")

if __name__ == "__main__":
    # Example usage
    logging.info("Updated Scene Graph Visualization Script loaded successfully")
    logging.info("Use process_puzzle_visualization() for complete puzzle processing")
