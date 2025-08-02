def _robust_edge_unpack(edges):
    """Yield (u, v, k, data) for all edge tuples, handling 2, 3, 4-item cases and non-dict data."""
    for edge in edges:
        if len(edge) == 4:
            u, v, k, data = edge
        elif len(edge) == 3:
            u, v, data = edge
            k = None
        elif len(edge) == 2:
            u, v = edge
            k = None
            data = {}
        else:
            logging.warning(f"[LOGO Visualization] Skipping unexpectedly short/long edge tuple: {edge}")
            continue
        if not isinstance(data, dict):
            logging.warning(f"[LOGO Visualization] Edge data not dict: {repr(data)} for edge {edge}; using empty dict.")
            data = {}
        yield u, v, k, data

import os
import logging
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import glob

__all__ = [
    'save_scene_graph_visualization',
    'save_scene_graph_csv',
    'save_enhanced_scene_graph_visualization',
    'create_multi_puzzle_visualization',
    'create_missing_data_analysis',
    'analyze_puzzle_completeness'
]

def get_node_style(data):
    """Enhanced node styling based on object properties and data completeness"""
    node_type = data.get('object_type', 'unknown')
    source = data.get('source', 'unknown')
    completeness = data.get('data_completeness_score', 0.5)
    
    # Base color by type
    color_map = {
        'line': 'lightcoral',
        'curve': 'lightblue', 
        'circle': 'lightgreen',
        'arc': 'lightyellow',
        'polygon': 'lightpink',
        'composite': 'lightgray',
        'motif': 'gold'
    }
    
    base_color = color_map.get(node_type, 'white')
    
    # Adjust alpha based on data completeness
    alpha = 0.3 + 0.7 * completeness  # Range from 0.3 to 1.0
    
    # Border color based on source
    border_colors = {
        'program': 'blue',
        'geometric_grouping': 'green', 
        'motif_mining': 'orange',
        'vision_language': 'purple'
    }
    border_color = border_colors.get(source, 'black')
    
    return {
        'color': base_color,
        'alpha': alpha,
        'border_color': border_color,
        'border_width': 2 if completeness > 0.8 else 1
    }

def get_edge_style(predicate):
    """Enhanced edge styling based on predicate type"""
    predicate_styles = {
        # Geometric relationships
        'is_above': {'color': 'red', 'style': '-', 'width': 2},
        'is_below': {'color': 'red', 'style': '-', 'width': 2},
        'is_left_of': {'color': 'blue', 'style': '-', 'width': 2},
        'is_right_of': {'color': 'blue', 'style': '-', 'width': 2},
        'contains': {'color': 'green', 'style': '-', 'width': 3},
        'intersects': {'color': 'orange', 'style': '--', 'width': 2},
        
        # Structural relationships
        'part_of': {'color': 'purple', 'style': '-', 'width': 2},
        'adjacent_endpoints': {'color': 'brown', 'style': ':', 'width': 1},
        'part_of_motif': {'color': 'gold', 'style': '-', 'width': 3},
        
        # Similarity relationships
        'same_shape_class': {'color': 'cyan', 'style': '--', 'width': 1},
        'visual_similarity': {'color': 'magenta', 'style': ':', 'width': 1},
        'forms_symmetry': {'color': 'lime', 'style': '-.', 'width': 2},
        
        # Pattern relationships
        'forms_bridge_pattern': {'color': 'navy', 'style': '-', 'width': 2},
        'forms_apex_pattern': {'color': 'maroon', 'style': '-', 'width': 2},
        'forms_symmetry_pattern': {'color': 'teal', 'style': '-', 'width': 2}
    }
    
    return predicate_styles.get(predicate, {'color': 'gray', 'style': '-', 'width': 1})

def save_enhanced_scene_graph_visualization(G, image_path, out_dir, problem_id, 
                                          show_missing_data=True, show_patterns=True):
    """
    Save enhanced visualization with comprehensive node/edge styling, 
    missing data highlighting, and multi-view analysis
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Node positions: use centroid, else first vertex, else layout
    pos = {}
    for n, data in G.nodes(data=True):
        if 'centroid' in data and data['centroid'] is not None:
            try:
                centroid = data['centroid']
                if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
                    pos[n] = (float(centroid[0]), float(centroid[1]))
                else:
                    pos[n] = (0, 0)
            except (ValueError, TypeError):
                pos[n] = (0, 0)
        elif 'vertices' in data and data['vertices']:
            try:
                pos[n] = (float(data['vertices'][0][0]), float(data['vertices'][0][1]))
            except (IndexError, ValueError, TypeError):
                pos[n] = (0, 0)
        else:
            pos[n] = (0, 0)
    
    # Use spring layout for nodes without positions
    nodes_without_pos = [n for n, p in pos.items() if p == (0, 0)]
    if nodes_without_pos:
        spring_pos = nx.spring_layout(G.subgraph(nodes_without_pos), k=1, iterations=50)
        for n in nodes_without_pos:
            pos[n] = spring_pos[n]
    
    # Create enhanced visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()
    
    # 1. Full graph with all relationships
    ax = axes[0]
    ax.set_title(f"Complete Scene Graph - {problem_id}", fontsize=14, fontweight='bold')
    
    # Draw nodes with enhanced styling
    for n, data in G.nodes(data=True):
        style = get_node_style(data)
        x, y = pos[n]
        
        # Create fancy node shape
        bbox = FancyBboxPatch(
            (x-0.05, y-0.03), 0.1, 0.06,
            boxstyle="round,pad=0.01",
            facecolor=style['color'],
            edgecolor=style['border_color'],
            linewidth=style['border_width'],
            alpha=style['alpha']
        )
        ax.add_patch(bbox)
        
        # Node label with data quality indicator
        object_type = data.get('object_type', 'obj')
        completeness = data.get('data_completeness_score', 0.0)
        if show_missing_data and completeness < 0.5:
            label = f"{object_type}*"  # Mark incomplete data
        else:
            label = object_type
        
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw edges with enhanced styling
    for u, v, data in G.edges(data=True):
        predicate = data.get('predicate', 'unknown')
        style = get_edge_style(predicate)
        
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        ax.plot([x1, x2], [y1, y2], 
               color=style['color'], 
               linestyle=style['style'],
               linewidth=style['width'],
               alpha=0.7)
        
        # Add edge label for important relationships
        if predicate in ['part_of', 'contains', 'part_of_motif']:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, predicate[:4], fontsize=6, ha='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 2. Relationship network (filtered view)
    ax = axes[1]
    ax.set_title(f"Key Relationships - {problem_id}", fontsize=14, fontweight='bold')
    
    # Filter for important relationships
    important_predicates = ['part_of', 'contains', 'part_of_motif', 'forms_symmetry', 'visual_similarity']
    filtered_edges = [(u, v, data) for u, v, data in G.edges(data=True) 
                     if data.get('predicate', '') in important_predicates]
    
    # Draw filtered graph
    for n, data in G.nodes(data=True):
        style = get_node_style(data)
        x, y = pos[n]
        ax.scatter(x, y, c=style['color'], s=300, alpha=style['alpha'], 
                  edgecolors=style['border_color'], linewidth=style['border_width'])
        ax.text(x, y-0.08, data.get('object_type', 'obj')[:6], ha='center', fontsize=8)
    
    for u, v, data in filtered_edges:
        predicate = data.get('predicate', 'unknown')
        style = get_edge_style(predicate)
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color=style['color'], linewidth=style['width'], alpha=0.8)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 3. Missing data analysis
    ax = axes[2]
    ax.set_title(f"Data Completeness Analysis - {problem_id}", fontsize=14, fontweight='bold')
    
    if show_missing_data:
        completeness_scores = []
        node_types = []
        
        for n, data in G.nodes(data=True):
            completeness_scores.append(data.get('data_completeness_score', 0.0))
            node_types.append(data.get('object_type', 'unknown'))
        
        # Create bar chart of completeness by node type
        type_completeness = defaultdict(list)
        for ntype, score in zip(node_types, completeness_scores):
            type_completeness[ntype].append(score)
        
        types = list(type_completeness.keys())
        avg_scores = [np.mean(type_completeness[t]) for t in types]
        
        bars = ax.bar(types, avg_scores, alpha=0.7)
        
        # Color bars based on completeness
        for bar, score in zip(bars, avg_scores):
            if score < 0.3:
                bar.set_color('red')
            elif score < 0.7:
                bar.set_color('orange') 
            else:
                bar.set_color('green')
        
        ax.set_ylabel('Average Completeness Score')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # Add horizontal line at 0.5 (threshold)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()
    
    # 4. Pattern and motif hierarchy
    ax = axes[3]
    ax.set_title(f"Patterns & Motifs - {problem_id}", fontsize=14, fontweight='bold')
    
    if show_patterns:
        # Show only motif nodes and pattern relationships
        motif_nodes = [n for n, data in G.nodes(data=True) 
                      if data.get('object_type') == 'motif' or 'motif' in data.get('source', '')]
        pattern_edges = [(u, v, data) for u, v, data in G.edges(data=True)
                        if 'pattern' in data.get('predicate', '') or 'motif' in data.get('predicate', '')]
        
        # Draw motif hierarchy
        if motif_nodes:
            for n in motif_nodes:
                data = G.nodes[n]
                x, y = pos[n]
                ax.scatter(x, y, c='gold', s=500, alpha=0.8, edgecolors='orange', linewidth=3)
                
                member_count = data.get('motif_member_count', 0)
                ax.text(x, y, f"M{member_count}", ha='center', va='center', fontweight='bold')
        
        # Draw pattern connections
        for u, v, data in pattern_edges:
            if u in pos and v in pos:
                predicate = data.get('predicate', 'unknown')
                style = get_edge_style(predicate)
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                ax.plot([x1, x2], [y1, y2], color=style['color'], linewidth=3, alpha=0.8)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save enhanced visualization
    out_path = os.path.join(out_dir, f"{problem_id}_enhanced_scene_graph.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    logging.info(f"Saved enhanced scene graph visualization to {out_path}")
    return out_path

def create_multi_puzzle_visualization(csv_dir, out_dir, puzzle_pattern="*", max_puzzles=12):
    """
    Create comprehensive multi-puzzle visualization showing relationships across all images
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Find CSV files matching pattern
    node_files = glob.glob(os.path.join(csv_dir, f"{puzzle_pattern}_nodes.csv"))
    edge_files = glob.glob(os.path.join(csv_dir, f"{puzzle_pattern}_edges.csv"))
    
    if not node_files or not edge_files:
        logging.warning(f"No CSV files found matching pattern {puzzle_pattern} in {csv_dir}")
        return None
    
    # Load data from multiple puzzles
    all_graphs = {}
    puzzle_stats = {}
    
    for i, (node_file, edge_file) in enumerate(zip(node_files[:max_puzzles], edge_files[:max_puzzles])):
        try:
            puzzle_id = os.path.basename(node_file).replace('_nodes.csv', '')
            
            # Load nodes and edges
            nodes_df = pd.read_csv(node_file)
            edges_df = pd.read_csv(edge_file)
            
            # Create graph
            G = nx.MultiDiGraph()
            
            # Add nodes
            for _, row in nodes_df.iterrows():
                node_data = row.to_dict()
                # Handle NaN values
                for key, value in node_data.items():
                    if pd.isna(value):
                        node_data[key] = None
                G.add_node(row['object_id'], **node_data)
            
            # Add edges  
            for _, row in edges_df.iterrows():
                edge_data = row.to_dict()
                for key, value in edge_data.items():
                    if pd.isna(value):
                        edge_data[key] = None
                G.add_edge(row['source'], row['target'], **edge_data)
            
            all_graphs[puzzle_id] = G
            
            # Collect stats
            puzzle_stats[puzzle_id] = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'avg_completeness': np.mean([data.get('data_completeness_score', 0.0) 
                                           for _, data in G.nodes(data=True)]),
                'predicates': len(set([data.get('predicate', 'unknown') 
                                     for _, _, data in G.edges(data=True)]))
            }
            
        except Exception as e:
            logging.error(f"Error loading puzzle {puzzle_id}: {e}")
            continue
    
    if not all_graphs:
        logging.error("No graphs were successfully loaded")
        return None
    
    # Create multi-puzzle visualization
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    axes = axes.ravel()
    
    for i, (puzzle_id, G) in enumerate(list(all_graphs.items())[:12]):
        ax = axes[i]
        
        try:
            # Create layout
            pos = nx.spring_layout(G, k=1, iterations=30)
            
            # Draw nodes
            node_colors = []
            node_sizes = []
            for n, data in G.nodes(data=True):
                completeness = data.get('data_completeness_score', 0.5)
                node_colors.append(completeness)
                node_sizes.append(100 + 200 * completeness)
            
            nx.draw_networkx_nodes(G, pos, ax=ax, 
                                 node_color=node_colors, 
                                 node_size=node_sizes,
                                 cmap='RdYlGn', vmin=0, vmax=1, alpha=0.8)
            
            # Draw edges
            edge_colors = []
            for _, _, data in G.edges(data=True):
                predicate = data.get('predicate', 'unknown')
                if 'pattern' in predicate or 'motif' in predicate:
                    edge_colors.append('red')
                elif predicate in ['part_of', 'contains']:
                    edge_colors.append('blue')
                else:
                    edge_colors.append('gray')
            
            nx.draw_networkx_edges(G, pos, ax=ax, 
                                 edge_color=edge_colors, alpha=0.6, width=0.5)
            
            # Title with stats
            stats = puzzle_stats[puzzle_id]
            ax.set_title(f"{puzzle_id}\n{stats['nodes']}N, {stats['edges']}E, {stats['avg_completeness']:.2f}C",
                        fontsize=10)
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {puzzle_id}", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(all_graphs), 12):
        axes[i].axis('off')
    
    plt.suptitle(f"Multi-Puzzle Scene Graph Overview ({len(all_graphs)} puzzles)", fontsize=16)
    plt.tight_layout()
    
    # Save visualization
    out_path = os.path.join(out_dir, f"multi_puzzle_overview_{puzzle_pattern}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    logging.info(f"Saved multi-puzzle visualization to {out_path}")
    return out_path

def analyze_puzzle_completeness(csv_dir, puzzle_pattern="*"):
    """
    Analyze data completeness across puzzles and generate report
    """
    node_files = glob.glob(os.path.join(csv_dir, f"{puzzle_pattern}_nodes.csv"))
    
    if not node_files:
        logging.warning(f"No node CSV files found matching pattern {puzzle_pattern}")
        return None
    
    completeness_analysis = {
        'puzzles_analyzed': len(node_files),
        'field_completeness': defaultdict(list),
        'puzzle_scores': {},
        'missing_patterns': defaultdict(int)
    }
    
    for node_file in node_files:
        try:
            puzzle_id = os.path.basename(node_file).replace('_nodes.csv', '')
            nodes_df = pd.read_csv(node_file)
            
            # Analyze field completeness
            field_scores = {}
            for col in nodes_df.columns:
                if col == 'object_id':
                    continue
                
                non_null_count = nodes_df[col].notna().sum()
                non_zero_count = (nodes_df[col] != 0).sum() if nodes_df[col].dtype in ['int64', 'float64'] else non_null_count
                
                completeness = non_zero_count / len(nodes_df) if len(nodes_df) > 0 else 0
                field_scores[col] = completeness
                completeness_analysis['field_completeness'][col].append(completeness)
            
            # Overall puzzle score
            puzzle_score = np.mean(list(field_scores.values()))
            completeness_analysis['puzzle_scores'][puzzle_id] = puzzle_score
            
            # Identify missing patterns
            for col, score in field_scores.items():
                if score < 0.1:  # Less than 10% data
                    completeness_analysis['missing_patterns'][col] += 1
                    
        except Exception as e:
            logging.error(f"Error analyzing {node_file}: {e}")
    
    # Generate summary statistics
    summary = {
        'avg_field_completeness': {
            field: np.mean(scores) 
            for field, scores in completeness_analysis['field_completeness'].items()
        },
        'worst_fields': sorted(
            completeness_analysis['field_completeness'].items(),
            key=lambda x: np.mean(x[1])
        )[:10],
        'best_puzzles': sorted(
            completeness_analysis['puzzle_scores'].items(),
            key=lambda x: x[1], reverse=True
        )[:5],
        'worst_puzzles': sorted(
            completeness_analysis['puzzle_scores'].items(),
            key=lambda x: x[1]
        )[:5]
    }
    
    return completeness_analysis, summary

def create_missing_data_analysis(csv_dir, out_dir, puzzle_pattern="*"):
    """
    Create comprehensive missing data analysis visualization
    """
    os.makedirs(out_dir, exist_ok=True)
    
    analysis, summary = analyze_puzzle_completeness(csv_dir, puzzle_pattern)
    if not analysis:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Field completeness heatmap
    ax = axes[0, 0]
    field_names = list(summary['avg_field_completeness'].keys())[:20]  # Top 20 fields
    field_scores = [summary['avg_field_completeness'][f] for f in field_names]
    
    bars = ax.barh(field_names, field_scores)
    
    # Color bars based on completeness
    for bar, score in zip(bars, field_scores):
        if score < 0.3:
            bar.set_color('red')
        elif score < 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    ax.set_xlabel('Average Completeness Score')
    ax.set_title('Field Completeness Across All Puzzles')
    ax.set_xlim(0, 1)
    
    # 2. Puzzle quality distribution
    ax = axes[0, 1]
    puzzle_scores = list(analysis['puzzle_scores'].values())
    ax.hist(puzzle_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(np.mean(puzzle_scores), color='red', linestyle='--', label=f'Mean: {np.mean(puzzle_scores):.3f}')
    ax.set_xlabel('Puzzle Completeness Score')
    ax.set_ylabel('Number of Puzzles')
    ax.set_title('Distribution of Puzzle Quality Scores')
    ax.legend()
    
    # 3. Missing data patterns
    ax = axes[1, 0]
    missing_fields = list(analysis['missing_patterns'].keys())[:15]
    missing_counts = [analysis['missing_patterns'][f] for f in missing_fields]
    
    bars = ax.bar(range(len(missing_fields)), missing_counts, color='red', alpha=0.7)
    ax.set_xticks(range(len(missing_fields)))
    ax.set_xticklabels(missing_fields, rotation=45, ha='right')
    ax.set_ylabel('Number of Puzzles with Missing Data')
    ax.set_title('Most Frequently Missing Fields')
    
    # 4. Best vs Worst puzzles
    ax = axes[1, 1]
    best_puzzles = [p[0] for p in summary['best_puzzles']]
    best_scores = [p[1] for p in summary['best_puzzles']]
    worst_puzzles = [p[0] for p in summary['worst_puzzles']]
    worst_scores = [p[1] for p in summary['worst_puzzles']]
    
    x = range(5)
    ax.bar([i - 0.2 for i in x], best_scores, width=0.4, label='Best', color='green', alpha=0.7)
    ax.bar([i + 0.2 for i in x], worst_scores, width=0.4, label='Worst', color='red', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(best_puzzles, rotation=45, ha='right')
    ax.set_ylabel('Completeness Score')
    ax.set_title('Best vs Worst Puzzle Quality')
    ax.legend()
    
    plt.tight_layout()
    
    # Save analysis
    out_path = os.path.join(out_dir, f"missing_data_analysis_{puzzle_pattern}.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Save analysis report
    report_path = os.path.join(out_dir, f"completeness_report_{puzzle_pattern}.json")
    import json
    with open(report_path, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        json_summary = {}
        for key, value in summary.items():
            if isinstance(value, dict):
                json_summary[key] = {str(k): float(v) if isinstance(v, (np.float64, np.float32)) else v 
                                   for k, v in value.items()}
            else:
                json_summary[key] = value
        json.dump(json_summary, f, indent=2)
    
    logging.info(f"Saved missing data analysis to {out_path}")
    logging.info(f"Saved completeness report to {report_path}")
    
    return out_path, report_path

def save_scene_graph_visualization(G, image_path, out_dir, problem_id, abstract_view=True):
    """Save a visualization of the scene graph G overlaid on the real image."""
    os.makedirs(out_dir, exist_ok=True)
    # Node positions: use centroid, else first vertex, else (0,0)
    pos = {}
    for n, data in G.nodes(data=True):
        if 'centroid' in data and data['centroid'] is not None:
            pos[n] = data['centroid']
        elif 'vertices' in data and data['vertices']:
            pos[n] = data['vertices'][0]
        else:
            pos[n] = (0, 0)

    # For visualization: if MultiDiGraph, convert to DiGraph for plotting edge labels
    # This simplifies the visualization by showing one edge between nodes, labeled with all predicates.
    plotG = nx.DiGraph()
    if hasattr(G, 'is_multigraph') and G.is_multigraph():
        plotG.add_nodes_from(G.nodes(data=True))
        # Aggregate edges
        edge_pred_map = defaultdict(list)
        for u, v, data in G.edges(data=True):
            pred = data.get('predicate', 'unknown')
            edge_pred_map[(u, v)].append(pred)
        
        for (u, v), preds in edge_pred_map.items():
            plotG.add_edge(u, v, label=', '.join(sorted(list(set(preds)))))
    else:
        plotG.add_nodes_from(G.nodes(data=True))
        for u, v, data in G.edges(data=True):
            plotG.add_edge(u, v, label=data.get('predicate', 'unknown'))

    # --- Node color/border/label logic ---
    node_labels = {}
    node_colors = []
    node_border_colors = []
    for n, data in plotG.nodes(data=True):
        label = data.get('object_type', 'obj')
        if data.get('source') == 'geometric_grouping':
            node_colors.append('skyblue' if not data.get('is_closed') else 'lightgreen')
            node_border_colors.append('blue')
            label += f"\n(strokes: {data.get('stroke_count', 1)})"
        else: # Primitive
            node_colors.append('lightcoral')
            node_border_colors.append('red')
        node_labels[n] = label

    # --- Edge color/label logic ---
    edge_labels = nx.get_edge_attributes(plotG, 'label')
    edge_colors = []
    color_map = {
        'part_of': 'orange',
        'adjacent_endpoints': 'blue',
        'is_above': 'purple',
        'is_parallel': 'green',
        'contains': 'brown',
        'intersects': 'red',
    }
    for u, v, data in plotG.edges(data=True):
        pred = data.get('label', '').split(',')[0] # Color by first predicate
        edge_colors.append(color_map.get(pred, 'gray'))

    # Filter out labels for self-loops or edges between co-located nodes to prevent drawing errors
    drawable_edge_labels = {
        (u, v): label for (u, v), label in edge_labels.items()
        if u != v and pos.get(u) != pos.get(v)
    }
    if len(drawable_edge_labels) < len(edge_labels):
        logging.warning(f"Skipped {len(edge_labels) - len(drawable_edge_labels)} edge labels for self-loops or co-located nodes in {problem_id}.")

    # 1. Save graph overlayed on real image (if image exists)
    if image_path and os.path.exists(image_path):
        try:
            with Image.open(image_path) as img:
                plt.figure(figsize=(12, 12))
                plt.imshow(img)
                nx.draw_networkx_nodes(plotG, pos, node_color=node_colors, node_size=500, edgecolors=node_border_colors, linewidths=1.5)
                nx.draw_networkx_edges(plotG, pos, edge_color=edge_colors, arrows=True, arrowstyle='-|>', width=1.5, alpha=0.8)
                nx.draw_networkx_labels(plotG, pos, labels=node_labels, font_size=7)
                nx.draw_networkx_edge_labels(plotG, pos, edge_labels=drawable_edge_labels, font_color='black', font_size=6, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.1))
                plt.title(f"Full Scene Graph - {problem_id}")
                plt.axis('on')
                out_path_full = os.path.join(out_dir, f"{problem_id}_scene_graph.png")
                plt.savefig(out_path_full, bbox_inches='tight', dpi=150)
                plt.close()
        except Exception as e:
            logging.error(f"Failed to create full visualization for {problem_id}: {e}")

    # 2. Save pure matplotlib graph visualization (no image)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(plotG, pos, node_color=node_colors, node_size=800, edgecolors=node_border_colors, linewidths=1.5)
    nx.draw_networkx_edges(plotG, pos, edge_color=edge_colors, arrows=True, arrowstyle='-|>', width=1.5, alpha=0.8)
    nx.draw_networkx_labels(plotG, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edge_labels(plotG, pos, edge_labels=drawable_edge_labels, font_color='black', font_size=7)
    plt.title(f"Graph Only - {problem_id}")
    plt.axis('off')
    out_path_graph = os.path.join(out_dir, f"{problem_id}_graph_only.png")
    plt.savefig(out_path_graph, bbox_inches='tight')
    plt.close()

    # 3. Save abstract graph view (if enabled and applicable)
    if abstract_view:
        abstract_nodes = [n for n, data in G.nodes(data=True) if data.get('source') == 'geometric_grouping']
        if abstract_nodes:
            abstract_G = G.subgraph(abstract_nodes).copy()
            
            # Since we are creating a new figure, we need to redefine plot elements
            abstract_pos = {n: pos[n] for n in abstract_nodes}
            abstract_node_colors = ['lightgreen' if data.get('is_closed') else 'skyblue' for n, data in abstract_G.nodes(data=True)]
            abstract_node_labels = {n: f"{data.get('object_type')}\n(strokes: {data.get('stroke_count', 1)})" for n, data in abstract_G.nodes(data=True)}
            
            # Aggregate edges for abstract view
            abstract_edge_map = defaultdict(list)
            for u, v, data in abstract_G.edges(data=True):
                abstract_edge_map[(u,v)].append(data.get('predicate', 'unknown'))
            
            abstract_plot_G = nx.DiGraph()
            abstract_plot_G.add_nodes_from(abstract_G.nodes(data=True))
            for (u,v), preds in abstract_edge_map.items():
                abstract_plot_G.add_edge(u, v, label=', '.join(sorted(list(set(preds)))))

            abstract_edge_labels = nx.get_edge_attributes(abstract_plot_G, 'label')
            abstract_edge_colors = [color_map.get(l.split(',')[0], 'gray') for l in abstract_edge_labels.values()]

            plt.figure(figsize=(8, 8))
            nx.draw_networkx_nodes(abstract_plot_G, abstract_pos, node_color=abstract_node_colors, node_size=1200, edgecolors='black', linewidths=1.5)
            nx.draw_networkx_edges(abstract_plot_G, abstract_pos, edge_color=abstract_edge_colors, arrows=True, arrowstyle='-|>', width=1.5, alpha=0.9)
            nx.draw_networkx_labels(abstract_plot_G, abstract_pos, labels=abstract_node_labels, font_size=8)
            nx.draw_networkx_edge_labels(abstract_plot_G, abstract_pos, edge_labels=abstract_edge_labels, font_color='red', font_size=7)
            
            plt.title(f"Abstract View - {problem_id}")
            plt.axis('off')
            out_path_abstract = os.path.join(out_dir, f"{problem_id}_graph_abstract.png")
            plt.savefig(out_path_abstract, bbox_inches='tight')
            plt.close()

def save_scene_graph_csv(G, out_dir, problem_id):
    """Save node and edge data of the scene graph as CSV files."""
    logging.info(f"[LOGO Visualization] save_scene_graph_csv called for problem_id={problem_id}")
    import json
    import numpy as np

    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        else:
            return obj

    nodes = []
    for n, data in G.nodes(data=True):
        d = sanitize(data.copy())
        d['id'] = n
        d['all_data'] = json.dumps(d, ensure_ascii=False)
        nodes.append(d)
    edges = []
    logging.info(f"[LOGO Visualization] (CSV) Graph type: {type(G)}. is_multigraph={isinstance(G, (nx.MultiDiGraph, nx.MultiGraph))}")
    try:
        if hasattr(G, 'is_multigraph') and G.is_multigraph():
            for u, v, k, data in _robust_edge_unpack(G.edges(keys=True, data=True)):
                d = sanitize(data.copy())
                d['source'] = u
                d['target'] = v
                d['key'] = k
                d['all_data'] = json.dumps(d, ensure_ascii=False)
                edges.append(d)
        else:
            for u, v, k, data in _robust_edge_unpack(G.edges(data=True)):
                d = sanitize(data.copy())
                d['source'] = u
                d['target'] = v
                d['all_data'] = json.dumps(d, ensure_ascii=False)
                edges.append(d)
    except Exception as e:
        logging.error(f"[LOGO Visualization] Error unpacking edges for CSV: {e}")
        logging.error(f"[LOGO Visualization] Edges: {list(G.edges(data=True))}")
        raise
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(nodes).to_csv(os.path.join(out_dir, f"{problem_id}_nodes.csv"), index=False)
    if edges:
        pd.DataFrame(edges).to_csv(os.path.join(out_dir, f"{problem_id}_edges.csv"), index=False)
    else:
        logging.warning(f"[LOGO Visualization] No edges found for problem {problem_id}, skipping edge CSV.")
