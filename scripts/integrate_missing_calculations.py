"""
Integration Script for Missing Data Calculations
Integrates the new calculation functions with the existing scene graph pipeline
"""

import os
import sys
import logging
import pandas as pd
import networkx as nx
import json
from typing import Dict, List, Any, Tuple
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.missing_calculations import DataCalculationPipeline, VLEmbeddingComputer, OpenCurveCalculator, MotifFeatureCalculator
from scripts.updated_scene_graph_visualization import (
    save_enhanced_scene_graph_visualization, 
    save_comprehensive_scene_graph_csv,
    create_puzzle_overview_visualization,
    analyze_missing_data_report
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedBongardProcessor:
    """Enhanced processor that integrates missing data calculations with visualization"""
    
    def __init__(self, enable_vl_embeddings=True, data_dir="data", output_dir="feedback/visualizations_enhanced"):
        self.enable_vl_embeddings = enable_vl_embeddings
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.calculation_pipeline = DataCalculationPipeline(enable_vl_embeddings)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def process_single_problem(self, problem_id: str, graph_path: str = None, image_path: str = None):
        """
        Process a single Bongard problem with enhanced calculations and visualization
        
        Args:
            problem_id: Problem identifier (e.g., 'bd_asymmetric_unbala_x_0000')
            graph_path: Path to existing graph file (optional)
            image_path: Path to source image (optional)
        """
        logging.info(f"Processing problem: {problem_id}")
        
        try:
            # Step 1: Load or build scene graph
            G = self._load_or_build_graph(problem_id, graph_path)
            if G is None:
                logging.error(f"Could not load graph for {problem_id}")
                return
            
            # Step 2: Apply missing data calculations
            enhanced_G = self._apply_enhanced_calculations(G, problem_id, image_path)
            
            # Step 3: Generate enhanced visualizations
            self._generate_enhanced_visualizations(enhanced_G, problem_id, image_path)
            
            # Step 4: Export enhanced CSV data
            self._export_enhanced_csv(enhanced_G, problem_id)
            
            # Step 5: Generate analysis reports
            self._generate_analysis_reports(enhanced_G, problem_id)
            
            logging.info(f"Completed processing for {problem_id}")
            
        except Exception as e:
            logging.error(f"Error processing {problem_id}: {e}")
            raise
    
    def process_puzzle_batch(self, puzzle_pattern: str = "bd_*"):
        """
        Process multiple problems matching a pattern
        
        Args:
            puzzle_pattern: Glob pattern for problem files
        """
        # Find CSV files matching pattern
        csv_files = list(Path(self.data_dir).glob(f"**/*{puzzle_pattern}*nodes.csv"))
        
        logging.info(f"Found {len(csv_files)} problems to process")
        
        puzzle_groups = {}
        
        for csv_file in csv_files:
            # Extract problem ID
            problem_id = csv_file.stem.replace('_nodes', '').replace('_enhanced_nodes', '')
            
            # Group by puzzle (remove image number suffix)
            puzzle_base = '_'.join(problem_id.split('_')[:-1]) if problem_id.count('_') > 3 else problem_id
            
            if puzzle_base not in puzzle_groups:
                puzzle_groups[puzzle_base] = []
            puzzle_groups[puzzle_base].append(problem_id)
        
        # Process each puzzle group
        for puzzle_base, problem_ids in puzzle_groups.items():
            logging.info(f"Processing puzzle group: {puzzle_base} ({len(problem_ids)} images)")
            
            try:
                self._process_puzzle_group(puzzle_base, problem_ids)
            except Exception as e:
                logging.error(f"Error processing puzzle group {puzzle_base}: {e}")
    
    def _load_or_build_graph(self, problem_id: str, graph_path: str = None) -> nx.Graph:
        """Load existing graph or build from CSV data"""
        
        # Try to find graph files
        if graph_path and os.path.exists(graph_path):
            return self._load_graph_from_file(graph_path)
        
        # Try to find CSV files
        nodes_csv = self._find_csv_file(problem_id, 'nodes')
        edges_csv = self._find_csv_file(problem_id, 'edges')
        
        if nodes_csv:
            return self._load_graph_from_csv(nodes_csv, edges_csv)
        
        # Try to build from existing scene graph pipeline
        return self._build_graph_from_pipeline(problem_id)
    
    def _find_csv_file(self, problem_id: str, file_type: str) -> str:
        """Find CSV file for problem"""
        # Search in data directory
        search_patterns = [
            f"{problem_id}_{file_type}.csv",
            f"{problem_id}_enhanced_{file_type}.csv",
            f"**/{problem_id}_{file_type}.csv",
            f"**/{problem_id}_enhanced_{file_type}.csv"
        ]
        
        for pattern in search_patterns:
            files = list(Path(self.data_dir).glob(pattern))
            if files:
                return str(files[0])
        
        return None
    
    def _load_graph_from_csv(self, nodes_csv: str, edges_csv: str = None) -> nx.Graph:
        """Load graph from CSV files"""
        try:
            G = nx.MultiDiGraph()
            
            # Load nodes
            nodes_df = pd.read_csv(nodes_csv)
            for _, row in nodes_df.iterrows():
                node_data = self._parse_csv_row(row)
                node_id = node_data.pop('id', str(row.name))
                G.add_node(node_id, **node_data)
            
            # Load edges if available
            if edges_csv and os.path.exists(edges_csv):
                edges_df = pd.read_csv(edges_csv)
                for _, row in edges_df.iterrows():
                    edge_data = self._parse_csv_row(row)
                    source = edge_data.pop('source', None)
                    target = edge_data.pop('target', None)
                    if source and target:
                        G.add_edge(source, target, **edge_data)
            
            logging.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logging.error(f"Error loading graph from CSV: {e}")
            return None
    
    def _parse_csv_row(self, row: pd.Series) -> Dict:
        """Parse CSV row, handling JSON strings and special types"""
        data = {}
        
        for key, value in row.items():
            if pd.isna(value):
                data[key] = None
            elif isinstance(value, str):
                # Try to parse JSON strings
                if value.startswith('[') or value.startswith('{'):
                    try:
                        data[key] = json.loads(value)
                    except:
                        data[key] = value
                else:
                    data[key] = value
            else:
                data[key] = value
        
        return data
    
    def _build_graph_from_pipeline(self, problem_id: str) -> nx.Graph:
        """Build graph using existing pipeline (if available)"""
        # This would integrate with the existing scene graph building pipeline
        # For now, return None to indicate we need CSV files
        logging.warning(f"Could not find graph or CSV files for {problem_id}")
        return None
    
    def _apply_enhanced_calculations(self, G: nx.Graph, problem_id: str, image_path: str = None) -> nx.Graph:
        """Apply enhanced calculations to all nodes in the graph"""
        logging.info(f"Applying enhanced calculations to {G.number_of_nodes()} nodes")
        
        enhanced_G = G.copy()
        nodes_processed = 0
        
        for node_id, node_data in G.nodes(data=True):
            try:
                # Determine image path for this node
                node_image_path = image_path
                if not node_image_path and 'image_path' in node_data:
                    node_image_path = node_data['image_path']
                
                # Apply calculations
                updated_data = self.calculation_pipeline.process_node_data(
                    dict(node_data), node_image_path
                )
                
                # Update node in graph
                enhanced_G.nodes[node_id].update(updated_data)
                nodes_processed += 1
                
            except Exception as e:
                logging.warning(f"Error processing node {node_id}: {e}")
                enhanced_G.nodes[node_id]['calculation_error'] = str(e)
        
        # Apply motif-level calculations
        enhanced_G = self._apply_motif_calculations(enhanced_G)
        
        logging.info(f"Enhanced calculations applied to {nodes_processed} nodes")
        return enhanced_G
    
    def _apply_motif_calculations(self, G: nx.Graph) -> nx.Graph:
        """Apply motif-level calculations"""
        motif_nodes = [n for n, d in G.nodes(data=True) if d.get('is_motif', False)]
        
        for motif_node in motif_nodes:
            try:
                # Find motif members
                motif_members = []
                member_relationships = []
                
                # Look for nodes that are part of this motif
                for node, data in G.nodes(data=True):
                    if data.get('motif_id') == G.nodes[motif_node].get('motif_id'):
                        motif_members.append(data)
                
                # Get relationships between members
                for u, v, edge_data in G.edges(data=True):
                    if (G.nodes[u].get('motif_id') == G.nodes[motif_node].get('motif_id') and
                        G.nodes[v].get('motif_id') == G.nodes[motif_node].get('motif_id')):
                        member_relationships.append(edge_data)
                
                # Calculate motif features
                if motif_members:
                    motif_features = self.calculation_pipeline.motif_calculator.calculate_motif_features(
                        motif_members, member_relationships
                    )
                    G.nodes[motif_node].update(motif_features)
                    
                    # Calculate motif VL embedding if enabled
                    if self.calculation_pipeline.vl_computer:
                        member_images = [m.get('image_path') for m in motif_members if m.get('image_path')]
                        member_bboxes = [m.get('bbox') for m in motif_members if m.get('bbox')]
                        
                        if member_images:
                            motif_embedding = self.calculation_pipeline.vl_computer.compute_motif_embedding(
                                member_images, member_bboxes
                            )
                            G.nodes[motif_node]['vl_embed'] = motif_embedding
                
            except Exception as e:
                logging.warning(f"Error calculating motif features for {motif_node}: {e}")
                G.nodes[motif_node]['motif_calculation_error'] = str(e)
        
        return G
    
    def _generate_enhanced_visualizations(self, G: nx.Graph, problem_id: str, image_path: str = None):
        """Generate enhanced visualizations"""
        output_dir = os.path.join(self.output_dir, problem_id)
        
        # Generate comprehensive visualization
        save_enhanced_scene_graph_visualization(
            G, image_path, output_dir, problem_id,
            show_all_relationships=True,
            highlight_missing_data=True
        )
        
        logging.info(f"Enhanced visualizations saved to {output_dir}")
    
    def _export_enhanced_csv(self, G: nx.Graph, problem_id: str):
        """Export enhanced CSV data"""
        output_dir = os.path.join(self.output_dir, problem_id)
        
        save_comprehensive_scene_graph_csv(G, output_dir, problem_id)
        
        logging.info(f"Enhanced CSV data saved to {output_dir}")
    
    def _generate_analysis_reports(self, G: nx.Graph, problem_id: str):
        """Generate analysis reports"""
        output_dir = os.path.join(self.output_dir, problem_id)
        
        # Generate missing data analysis
        report_file = os.path.join(output_dir, f"{problem_id}_analysis_report.json")
        
        # Basic analysis of the enhanced graph
        analysis = {
            'problem_id': problem_id,
            'timestamp': pd.Timestamp.now().isoformat(),
            'graph_stats': {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'motif_nodes': len([n for n, d in G.nodes(data=True) if d.get('is_motif', False)])
            },
            'data_quality': self._analyze_data_quality(G),
            'missing_data_summary': self._analyze_missing_data(G)
        }
        
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logging.info(f"Analysis report saved to {report_file}")
    
    def _analyze_data_quality(self, G: nx.Graph) -> Dict:
        """Analyze data quality in the graph"""
        total_nodes = G.number_of_nodes()
        
        quality_metrics = {
            'nodes_with_vl_embeddings': 0,
            'nodes_with_valid_geometry': 0,
            'nodes_with_curvature_data': 0,
            'motif_nodes_with_features': 0
        }
        
        for node, data in G.nodes(data=True):
            # Check VL embeddings
            vl_embed = data.get('vl_embed', [])
            if vl_embed and not all(x == 0 for x in vl_embed):
                quality_metrics['nodes_with_vl_embeddings'] += 1
            
            # Check geometry validity
            if data.get('geometry_valid', True):
                quality_metrics['nodes_with_valid_geometry'] += 1
            
            # Check curvature data for curves
            if data.get('object_type') in ['curve', 'open_curve', 'arc']:
                if data.get('curvature_score', 0) > 0:
                    quality_metrics['nodes_with_curvature_data'] += 1
            
            # Check motif features
            if data.get('is_motif', False) and data.get('motif_features'):
                quality_metrics['motif_nodes_with_features'] += 1
        
        # Calculate percentages
        quality_percentages = {}
        for key, count in quality_metrics.items():
            quality_percentages[f"{key}_percentage"] = (count / total_nodes * 100) if total_nodes > 0 else 0
        
        quality_metrics.update(quality_percentages)
        return quality_metrics
    
    def _analyze_missing_data(self, G: nx.Graph) -> Dict:
        """Analyze missing data patterns"""
        missing_data = {
            'vl_embed_missing': 0,
            'curvature_missing': 0,
            'centroid_missing': 0,
            'motif_id_missing': 0
        }
        
        for node, data in G.nodes(data=True):
            # VL embedding missing
            vl_embed = data.get('vl_embed', [])
            if not vl_embed or all(x == 0 for x in vl_embed):
                missing_data['vl_embed_missing'] += 1
            
            # Curvature missing for curves
            if (data.get('object_type') in ['curve', 'open_curve', 'arc'] and 
                data.get('curvature_score', 0) == 0):
                missing_data['curvature_missing'] += 1
            
            # Centroid missing
            if not data.get('centroid'):
                missing_data['centroid_missing'] += 1
            
            # Motif ID missing for motifs
            if data.get('is_motif', False) and not data.get('motif_id'):
                missing_data['motif_id_missing'] += 1
        
        return missing_data
    
    def _process_puzzle_group(self, puzzle_base: str, problem_ids: List[str]):
        """Process a group of problems belonging to the same puzzle"""
        logging.info(f"Processing puzzle group: {puzzle_base}")
        
        # Process individual problems
        graph_data = []
        for problem_id in problem_ids:
            try:
                # Load and process individual graph
                nodes_csv = self._find_csv_file(problem_id, 'nodes')
                edges_csv = self._find_csv_file(problem_id, 'edges')
                
                if nodes_csv:
                    G = self._load_graph_from_csv(nodes_csv, edges_csv)
                    if G:
                        enhanced_G = self._apply_enhanced_calculations(G, problem_id)
                        
                        # Find corresponding image
                        image_path = self._find_image_for_problem(problem_id)
                        graph_data.append((enhanced_G, image_path))
                        
                        # Process individual problem
                        self._generate_enhanced_visualizations(enhanced_G, problem_id, image_path)
                        self._export_enhanced_csv(enhanced_G, problem_id)
                        self._generate_analysis_reports(enhanced_G, problem_id)
                        
            except Exception as e:
                logging.error(f"Error processing {problem_id}: {e}")
        
        # Create puzzle overview visualization
        if graph_data:
            overview_output_dir = os.path.join(self.output_dir, puzzle_base)
            create_puzzle_overview_visualization(graph_data, overview_output_dir, puzzle_base)
            
            # Generate puzzle-level missing data report
            analyze_missing_data_report(overview_output_dir, 
                                      os.path.join(overview_output_dir, f"{puzzle_base}_missing_data_report.json"))
    
    def _find_image_for_problem(self, problem_id: str) -> str:
        """Find image file for a problem"""
        # Search for images in common directories
        search_dirs = [
            self.data_dir,
            os.path.join(self.data_dir, 'raw'),
            os.path.join(self.data_dir, 'images'),
            'data/raw',
            'data/images'
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                # Look for PNG, JPG files
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    files = list(Path(search_dir).glob(f"**/{problem_id}.{ext[2:]}"))
                    if files:
                        return str(files[0])
        
        return None

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Enhanced Bongard Problem Processor")
    parser.add_argument('--problem_id', type=str, help="Process specific problem ID")
    parser.add_argument('--puzzle_pattern', type=str, default="bd_*", 
                       help="Pattern for batch processing (default: bd_*)")
    parser.add_argument('--data_dir', type=str, default="data", 
                       help="Data directory (default: data)")
    parser.add_argument('--output_dir', type=str, default="feedback/visualizations_enhanced", 
                       help="Output directory (default: feedback/visualizations_enhanced)")
    parser.add_argument('--disable_vl', action='store_true', 
                       help="Disable VL embedding computation")
    parser.add_argument('--batch', action='store_true', 
                       help="Process in batch mode")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = EnhancedBongardProcessor(
        enable_vl_embeddings=not args.disable_vl,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    if args.problem_id:
        # Process single problem
        processor.process_single_problem(args.problem_id)
    elif args.batch:
        # Process batch
        processor.process_puzzle_batch(args.puzzle_pattern)
    else:
        print("Please specify --problem_id for single processing or --batch for batch processing")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
