import json
import pandas as pd
import numpy as np

print("=== SCENE GRAPH DATA QUALITY ANALYSIS ===")

# Load the edges CSV from the attachment
try:
    edges_df = pd.read_csv('feedback/visualizations_logo/bd_asymmetric_unbala_x_0000_edges.csv')
    print('\n1. Scene Graph Edges Analysis:')
    print(f'   Total edges: {len(edges_df)}')
    print(f'   Unique predicates: {edges_df["predicate"].nunique()}')
    print('\n   Predicate distribution:')
    print(edges_df['predicate'].value_counts())
    print('\n   Missing data in edges:')
    print(edges_df.isnull().sum())
    
    # Check for empty values
    print('\n   Empty string values:')
    for col in edges_df.columns:
        empty_count = (edges_df[col] == '').sum()
        if empty_count > 0:
            print(f'   {col}: {empty_count} empty values')
            
except Exception as e:
    print(f"Error loading edges CSV: {e}")

# Load nodes CSV if available
try:
    nodes_df = pd.read_csv('feedback/visualizations_logo/bd_asymmetric_unbala_x_0000_nodes.csv')
    print(f'\n2. Scene Graph Nodes Analysis:')
    print(f'   Total nodes: {len(nodes_df)}')
    print(f'   Node columns: {list(nodes_df.columns)}')
    print('\n   Missing data in nodes:')
    print(nodes_df.isnull().sum())
    
    # Check object types
    if 'object_type' in nodes_df.columns:
        print('\n   Object type distribution:')
        print(nodes_df['object_type'].value_counts())
        
except Exception as e:
    print(f"Error loading nodes CSV: {e}")

# Check derived labels structure
try:
    with open('data/derived_labels.json', 'r') as f:
        sample_data = json.load(f)
    
    print(f'\n3. Derived Labels Analysis:')
    print(f'   Total records: {len(sample_data)}')
    
    # Sample the first record to understand structure
    if sample_data:
        sample = sample_data[0]
        print(f'   Sample record keys: {list(sample.keys())}')
        if 'features' in sample:
            print(f'   Features available: {list(sample["features"].keys())}')
        if 'geometry' in sample:
            print(f'   Geometry points: {len(sample["geometry"])}')
            
except Exception as e:
    print(f"Error loading derived labels: {e}")

# Check action programs if available
print('\n4. Action Program Data Analysis:')
try:
    # Look for action program files
    import os
    action_files = []
    for root, dirs, files in os.walk('data/raw'):
        for file in files:
            if 'action' in file.lower():
                action_files.append(os.path.join(root, file))
    
    print(f'   Found action program files: {len(action_files)}')
    for f in action_files[:5]:  # Show first 5
        print(f'   - {f}')
        
except Exception as e:
    print(f"Error checking action programs: {e}")

print('\n=== ANALYSIS COMPLETE ===')
