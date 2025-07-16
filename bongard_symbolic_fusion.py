import numpy as np
import networkx as nx
import logging
import cv2 # For mask processing
from PIL import Image # For potential image operations, though masks are numpy arrays
import math # For geometric calculations
from collections import defaultdict # For generate_reasoning_chain
import json # For saving/loading symbolic annotations

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions for Mask Processing ---

def _mask_to_bbox(mask: np.ndarray) -> list:
    """
    Converts a binary mask to an axis-aligned bounding box [xmin, ymin, xmax, ymax].
    Args:
        mask (np.ndarray): A 2D binary mask.
    Returns:
        list: Bounding box coordinates [xmin, ymin, xmax, ymax], or [0,0,0,0] if mask is empty.
    """
    if mask.ndim != 2:
        logging.error(f"Input mask must be 2D, got {mask.ndim}D.")
        return [0, 0, 0, 0]

    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0, 0, 0, 0] # Return empty box if no foreground pixels

    xmin, ymin = int(xs.min()), int(ys.min())
    xmax, ymax = int(xs.max()), int(ys.max())
    return [xmin, ymin, xmax, ymax]

def _mask_centroid(mask: np.ndarray) -> list:
    """
    Calculates the centroid (mean x, mean y) of foreground pixels in a binary mask.
    Args:
        mask (np.ndarray): A 2D binary mask.
    Returns:
        list: Centroid coordinates [cx, cy], or [0,0] if mask is empty.
    """
    if mask.ndim != 2:
        logging.error(f"Input mask must be 2D, got {mask.ndim}D.")
        return [0, 0]

    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return [0, 0] # Return [0,0] if no foreground pixels

    return [float(xs.mean()), float(ys.mean())]

def _mask_area(mask: np.ndarray) -> int:
    """
    Calculates the area (number of foreground pixels) of a binary mask.
    Args:
        mask (np.ndarray): A 2D binary mask.
    Returns:
        int: The area of the mask.
    """
    if mask.ndim != 2:
        logging.error(f"Input mask must be 2D, got {mask.ndim}D.")
        return 0
    return int(np.sum(mask > 0))

def _mask_aspect_ratio(mask: np.ndarray) -> float:
    """
    Calculates the aspect ratio (width / height) of the bounding box of a mask.
    Args:
        mask (np.ndarray): A 2D binary mask.
    Returns:
        float: Aspect ratio, or 0.0 if mask is empty or height is zero.
    """
    bbox = _mask_to_bbox(mask)
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    if height == 0:
        return 0.0 # Avoid division by zero
    return float(width / height)

def _mask_solidity(mask: np.ndarray) -> float:
    """
    Calculates the solidity of a mask (area / bounding box area).
    Args:
        mask (np.ndarray): A 2D binary mask.
    Returns:
        float: Solidity, or 0.0 if mask or bbox is empty.
    """
    area = _mask_area(mask)
    bbox = _mask_to_bbox(mask)
    xmin, ymin, xmax, ymax = bbox
    bbox_area = (xmax - xmin) * (ymax - ymin)
    if bbox_area == 0:
        return 0.0
    return float(area / bbox_area)

# --- Geometric Relation Helpers (using xyxy bbox format) ---
def iou_xyxy(bbox1: list, bbox2: list) -> float:
    """
    Compute Intersection-over-Union (IoU) for two bounding boxes.
    Boxes are expected in [x0, y0, x1, y1] format (top-left, bottom-right corners).
    Args:
        bbox1 (list): Bounding box 1.
        bbox2 (list): Bounding box 2.
    Returns:
        float: IoU value.
    """
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def _get_spatial_relations(bbox1: list, bbox2: list, threshold_iou: float = 0.01,
                           center_dist_thresh_px: float = 10.0) -> list:
    """
    Determines common spatial relations between two bounding boxes.
    Args:
        bbox1 (list): [xmin, ymin, xmax, ymax] for object 1.
        bbox2 (list): [xmin, ymin, xmax, ymax] for object 2.
        threshold_iou (float): IoU threshold to consider boxes as "overlapping".
        center_dist_thresh_px (float): Pixel threshold for "aligned" or "near" relations.
    Returns:
        list: A list of strings representing detected relations.
    """
    relations = []
    
    # Calculate centroids
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2

    # Horizontal relations
    if cx1 < cx2 - center_dist_thresh_px:
        relations.append('left-of')
    elif cx1 > cx2 + center_dist_thresh_px:
        relations.append('right-of')
    
    # Vertical relations (smaller y is higher)
    if cy1 < cy2 - center_dist_thresh_px:
        relations.append('above')
    elif cy1 > cy2 + center_dist_thresh_px:
        relations.append('below')

    # Overlap
    current_iou = iou_xyxy(bbox1, bbox2)
    if current_iou > threshold_iou:
        relations.append('overlaps')
        if current_iou > 0.8: # High overlap
            relations.append('highly-overlaps')

    # Contains (A contains B)
    if (bbox1[0] <= bbox2[0] and bbox1[1] <= bbox2[1] and
        bbox1[2] >= bbox2[2] and bbox1[3] >= bbox2[3]):
        relations.append('contains')
    
    # Is contained by (A is contained by B)
    if (bbox2[0] <= bbox1[0] and bbox2[1] <= bbox1[1] and
        bbox2[2] >= bbox1[2] and bbox2[3] >= bbox1[3]):
        relations.append('is-contained-by')

    # Alignment (horizontal or vertical)
    # Check if centroids are roughly aligned
    if abs(cx1 - cx2) < center_dist_thresh_px:
        relations.append('vertically-aligned')
    if abs(cy1 - cy2) < center_dist_thresh_px:
        relations.append('horizontally-aligned')

    return relations

# --- Clustering/Grouping Helpers ---
def _cluster_by_proximity(G: nx.Graph, threshold: float = 50.0) -> dict:
    """
    Clusters nodes in a graph based on edge 'distance' attribute using connected components.
    Args:
        G (nx.Graph): Graph where nodes have 'centroid' properties and edges can have 'distance'.
        threshold (float): Maximum distance for two nodes to be considered in the same cluster.
    Returns:
        dict: A dictionary mapping node index to cluster ID.
    """
    clusters = {}
    cluster_id = 0
    visited = set()

    for node in G.nodes:
        if node in visited:
            continue

        # Start a BFS from the current unvisited node
        queue = [node]
        current_cluster_nodes = []

        while queue:
            n = queue.pop(0) # Use pop(0) for BFS (queue behavior)
            if n in visited:
                continue
            
            visited.add(n)
            current_cluster_nodes.append(n)
            clusters[n] = cluster_id

            for nbr in G.neighbors(n):
                # Ensure 'distance' attribute exists and is below threshold
                edge_data = G[n][nbr]
                if 'distance' in edge_data and edge_data['distance'] < threshold and nbr not in visited:
                    queue.append(nbr)
        
        if current_cluster_nodes: # Increment cluster_id only if a new cluster was found
            cluster_id += 1
    
    return clusters

# --- Main Symbolic Fusion Logic ---

def symbolic_fusion(masks: list[np.ndarray], image: np.ndarray = None, 
                    attributes: list[dict] = None, relations: list[dict] = None) -> list[dict]:
    """
    Professional symbolic fusion logic for Bongard problems or general object scene understanding.
    Extracts geometric properties, builds a relational graph, performs grouping, and adds symbolic reasoning.
    
    Args:
        masks (List[np.ndarray]): List of binary masks (H,W) for detected objects.
        image (np.ndarray, optional): Original image (H,W,3) for context. Not directly used for properties,
                                      but could be for visual debugging or color analysis.
        attributes (List[dict], optional): List of pre-computed attribute dicts for each object.
                                           If provided, these are merged with extracted properties.
                                           Expected format: [{'color': 'red', 'shape': 'circle'}, ...].
        relations (List[dict], optional): List of pre-defined relation dicts between objects.
                                          If provided, these are used to build the graph.
                                          Expected format: [{'source': 0, 'target': 1, 'type': 'left-of'}, ...].
    Returns:
        List[dict]: Symbolic annotations for each object, including extracted properties,
                    relations, and grouping information.
                    Example: [{'id': 0, 'area': 100, 'bbox': [x,y,x,y], 'centroid': [cx,cy],
                               'attributes': {...}, 'group': 0, 'relations': [...], 'is_key': True}, ...]
    """
    results = []
    if not masks:
        logging.info("No masks provided for symbolic fusion.")
        return []

    try:
        # 1. Extract basic geometric properties from masks
        for idx, mask in enumerate(masks):
            props = {
                'id': idx, # Assign an ID to each object
                'area': _mask_area(mask),
                'bbox_xyxy': _mask_to_bbox(mask), # [xmin, ymin, xmax, ymax]
                'centroid': _mask_centroid(mask),
                'aspect_ratio': _mask_aspect_ratio(mask),
                'solidity': _mask_solidity(mask),
                # Add more properties here if needed (e.g., perimeter, orientation using skimage.measure.regionprops)
            }
            # Merge with provided attributes if any
            if attributes and idx < len(attributes):
                props['attributes'] = attributes[idx]
            else:
                props['attributes'] = {} # Ensure attributes key exists
            results.append(props)
        logging.info(f"Extracted properties for {len(results)} objects.")

        # 2. Build relation graph (spatial, attribute, etc.)
        G = nx.Graph()
        for i, props_i in enumerate(results):
            G.add_node(i, **props_i) # Add all properties as node attributes

        if relations:
            # Use pre-defined relations
            for rel in relations:
                if 'source' in rel and 'target' in rel and 'type' in rel:
                    G.add_edge(rel['source'], rel['target'], relation_type=rel['type'], **rel)
                else:
                    logging.warning(f"Malformed pre-defined relation: {rel}. Skipping.")
        else:
            # Compute spatial relations between all pairs of objects
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    bbox_i = results[i]['bbox_xyxy']
                    bbox_j = results[j]['bbox_xyxy']
                    
                    # Calculate Euclidean distance between centroids for proximity clustering
                    dist = np.linalg.norm(np.array(results[i]['centroid']) - np.array(results[j]['centroid']))
                    G.add_edge(i, j, distance=float(dist)) # Add distance as an edge attribute

                    # Add specific spatial relation types as edge attributes
                    spatial_rels = _get_spatial_relations(bbox_i, bbox_j)
                    for s_rel in spatial_rels:
                        # Add a separate edge or update existing one with relation type
                        # For simplicity, we can add multiple attributes to the same edge
                        # or create directional edges if relations are asymmetric
                        if s_rel not in G[i][j]: # Avoid overwriting if already set by pre-defined relations
                            G[i][j][s_rel] = True # Mark presence of this relation

        logging.info(f"Built relational graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        # 3. Grouping logic (e.g., cluster by proximity, attribute similarity)
        # Example: group objects within a threshold distance
        # Assuming image is provided and has a shape
        image_height = image.shape[0] if image is not None else 640 # Default if no image
        proximity_threshold = image_height * 0.1 # Example: 10% of image height as proximity threshold
        
        clusters = _cluster_by_proximity(G, threshold=proximity_threshold)
        for idx, props in enumerate(results):
            props['group_id'] = clusters.get(idx, -1) # Assign cluster ID
            
            # Collect all relations for the current object from the graph
            obj_relations = []
            for nbr in G.neighbors(idx):
                edge_data = G[idx][nbr]
                relation_info = {'target_id': nbr}
                # Copy all edge attributes (e.g., 'distance', 'left-of': True)
                relation_info.update(edge_data) 
                obj_relations.append(relation_info)
            props['relations'] = obj_relations
        logging.info(f"Performed object grouping. Found {len(set(clusters.values()))} groups.")

        # 4. Symbolic reasoning (e.g., rule induction, pattern detection)
        # This is where high-level Bongard problem-solving logic would reside.
        # Examples:
        # - Identify "dominant" objects (e.g., largest, most central, most connected).
        # - Detect patterns (e.g., "three circles in a row", "object inside another object").
        # - Infer abstract properties (e.g., "all objects are open shapes").

        # Placeholder: mark objects with largest area as 'key'
        max_area = max([p['area'] for p in results]) if results else 0
        for props in results:
            props['is_largest_area'] = props['area'] == max_area
            # Example: check if object has 'contains' relation
            props['is_container'] = any('contains' in r for r in props['relations'])
            props['is_contained'] = any('is-contained-by' in r for r in props['relations'])

        logging.info("Applied basic symbolic reasoning rules.")
        return results

    except Exception as e:
        logging.error(f"Symbolic fusion failed: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return [{"error": str(e)} for _ in masks] # Return error for each mask if total failure

def generate_reasoning_chain(symbolic_annotations: list[dict]) -> str:
    """
    Generates a human-readable reasoning chain or summary from symbolic annotations.
    This is a placeholder for a more sophisticated rule-based or LLM-driven explanation.
    Args:
        symbolic_annotations (list[dict]): Output from symbolic_fusion.
    Returns:
        str: A string describing the inferred properties and relations.
    """
    if not symbolic_annotations:
        return "No objects detected for reasoning."

    summary_lines = []
    summary_lines.append("--- Symbolic Reasoning Chain ---")
    
    # Global observations
    num_objects = len(symbolic_annotations)
    summary_lines.append(f"Total objects detected: {num_objects}")

    # Group-based observations
    groups = defaultdict(list)
    for obj in symbolic_annotations:
        if 'group_id' in obj and obj['group_id'] != -1:
            groups[obj['group_id']].append(obj)
    
    if groups:
        summary_lines.append(f"Identified {len(groups)} distinct groups:")
        for group_id, objects_in_group in groups.items():
            obj_ids = [obj['id'] for obj in objects_in_group]
            summary_lines.append(f"  Group {group_id}: Objects {obj_ids}")
            # Add more detail about group properties (e.g., avg size, common attributes)
    else:
        summary_lines.append("No distinct object groupings found.")

    # Object-specific observations
    for obj in symbolic_annotations:
        obj_id = obj.get('id', 'N/A')
        line = f"Object {obj_id} (Group {obj.get('group_id', -1)}):"
        
        # Basic properties
        if 'area' in obj: line += f" Area={obj['area']}"
        if 'aspect_ratio' in obj: line += f" AspectRatio={obj['aspect_ratio']:.2f}"
        if 'solidity' in obj: line += f" Solidity={obj['solidity']:.2f}"
        
        # Attributes
        if obj.get('attributes'):
            attr_str = ", ".join([f"{k}:{v}" for k,v in obj['attributes'].items()])
            line += f" Attributes=[{attr_str}]"
        
        # Symbolic flags
        if obj.get('is_largest_area'): line += " [LARGEST]"
        if obj.get('is_container'): line += " [CONTAINER]"
        if obj.get('is_contained'): line += " [CONTAINED]"

        summary_lines.append(line)

        # Relations
        if obj.get('relations'):
            for rel in obj['relations']:
                target_id = rel.get('target_id', 'N/A')
                rel_types = [k for k,v in rel.items() if v is True and k != 'target_id' and k != 'distance']
                dist_info = f" (dist={rel.get('distance', -1):.1f})" if 'distance' in rel else ""
                if rel_types:
                    summary_lines.append(f"  -> Relates to Object {target_id}: {', '.join(rel_types)}{dist_info}")
                elif 'distance' in rel:
                    summary_lines.append(f"  -> Near Object {target_id}{dist_info}")

    summary_lines.append("--- End Reasoning Chain ---")
    return "\n".join(summary_lines)


# --- CLI Entry Point ---
if __name__ == "__main__":
    import argparse
    import os
    import glob # For listing files
    
    parser = argparse.ArgumentParser(description="Symbolic Fusion and Reasoning Utility")
    parser.add_argument('--image_path', type=str, help='Path to the image (optional, for context/display)')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing binary mask files (.png)')
    parser.add_argument('--output_dir', type=str, default='symbolic_output', help='Output directory for results')
    parser.add_argument('--proximity_threshold', type=float, default=0.1, 
                        help='Proximity threshold for grouping (relative to image height, 0.1 means 10% of height)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load image (optional)
    image_np = None
    if args.image_path and os.path.exists(args.image_path):
        image_pil = Image.open(args.image_path).convert('RGB')
        image_np = np.array(image_pil)
        logging.info(f"Loaded image from {args.image_path} with shape {image_np.shape}")
    else:
        logging.warning(f"Image path '{args.image_path}' not found or not provided. Image context will be limited. Defaulting to 640x640 for relative thresholds.")
        # Create a dummy image shape if not provided, for relative thresholds
        image_np = np.zeros((640, 640, 3), dtype=np.uint8) # Default to 640x640

    # Load masks from directory
    masks = []
    mask_files = sorted([f for f in os.listdir(args.mask_dir) if f.lower().endswith('.png')])
    if not mask_files:
        logging.error(f"No PNG mask files found in {args.mask_dir}. Exiting.")
        exit()

    for mask_file in mask_files:
        mask_path = os.path.join(args.mask_dir, mask_file)
        try:
            # Read mask as grayscale, ensure it's binary (0 or 255)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logging.warning(f"Could not read mask file: {mask_path}. Skipping.")
                continue
            masks.append((mask > 0).astype(np.uint8)) # Convert to binary 0/1 mask
        except Exception as e:
            logging.error(f"Error loading mask {mask_file}: {e}. Skipping.")
            continue
    
    if not masks:
        logging.error("No valid masks loaded. Exiting.")
        exit()

    # Run symbolic fusion
    # The `proximity_threshold` argument to symbolic_fusion is derived from image_height * args.proximity_threshold
    # The function itself handles the `_cluster_by_proximity` call with this calculated threshold.
    
    symbolic_annotations = symbolic_fusion(
        masks=masks, 
        image=image_np, # Pass image_np for context-aware thresholding
        attributes=None, 
        relations=None # Let the function compute spatial relations
    )

    # Save symbolic annotations
    output_json_path = os.path.join(args.output_dir, 'symbolic_annotations.json')
    with open(output_json_path, 'w') as f:
        json.dump(symbolic_annotations, f, indent=4)
    logging.info(f"Symbolic annotations saved to {output_json_path}")

    # Generate reasoning chain
    reasoning_chain = generate_reasoning_chain(symbolic_annotations)
    output_txt_path = os.path.join(args.output_dir, 'reasoning_chain.txt')
    with open(output_txt_path, 'w') as f:
        f.write(reasoning_chain)
    logging.info(f"Reasoning chain saved to {output_txt_path}")

    print("\n--- Symbolic Fusion Results ---")
    print(f"Processed {len(masks)} masks.")
    print(f"Output saved to: {args.output_dir}")
    print("\nExample Symbolic Annotation (first object):")
    if symbolic_annotations:
        print(json.dumps(symbolic_annotations[0], indent=2))
    else:
        print("No symbolic annotations generated.")
    print("\nReasoning Chain Preview:")
    print(reasoning_chain)
