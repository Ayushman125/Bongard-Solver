import logging
import os
import numpy as np
import traceback
from typing import List, Dict, Any
from src.scene_graphs_building.data_loading import remap_path




def are_points_collinear(verts, tol=1e-6):
    arr = np.asarray(verts)
    if len(arr) < 3:
        return True
    v0 = arr[0]
    v1 = arr[1]
    direction = v1 - v0
    norm = np.linalg.norm(direction)
    if norm < tol:
        return True
    direction = direction / norm
    for v in arr[2:]:
        rel = v - v0
        proj = np.dot(rel, direction)
        perp = rel - proj * direction
        if np.linalg.norm(perp) > tol:
            return False
    return True

def assign_object_type(verts):
    arr = np.asarray(verts)
    if len(arr) == 1 or np.allclose(np.std(arr, axis=0), 0, atol=1e-8):
        return "point"
    elif len(arr) == 2 or are_points_collinear(arr):
        return "line"
    elif len(arr) >= 3 and not are_points_collinear(arr):
        return "polygon"
    else:
        # SOTA: treat degenerate/ambiguous as 'curve' if possible
        return "curve" if len(arr) > 1 else "unknown"

def parse_action_command(cmd):
    if isinstance(cmd, dict):
        return cmd
    if not isinstance(cmd, str):
        return None
    parts = cmd.split('_', 2)
    # Handle formats like "line_normal_0.2-0.3" (3 parts) and "start_0.1-0.2" (2 parts)
    if len(parts) == 3:
        shape, mode, rest = parts
    elif len(parts) == 2:
        shape, rest = parts
        mode = None # No mode specified
    else:
        return None

    if shape == "line" and '-' in rest:
        try:
            a, b = rest.split('-',1)
            return {'type':'line', 'mode':mode, 'x':float(a), 'y':float(b)}
        except Exception:
            return None
    elif shape == "start" and '-' in rest:
        try:
            a, b = rest.split('-',1)
            return {'type':'start', 'x':float(a), 'y':float(b)}
        except Exception:
            return None
    elif shape == "arc" and '-' in rest:
        try:
            radius, angle = rest.split('-',1)
            return {'type':'arc', 'mode':mode, 'radius':float(radius), 'angle':float(angle)}
        except Exception:
            return None
    elif shape == "turn" and '-' in rest:
        try:
            angle = rest
            return {'type':'turn', 'mode':mode, 'angle':float(angle)}
        except Exception:
            return None
    # Extend for other commands as needed
    return None




async def _process_single_problem(problem_id: str, problem_records: List[Dict[str, Any]], feature_cache, *, args=None):
    # ...existing code...

    objects = []
    required_fields = ['action_index', 'parent_shape_id', 'turn_direction', 'action_command', 'programmatic_label', 'stroke_type', 'endpoints', 'length', 'orientation', 'label', 'vertices',
                      'geometry_reason', 'is_closed', 'compactness', 'feature_valid', 'stroke_count', 'clustered', 'motif_membership', 'image_path', 'is_valid',
                      'fallback_geometry', 'bounding_box', 'centroid', 'area', 'perimeter', 'aspect_ratio', 'curvature', 'skeleton_length', 'symmetry_axis',
                      'relationships', 'predicate', 'source', 'parent_id']
    # Diversity statistics tracker
    from collections import defaultdict
    diversity_stats = defaultdict(set)
    # ConceptNet mapping utility (stub)
    def enrich_kb_concept(concept):
        # TODO: integrate ConceptNet API
        return concept
    # --- Perform predicate induction ONCE for the entire problem ---
    predicate_val = None
    try:
        from src.scene_graphs_building.predicate_induction import induce_predicate_for_problem
        if args is not None and hasattr(args, 'mode') and args.mode == 'logo':
            # Pass all records for the problem to get meaningful stats
            predicate_val, _ = induce_predicate_for_problem(problem_records)
    except Exception as e:
        logging.error(f"Predicate induction failed for problem {problem_id}: {e}")
        predicate_val = None

    for idx, rec in enumerate(problem_records):
        parent_shape_id = f"{problem_id}_{idx}"
        action_program = rec.get('action_program', [])
        used_fallback = False
        
        label = rec.get('label', '')
        shape_label = rec.get('shape_label', label)
        category = rec.get('category', label)
        stroke_type = rec.get('stroke_type', '')
        programmatic_label = rec.get('programmatic_label', '')
        object_color = rec.get('object_color', '')
        component_index = rec.get('component_index', idx)
        # --- Compute cluster_label, motif_type, motif_membership, compactness if not present ---
        cluster_label = rec.get('cluster_label', None)
        clustered = cluster_label is not None
        # Example: fallback to index for cluster_label if not present
        if cluster_label is None:
            cluster_label = idx
            clustered = False
        kb_concept = enrich_kb_concept(rec.get('kb_concept', None))
        global_stat = rec.get('global_stat', None)
        compactness = rec.get('compactness', None)
        # Compute compactness if not present and geometry available
        if compactness is None:
            verts = rec.get('geometry') or rec.get('vertices') or []
            if len(verts) >= 3:
                try:
                    from shapely.geometry import Polygon
                    poly = Polygon(verts)
                    area = poly.area
                    perimeter = poly.length
                    if perimeter > 0:
                        compactness = 4 * np.pi * area / (perimeter ** 2)
                except Exception:
                    compactness = None
        # Compute additional geometry attributes if possible
        # Compute compactness if not present and geometry available
        if compactness is None:
            verts = rec.get('geometry') or rec.get('vertices') or []
            if len(verts) >= 3:
                try:
                    # Placeholder for compactness calculation
                    pass
                except Exception:
                    pass
        # Compute additional geometry attributes if possible
        bounding_box = None
        centroid = None
        area = None
        perimeter = None
        aspect_ratio = None
        curvature = None
        skeleton_length = None
        symmetry_axis = None
        fallback_geometry = None
        verts_for_geom = rec.get('geometry') or rec.get('vertices') or []
        if len(verts_for_geom) >= 3:
            try:
                from shapely.geometry import Polygon
                poly = Polygon(verts_for_geom)
                bounding_box = poly.bounds
                centroid = list(poly.centroid.coords)[0]
                area = poly.area
                perimeter = poly.length
                minx, miny, maxx, maxy = poly.bounds
                width = maxx - minx
                height = maxy - miny
                aspect_ratio = width / height if height != 0 else None
                # curvature, skeleton_length, symmetry_axis: placeholders
                curvature = None
                skeleton_length = None
                symmetry_axis = None
                fallback_geometry = verts_for_geom
            except Exception:
                bounding_box = None
                centroid = None
                area = None
                perimeter = None
                aspect_ratio = None
                curvature = None
                skeleton_length = None
                symmetry_axis = None
                fallback_geometry = verts_for_geom
        motif_type = rec.get('motif_type', None)
        if motif_type is None:
            motif_type = 'unknown'
        motif_membership = rec.get('motif_membership', None)
        if motif_membership is None:
            motif_membership = f"motif_{idx}"
        subcategory = rec.get('subcategory', None)
        supercategory = rec.get('supercategory', None)
        semantic_type = rec.get('semantic_type', None)
        instance_id = rec.get('instance_id', None)
        group_id = rec.get('group_id', None)
        part_of = rec.get('part_of', None)
        relationship_type = rec.get('relationship_type', None)
        symmetry_type = rec.get('symmetry_type', None)
        motif_membership = rec.get('motif_membership', None)
        diversity_stat = rec.get('diversity_stat', None)
        image_path = rec.get('image_path', None)
        geometry_reason = None
        feature_valid = {}
        is_valid = True
        stroke_count = 0
        # --- LOGO simulation ---
        if action_program:
            turtle_pos = None
            turtle_heading = 0.0
            pen_down = True
            prev_pos = None
            verts_seq = []
            for stroke_idx, cmd in enumerate(action_program):
                parsed_cmd = parse_action_command(cmd)
                if not parsed_cmd:
                    continue
                cmd_type = parsed_cmd.get('type')
                command_mode = parsed_cmd.get('mode', None)
                command_params = {k: v for k, v in parsed_cmd.items() if k not in ['type', 'mode']}
                if cmd_type == 'start':
                    turtle_pos = [parsed_cmd['x'], parsed_cmd['y']]
                    turtle_heading = 0.0
                    prev_pos = list(turtle_pos)
                    verts_seq.append(list(turtle_pos))
                    continue
                elif cmd_type == 'line':
                    if turtle_pos is None:
                        turtle_pos = [0.0, 0.0]
                        verts_seq.append(list(turtle_pos))
                    dx = parsed_cmd['x']
                    dy = parsed_cmd['y']
                    new_pos = [turtle_pos[0] + dx, turtle_pos[1] + dy]
                    verts = [list(turtle_pos), list(new_pos)]
                    verts_seq.append(list(new_pos))
                    obj_id = f"{problem_id}_{idx}_{stroke_idx}"
                    stroke_count += 1
                    geometry_reason = 'line'
                    feature_valid = {'vertices': True, 'endpoints': True, 'length': True, 'orientation': True}
                    is_closed = False
                    if len(verts) >= 3:
                        arr = np.array(verts)
                        is_closed = np.allclose(arr[0], arr[-1], atol=1e-6)
                    bounding_box, centroid, area, perimeter, aspect_ratio, compactness_stroke = _calculate_stroke_geometry(verts)
                    # Motif membership: parse from action_program if available, else default
                    motif_membership_line = motif_membership if motif_membership is not None else f"motif_{idx}"
                    # Fallback geometry: always use verts
                    fallback_geometry_line = verts
                    # Symmetry axis: infer for regular shapes
                    symmetry_axis_line = None
                    if command_mode in ['triangle', 'square', 'circle']:
                        symmetry_axis_line = command_mode
                    # Relationships: infer adjacency (previous stroke)
                    relationships_line = []
                    if stroke_idx > 0:
                        relationships_line.append(f"adjacent_to_{problem_id}_{idx}_{stroke_idx-1}")
                    # Induce predicate for stroke
                    
                    obj = {
                        'object_id': obj_id,
                        'parent_shape_id': parent_shape_id,
                        'action_index': stroke_idx,
                        'vertices': verts,
                        'object_type': assign_object_type(verts),
                        'turn_direction': command_mode,
                        'action_command': cmd,
                        'command_type': cmd_type,
                        'command_mode': command_mode,
                        'command_params': command_params,
                        'programmatic_label': programmatic_label if programmatic_label else cmd_type,
                        'stroke_type': stroke_type if stroke_type else cmd_type,
                        'label': label,
                        'shape_label': shape_label,
                        'category': category,
                        'original_image_path': remap_path(rec.get('image_path', '')),
                        'image_path': image_path,
                        'original_record_idx': idx,
                        'action_program': action_program,
                        'endpoints': [verts[0], verts[-1]],
                        'length': float(np.linalg.norm(np.array(verts[-1]) - np.array(verts[0]))),
                        'orientation': float(np.degrees(np.arctan2(dy, dx))),
                        'object_color': object_color,
                        'component_index': component_index,
                        'cluster_label': cluster_label,
                        'clustered': clustered,
                        'kb_concept': kb_concept,
                        'global_stat': global_stat,
                        'compactness': compactness_stroke,
                        'motif_type': motif_type,
                        'motif_membership': motif_membership_line if motif_membership_line is not None else '',
                        'stroke_count': stroke_count,
                        'subcategory': subcategory,
                        'supercategory': supercategory,
                        'semantic_type': semantic_type,
                        'instance_id': instance_id,
                        'group_id': group_id,
                        'part_of': part_of,
                        'relationship_type': relationship_type,
                        'symmetry_type': symmetry_type,
                        'diversity_stat': diversity_stat,
                        'geometry_reason': geometry_reason,
                        'is_valid': is_valid,
                        'feature_valid': feature_valid,
                        'is_closed': is_closed,
                        'bounding_box': bounding_box,
                        'centroid': centroid,
                        'area': area,
                        'perimeter': perimeter,
                        'aspect_ratio': aspect_ratio,
                        'curvature': 0.0,
                        'skeleton_length': float(np.linalg.norm(np.array(verts[-1]) - np.array(verts[0]))),
                        'symmetry_axis': symmetry_axis_line,
                        'relationships': relationships_line,
                        'parent_id': parent_shape_id,
                        'fallback_geometry': fallback_geometry_line,
                        'predicate': predicate_val,
                        'source': 'action_program',
                    }
                    # Track diversity statistics
                    for k, v in obj.items():
                        diversity_stats[k].add(str(v))
                    missing = [f for f in required_fields if obj.get(f) is None]
                    if missing:
                        logging.warning(f"[LOGO] Node {obj['object_id']} is missing required fields: {missing}")
                        objects.append(obj)
                    turtle_pos = list(new_pos)
                    prev_pos = list(turtle_pos)
                elif cmd_type == 'arc':
                    if turtle_pos is None:
                        turtle_pos = [0.0, 0.0]
                        verts_seq.append(list(turtle_pos))
                    radius = parsed_cmd.get('radius', 1.0)
                    angle = parsed_cmd.get('angle', 0.0)
                    num_points = max(6, int(abs(angle) // 10))
                    verts = [list(turtle_pos)]
                    start_angle = turtle_heading
                    for i in range(1, num_points+1):
                        theta = start_angle + (angle/num_points)*i
                        rad = np.radians(theta)
                        x = turtle_pos[0] + radius * np.cos(rad)
                        y = turtle_pos[1] + radius * np.sin(rad)
                        verts.append([x, y])
                    verts_seq.extend(verts[1:])
                    arc_length = abs(np.radians(angle) * radius)
                    obj_id = f"{problem_id}_{idx}_{stroke_idx}"
                    stroke_count += 1
                    geometry_reason = 'arc'
                    feature_valid = {'vertices': True, 'endpoints': True, 'length': True, 'orientation': True}
                    # Compute is_closed for this object
                    is_closed = False
                    if len(verts) >= 3:
                        arr = np.array(verts)
                        is_closed = np.allclose(arr[0], arr[-1], atol=1e-6)
                    
                    bounding_box, centroid, area, perimeter, aspect_ratio, compactness_stroke = _calculate_stroke_geometry(verts)
                    radius = parsed_cmd.get('radius', 1.0)
                    motif_membership_arc = motif_membership if motif_membership is not None else f"motif_{idx}"
                    fallback_geometry_arc = verts
                    symmetry_axis_arc = None
                    if command_mode in ['triangle', 'square', 'circle']:
                        symmetry_axis_arc = command_mode
                    relationships_arc = []
                    if stroke_idx > 0:
                        relationships_arc.append(f"adjacent_to_{problem_id}_{idx}_{stroke_idx-1}")
                    # Induce predicate for arc
                    
                    obj = {
                        'object_id': obj_id,
                        'parent_shape_id': parent_shape_id,
                        'action_index': stroke_idx,
                        'vertices': verts,
                        'object_type': assign_object_type(verts),
                        'turn_direction': command_mode,
                        'action_command': cmd,
                        'command_type': cmd_type,
                        'command_mode': command_mode,
                        'command_params': command_params,
                        'programmatic_label': programmatic_label if programmatic_label else cmd_type,
                        'stroke_type': stroke_type if stroke_type else cmd_type,
                        'label': label,
                        'shape_label': shape_label,
                        'category': category,
                        'original_image_path': remap_path(rec.get('image_path', '')),
                        'image_path': image_path,
                        'original_record_idx': idx,
                        'action_program': action_program,
                        'endpoints': [verts[0], verts[-1]],
                        'length': arc_length,
                        'orientation': float(np.degrees(np.arctan2(verts[-1][1] - verts[0][1], verts[-1][0] - verts[0][0]))),
                        'object_color': object_color,
                        'component_index': component_index,
                        'cluster_label': cluster_label,
                        'clustered': clustered,
                        'kb_concept': kb_concept,
                        'global_stat': global_stat,
                        'compactness': compactness_stroke,
                        'motif_type': motif_type,
                        'motif_membership': motif_membership_arc if motif_membership_arc is not None else '',
                        'stroke_count': stroke_count,
                        'subcategory': subcategory,
                        'supercategory': supercategory,
                        'semantic_type': semantic_type,
                        'instance_id': instance_id,
                        'group_id': group_id,
                        'part_of': part_of,
                        'relationship_type': relationship_type,
                        'symmetry_type': symmetry_type,
                        'diversity_stat': diversity_stat,
                        'geometry_reason': geometry_reason,
                        'is_valid': is_valid,
                        'feature_valid': feature_valid,
                        'is_closed': is_closed,
                        'bounding_box': bounding_box,
                        'centroid': centroid,
                        'area': area,
                        'perimeter': perimeter,
                        'aspect_ratio': aspect_ratio,
                        'curvature': 1.0 / radius if radius != 0 else float('inf'),
                        'skeleton_length': arc_length,
                        'symmetry_axis': symmetry_axis_arc,
                        'relationships': relationships_arc,
                        'parent_id': parent_shape_id,
                        'fallback_geometry': fallback_geometry_arc,
                        'predicate': predicate_val,
                        'source': 'action_program',
                    }
                    for k, v in obj.items():
                        diversity_stats[k].add(str(v))
                    missing = [f for f in required_fields if obj.get(f) is None]
                    if missing:
                        logging.warning(f"[LOGO] Node {obj['object_id']} is missing required fields: {missing}")
                    objects.append(obj)
                    turtle_pos = list(verts[-1])
                    turtle_heading += angle
                    prev_pos = list(turtle_pos)
                elif cmd_type == 'turn':
                    turtle_heading += parsed_cmd.get('angle', 0.0)
                    continue
                else:
                    logging.info(f"[LOGO] Skipping non-geometric or unknown command at idx={stroke_idx}: {cmd}")
            if len(verts_seq) >= 2:
                obj_id = f"{problem_id}_{idx}_fullpath"
                geometry_reason = 'fullpath'
                feature_valid = {'vertices': True, 'endpoints': True, 'length': True, 'orientation': True}
                is_closed = False
                # Induce turn_direction and action_command from action_program
                turn_direction = None
                action_command = None
                if action_program:
                    for cmd in reversed(action_program):
                        parsed = parse_action_command(cmd)
                        if parsed and parsed.get('mode', None) is not None:
                            turn_direction = parsed['mode']
                            break
                    action_command = action_program[-1]
                # Induce predicate for fullpath
                
                bounding_box, centroid, area, perimeter, aspect_ratio, compactness_stroke = _calculate_stroke_geometry(verts_seq)
                obj = {
                    'object_id': obj_id,
                    'parent_shape_id': parent_shape_id,
                    'action_index': -1,
                    'vertices': verts_seq,
                    'object_type': assign_object_type(verts_seq),
                    'turn_direction': turn_direction,
                    'action_command': action_command,
                    'command_type': None,
                    'command_mode': None,
                    'command_params': None,
                    'programmatic_label': programmatic_label,
                    'stroke_type': stroke_type,
                    'label': label,
                    'shape_label': shape_label,
                    'category': category,
                    'original_image_path': remap_path(rec.get('image_path', '')),
                    'image_path': image_path,
                    'original_record_idx': idx,
                    'action_program': action_program,
                    'endpoints': [verts_seq[0], verts_seq[-1]],
                    'length': float(np.sum([np.linalg.norm(np.array(verts_seq[i+1]) - np.array(verts_seq[i])) for i in range(len(verts_seq)-1)])),
                    'orientation': float(np.degrees(np.arctan2(verts_seq[-1][1] - verts_seq[0][1], verts_seq[-1][0] - verts_seq[0][0]))),
                    'object_color': object_color,
                    'component_index': component_index,
                    'cluster_label': cluster_label,
                    'clustered': clustered,
                    'kb_concept': kb_concept,
                    'global_stat': global_stat,
                    'compactness': compactness_stroke,
                    'motif_type': motif_type,
                    'motif_membership': motif_membership if motif_membership is not None else '',
                    'stroke_count': stroke_count,
                    'subcategory': subcategory,
                    'supercategory': supercategory,
                    'semantic_type': semantic_type,
                    'instance_id': instance_id,
                    'group_id': group_id,
                    'part_of': part_of,
                    'relationship_type': relationship_type,
                    'symmetry_type': symmetry_type,
                    'diversity_stat': diversity_stat,
                    'geometry_reason': geometry_reason,
                    'is_valid': is_valid,
                    'feature_valid': feature_valid,
                    'is_closed': is_closed,
                    'fallback_geometry': verts_seq,
                    'bounding_box': bounding_box,
                    'centroid': centroid,
                    'area': area,
                    'perimeter': perimeter,
                    'aspect_ratio': aspect_ratio,
                    'curvature': 0.0,
                    'skeleton_length': perimeter,
                    'symmetry_axis': None,
                    'relationships': [],
                    'predicate': predicate_val,
                    'source': 'action_program',
                    'parent_id': parent_shape_id,
                }
                for k, v in obj.items():
                    diversity_stats[k].add(str(v))
                missing = [f for f in required_fields if obj.get(f) is None]
                if missing:
                    logging.warning(f"[LOGO] Full path node {obj['object_id']} is missing required fields: {missing}")
            objects.append(obj)
    # --- Category mapping and diversity extension ---
    import networkx as nx
    from src.scene_graphs_building import graph_building
    from src.scene_graphs_building.config import BASIC_LOGO_PREDICATES
    G = nx.MultiDiGraph()
    for obj in objects:
        # Guarantee required node keys for schema validation
        obj.setdefault('predicate', None)
        obj.setdefault('source', None)
        obj.setdefault('symmetry_axis', None)
        G.add_node(obj['object_id'], **obj)

    # ------------------------------------------------------------------
    # 1. Build edges from node relationships / predicates
    # ------------------------------------------------------------------
    for u, data in G.nodes(data=True):
        for rel in data.get('relationships', []):
            if rel.startswith('adjacent_to_'):
                v_id = rel.split('adjacent_to_')[1]
                if G.has_node(v_id):
                    G.add_edge(u, v_id, predicate='adjacent_endpoints', source='program')

    # 2. Run the generic predicate engine as in the full pipeline
    graph_building.add_predicate_edges(G, BASIC_LOGO_PREDICATES)

    # 3. (optional) commonsense or VL edges
    # graph_building.add_commonsense_edges(G, top_k=2)

    # 4. Store edge count for later CSV export
    G.graph['edge_count'] = G.number_of_edges()
    # Diversity statistics logging
    logging.info(f"[LOGO] Diversity statistics for problem_id={problem_id}:")
    for k, vset in diversity_stats.items():
        logging.info(f"  {k}: {len(vset)} unique values")
    # Visualization and CSV saving for LOGO mode (no advanced features)
    try:
        import importlib
        import sys
        import glob
        for pycache in glob.glob(os.path.join(os.path.dirname(__file__), '..', '**', '__pycache__'), recursive=True):
            import shutil
            try:
                shutil.rmtree(pycache)
            except Exception as e:
                print(f"[DEBUG] Could not remove {pycache}: {e}")
        from scripts.scene_graph_visualization import save_scene_graph_visualization, save_scene_graph_csv
        print(f"[DEBUG] Using scene_graph_visualization from: {importlib.util.find_spec('scripts.scene_graph_visualization').origin}")
        
        image_path_vis = None
        for rec in problem_records:
            if 'image_path' in rec and rec['image_path']:
                image_path_vis = remap_path(rec['image_path'])
                break
        feedback_vis_dir = os.path.join('feedback', 'visualizations_logo')
        if image_path_vis:
            save_scene_graph_visualization(G, image_path_vis, feedback_vis_dir, problem_id)
        save_scene_graph_csv(G, feedback_vis_dir, problem_id)
    except Exception as e:
        logging.warning(f"[LOGO Visualization] Failed to save visualization or CSV for {problem_id}: {e}")
        logging.warning(traceback.format_exc())
    return {'scene_graph': {'graph': G, 'objects': objects, 'relationships': [], 'mode': 'logo', 'diversity_stats': {k: len(vset) for k, vset in diversity_stats.items()}}, 'rules': None}

def _calculate_stroke_geometry(verts):
    from shapely.geometry import Polygon, LineString
    bounding_box = None
    centroid = None
    area = None
    perimeter = None
    aspect_ratio = None
    compactness = None
    
    if len(verts) < 2:
        return bounding_box, centroid, area, perimeter, aspect_ratio, compactness

    try:
        if len(verts) >= 3 and assign_object_type(verts) == 'polygon':
            poly = Polygon(verts)
            bounding_box = poly.bounds
            centroid = list(poly.centroid.coords)[0]
            area = poly.area
            perimeter = poly.length
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter ** 2)
            else:
                compactness = 0.0
            minx, miny, maxx, maxy = poly.bounds
            width = maxx - minx
            height = maxy - miny
            aspect_ratio = width / height if height != 0 else None
        else: # It's a line or a curve
            line = LineString(verts)
            bounding_box = line.bounds
            centroid = list(line.centroid.coords)[0]
            area = 0.0
            perimeter = line.length
            minx, miny, maxx, maxy = line.bounds
            width = maxx - minx
            height = maxy - miny
            aspect_ratio = width / height if height != 0 else None
            compactness = 0.0
    except Exception as e:
        logging.warning(f"Could not compute geometry for stroke with {len(verts)} verts: {e}")

    return bounding_box, centroid, area, perimeter, aspect_ratio, compactness