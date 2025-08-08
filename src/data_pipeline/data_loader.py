# --- Main Functions for Data Loading and Processing ---
import os
import pickle
import json

def load_json_data(input_path):
    """
    Load data from JSON or pickle file.
    Supports both .json and .pkl file extensions.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Data file not found: {input_path}")
    
    if input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif input_path.endswith('.pkl'):
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Try to load as JSON first, then pickle
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            with open(input_path, 'rb') as f:
                return pickle.load(f)

# ACTION PROGRAMS ONLY: Use load_action_programs() instead

def remap_path(path):
    """Remaps image paths to match expected dataset structure. FIXED: No replacement, use actual folder names."""
    return path

def robust_image_open(path, *args, **kwargs):
    """
    Open an image file after remapping the path. Use this everywhere in the pipeline for image loading.
    Example: img = robust_image_open(image_path)
    """
    from PIL import Image
    remapped = remap_path(path)
    if not os.path.exists(remapped):
        raise FileNotFoundError(f"Image file not found after remapping: {remapped}")
    return Image.open(remapped, *args, **kwargs)

# --- Bongard LOGO dataset utilities ---
def load_split_file(split_file_path):
    """
    Load the ShapeBongard_V2_split.json file to get train/validation/test splits.
    Returns a dict with 'train', 'val', 'test' keys containing problem ID lists.
    """
    try:
        with open(split_file_path, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        
        print(f"[INFO] Loaded split file from {split_file_path}")
        for split_name, problem_list in split_data.items():
            print(f"[INFO] {split_name}: {len(problem_list)} problems")
            
        return split_data
    except Exception as e:
        print(f"[WARN] Failed to load split file {split_file_path}: {e}")
        return {}

def get_problem_ids_from_split(split_data, split_name='val', max_problems=None):
    """
    Get problem IDs from a specific split.
    
    Args:
        split_data: Dict from load_split_file()
        split_name: 'train', 'val', 'test', or 'test_hd_novel'
        max_problems: Maximum number of problems to return (None for all)
    
    Returns:
        List of problem IDs
    """
    if split_name not in split_data:
        print(f"[WARN] Split '{split_name}' not found in split data. Available: {list(split_data.keys())}")
        return []
    
    problem_ids = split_data[split_name]
    
    if max_problems is not None and len(problem_ids) > max_problems:
        problem_ids = problem_ids[:max_problems]
        print(f"[INFO] Limited to {max_problems} problems from {split_name} split")
    
    print(f"[INFO] Using {len(problem_ids)} problems from {split_name} split")
    return problem_ids
def load_action_programs(base_dir, categories=('bd', 'ff', 'hd')):
    """
    Loads action program JSONs for Bongard LOGO dataset from the specified base directory.
    Returns a dict mapping problem_id to action program data with proper structure.
    
    Expected structure: {problem_id: [positive_examples, negative_examples]}
    """
    import glob
    action_programs = {}
    for cat in categories:
        # Look for action program files in the category subdirectory
        action_file_pattern = os.path.join(base_dir, cat, f"{cat}_action_programs.json")
        
        # Also try direct pattern match for backward compatibility
        if not glob.glob(action_file_pattern):
            action_file_pattern = os.path.join(base_dir, f"{cat}_action_programs.json")
        
        for fname in glob.glob(action_file_pattern):
            print(f"[INFO] Loading action programs from {fname}")
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    action_data = json.load(f)
                    
                # Merge action programs from this category
                for problem_id, action_program_data in action_data.items():
                    # Validate structure: should be [positive_examples, negative_examples]
                    if isinstance(action_program_data, list) and len(action_program_data) == 2:
                        action_programs[problem_id] = action_program_data
                        # Removed per-problem log for cleaner output
                    else:
                        print(f"[WARN] Unexpected action program structure for {problem_id}: {type(action_program_data)}")
                        
            except Exception as e:
                print(f"[WARN] Failed to load action program {fname}: {e}")
    
    # Removed misleading log; filtered count is now logged in pipeline
    return action_programs

def parse_action_command(command):
    """
    Parse a single action command to extract semantic features for all Bongard-LOGO command types.
    
    Action commands have format: 
    - line_<shape>_<params>: Line commands with various shape types
    - arc_<shape>_<params>: Arc commands with various shape types
    - start_<x>_<y>: Starting position commands
    - turn_<angle>: Turn commands
    
    Supported shape types (5 total discovered in Bongard-LOGO dataset):
    - normal: Standard straight/curved lines (24,107 occurrences - most common)
    - zigzag: Zigzag patterns (6,729 occurrences)
    - square: Square-based shapes (6,519 occurrences)
    - circle: Circular shapes or circular arcs (6,256 occurrences)
    - triangle: Triangular shapes (5,837 occurrences)
    
    Command type distribution:
    - line: 39,262 total occurrences (79.4%)
    - arc: 10,186 total occurrences (20.6%)
    
    Returns:
        dict with parsed command information including command_type, shape, parameters, coordinates, etc.
    """
    if not command or not isinstance(command, str):
        return {'command_type': 'unknown', 'shape': 'unknown', 'parameters': str(command), 'full_command': str(command)}
    
    parts = command.split('_')
    if len(parts) < 2:
        return {'command_type': 'unknown', 'shape': 'unknown', 'parameters': command, 'full_command': command}
    
    command_type = parts[0].lower()  # 'line', 'arc', 'start', 'turn', etc.
    
    # Handle different command types
    if command_type == 'start':
        # start_x_y format
        x, y = (float(parts[1]), float(parts[2])) if len(parts) >= 3 else (0.0, 0.0)
        return {
            'command_type': 'start',
            'shape': 'start',
            'parameters': f"{x}_{y}",
            'x': x,
            'y': y,
            'full_command': command
        }
    elif command_type == 'turn':
        # turn_angle format
        angle = float(parts[1]) if len(parts) >= 2 else 0.0
        return {
            'command_type': 'turn', 
            'shape': 'turn',
            'parameters': str(angle),
            'angle': angle,
            'full_command': command
        }
    elif command_type in ['line', 'arc']:
        shape = parts[1] if len(parts) >= 2 else 'unknown'  # 'normal', 'circle', 'square', 'triangle', 'zigzag', etc.
        
        # Join remaining parts as parameters
        parameters = '_'.join(parts[2:]) if len(parts) > 2 else ''
        
        # Parse coordinate parameters for line/arc commands
        coords = None
        size = None
        thickness = None
        
        if parameters and '-' in parameters:
            try:
                # Format: size-thickness (e.g., "1.000-0.500")
                coord_parts = parameters.split('-')
                if len(coord_parts) >= 2:
                    size = float(coord_parts[0])
                    thickness = float(coord_parts[1])
                    coords = (size, thickness)
            except ValueError:
                pass
        
        # Determine geometric properties based on the 5 discovered shape types
        is_closed = shape in ['circle', 'square', 'triangle']  # These form closed shapes
        is_curved = command_type == 'arc' or shape == 'circle'  # Arcs and circles are curved
        is_regular = shape in ['circle', 'square', 'triangle']  # Regular geometric shapes
        is_linear_pattern = shape in ['normal', 'zigzag']  # Linear patterns
        
        # Shape complexity categorization
        complexity_level = {
            'normal': 1,      # Simplest - straight lines
            'circle': 2,      # Simple geometric shape
            'square': 2,      # Simple geometric shape  
            'triangle': 2,    # Simple geometric shape
            'zigzag': 3       # Most complex - irregular pattern
        }.get(shape, 1)
        
        return {
            'command_type': command_type,
            'shape': shape,
            'parameters': parameters,
            'coords': coords,
            'size': size,
            'thickness': thickness,
            'is_closed': is_closed,
            'is_curved': is_curved,
            'is_regular': is_regular,
            'is_linear_pattern': is_linear_pattern,
            'complexity_level': complexity_level,
            'full_command': command
        }
    else:
        # Unknown command type
        return {
            'command_type': command_type,
            'shape': 'unknown',
            'parameters': '_'.join(parts[1:]) if len(parts) > 1 else '',
            'full_command': command
        }

def extract_features_from_actions(action_sequence):
    """
    Extract comprehensive semantic features from a sequence of action commands.
    Handles all Bongard-LOGO command types and their subcategories.
    
    Args:
        action_sequence: List of action command strings
        
    Returns:
        dict with extracted features including command counts, shape types, geometric properties, etc.
    """
    import logging
    try:
        from src.Derive_labels.features import ensure_str_list
        action_sequence = ensure_str_list(action_sequence)
        logging.debug(f"[DATA_LOADER] Action sequence after ensure_str_list: {[type(a) for a in action_sequence]}")
    except Exception as e:
        logging.error(f"[DATA_LOADER] ensure_str_list failed: {e}")
        # Fallback: try shape_utils
        try:
            from src.Derive_labels.shape_utils import ensure_flat_str_list
            action_sequence = ensure_flat_str_list(action_sequence)
            logging.debug(f"[DATA_LOADER] Action sequence after ensure_flat_str_list: {[type(a) for a in action_sequence]}")
        except Exception as e2:
            logging.error(f"[DATA_LOADER] ensure_flat_str_list failed: {e2}")
            # Fallback: convert all to str
            action_sequence = [str(a) for a in action_sequence]
            logging.debug(f"[DATA_LOADER] Action sequence after str fallback: {[type(a) for a in action_sequence]}")
    parsed_commands = [parse_action_command(cmd) for cmd in action_sequence]
    
    # Count command types and shapes
    command_types = [cmd['command_type'] for cmd in parsed_commands]
    shape_types = [cmd['shape'] for cmd in parsed_commands]
    
    from collections import Counter
    
    # Basic counts
    command_type_counts = dict(Counter(command_types))
    shape_type_counts = dict(Counter(shape_types))
    
    # Geometric property analysis based on discovered shape types
    closed_shapes = [cmd for cmd in parsed_commands if cmd.get('is_closed', False)]
    curved_shapes = [cmd for cmd in parsed_commands if cmd.get('is_curved', False)]
    regular_shapes = [cmd for cmd in parsed_commands if cmd.get('is_regular', False)]
    linear_patterns = [cmd for cmd in parsed_commands if cmd.get('is_linear_pattern', False)]
    
    # Shape complexity analysis using actual complexity levels
    complexity_scores = [cmd.get('complexity_level', 1) for cmd in parsed_commands if cmd.get('complexity_level')]
    avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 1.0
    
    # Shape complexity analysis
    unique_command_types = list(set(command_types))
    unique_shape_types = list(set(shape_types))
    
    # Size and thickness statistics
    sizes = [cmd.get('size') for cmd in parsed_commands if cmd.get('size') is not None]
    thicknesses = [cmd.get('thickness') for cmd in parsed_commands if cmd.get('thickness') is not None]
    
    # Advanced geometric features
    has_lines = 'line' in command_types
    has_arcs = 'arc' in command_types
    has_mixed_geometry = has_lines and has_arcs
    
    # Shape category analysis using the 5 discovered types
    bongard_geometric_shapes = ['circle', 'square', 'triangle']  # Closed geometric shapes
    bongard_line_patterns = ['normal', 'zigzag']  # Linear patterns
    
    has_geometric_shapes = any(shape in shape_types for shape in bongard_geometric_shapes)
    has_line_patterns = any(shape in shape_types for shape in bongard_line_patterns)
    
    # Calculate shape type distribution
    geometric_count = len([s for s in shape_types if s in bongard_geometric_shapes])
    pattern_count = len([s for s in shape_types if s in bongard_line_patterns])
    total_shapes = len([s for s in shape_types if s in bongard_geometric_shapes + bongard_line_patterns])
    
    # Composite pattern detection
    pattern_complexity = len(unique_shape_types) + len(unique_command_types)
    is_simple_pattern = pattern_complexity <= 3
    is_complex_pattern = pattern_complexity > 6
    
    features = {
        # Basic counts
        'total_commands': len(action_sequence),
        'command_type_counts': command_type_counts,
        'shape_type_counts': shape_type_counts,
        'unique_command_types': unique_command_types,
        'unique_shape_types': unique_shape_types,
        
        # Geometric properties
        'closed_shape_count': len(closed_shapes),
        'curved_shape_count': len(curved_shapes),
        'regular_shape_count': len(regular_shapes),
        'linear_pattern_count': len(linear_patterns),
        'has_lines': has_lines,
        'has_arcs': has_arcs,
        'has_mixed_geometry': has_mixed_geometry,
        'avg_complexity': avg_complexity,
        
        # Shape categories (Bongard-LOGO specific)
        'has_geometric_shapes': has_geometric_shapes,
        'has_line_patterns': has_line_patterns,
        'geometric_shape_ratio': geometric_count / max(total_shapes, 1),
        'pattern_shape_ratio': pattern_count / max(total_shapes, 1),
        
        # Size and thickness analysis
        'avg_size': sum(sizes) / len(sizes) if sizes else 0.0,
        'max_size': max(sizes) if sizes else 0.0,
        'min_size': min(sizes) if sizes else 0.0,
        'avg_thickness': sum(thicknesses) / len(thicknesses) if thicknesses else 0.0,
        'size_variance': max(sizes) - min(sizes) if len(sizes) > 1 else 0.0,
        
        # Pattern complexity
        'pattern_complexity': pattern_complexity,
        'is_simple_pattern': is_simple_pattern,
        'is_complex_pattern': is_complex_pattern,
        
        # Detailed parsed commands for downstream processing
        'parsed_commands': parsed_commands,
        
        # Bongard-specific features
        'symmetry_indicators': _extract_symmetry_indicators(parsed_commands),
        'topological_features': _extract_topological_features(parsed_commands),
        'semantic_categories': _extract_semantic_categories(parsed_commands)
    }
    
    return features

def _extract_symmetry_indicators(parsed_commands):
    """Extract potential symmetry indicators from parsed commands using actual Bongard-LOGO shape types"""
    # Only circle, square, and triangle are symmetric in Bongard-LOGO
    symmetry_shapes = ['circle', 'square', 'triangle']
    
    indicators = {
        'has_symmetric_shapes': any(cmd.get('shape') in symmetry_shapes for cmd in parsed_commands),
        'regular_shape_count': len([cmd for cmd in parsed_commands if cmd.get('is_regular', False)]),
        'symmetric_shape_types': list(set(cmd.get('shape') for cmd in parsed_commands if cmd.get('shape') in symmetry_shapes)),
        'asymmetric_pattern_count': len([cmd for cmd in parsed_commands if cmd.get('shape') == 'zigzag']),
        'regular_vs_irregular_ratio': len([cmd for cmd in parsed_commands if cmd.get('is_regular', False)]) / max(len(parsed_commands), 1)
    }
    
    return indicators

def _extract_topological_features(parsed_commands):
    """Extract topological features from parsed commands"""
    closed_count = len([cmd for cmd in parsed_commands if cmd.get('is_closed', False)])
    total_shapes = len([cmd for cmd in parsed_commands if cmd.get('command_type') in ['line', 'arc']])
    
    features = {
        'closed_ratio': closed_count / max(total_shapes, 1),
        'open_shape_count': total_shapes - closed_count,
        'has_mixed_topology': closed_count > 0 and (total_shapes - closed_count) > 0,
        'connectivity_complexity': len(set(cmd.get('shape') for cmd in parsed_commands))
    }
    
    return features

def _extract_semantic_categories(parsed_commands):
    """Extract semantic categories based on the 5 actual Bongard-LOGO shape types"""
    # Updated categories based on actual discovered shape types
    categories = {
        'basic_geometry': ['triangle', 'square', 'circle'],  # The 3 geometric shapes
        'line_patterns': ['normal', 'zigzag'],  # The 2 line pattern types
        'curves': ['circle'],  # Only circle is curved in the basic shapes
        'straight_lines': ['normal', 'square', 'triangle'],  # Shapes made with straight lines
        'complex_patterns': ['zigzag']  # Complex irregular patterns
    }
    
    detected_categories = {}
    shape_types = [cmd.get('shape') for cmd in parsed_commands]
    
    for category, shapes in categories.items():
        detected_categories[f'has_{category}'] = any(shape in shape_types for shape in shapes)
        detected_categories[f'{category}_count'] = len([shape for shape in shape_types if shape in shapes])
    
    # Add specific Bongard-LOGO pattern analysis
    detected_categories['pure_geometric'] = all(shape in ['circle', 'square', 'triangle'] for shape in shape_types if shape)
    detected_categories['pure_patterns'] = all(shape in ['normal', 'zigzag'] for shape in shape_types if shape)
    detected_categories['mixed_types'] = len(set(shape_types)) > 1
    
    return detected_categories

def get_problem_data(problem_id, action_programs):
    """
    Fetch per-problem data from action_programs ONLY.
    
    ACTION PROGRAMS ONLY: No derived_labels dependency. 
    All action command types (line, arc) and shape types (normal, circle, square, etc.) are parsed.
    """
    # Get action program data
    action_prog = action_programs.get(problem_id)
    if not action_prog:
        print(f"[WARN] No action program found for problem_id: {problem_id}")
        return None
    
    # Validate action program structure
    if not isinstance(action_prog, list) or len(action_prog) != 2:
        print(f"[WARN] Invalid action program structure for {problem_id}")
        return None
    
    positive_examples, negative_examples = action_prog
    
    # Create records for each example
    records = []
    
    # Process positive examples
    for idx, example_actions in enumerate(positive_examples):
        # Each example_actions is a list of command lists
        flattened_actions = []
        for cmd_list in example_actions:
            if isinstance(cmd_list, list):
                flattened_actions.extend(cmd_list)
            else:
                flattened_actions.append(cmd_list)
        
        # Extract features from action sequence
        action_features = extract_features_from_actions(flattened_actions)
        
        record = {
            'problem_id': problem_id,
            'example_type': 'positive',
            'example_index': idx,
            'action_program': example_actions,
            'flattened_actions': flattened_actions,
            'image_path': None,  # No actual image file for action program data
            'label': 'category_1',
            'category': 'positive',
            'features': action_features,
            'shape_label': f"positive_example_{idx}",
            'programmatic_label': f"pos_{action_features['total_commands']}_commands"
        }
        
        # ACTION PROGRAMS ONLY: No derived labels fallback needed
        
        records.append(record)
    
    # Process negative examples  
    for idx, example_actions in enumerate(negative_examples):
        # Each example_actions is a list of command lists
        flattened_actions = []
        for cmd_list in example_actions:
            if isinstance(cmd_list, list):
                flattened_actions.extend(cmd_list)
            else:
                flattened_actions.append(cmd_list)
        
        # Extract features from action sequence
        action_features = extract_features_from_actions(flattened_actions)
        
        record = {
            'problem_id': problem_id,
            'example_type': 'negative', 
            'example_index': idx,
            'action_program': example_actions,
            'flattened_actions': flattened_actions,
            'image_path': None,  # No actual image file for action program data
            'label': 'category_0',
            'category': 'negative',
            'features': action_features,
            'shape_label': f"negative_example_{idx}",
            'programmatic_label': f"neg_{action_features['total_commands']}_commands"
        }
        
        # ACTION PROGRAMS ONLY: No derived labels fallback needed
        
        records.append(record)
    
    return {
        'problem_id': problem_id,
        'records': records,
        'action_programs': action_prog,
        'total_positive': len(positive_examples),
        'total_negative': len(negative_examples)
    }
