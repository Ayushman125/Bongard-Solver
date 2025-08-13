import os
import csv

import re
class FileIO:
    _shape_attributes = None
    _shape_defs = None
    _shape_defs_length_cache = None  # List of rounded lengths for each row
    _shape_defs_stroke_index = None  # Dict: stroke_count -> [row indices]

    @staticmethod
    def _load_tsv(path):
        print(f"[TSV LOAD][CHECK] Attempting to load TSV from: {path}")
        file_exists = os.path.exists(path)
        print(f"[TSV LOAD][CHECK] File exists: {file_exists}")
        if not file_exists:
            print(f"[TSV LOAD][WARNING] TSV file does NOT exist: {path}")
            return []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)
            print(f"[TSV LOAD] Path: {path}")
            print(f"[TSV LOAD] Header: {repr(reader.fieldnames)}")
            print(f"[TSV LOAD] Loaded {len(rows)} rows.")
            if not rows:
                print(f"[TSV LOAD][WARNING] TSV file is EMPTY: {path}")
            else:
                print(f"[TSV LOAD] First row: {repr(rows[0])}")
            return rows

    @classmethod
    def get_shape_attributes(cls):
        if cls._shape_attributes is None:
            # Fix: Bongard-LOGO is in project root, not under src
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            tsv_path = os.path.join(project_root, 'Bongard-LOGO', 'data', 'human_designed_shapes_attributes.tsv')
            print(f"[TSV LOAD][PATH FIX] attributes path: {tsv_path}")
            cls._shape_attributes = cls._load_tsv(tsv_path)
        return cls._shape_attributes

    @classmethod
    def get_shape_defs(cls):
        if cls._shape_defs is None:
            # Fix: Bongard-LOGO is in project root, not under src
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            tsv_path = os.path.join(project_root, 'Bongard-LOGO', 'data', 'human_designed_shapes.tsv')
            print(f"[TSV LOAD][PATH FIX] defs path: {tsv_path}")
            cls._shape_defs = cls._load_tsv(tsv_path)
            # Build length cache and stroke index
            cls._shape_defs_length_cache = []
            cls._shape_defs_stroke_index = {}
            for idx, row in enumerate(cls._shape_defs):
                raw_actions = row.get('set of base actions', '')
                lengths = [round(l, 1) for l in cls.parse_base_actions(raw_actions)]
                cls._shape_defs_length_cache.append(lengths)
                stroke_count = len(lengths)
                if stroke_count not in cls._shape_defs_stroke_index:
                    cls._shape_defs_stroke_index[stroke_count] = []
                cls._shape_defs_stroke_index[stroke_count].append(idx)
        return cls._shape_defs

    @classmethod
    def get_shape_attribute_map(cls):
        # Map from shape function name to attribute dict
        return {row['shape function name']: row for row in cls.get_shape_attributes() if row.get('shape function name')}

    @classmethod
    def get_shape_def_map(cls):
        return {row['shape function name']: row for row in cls.get_shape_defs() if row.get('shape function name')}

    @staticmethod
    def parse_base_actions(actions_str):
        # e.g. 'line_1.0, line_0.732, line_0.897' -> [1.0, 0.7, 0.9]
        actions = [a.strip() for a in actions_str.split(',')]
        lengths = []
        for a in actions:
            m = re.match(r'(?:line|arc)_([0-9.]+)', a)
            if m:
                val = round(float(m.group(1)), 1)
                lengths.append(val)
            else:
                # For arc actions like arc_0.5_30.0, take the first number as radius
                m2 = re.match(r'arc_([0-9.]+)_([0-9.]+)', a)
                if m2:
                    val = round(float(m2.group(1)), 1)
                    lengths.append(val)
        return lengths

    @staticmethod
    def parse_turn_angles(turn_str):
        # e.g. 'L0--L120.0--L120.0' -> [0.0, 120.0, 120.0]
        parts = [p.strip() for p in turn_str.split('--')]
        angles = []
        for p in parts:
            m = re.match(r'[LR]?([0-9.\-]+)', p)
            if m:
                angles.append(float(m.group(1)))
        return angles

    @staticmethod
    def parse_action_command(cmd):
        # e.g. 'line_triangle_1.000-0.500' -> (1.0, angle)
        # Ignore style modifier, extract length and turn
        m = re.match(r'(?:line|arc)_[^_]+_([0-9.]+)-([0-9.]+)', cmd)
        if m:
            length = round(float(m.group(1)), 1)
            turn = float(m.group(2))
            # Convert normalized turn to degrees (assuming 1.0 = 360)
            angle = round(turn * 360.0, 3)
            return (length, angle)
        # Fallback: just extract first number as length
        m2 = re.match(r'(?:line|arc)_([0-9.]+)', cmd)
        if m2:
            length = round(float(m2.group(1)), 1)
            return (length, None)
        return None

    @classmethod
    def match_shape_from_actions(cls, action_cmds, problem_name=None, image_path=None):
        """
        Improved matching: strip modifiers, normalize lengths/angles, compare with tolerances, ignore direction prefix, log all parsing steps and failures. Logs also include problem name and image path for debugging.
        """
        import logging
        logger = logging.getLogger(__name__)
        log_prefix = f"[CANONICAL MAPPING]"
        if problem_name:
            log_prefix += f"[PROBLEM] {problem_name}"
        if image_path:
            log_prefix += f"[IMAGE] {image_path}"
        logger.warning(f"{log_prefix} [SHAPE INPUT] Trying to map actions: {action_cmds}")
        sig = []
        for cmd in action_cmds:
            parts = cmd.split('_')
            cmd_geom = None
            if parts[0] == 'line' and len(parts) >= 4:
                num_angle = parts[-1].split('-')
                if len(num_angle) == 2:
                    cmd_geom = f"line_{num_angle[0]}-{num_angle[1]}"
            elif parts[0] == 'arc':
                arc_match = re.match(r'arc_[^_]+_([0-9.]+)(?:_([0-9.]+))?-([0-9.]+)', cmd)
                if arc_match:
                    radius = arc_match.group(1)
                    angle = arc_match.group(3)
                    cmd_geom = f"arc_{radius}-{angle}"
                else:
                    num_angle = parts[-1].split('-')
                    if len(num_angle) == 2:
                        cmd_geom = f"arc_{num_angle[0]}-{num_angle[1]}"
            if not cmd_geom:
                cmd_geom = cmd
            parsed = cls.parse_action_command(cmd_geom)
            logger.warning(f"{log_prefix} [PARSE ACTION] Raw: {cmd} | Geom: {cmd_geom} | Parsed: {parsed}")
            if parsed:
                sig.append(parsed)
        logger.warning(f"{log_prefix} [ACTION SIGNATURE] Parsed signature: {sig}")
        matches = []
        sig_lengths = [round(length, 1) for length, _ in sig]
        shape_defs = cls.get_shape_defs()
        length_cache = cls._shape_defs_length_cache
        stroke_index = cls._shape_defs_stroke_index
        stroke_count = len(sig_lengths)
        candidate_indices = stroke_index.get(stroke_count, []) if stroke_index else range(len(shape_defs))
        if shape_defs:
            print("[STDOUT TSV FIELDNAMES]", repr(list(shape_defs[0].keys())))
            print("[STDOUT TSV FIRST ROW]", repr(shape_defs[0]))
        for idx in candidate_indices:
            row = shape_defs[idx]
            lengths = length_cache[idx] if length_cache else [round(l, 1) for l in cls.parse_base_actions(row.get('set of base actions', ''))]
            tsv_name = row.get('shape function name')
            raw_actions = row.get('set of base actions', '')
            raw_angles = row.get('turn angles', '')
            logger.warning(f"{log_prefix} [TSV SHAPE] {tsv_name} | RAW ACTIONS: '{raw_actions}' | RAW ANGLES: '{raw_angles}' | TSV Lengths: {lengths} | Action Lengths: {sig_lengths}")
            if len(lengths) != stroke_count:
                logger.warning(f"{log_prefix} [SKIP] {tsv_name}: Length mismatch (TSV {len(lengths)} vs Actions {stroke_count})")
                continue
            # Use normalized length ratios for robust matching
            def normalize_ratios(lengths):
                s = sum(lengths)
                return [l/s for l in lengths] if s > 0 else [0 for l in lengths]

            tsv_ratios = normalize_ratios(lengths)
            act_ratios = normalize_ratios(sig_lengths)
            if len(tsv_ratios) != len(act_ratios):
                logger.warning(f"{log_prefix} [SKIP] {tsv_name}: Length count mismatch (TSV {len(tsv_ratios)} vs Actions {len(act_ratios)})")
                continue
            match = True
            # Extract arc types from TSV and action_cmds
            def extract_arc_types(actions_str):
                actions = [a.strip() for a in actions_str.split(',')]
                types = []
                for a in actions:
                    if a.startswith('arc'):
                        parts = a.split('_')
                        if len(parts) > 2:
                            types.append(parts[1])
                        else:
                            types.append('normal')
                    else:
                        types.append(None)
                return types

            def extract_arc_types_cmds(cmds):
                types = []
                for cmd in cmds:
                    if cmd.startswith('arc'):
                        parts = cmd.split('_')
                        if len(parts) > 2:
                            types.append(parts[1])
                        else:
                            types.append('normal')
                    else:
                        types.append(None)
                return types

            tsv_arc_types = extract_arc_types(raw_actions)
            act_arc_types = extract_arc_types_cmds(action_cmds)

            for idx2, (tsv_r, act_r, tsv_type, act_type) in enumerate(zip(tsv_ratios, act_ratios, tsv_arc_types, act_arc_types)):
                if abs(tsv_r - act_r) > 0.05:
                    logger.warning(f"{log_prefix} [FAIL] {tsv_name}: Stroke {idx2} ratio mismatch (TSV {tsv_r} vs Action {act_r})")
                    match = False
                    break
                # If both are arcs, require type match
                if tsv_type is not None or act_type is not None:
                    if tsv_type != act_type:
                        logger.warning(f"{log_prefix} [FAIL] {tsv_name}: Arc type mismatch at stroke {idx2} (TSV {tsv_type} vs Action {act_type})")
                        match = False
                        break
            if match:
                logger.warning(f"{log_prefix} [MATCH] {tsv_name}: Matched with Action Length Ratios {act_ratios} and Arc Types {act_arc_types}")
                matches.append(row)
        if not matches:
            logger.warning(f"{log_prefix} [NOT FOUND] No canonical mapping found for actions: {action_cmds}")
        return matches

    @classmethod
    def get_shape_labels_and_attributes(cls, action_cmds, problem_name=None, image_path=None):
        # Returns all matching shape labels and attributes, logs problem_name and image_path for debugging
        matches = cls.match_shape_from_actions(action_cmds, problem_name=problem_name, image_path=image_path)
        attr_map = cls.get_shape_attribute_map()
        results = []
        for row in matches:
            shape_func = row['shape function name']
            label = {
                'shape_function': shape_func,
                'shape_name': row.get('shape name'),
                'super_class': row.get('super class'),
                'base_actions': row.get('set of base actions'),
                'turn_angles': row.get('turn angles'),
                'attributes': attr_map.get(shape_func, {})
            }
            results.append(label)
        return results