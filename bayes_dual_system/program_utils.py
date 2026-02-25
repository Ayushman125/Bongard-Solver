"""
LOGO Action Program Feature Extraction

This module extracts symbolic features from LOGO action programs, enabling
System 2 to reason about stroke sequences, shape categories, and geometric
properties - the actual concepts in Bongard-LOGO.

References:
- Bongard-LOGO paper Section 2.3: Action programs as ground-truth concepts
- Tenenbaum framework: Concepts as generative programs
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import re


def flatten_program(program: Any) -> List[str]:
    """
    Flatten nested program structure to list of action strings.
    
    Programs can be:
    - [[["line_square_1.000-0.083", ...]]]  (nested lists)
    - ["line_square_1.000-0.083", ...]       (flat list)
    - Single action string
    
    Returns:
        List of action command strings
    """
    if isinstance(program, str):
        return [program]
    
    if not isinstance(program, list):
        return []
    
    # Recursively flatten nested lists
    result = []
    for item in program:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, list):
            result.extend(flatten_program(item))
    
    return result


def parse_action_command(cmd: str) -> Optional[dict]:
    """
    Parse LOGO action command string.
    
    Format: {action}_{moving_type}_{length}-{angle}
    Examples:
        "line_square_1.000-0.083"
        "arc_triangle_0.500_0.625-0.750"
    
    Returns:
        dict with keys: action, moving_type, length, angle
        None if parsing fails
    """
    # Pattern: action_movingtype_length-angle
    # Length can have underscore for arcs: 0.500_0.625
    pattern = r'^(line|arc)_(normal|zigzag|triangle|circle|square)_([\d.\_]+)-([\d.]+)$'
    match = re.match(pattern, cmd)
    
    if not match:
        return None
    
    action, moving_type, length_str, angle_str = match.groups()
    
    # Parse length (may have underscore for arc radius)
    try:
        if '_' in length_str:
            # Arc with radius specification
            parts = length_str.split('_')
            length = float(parts[0])  # Use first value
        else:
            length = float(length_str)
        
        angle = float(angle_str)
    except ValueError:
        return None
    
    return {
        'action': action,
        'moving_type': moving_type,
        'length': length,
        'angle': angle,
    }


def extract_program_features(program: Any) -> Optional[np.ndarray]:
    """
    Extract symbolic features from LOGO action program.
    
    These features capture:
    - Stroke counts (line vs arc)
    - Moving types (normal, zigzag, triangle, circle, square)
    - Sequence properties (length, complexity)
    - Length/angle statistics
    
    This aligns with Bongard-LOGO concepts:
    - Free-form: Stroke sequences matter
    - Basic shapes: Shape categories from stroke patterns
    - Abstract: Geometric properties from stroke analysis
    
    Args:
        program: LOGO action program (nested lists or flat list of strings)
    
    Returns:
        Feature vector (16 dimensions) or None if invalid
    """
    if program is None:
        return None
    
    flat_program = flatten_program(program)
    if not flat_program:
        return None
    
    # Parse all commands
    parsed = [parse_action_command(cmd) for cmd in flat_program]
    parsed = [p for p in parsed if p is not None]
    
    if not parsed:
        return None
    
    # Count action types
    line_count = sum(1 for p in parsed if p['action'] == 'line')
    arc_count = sum(1 for p in parsed if p['action'] == 'arc')
    
    # Count moving types (stroke styles)
    normal_count = sum(1 for p in parsed if p['moving_type'] == 'normal')
    zigzag_count = sum(1 for p in parsed if p['moving_type'] == 'zigzag')
    triangle_count = sum(1 for p in parsed if p['moving_type'] == 'triangle')
    circle_count = sum(1 for p in parsed if p['moving_type'] == 'circle')
    square_count = sum(1 for p in parsed if p['moving_type'] == 'square')
    
    # Sequence properties
    sequence_length = len(parsed)
    
    # Length statistics
    lengths = [p['length'] for p in parsed]
    length_mean = float(np.mean(lengths)) if lengths else 0.0
    length_std = float(np.std(lengths)) if len(lengths) > 1 else 0.0
    
    # Angle statistics
    angles = [p['angle'] for p in parsed]
    angle_mean = float(np.mean(angles)) if angles else 0.0
    angle_std = float(np.std(angles)) if len(angles) > 1 else 0.0
    
    # Complexity metrics
    unique_actions = len(set(p['action'] for p in parsed))
    unique_types = len(set(p['moving_type'] for p in parsed))
    
    # Ratio features
    line_ratio = line_count / max(sequence_length, 1)
    arc_ratio = arc_count / max(sequence_length, 1)
    
    features = np.array([
        sequence_length,      # 0: Total stroke count
        line_count,           # 1: Line stroke count
        arc_count,            # 2: Arc stroke count
        normal_count,         # 3: Normal style count
        zigzag_count,         # 4: Zigzag style count
        triangle_count,       # 5: Triangle style count
        circle_count,         # 6: Circle style count
        square_count,         # 7: Square style count
        length_mean,          # 8: Average stroke length
        length_std,           # 9: Stroke length variation
        angle_mean,          # 10: Average angle
        angle_std,           # 11: Angle variation
        unique_actions,      # 12: Action diversity
        unique_types,        # 13: Style diversity
        line_ratio,          # 14: Proportion of lines
        arc_ratio,           # 15: Proportion of arcs
    ], dtype=np.float32)
    
    return features


def compute_program_similarity(prog1: Any, prog2: Any, threshold: float = 0.8) -> float:
    """
    Compute similarity between two LOGO programs.
    
    This enables concept matching: does test image's program
    match the consensus program from positive examples?
    
    Args:
        prog1, prog2: LOGO action programs
        threshold: Minimum substring match ratio
    
    Returns:
        Similarity score in [0, 1]
    """
    flat1 = flatten_program(prog1)
    flat2 = flatten_program(prog2)
    
    if not flat1 or not flat2:
        return 0.0
    
    # Exact sequence match
    if flat1 == flat2:
        return 1.0
    
    # Longest common subsequence
    lcs_len = _lcs_length(flat1, flat2)
    max_len = max(len(flat1), len(flat2))
    
    return lcs_len / max_len if max_len > 0 else 0.0


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute longest common subsequence length."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Compute Levenshtein edit distance between two sequences.
    
    Counts minimum insertions, deletions, and substitutions
    to transform seq1 into seq2.
    
    Args:
        seq1, seq2: Sequences of action commands
    
    Returns:
        Edit distance (0 = identical sequences)
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete from seq1
                    dp[i][j - 1],      # Insert into seq1
                    dp[i - 1][j - 1]   # Substitute
                )
    
    return dp[m][n]


def compute_sequence_features(prog1: Any, prog2: Any) -> np.ndarray:
    """
    Extract sequence-based similarity features between two programs.
    
    Improvement #2: Sequence matching for better free-form concept recognition.
    Uses multiple alignment metrics beyond just LCS.
    
    Args:
        prog1, prog2: LOGO action programs
    
    Returns:
        6-dimensional feature vector:
        [lcs_ratio, edit_similarity, exact_match, length_ratio, prefix_match, suffix_match]
    """
    flat1 = flatten_program(prog1)
    flat2 = flatten_program(prog2)
    
    if not flat1 or not flat2:
        return np.zeros(6, dtype=np.float32)
    
    len1, len2 = len(flat1), len(flat2)
    max_len = max(len1, len2)
    min_len = min(len1, len2)
    
    # 1. LCS ratio (existing)
    lcs_len = _lcs_length(flat1, flat2)
    lcs_ratio = lcs_len / max_len if max_len > 0 else 0.0
    
    # 2. Edit distance similarity
    edit_dist = edit_distance(flat1, flat2)
    edit_similarity = 1.0 - (edit_dist / max_len) if max_len > 0 else 0.0
    
    # 3. Exact match
    exact_match = 1.0 if flat1 == flat2 else 0.0
    
    # 4. Length ratio (penalty for different lengths)
    length_ratio = min_len / max_len if max_len > 0 else 0.0
    
    # 5. Prefix match (first k actions)
    prefix_len = min(min_len, 3)  # Check first 3 actions
    prefix_match = 1.0 if flat1[:prefix_len] == flat2[:prefix_len] else 0.0
    
    # 6. Suffix match (last k actions)
    suffix_len = min(min_len, 3)  # Check last 3 actions
    suffix_match = 1.0 if flat1[-suffix_len:] == flat2[-suffix_len:] else 0.0
    
    return np.array([
        lcs_ratio,
        edit_similarity,
        exact_match,
        length_ratio,
        prefix_match,
        suffix_match,
    ], dtype=np.float32)


def find_consensus_program(programs: List[Any], min_support: float = 0.5) -> Optional[List[str]]:
    """
    Find consensus program from multiple examples.
    
    For free-form shapes, all positive images should share the same
    stroke sequence. This function finds the most common subsequences.
    
    Args:
        programs: List of LOGO action programs
        min_support: Minimum fraction of programs that must contain a subsequence
    
    Returns:
        Consensus program (list of action commands) or None
    """
    if not programs:
        return None
    
    flat_programs = [flatten_program(p) for p in programs if p]
    flat_programs = [p for p in flat_programs if p]
    
    if not flat_programs:
        return None
    
    # Simple heuristic: Return the most common full sequence
    from collections import Counter
    
    # Convert to tuples for hashing
    program_tuples = [tuple(p) for p in flat_programs]
    counter = Counter(program_tuples)
    
    if counter:
        most_common_tuple, count = counter.most_common(1)[0]
        support = count / len(program_tuples)
        
        if support >= min_support:
            return list(most_common_tuple)
    
    # Fallback: Return longest program (assumes longest captures most strokes)
    return max(flat_programs, key=len)
