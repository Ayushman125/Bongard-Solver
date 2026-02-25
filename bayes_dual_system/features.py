from typing import Any, Dict, List


LINE_TYPES = ["normal", "zigzag", "circle", "triangle", "square"]


def _flatten_program_tokens(program: Any) -> List[str]:
    tokens: List[str] = []
    if not isinstance(program, list):
        return tokens

    for maybe_subprog in program:
        if isinstance(maybe_subprog, list):
            for token in maybe_subprog:
                if isinstance(token, str):
                    tokens.append(token)
        elif isinstance(maybe_subprog, str):
            tokens.append(maybe_subprog)
    return tokens


def extract_predicates(program: Any) -> Dict[str, bool]:
    tokens = _flatten_program_tokens(program)
    n_tokens = len(tokens)
    n_shapes = len(program) if isinstance(program, list) else 0

    line_count = 0
    arc_count = 0
    type_counts = {line_type: 0 for line_type in LINE_TYPES}

    for token in tokens:
        parts = token.split("_")
        if not parts:
            continue
        primitive = parts[0]
        if primitive == "line":
            line_count += 1
        elif primitive == "arc":
            arc_count += 1

        if len(parts) > 1 and parts[1] in type_counts:
            type_counts[parts[1]] += 1

    predicates: Dict[str, bool] = {
        "has_line": line_count > 0,
        "has_arc": arc_count > 0,
        "line_dominant": line_count > arc_count,
        "arc_dominant": arc_count > line_count,
        "n_shapes_ge_2": n_shapes >= 2,
        "n_tokens_ge_5": n_tokens >= 5,
        "n_tokens_ge_8": n_tokens >= 8,
        "n_tokens_ge_10": n_tokens >= 10,
    }

    for line_type, count in type_counts.items():
        predicates[f"has_type_{line_type}"] = count > 0
        predicates[f"type_{line_type}_dominant"] = count > (n_tokens / 2.0 if n_tokens else 0)

    return predicates
