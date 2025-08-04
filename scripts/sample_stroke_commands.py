"""
Stroke Command Sampler for Bongard-LOGO Dataset

This script parses bd_action_programs.json and collects at least 10 examples for each stroke type (line, arc) and each shape modifier (normal, circle, square, triangle, zigzag).
Output: stroke_examples_by_type.txt with clear section headers for each type/modifier.
"""
import json
from collections import defaultdict

# Path to the action program file
ACTION_FILE = "data/raw/ShapeBongard_V2/bd/bd_action_programs.json"
OUTPUT_FILE = "stroke_examples_by_type.txt"

# Stroke types and modifiers to collect
STROKE_TYPES = ["line", "arc"]
SHAPE_MODIFIERS = ["normal", "circle", "square", "triangle", "zigzag"]

# How many examples to collect for each
EXAMPLES_PER_GROUP = 10

def collect_strokes(data):
    # Dict: (stroke_type, shape_modifier) -> list of commands
    groups = defaultdict(list)
    def recurse(obj):
        if isinstance(obj, list):
            for item in obj:
                recurse(item)
        elif isinstance(obj, str):
            parts = obj.split("_")
            if len(parts) >= 3:
                stroke_type = parts[0]
                shape_modifier = parts[1]
                key = (stroke_type, shape_modifier)
                if stroke_type in STROKE_TYPES and shape_modifier in SHAPE_MODIFIERS:
                    if len(groups[key]) < EXAMPLES_PER_GROUP:
                        groups[key].append(obj)
    for problem_id, problem_data in data.items():
        recurse(problem_data)
    return groups

def main():
    with open(ACTION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    groups = collect_strokes(data)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for stroke_type in STROKE_TYPES:
            for shape_modifier in SHAPE_MODIFIERS:
                key = (stroke_type, shape_modifier)
                out.write(f"=== {stroke_type.upper()}_{shape_modifier.upper()} ===\n")
                examples = groups.get(key, [])
                if examples:
                    for cmd in examples:
                        out.write(cmd + "\n")
                else:
                    out.write("(No examples found)\n")
                out.write("\n")
    print(f"Done. Output written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
