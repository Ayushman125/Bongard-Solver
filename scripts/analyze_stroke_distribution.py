"""
Analyze stroke type and shape modifier distribution in Bongard-LOGO action program files.
"""
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import os

action_file = os.path.join('data', 'raw', 'ShapeBongard_V2', 'bd', 'bd_action_programs.json')

def parse_stroke(command):
    if not isinstance(command, str):
        return None, None
    parts = command.split('_')
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]

def analyze_action_programs(action_file):
    with open(action_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    stroke_type_counter = Counter()
    shape_modifier_counter = Counter()
    combo_counter = Counter()
    per_problem_stats = defaultdict(lambda: Counter())
    total_images = 0
    for problem_id, problem_data in data.items():
        if not isinstance(problem_data, list) or len(problem_data) != 2:
            continue
        for category_idx, category in enumerate(problem_data):
            for image_data in category:
                # Unwrap extra nesting
                if isinstance(image_data, list) and len(image_data) > 0:
                    strokes = image_data[0]
                else:
                    strokes = image_data
                total_images += 1
                for stroke in strokes:
                    stroke_type, shape_mod = parse_stroke(stroke)
                    if stroke_type and shape_mod:
                        stroke_type_counter[stroke_type] += 1
                        shape_modifier_counter[shape_mod] += 1
                        combo_counter[(stroke_type, shape_mod)] += 1
                        per_problem_stats[problem_id][(stroke_type, shape_mod)] += 1
    print(f"Total images: {total_images}")
    print("Stroke type distribution:")
    for k, v in stroke_type_counter.items():
        print(f"  {k}: {v}")
    print("\nShape modifier distribution:")
    for k, v in shape_modifier_counter.items():
        print(f"  {k}: {v}")
    print("\nStroke type + shape modifier combos:")
    for k, v in combo_counter.items():
        print(f"  {k}: {v}")
    # Plot bar charts
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.bar(stroke_type_counter.keys(), stroke_type_counter.values())
    plt.title('Stroke Type Distribution')
    plt.subplot(1,2,2)
    plt.bar(shape_modifier_counter.keys(), shape_modifier_counter.values())
    plt.title('Shape Modifier Distribution')
    plt.tight_layout()
    plt.show()
    # Optionally: print a few sample images
    print("\nSample image stroke sequences:")
    shown = 0
    for problem_id, problem_data in data.items():
        for category in problem_data:
            for image_data in category:
                if shown >= 5:
                    return
                strokes = image_data[0] if isinstance(image_data, list) and len(image_data) > 0 else image_data
                print(f"Problem: {problem_id}, Strokes: {strokes}")
                shown += 1

if __name__ == "__main__":
    analyze_action_programs(action_file)
