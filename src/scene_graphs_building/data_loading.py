
# --- Main Functions for Data Loading and Processing ---
import os
import pickle
import json

def load_data(input_path):
    """Simple pickle/json loader for demonstration; replace as needed."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.endswith('.pkl'):
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    elif input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported input file type: {input_path}")

def remap_path(path):
    """Remaps image paths to match expected dataset structure."""
    return path.replace('category_1', '1').replace('category_0', '0')

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
def load_action_programs(base_dir, categories=('bd', 'ff', 'hd')):
    """
    Loads action program JSONs for Bongard LOGO dataset from the specified base directory.
    Returns a dict mapping problem_id to list of action programs.
    """
    import glob
    action_programs = {}
    for cat in categories:
        pattern = os.path.join(base_dir, f"{cat}_*.json")
        for fname in glob.glob(pattern):
            problem_id = os.path.splitext(os.path.basename(fname))[0]
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    action_programs[problem_id] = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load action program {fname}: {e}")
    return action_programs

def get_problem_data(problem_id, derived_labels, action_programs):
    """Fetch per-problem data from derived_labels and action_programs."""
    entry = next((item for item in derived_labels if item['problem_id'] == problem_id), None)
    if not entry:
        return None
    action_prog = action_programs.get(problem_id)
    return {
        'problem_id': problem_id,
        'label_data': entry,
        'action_programs': action_prog
    }
