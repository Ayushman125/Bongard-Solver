import json
import sys

def mask_truncate(arr, max_len=8):
    # Truncate any long list of 1s (or any int) to a compact summary
    if isinstance(arr, list):
        # 1D list of 1s
        if all(isinstance(x, int) and x == 1 for x in arr):
            if len(arr) > max_len:
                return f"[1, 1, 1, ... 1, 1, 1] (len={len(arr)})"
            else:
                return arr
        # 1D list of ints (not all 1s)
        if all(isinstance(x, int) for x in arr):
            if len(arr) > max_len:
                return f"[{', '.join(str(x) for x in arr[:3])} ... {', '.join(str(x) for x in arr[-3:])}] (len={len(arr)})"
            else:
                return arr
        # 2D mask: list of lists of 1s
        if all(isinstance(x, list) and all(isinstance(xx, int) and xx == 1 for xx in x) for x in arr):
            if len(arr) > max_len:
                return [f"[1, 1, 1, ... 1, 1, 1] (len={len(arr[0])})"] * 2 + ["...", f"[1, 1, 1, ... 1, 1, 1] (len={len(arr[0])})"] * 2 + [f"(len={len(arr)})"]
            else:
                return [mask_truncate(x, max_len) for x in arr]
        # 2D mask: list of lists of ints (not all 1s)
        if all(isinstance(x, list) and all(isinstance(xx, int) for xx in x) for x in arr):
            if len(arr) > max_len:
                return [f"[{', '.join(str(xx) for xx in x[:3])} ... {', '.join(str(xx) for xx in x[-3:])}] (len={len(x)})" for x in arr[:2]] + ["...", f"[{', '.join(str(xx) for xx in arr[-1][:3])} ... {', '.join(str(xx) for xx in arr[-1][-3:])}] (len={len(arr[-1])})"] + [f"(len={len(arr)})"]
            else:
                return [mask_truncate(x, max_len) for x in arr]
    return arr

def process(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "mask":
                try:
                    # Try to parse mask if it's a string
                    if isinstance(v, str):
                        v = json.loads(v.replace("'", '"'))
                except Exception:
                    pass
                out[k] = mask_truncate(v)
            else:
                out[k] = process(v)
        return out
    elif isinstance(obj, list):
        return [process(x) for x in obj]
    else:
        return obj

with open("experiments/phase0_spotcheck/spotcheck_bundles.json") as f:
    d = json.load(f)

# Only process first two bundles for brevity
for i, bundle in enumerate(d[:2]):
    print(f"\n--- Spotcheck JSON bundle {i+1} ---")
    print(json.dumps(process(bundle), indent=2))
