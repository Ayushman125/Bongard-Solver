import json
import sys

if len(sys.argv) < 2:
    print("Usage: python print_horizontal.py <path_to_json> [entry_index]")
    sys.exit(1)

json_path = sys.argv[1]
entry_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

with open(json_path) as f:
    data = json.load(f)

if entry_index >= len(data):
    print(f"Index {entry_index} out of range. File has {len(data)} entries.")
    sys.exit(1)

entry = data[entry_index]
for k, v in entry.items():
    print(f"{k}: {v}", end=" | ")
print()
