import json

with open('logs/deep_analysis_results.json', 'r') as f:
    d = json.load(f)

print("Checking system1 ['confidences'] structure:")
sys1_conf = d['raw_data_program']['system1']['confidences']
print(f"  Type: {type(sys1_conf)}")
print(f"  Length: {len(sys1_conf) if isinstance(sys1_conf, (list, dict)) else 'N/A'}")
if isinstance(sys1_conf, dict):
    print(f"  Keys: {list(sys1_conf.keys())}")
elif isinstance(sys1_conf, list):
    print(f"  First item type: {type(sys1_conf[0])}")
    print(f"  First item: {sys1_conf[0]}")

print("\n\nChecking system1 ['accuracies'] structure:")
sys1_acc = d['raw_data_program']['system1']['accuracies']
if isinstance(sys1_acc, dict):
    print(f"  Type: dict with keys: {list(sys1_acc.keys())}")
    for split_name in list(sys1_acc.keys())[:2]:
        split_data = sys1_acc[split_name]
        if isinstance(split_data, list):
            print(f"    {split_name}: list with {len(split_data)} items")
        else:
            print(f"    {split_nме}: {type(split_data)}")
elif isinstance(sys1_acc, list):
    print(f"  Type: list with {len(sys1_acc)} items")

print("\n\nChecking episodes keys more carefully:")
eps = d['raw_data_program']['episodes']
for key in eps.keys():
    val = eps[key]
    if isinstance(val, dict):
        print(f"  {key}: dict with {len(val)} sub-keys")
        print(f"    First 5 keys: {list(val.keys())[:5]}")
    elif isinstance(val, list):
        print(f"  {key}: list with {len(val)} items")
