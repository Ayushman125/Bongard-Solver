import json
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

DERIVED = Path("data/derived_labels.json")

def induce_tree(problem_id):
    # 1. Gather X, y
    data = [r for r in json.load(DERIVED.open()) if r["problem_id"] == problem_id]
    if not data:
        raise ValueError(f"No data for problem_id {problem_id}")
    # Use all features, including action_count
    X = []
    feature_names = list(data[0]["features"].keys())
    for r in data:
        feats = dict(r["features"])
        # Ensure action_count is present
        if "action_count" not in feats:
            feats["action_count"] = len(r.get("action_program", feats.get("action_program", [])))
        X.append([feats[n] for n in feature_names])
    y = [1 if r["label"] == "positive" else 0 for r in data]

    # 2. Train shallow tree
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)
    if clf.score(X, y) < 1.0:
        raise ValueError(f"No perfect split for {problem_id}")

    # 3. Extract rule as a Python function
    tree_ = clf.tree_
    def recurse(node):
        if tree_.feature[node] == -2:  # leaf
            return bool(round(tree_.value[node][0][1]))
        feat = feature_names[tree_.feature[node]]
        thr  = tree_.threshold[node]
        left = recurse(tree_.children_left[node])
        right= recurse(tree_.children_right[node])
        return (lambda f, feat=feat, thr=thr, left=left, right=right:
                  left(f) if f[feat] <= thr else right(f))
    return recurse(0)
