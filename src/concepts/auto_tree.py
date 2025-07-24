
import json, operator
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
import numpy as np

DERIVED = Path("data/derived_labels.json")

def is_scalar(val):
    return isinstance(val, (float, int, bool, np.floating, np.integer, np.bool_))

def induce_tree(problem_id):
    # 1. Gather X, y
    data = [r for r in json.load(DERIVED.open()) if r["problem_id"] == problem_id]
    X = []
    for r in data:
        feats = {}
        for k, v in r["features"].items():
            if is_scalar(v):
                feats[k] = v
        # Also add hand-picked scalar meta-features if needed
        if "action_count" in r["features"] and is_scalar(r["features"]["action_count"]):
            feats["action_count"] = r["features"]["action_count"]
        X.append(feats)
    # Consistent feature ordering
    feature_names = sorted(set().union(*(x.keys() for x in X)))
    X_mat = []
    for fdict in X:
        X_mat.append([fdict.get(k, 0.0) for k in feature_names])  # fill missing with 0.0

    y = [1 if r["label"] == "positive" else 0 for r in data]

    # 2. Train shallow tree
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X_mat, y)
    if clf.score(X_mat, y) < 1.0:
        # Fallback: program membership predicate
        # Load all positive action_programs for this problem
        pos_seqs = set()
        for r in data:
            if r["label"] == "positive":
                prog = r.get("action_program")
                if prog is None and "features" in r:
                    prog = r["features"].get("action_program", [])
                pos_seqs.add(tuple(prog))
        def membership_predicate(f):
            prog = f.get("action_program")
            if prog is None and "features" in f:
                prog = f["features"].get("action_program", [])
            return tuple(prog) in pos_seqs
        return membership_predicate

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
