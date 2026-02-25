
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
        # Fallback: program membership predicate (must be pickleable for multiprocessing)
        pos_seqs = set()
        for r in data:
            if r["label"] == "positive":
                prog = r.get("action_program")
                if prog is None and "features" in r:
                    prog = r["features"].get("action_program", [])
                pos_seqs.add(tuple(prog))
        # Return a pure function (no closure over local state)
        def membership_predicate(f, pos_seqs=pos_seqs):
            prog = f.get("action_program")
            if prog is None and "features" in f:
                prog = f["features"].get("action_program", [])
            return tuple(prog) in pos_seqs
        membership_predicate.__name__ = f"membership_predicate_{problem_id}"
        membership_predicate._rule_type = "membership"
        membership_predicate._pos_seqs = list(pos_seqs)
        return membership_predicate

    # 3. Extract rule as a Python function (pure, pickleable)
    tree_ = clf.tree_
    def make_tree_predicate(tree_, feature_names):
        # Recursively build a pure function for the decision tree
        def recurse(node):
            if tree_.feature[node] == -2:  # leaf
                return bool(round(tree_.value[node][0][1]))
            feat = feature_names[tree_.feature[node]]
            thr  = tree_.threshold[node]
            left = recurse(tree_.children_left[node])
            right= recurse(tree_.children_right[node])
            def node_fn(f, feat=feat, thr=thr, left=left, right=right):
                val = f.get(feat, 0.0)
                return left(f) if val <= thr else right(f)
            return node_fn
        pred = recurse(0)
        pred.__name__ = f"tree_predicate_{problem_id}"
        pred._rule_type = "tree"
        pred._feature_names = feature_names
        pred._tree_params = clf.get_params()
        return pred
    return make_tree_predicate(tree_, feature_names)
