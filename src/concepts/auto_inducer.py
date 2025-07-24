
import itertools
import operator as op
import re
import logging
TEMPLATES = [
    # ... other templates ...
    # Map nactX_Y templates to your 'num_straight' feature
    ("nact_range[{lo},{hi}]", 
         lambda f, rng: rng[0] <= f["num_straight"] <= rng[1],
         lambda pos: {(min(x["features"]["num_straight"] for x in pos),
                       max(x["features"]["num_straight"] for x in pos))}),
]

def _parse_range_token(tkn):
    # 'nact5' → (5,5); 'nact2_5' → (2,5)
    parts = [int(n) for n in re.findall(r"\d+", tkn)]
    if len(parts) == 1:
        return (parts[0], parts[0])
    return (min(parts), max(parts))

def _get_feature_types(samples):
    # Returns dict: feature -> type ('bool', 'numeric', 'other')
    types = {}
    for f in samples:
        for k, v in f.items():
            if isinstance(v, bool):
                types[k] = 'bool'
            elif isinstance(v, (int, float)):
                types[k] = 'numeric'
            else:
                types.setdefault(k, 'other')
    return types

def induce(problem_id, positives, negatives):
    # 1. Try all boolean features
    pos_feats = positives
    neg_feats = negatives
    types = _get_feature_types(pos_feats + neg_feats)
    for feat, typ in types.items():
        if typ == 'bool':
            if all(f.get(feat, False) for f in pos_feats) and not any(f.get(feat, False) for f in neg_feats):
                return {"problem_id":problem_id, "signature":f"{feat}", "param":True, "features":[feat], "type":"bool"}
            if not any(f.get(feat, False) for f in pos_feats) and all(f.get(feat, False) for f in neg_feats):
                return {"problem_id":problem_id, "signature":f"not_{feat}", "param":False, "features":[feat], "type":"bool"}

    # 2. Try all numeric features with threshold/range
    for feat, typ in types.items():
        if typ == 'numeric':
            pos_vals = [f[feat] for f in pos_feats if feat in f]
            neg_vals = [f[feat] for f in neg_feats if feat in f]
            if not pos_vals or not neg_vals:
                continue
            # Try threshold: f[feat] > θ
            min_pos, max_neg = min(pos_vals), max(neg_vals)
            if min_pos > max_neg:
                thresh = (min_pos + max_neg) / 2
                return {"problem_id":problem_id, "signature":f"{feat}>{thresh}", "param":thresh, "features":[feat], "type":"threshold"}
            # Try threshold: f[feat] < θ
            max_pos, min_neg = max(pos_vals), min(neg_vals)
            if max_pos < min_neg:
                thresh = (max_pos + min_neg) / 2
                return {"problem_id":problem_id, "signature":f"{feat}<{thresh}", "param":thresh, "features":[feat], "type":"threshold"}
            # Try range: low <= f[feat] <= high
            lo, hi = min(pos_vals), max(pos_vals)
            if all(lo <= v <= hi for v in pos_vals) and not any(lo <= v <= hi for v in neg_vals):
                return {"problem_id":problem_id, "signature":f"{lo}<={feat}<={hi}", "param":(lo,hi), "features":[feat], "type":"range"}

    # 3. Try conjunctions of two features (AND)
    feat_keys = [k for k in types if types[k] in ('bool','numeric')]
    for f1, f2 in itertools.combinations(feat_keys, 2):
        # Try all pairs of bool/numeric features
        for v1 in [True, False] if types[f1]=='bool' else [None]:
            for v2 in [True, False] if types[f2]=='bool' else [None]:
                def pred(x):
                    ok1 = (x.get(f1, False)==v1) if types[f1]=='bool' else True
                    ok2 = (x.get(f2, False)==v2) if types[f2]=='bool' else True
                    return ok1 and ok2
                if all(pred(f) for f in pos_feats) and not any(pred(f) for f in neg_feats):
                    # Construct signature string for boolean conjunctions
                    parts = []
                    if types[f1] == 'bool':
                        parts.append(f"{f1}" if v1 else f"not_{f1}")
                    if types[f2] == 'bool':
                        parts.append(f"{f2}" if v2 else f"not_{f2}")
                    signature_str = " AND ".join(parts)
                    return {"problem_id":problem_id, "signature":signature_str, "param":(v1,v2), "features":[f1,f2], "type":"and_bool"}
        # Try numeric threshold for both
        if types[f1]=='numeric' and types[f2]=='numeric':
            pos1 = [f[f1] for f in pos_feats if f1 in f]
            pos2 = [f[f2] for f in pos_feats if f2 in f]
            neg1 = [f[f1] for f in neg_feats if f1 in f]
            neg2 = [f[f2] for f in neg_feats if f2 in f]
            if pos1 and pos2 and neg1 and neg2:
                min1, max1 = min(pos1), max(pos1)
                min2, max2 = min(pos2), max(pos2)
                def pred(x):
                    # Handle cases where feature might not be present in x
                    val1 = x.get(f1)
                    val2 = x.get(f2)
                    
                    check1 = (min1 <= val1 <= max1) if val1 is not None else False
                    check2 = (min2 <= val2 <= max2) if val2 is not None else False
                    
                    return check1 and check2
                
                if all(pred(f) for f in pos_feats) and not any(pred(f) for f in neg_feats):
                    return {"problem_id":problem_id, "signature":f"{min1}<={f1}<={max1} AND {min2}<={f2}<={max2}", "param":((min1,max1),(min2,max2)), "features":[f1,f2], "type":"and_range"}

    # If nothing found, try fallback for ff_nact problems
    if problem_id.startswith("ff_nact"):
        try:
            tkn = problem_id.split("_")[1]
            lo, hi = _parse_range_token(tkn)
            def pred(x):
                return lo <= x.get("num_straight", -1) <= hi
            pos_pred = [pred(f) for f in positives]
            neg_pred = [pred(f) for f in negatives]
            logging.info(f"ff_nact fallback for {problem_id}: pos_pred={pos_pred}, neg_pred={neg_pred}, lo={lo}, hi={hi}")
            if all(pos_pred) and not any(neg_pred):
                logging.info("Fallback ff_nact predicate for %s → [%d,%d]", problem_id, lo, hi)
                return {"problem_id": problem_id, "signature": f"nact_range[{{lo}},{{hi}}]", "param": (lo, hi), "features": ["num_straight"], "type": "range"}
        except Exception as e:
            logging.warning(f"ff_nact fallback failed for {problem_id}: {e}")
    # If nothing found, fail
    raise ValueError(f"No separating predicate for {problem_id}")