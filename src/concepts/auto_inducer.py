import itertools
import operator as op

# Predicate templates: (signature, function, param generator)
TEMPLATES = [
    ("is_convex",           lambda f, _: f.get("is_convex", False)),
    ("num_straight=={n}",   lambda f, n: f.get("num_straight", None) == n,
                            lambda samples: {x["num_straight"] for x in samples if "num_straight" in x}),
    ("has_quadrangle",      lambda f, _: f.get("has_quadrangle", False)),
    ("nact_range[{lo},{hi}]",lambda f, rng: rng[0]<=f.get("num_strokes", 0)<=rng[1],
                            lambda s:{(min(x["num_strokes"] for x in s if "num_strokes" in x),
                                       max(x["num_strokes"] for x in s if "num_strokes" in x))}),
    ("symmetry_score<{t}",  lambda f, t: f.get("symmetry_score", 1e9) < t,
                            lambda s:{max(x["symmetry_score"] for x in s if "symmetry_score" in x)}),
]

def induce(problem_id, positives, negatives):
    for sig, fn, *param_fn in TEMPLATES:
        if not param_fn:
            params_iter = [None]
        else:
            try:
                params_iter = param_fn[0](positives)
                if not params_iter:
                    continue  # skip this template if no valid params
            except ValueError:
                continue  # skip if min/max fails due to empty sequence
            except Exception:
                continue
        for p in params_iter:
            try:
                if all(fn(x, p) for x in positives) and not any(fn(x, p) for x in negatives):
                    # Format signature for audit
                    if isinstance(p, tuple):
                        sig_fmt = sig.format(lo=p[0], hi=p[1])
                    elif isinstance(p, (int, float, str)):
                        sig_fmt = sig.format(n=p, t=p, shape=p)
                    else:
                        sig_fmt = sig
                    return {"problem_id":problem_id,"signature":sig_fmt,"param":p}
            except Exception:
                continue
    raise ValueError(f"No separating predicate for {problem_id}")
