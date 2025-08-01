import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_ind
import logging
def induce_predicate_statistical(objects):
    import logging
    pos = [o for o in objects if o.get('category') == 1]
    neg = [o for o in objects if o.get('category') == 0]
    features = [
        'area', 'aspect_ratio', 'compactness', 'orientation', 'length', 'cx', 'cy',
        'curvature', 'stroke_count', 'programmatic_label', 'kb_concept', 'global_stat'
    ]
    best_feature = None
    best_p = 1.0
    for feat in features:
        pos_vals = [o.get(feat, 0) for o in pos if o.get(feat) is not None]
        neg_vals = [o.get(feat, 0) for o in neg if o.get(feat) is not None]
        if len(pos_vals) < 4 or len(neg_vals) < 4:
            logging.warning(f"Statistical induction: Skipping {feat} due to small group size (pos={len(pos_vals)}, neg={len(neg_vals)})")
            continue
        # For categorical features, use mutual_info_score
        if feat in ('programmatic_label', 'kb_concept'):
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import mutual_info_score
            le = LabelEncoder()
            all_vals = pos_vals + neg_vals
            le.fit(all_vals)
            pos_enc = le.transform(pos_vals)
            neg_enc = le.transform(neg_vals)
            mi = mutual_info_score(pos_enc, neg_enc)
            if mi > 0.1 and mi > best_p:
                best_p = mi
                best_feature = feat
        else:
            from scipy.stats import ttest_ind
            stat, p = ttest_ind(pos_vals, neg_vals, equal_var=False)
            if p < best_p:
                best_p = p
                best_feature = feat
    if best_feature and best_p < 0.05:
        return f"{best_feature}_statistically_significant", best_feature
    return "same_shape", None

def induce_predicate_decision_tree(objects):
    import logging
    from sklearn.preprocessing import LabelEncoder
    # Prepare features and labels
    features = []
    labels = []
    feature_names = [
        'area', 'aspect_ratio', 'compactness', 'orientation', 'length', 'cx', 'cy',
        'curvature', 'stroke_count', 'programmatic_label', 'kb_concept', 'global_stat'
    ]
    # Robust categorical encoding
    program_labels = [o.get('programmatic_label', '') for o in objects]
    kb_labels = [o.get('kb_concept', '') for o in objects]
    program_le = LabelEncoder().fit(program_labels)
    kb_le = LabelEncoder().fit(kb_labels)
    for obj in objects:
        features.append([
            obj.get('area', 0),
            obj.get('aspect_ratio', 1),
            obj.get('compactness', 0),
            obj.get('orientation', 0),
            obj.get('length', 0),
            obj.get('cx', 0),
            obj.get('cy', 0),
            obj.get('curvature', 0),
            obj.get('stroke_count', 0),
            program_le.transform([obj.get('programmatic_label', '')])[0],
            kb_le.transform([obj.get('kb_concept', '')])[0],
            obj.get('global_stat', 0)
        ])
        labels.append(obj.get('category', 0))
    features = np.array(features)
    labels = np.array(labels)
    if len(labels) < 8 or min(np.bincount(labels)) < 4:
        logging.warning(f"Decision tree induction: Skipping due to small/imbalanced splits (labels={np.bincount(labels)})")
        return "same_shape", None
    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(features, labels)
    # Extract rules (predicates)
    if hasattr(clf, 'tree_'):
        tree = clf.tree_
        if tree.feature[0] != -2:  # -2 means leaf
            split_feature = feature_names[tree.feature[0]]
            threshold = tree.threshold[0]
            logging.info(f"Decision tree rule: {split_feature} > {threshold:.2f}")
            return f"{split_feature}_gt_{threshold:.2f}", (split_feature, threshold)
    return "same_shape", None

def induce_predicate_automl(objects, automl_type='tpot', max_time_mins=None, generations=None):
    """
    Uses TPOT or AutoSklearn to automate feature selection and rule induction for predicate induction.
    Returns the best feature and threshold found by the AutoML pipeline.
    """
    if max_time_mins is None or generations is None:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--tpot-max-time', type=int, default=5)
        parser.add_argument('--tpot-generations', type=int, default=5)
        args, _ = parser.parse_known_args()
        max_time_mins = args.tpot_max_time
        generations = args.tpot_generations
    try:
        features = []
        labels = []
        from sklearn.preprocessing import LabelEncoder
        program_labels = [o.get('programmatic_label', '') for o in objects]
        kb_labels = [o.get('kb_concept', '') for o in objects]
        program_le = LabelEncoder().fit(program_labels)
        kb_le = LabelEncoder().fit(kb_labels)
        for obj in objects:
            features.append([
                obj.get('area', 0),
                obj.get('aspect_ratio', 1),
                obj.get('compactness', 0),
                obj.get('orientation', 0),
                obj.get('length', 0),
                obj.get('cx', 0),
                obj.get('cy', 0),
                program_le.transform([obj.get('programmatic_label', '')])[0],
                kb_le.transform([obj.get('kb_concept', '')])[0]
            ])
            labels.append(obj.get('category', 0))
        import numpy as np
        features = np.array(features)
        labels = np.array(labels)
        feature_names = ['area', 'aspect_ratio', 'compactness', 'orientation', 'length', 'cx', 'cy', 'programmatic_label', 'kb_concept']
        if automl_type == 'tpot':
            try:
                from tpot import TPOTClassifier
                tpot = TPOTClassifier(generations=generations, population_size=20, max_time_mins=max_time_mins, random_state=42)
                tpot.fit(features, labels)
                # Check for both fitted_pipeline_ and fitted_pipeline attributes
                pipeline = None
                if hasattr(tpot, 'fitted_pipeline_'):
                    pipeline = tpot.fitted_pipeline_
                elif hasattr(tpot, 'fitted_pipeline'):
                    pipeline = tpot.fitted_pipeline
                if pipeline is not None and hasattr(pipeline, 'feature_importances_'):
                    importances = pipeline.feature_importances_
                    best_idx = int(np.argmax(importances))
                    best_feature = feature_names[best_idx]
                    return f"{best_feature}_automl_tpot", best_feature
                else:
                    logging.warning("TPOT did not produce a valid pipeline with feature_importances_. Returning fallback.")
                    return "same_shape", None
            except Exception as e:
                logging.warning(f"TPOT failed: {e}")
                return "same_shape", None
        elif automl_type == 'autosklearn':
            try:
                import autosklearn.classification
                automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=max_time_mins*60, per_run_time_limit=60)
                automl.fit(features, labels)
                # Extract feature importances if available
                if hasattr(automl, 'feature_importances_'):
                    importances = automl.feature_importances_
                    best_idx = int(np.argmax(importances))
                    best_feature = feature_names[best_idx]
                    return f"{best_feature}_automl_autosklearn", best_feature
                else:
                    return "same_shape", None
            except Exception as e:
                logging.warning(f"AutoSklearn failed: {e}")
                return "same_shape", None
        else:
            logging.warning(f"Unknown automl_type: {automl_type}")
            return "same_shape", None
    except Exception as e:
        logging.warning(f"AutoML predicate induction failed: {e}")
        return "same_shape", None

def induce_predicate_for_problem(objects, **kwargs):
    """
    Attempts predicate induction in the following order:
    1. Statistical
    2. Decision Tree
    3. (GNN or other methods can be added here)
    4. AutoML (TPOT/AutoSklearn)
    Returns the first predicate that is not 'same_shape'.
    """
    pred, params = induce_predicate_statistical(objects)
    if pred != "same_shape":
        return pred, params
    pred, params = induce_predicate_decision_tree(objects)
    if pred != "same_shape":
        return pred, params
    # Add GNN or other methods here if available
    # Remove keys not accepted by induce_predicate_automl
    automl_kwargs = {k: v for k, v in kwargs.items() if k in {'automl_type', 'max_time_mins', 'generations'}}
    pred, params = induce_predicate_automl(objects, **automl_kwargs)
    return pred, params
