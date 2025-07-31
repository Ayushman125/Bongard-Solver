import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_ind
import logging
def induce_predicate_statistical(objects):
    pos = [o for o in objects if o.get('category') == 1]
    neg = [o for o in objects if o.get('category') == 0]
    features = ['area', 'aspect_ratio', 'compactness', 'orientation']
    best_feature = None
    best_p = 1.0
    for feat in features:
        pos_vals = [o.get(feat, 0) for o in pos]
        neg_vals = [o.get(feat, 0) for o in neg]
        if len(pos_vals) > 1 and len(neg_vals) > 1:
            stat, p = ttest_ind(pos_vals, neg_vals, equal_var=False)
            if p < best_p:
                best_p = p
                best_feature = feat
    if best_feature and best_p < 0.05:
        return f"{best_feature}_statistically_significant", best_feature
    return "same_shape", None

def induce_predicate_decision_tree(objects):
    # Prepare features and labels
    features = []
    labels = []
    for obj in objects:
        # Example features: area, aspect_ratio, compactness, orientation
        features.append([
            obj.get('area', 0),
            obj.get('aspect_ratio', 1),
            obj.get('compactness', 0),
            obj.get('orientation', 0)
        ])
        labels.append(obj.get('category', 0))
    features = np.array(features)
    labels = np.array(labels)

    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(features, labels)

    # Extract rules (predicates)
    # You can use tree_ structure to extract splits and thresholds
    # For demonstration, return the most important feature and threshold
    feature_names = ['area', 'aspect_ratio', 'compactness', 'orientation']
    if hasattr(clf, 'tree_'):
        tree = clf.tree_
        if tree.feature[0] != -2:  # -2 means leaf
            split_feature = feature_names[tree.feature[0]]
            threshold = tree.threshold[0]
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
        for obj in objects:
            features.append([
                obj.get('area', 0),
                obj.get('aspect_ratio', 1),
                obj.get('compactness', 0),
                obj.get('orientation', 0)
            ])
            labels.append(obj.get('category', 0))
        import numpy as np
        features = np.array(features)
        labels = np.array(labels)
        feature_names = ['area', 'aspect_ratio', 'compactness', 'orientation']
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
