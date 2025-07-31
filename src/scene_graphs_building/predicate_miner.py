import numpy as np

class PredicateMiner:
    def fit(self, objects):
        # Fit thresholds for area, aspect, curvature, etc. using contrastive stats
        areas = np.array([o.get('area', 0) for o in objects])
        aspects = np.array([o.get('aspect_ratio', 1) for o in objects])
        curvs = np.array([o.get('mean_curvature', 0) for o in objects])
        # Use percentiles for thresholds
        area_thr = np.percentile(areas, 70) if len(areas) else 1
        aspect_thr = np.percentile(np.abs(aspects[:,None]-aspects), 30) if len(aspects) else 0.1
        curv_thr = np.percentile(curvs, 70) if len(curvs) else 0.1
        preds = {}
        preds['larger_than'] = lambda a,b: a.get('area',0) > area_thr and b.get('area',0) < area_thr
        preds['aspect_sim'] = lambda a,b: abs(a.get('aspect_ratio',1)-b.get('aspect_ratio',1)) < aspect_thr
        preds['curvature_sim'] = lambda a,b: abs(a.get('mean_curvature',0)-b.get('mean_curvature',0)) < curv_thr
        return preds
