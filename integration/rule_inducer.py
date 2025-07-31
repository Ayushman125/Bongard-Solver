import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RuleInducer:
    def induce(self, G):
        # Use node features and gnn_score to induce rules
        X, y = [], []
        for n, d in G.nodes(data=True):
            X.append([
                d.get('area',0),
                d.get('mean_curvature',0),
                d.get('gnn_score',0)
            ])
            y.append(1 if d.get('shape_label','')=='positive' else 0)
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        return clf
