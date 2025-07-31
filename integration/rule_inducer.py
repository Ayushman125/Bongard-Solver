import numpy as np
from sklearn.tree import DecisionTreeClassifier
import networkx as nx
from collections import Counter

class RuleInducer:
    def __init__(self):
        pass

    def induce_decision_tree(self, G):
        """Induce a decision tree rule from node features and gnn_score."""
        X, y = [], []
        for n, d in G.nodes(data=True):
            X.append([
                d.get('area',0),
                d.get('mean_curvature',0),
                d.get('gnn_score',0)
            ])
            y.append(1 if d.get('shape_label','')=='positive' else 0)
        if not X or not y:
            return None
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)
        return clf

    def induce_symbolic(self, G):
        """Symbolic rule induction: find most common predicate and output as rule."""
        if not isinstance(G, nx.MultiDiGraph):
            return None
        pred_counts = Counter([edata.get('predicate') for _,_,edata in G.edges(data=True)])
        if not pred_counts:
            return None
        most_common_pred, count = pred_counts.most_common(1)[0]
        rule = f"IF edge predicate == '{most_common_pred}' THEN likely positive"
        return {'rule': rule, 'predicate': most_common_pred, 'support': count, 'total': G.number_of_edges()}

    def induce(self, G, method='symbolic'):
        """Unified interface: method can be 'symbolic' or 'tree'."""
        if method == 'tree':
            return self.induce_decision_tree(G)
        else:
            return self.induce_symbolic(G)
