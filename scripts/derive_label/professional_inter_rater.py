"""
Professional module for inter-rater reliability and kappa statistics, compatible with scikit-learn and numpy.
Provides: cohen_kappa_score, fleiss_kappa, and related utilities.
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score

def fleiss_kappa(ratings):
    """
    Computes Fleiss' kappa for group agreement.
    ratings: 2D numpy array (subjects x raters), each entry is a category label (int or str)
    Returns: float (kappa)
    """
    ratings = np.asarray(ratings)
    n_subjects, n_raters = ratings.shape
    # Find all unique categories
    categories = np.unique(ratings)
    n_categories = len(categories)
    cat2idx = {cat: i for i, cat in enumerate(categories)}
    # Count ratings per category per subject
    counts = np.zeros((n_subjects, n_categories), dtype=int)
    for i in range(n_subjects):
        for j in range(n_raters):
            counts[i, cat2idx[ratings[i, j]]] += 1
    # Proportion of all assignments to each category
    p = np.sum(counts, axis=0) / (n_subjects * n_raters)
    # Agreement for each subject
    P = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    Pbar = np.mean(P)
    PbarE = np.sum(p ** 2)
    kappa = (Pbar - PbarE) / (1 - PbarE) if (1 - PbarE) != 0 else 0.0
    return kappa

# Example usage:
# ratings = np.array([[1, 2, 2], [2, 2, 2], [1, 1, 1]])
# print(fleiss_kappa(ratings))
