import numpy as np

def euclidean_embed(features):
    return np.array(features)

def hyperbolic_embed(features):
    return np.tanh(np.array(features))

def mine_hard(embeddings, topk=5):
    dists = np.linalg.norm(embeddings - embeddings.mean(axis=0), axis=1)
    idx = np.argsort(-dists)[:topk]
    return embeddings[idx]

def poincare_mixup(u, v, alpha=0.5):
    return (u + v) / 2 * alpha
