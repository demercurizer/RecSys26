import numpy as np


def recall_at_k(recommended, relevant, k=10):
    relevant = set(relevant)
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):
    relevant = set(relevant)
    dcg = sum(1 / np.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / idcg