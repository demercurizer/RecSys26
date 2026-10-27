import time

import numpy as np
import polars as pl
from scipy.linalg import inv
from scipy.sparse import csr_matrix


class EASE:
    def __init__(self, reg=300.0, rating_threshold=4):
        self.reg = reg
        self.rating_threshold = rating_threshold

    def fit(self, train):
        self.users = sorted(train["user_id"].unique().to_list())
        self.items = sorted(train["item_id"].unique().to_list())

        self.user_to_idx = {user_id: i for i, user_id in enumerate(self.users)}
        self.item_to_idx = {item_id: i for i, item_id in enumerate(self.items)}

        positive = train.filter(pl.col("rating") >= self.rating_threshold)

        rows = np.fromiter((self.user_to_idx[user_id] for user_id in positive["user_id"]), dtype=np.int32, count=positive.height)
        cols = np.fromiter((self.item_to_idx[item_id] for item_id in positive["item_id"]), dtype=np.int32, count=positive.height)

        self.X = csr_matrix((np.ones(positive.height, dtype=np.float32), (rows, cols)), shape=(len(self.users), len(self.items)), dtype=np.float32)

        seen_rows = np.fromiter((self.user_to_idx[user_id] for user_id in train["user_id"]), dtype=np.int32, count=train.height)
        seen_cols = np.fromiter((self.item_to_idx[item_id] for item_id in train["item_id"]), dtype=np.int32, count=train.height)
        self.seen = csr_matrix((np.ones(train.height, dtype=np.float32), (seen_rows, seen_cols)), shape=self.X.shape)

        print(f"users={len(self.users)} items={len(self.items)} positive_interactions={positive.height} reg={self.reg}", flush=True)

        started = time.perf_counter()
        print("stage=gram_matrix", flush=True)

        G = (self.X.T @ self.X).toarray().astype(np.float32, copy=False)

        print(f"stage=gram_matrix_done time={time.perf_counter() - started:.1f}s memory={G.nbytes / 1024 ** 3:.2f}GB", flush=True)

        diagonal_idx = np.arange(G.shape[0])
        G[diagonal_idx, diagonal_idx] += self.reg

        started = time.perf_counter()
        print("stage=inverse", flush=True)

        P = inv(G, overwrite_a=True, check_finite=False)

        print(f"stage=inverse_done time={time.perf_counter() - started:.1f}s", flush=True)

        diagonal = np.diag(P).copy()
        P *= -1.0
        P /= diagonal[None, :]
        np.fill_diagonal(P, 0.0)

        self.B = P.astype(np.float32, copy=False)
        return self

    def recommend(self, user_id, k=10):
        user_idx = self.user_to_idx[user_id]
        scores = np.asarray(self.X[user_idx] @ self.B).ravel()
        scores[self.seen[user_idx].indices] = -np.inf

        k = min(k, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]

        return [self.items[i] for i in top]