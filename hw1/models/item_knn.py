import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


class ItemKNN:
    def __init__(self, n_neighbors=100, rating_threshold=4):
        self.n_neighbors = n_neighbors
        self.rating_threshold = rating_threshold

    def fit(self, train):
        self.users = train["user_id"].unique().to_list()
        self.items = train["item_id"].unique().to_list()

        self.user_to_idx = {u: i for i, u in enumerate(self.users)}
        self.item_to_idx = {i: j for j, i in enumerate(self.items)}

        positive = train.filter(pl.col("rating") >= self.rating_threshold)

        rows = np.array([self.user_to_idx[u] for u in positive["user_id"]], dtype=np.int32)
        cols = np.array([self.item_to_idx[i] for i in positive["item_id"]], dtype=np.int32)
        data = np.ones(len(rows), dtype=np.float32)

        self.X = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.users), len(self.items)),
        )

        norms = np.sqrt(np.asarray(self.X.power(2).sum(axis=0)).ravel())
        norms[norms == 0] = 1

        X_norm = self.X.multiply(1 / norms)
        similarity = (X_norm.T @ X_norm).tocsr()
        similarity.setdiag(0)
        similarity.eliminate_zeros()

        self.similarity = self._keep_top_k(similarity, self.n_neighbors)

        rows = np.array([self.user_to_idx[u] for u in train["user_id"]], dtype=np.int32)
        cols = np.array([self.item_to_idx[i] for i in train["item_id"]], dtype=np.int32)

        self.seen = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=self.X.shape,
        )

        return self

    def _keep_top_k(self, matrix, k):
        rows, cols, values = [], [], []

        for i in range(matrix.shape[0]):
            start, end = matrix.indptr[i], matrix.indptr[i + 1]
            row_cols = matrix.indices[start:end]
            row_values = matrix.data[start:end]

            if len(row_values) > k:
                idx = np.argpartition(row_values, -k)[-k:]
                row_cols = row_cols[idx]
                row_values = row_values[idx]

            rows.extend([i] * len(row_cols))
            cols.extend(row_cols)
            values.extend(row_values)

        return csr_matrix(
            (values, (rows, cols)),
            shape=matrix.shape,
            dtype=np.float32,
        )

    def recommend(self, user_id, k=10):
        user_idx = self.user_to_idx[user_id]

        scores = (self.X[user_idx] @ self.similarity).toarray().ravel()

        seen_items = self.seen[user_idx].indices
        scores[seen_items] = -np.inf

        k = min(k, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]

        return [self.items[i] for i in top if np.isfinite(scores[i])]