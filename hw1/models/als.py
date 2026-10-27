import time

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csr_matrix


class ALS:
    def __init__(self, factors=32, reg=0.1, epochs=8, seed=42):
        self.factors = factors
        self.reg = reg
        self.epochs = epochs
        self.seed = seed

    def fit(self, train, target=None):
        self.users = sorted(train["user_id"].unique().to_list())
        self.items = sorted(train["item_id"].unique().to_list())

        self.user_to_idx = {user_id: i for i, user_id in enumerate(self.users)}
        self.item_to_idx = {item_id: i for i, item_id in enumerate(self.items)}

        rows = np.fromiter((self.user_to_idx[user_id] for user_id in train["user_id"]), dtype=np.int32, count=train.height)
        cols = np.fromiter((self.item_to_idx[item_id] for item_id in train["item_id"]), dtype=np.int32, count=train.height)
        values = train["rating"].to_numpy().astype(np.float32) if target is None else np.asarray(target, dtype=np.float32)

        self.R = csr_matrix((values, (rows, cols)), shape=(len(self.users), len(self.items)), dtype=np.float32)
        self.R_csc = self.R.tocsc()

        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.01, size=(len(self.users), self.factors)).astype(np.float32)
        self.item_factors = rng.normal(0, 0.01, size=(len(self.items), self.factors)).astype(np.float32)

        eye = np.eye(self.factors, dtype=np.float32)

        print(f"users={len(self.users)} items={len(self.items)} interactions={train.height} factors={self.factors} reg={self.reg}", flush=True)

        for epoch in range(1, self.epochs + 1):
            started = time.perf_counter()

            for u in range(len(self.users)):
                start, end = self.R.indptr[u], self.R.indptr[u + 1]
                item_indices = self.R.indices[start:end]
                ratings = self.R.data[start:end]

                if len(item_indices) == 0:
                    continue

                V = self.item_factors[item_indices]
                A = V.T @ V + self.reg * eye
                b = V.T @ ratings
                self.user_factors[u] = solve(A, b, assume_a="pos", check_finite=False)

            for i in range(len(self.items)):
                start, end = self.R_csc.indptr[i], self.R_csc.indptr[i + 1]
                user_indices = self.R_csc.indices[start:end]
                ratings = self.R_csc.data[start:end]

                if len(user_indices) == 0:
                    continue

                U = self.user_factors[user_indices]
                A = U.T @ U + self.reg * eye
                b = U.T @ ratings
                self.item_factors[i] = solve(A, b, assume_a="pos", check_finite=False)

            print(f"epoch={epoch}/{self.epochs} time={time.perf_counter() - started:.1f}s", flush=True)

        self.seen = csr_matrix((np.ones(train.height, dtype=np.float32), (rows, cols)), shape=self.R.shape)
        return self

    def recommend(self, user_id, k=10):
        user_idx = self.user_to_idx[user_id]
        scores = self.item_factors @ self.user_factors[user_idx]
        scores[self.seen[user_idx].indices] = -np.inf

        k = min(k, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]

        return [self.items[i] for i in top]