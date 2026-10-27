import numpy as np

from models.als import ALS


class SignalALS(ALS):
    def __init__(self, factors=32, reg=0.1, epochs=8, beta=0.5, tau_days=365.0, seed=42):
        super().__init__(factors=factors, reg=reg, epochs=epochs, seed=seed)
        self.beta = beta
        self.tau_days = tau_days

    def fit(self, train):
        ratings = train["rating"].to_numpy().astype(np.float32)
        base_target = (ratings - 3.0) / 2.0

        timestamps = train["timestamp"].to_numpy().astype(np.int64)
        train_end = timestamps.max()
        age_days = (train_end - timestamps) / (1000.0 * 60 * 60 * 24)

        recency = np.exp(-age_days / self.tau_days).astype(np.float32)
        target = base_target * (1.0 + self.beta * recency)

        print(f"signal=rating_recency beta={self.beta} tau_days={self.tau_days}", flush=True)
        return super().fit(train, target=target)