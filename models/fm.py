import torch
from torch import nn


class FM(nn.Module):
    def __init__(self, n_features, dim=32):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(n_features, 1)
        self.embedding = nn.Embedding(n_features, dim)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)

        v = self.embedding(x)
        interaction = 0.5 * (
            (v.sum(dim=1) ** 2 - (v ** 2).sum(dim=1))
            .sum(dim=1)
        )

        return self.bias + linear + interaction