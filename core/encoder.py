import torch
import torch.nn as nn

# -------------------------------
# Encoder
# -------------------------------
class TabREncoder(nn.Module):
    def __init__(self, in_dim, d, n_blocks=0, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d, d),
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        x = self.linear(x)
        for blk in self.blocks:
            x = x + blk(x)
        return x
