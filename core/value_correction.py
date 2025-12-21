import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Value Correction T
# -------------------------------
class ValueCorrection(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, d, bias=False)
        self.fc2 = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, delta):
        x = self.fc1(delta)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
