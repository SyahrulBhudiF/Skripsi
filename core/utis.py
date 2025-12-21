import torch
import pandas as pd

# -------------------------------
# Utils
# -------------------------------
def load_csv(path, label_col):
    df = pd.read_csv(path)
    y = torch.tensor(df[label_col].values, dtype=torch.long)
    X = torch.tensor(df.drop(columns=[label_col]).values, dtype=torch.float32)
    return X, y, df
