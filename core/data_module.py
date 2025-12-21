import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------
# DataModule
# -------------------------------
class TabRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X_train, y_train,
        X_val, y_val,
        ctx_train_X, ctx_train_y,
        ctx_val_X, ctx_val_y,
        batch_size=256
    ):
        super().__init__()
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.ctx_train_x, self.ctx_train_y = ctx_train_X, ctx_train_y
        self.ctx_val_x, self.ctx_val_y = ctx_val_X, ctx_val_y
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=self.batch_size,
            num_workers=8
        )
