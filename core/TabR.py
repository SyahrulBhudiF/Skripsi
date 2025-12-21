import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import faiss
import pandas as pd
import numpy as np

from core.encoder import TabREncoder
from core.value_correction import ValueCorrection


# -------------------------------
# Lightning TabR
# -------------------------------
class LitTabR(pl.LightningModule):
    def __init__(
        self,
        in_dim,
        num_classes=2,
        d=128,
        m=96,
        encoder_blocks=0,
        dropout=0.1,
        lr=1e-3,
        freeze_context_epoch=5
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = TabREncoder(in_dim, d, encoder_blocks, dropout)
        self.WK = nn.Linear(d, d, bias=False)
        self.WY = nn.Embedding(num_classes, d)
        self.T = ValueCorrection(d, dropout)

        self.predictor = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d)
        )
        self.head = nn.Linear(d, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        # context state
        self.ctx_frozen = False
        self.ctx_k = None
        self.ctx_y = None
        self.faiss_index = None
        self.ctx_val_ready = False

        # inference helpers
        self.imputer = None
        self.scaler = None
        self.feature_cols = None
        self.default_context = None

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    # ---- FAISS ----
    def build_faiss(self):
        k = self.ctx_k.detach().cpu().numpy().astype("float32")
        index = faiss.IndexFlatL2(k.shape[1])
        index.add(k)
        self.faiss_index = index

    def retrieve_topk(self, k_query):
        kq = k_query.detach().cpu().numpy().astype("float32")
        dist, idx = self.faiss_index.search(kq, self.hparams.m)
        return (
            torch.tensor(dist, device=k_query.device),
            torch.tensor(idx, device=k_query.device)
        )

    # ---- Forward ----
    def forward(self, x):
        z = self.encoder(x)
        k = self.WK(z)

        dist, idx = self.retrieve_topk(k)
        sim = -dist
        weights = F.softmax(sim, dim=1)
        weights = F.dropout(weights, p=0.1, training=self.training)

        k_i = self.ctx_k[idx]
        y_i = self.WY(self.ctx_y[idx])

        delta = k.unsqueeze(1) - k_i
        V = y_i + self.T(delta)
        R = torch.sum(weights.unsqueeze(-1) * V, dim=1)

        z_hat = z + R
        z_hat = z_hat + self.predictor(z_hat)
        return self.head(z_hat)

    # ---- Steps ----
    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc,  prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc,  prog_bar=True, on_epoch=True)

    # ---- Context Freeze ----
    def on_train_epoch_start(self):
        if (
            not self.ctx_frozen
            and self.current_epoch >= self.hparams.freeze_context_epoch
        ):
            self.freeze_context(self.trainer.datamodule.ctx_train_x,
                                self.trainer.datamodule.ctx_train_y)

    def freeze_context(self, ctx_x, ctx_y):
        self.ctx_frozen = True
        self.encoder.eval()

        ctx_x = ctx_x.to(self.get_device())
        ctx_y = ctx_y.to(self.get_device())

        with torch.no_grad():
            z_ctx = self.encoder(ctx_x)
            self.ctx_k = self.WK(z_ctx)
            self.ctx_y = ctx_y

        self.build_faiss()
        print("Context frozen & FAISS built")

    def on_validation_epoch_start(self):
        if self.ctx_val_ready:
            return

        device = self.get_device()
        ctx_x = self.trainer.datamodule.ctx_val_x.to(device)
        ctx_y = self.trainer.datamodule.ctx_val_y.to(device)

        self.encoder.eval()
        with torch.no_grad():
            z_ctx = self.encoder(ctx_x)
            self.ctx_k = self.WK(z_ctx)
            self.ctx_y = ctx_y

        self.build_faiss()
        self.ctx_val_ready = True
        print("Validation context frozen & FAISS built")

    def set_context(self, ctx_x, ctx_y):
        device = self.get_device()
        ctx_x = ctx_x.to(device)
        ctx_y = ctx_y.to(device)

        self.encoder.eval()
        with torch.no_grad():
            z_ctx = self.encoder(ctx_x)
            self.ctx_k = self.WK(z_ctx)
            self.ctx_y = ctx_y

        self.build_faiss()

    def set_preprocessor(self, imputer, scaler, feature_cols):
        self.imputer = imputer
        self.scaler = scaler
        self.feature_cols = feature_cols

    def set_default_context(self, ctx_x, ctx_y):
        self.default_context = (ctx_x, ctx_y)
        self.set_context(ctx_x, ctx_y)

    def get_device(self):
        return next(self.parameters()).device

    def predict_from_df(self, df, threshold=0.5):
        assert self.imputer is not None, "imputer not set"
        assert self.scaler is not None, "scaler not set"
        assert self.feature_cols is not None, "feature_cols not set"

        self.eval()

        # preprocess
        X_np = self.scaler.transform(
            self.imputer.transform(df[self.feature_cols])
        )

        x = torch.tensor(
            X_np,
            dtype=torch.float32,
            device=self.get_device()
        )

        # ensure context
        if self.default_context is not None:
            ctx_x, ctx_y = self.default_context
            self.set_context(ctx_x, ctx_y)

        with torch.no_grad():
            logits = self(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = (prob > threshold).long()

        return pred.cpu().numpy(), prob.cpu().numpy()

    def predict_from_csv(self, csv_path, **kwargs):
        df = pd.read_csv(csv_path)
        return self.predict_from_df(df, **kwargs)
