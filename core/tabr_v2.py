import math

import faiss
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class TabRv2Backbone(nn.Module):
    def __init__(self, in_dim: int, d_main: int, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(in_dim, d_main)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_main),
                    nn.Linear(d_main, d_main * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_main * 2, d_main),
                )
                for _ in range(n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(d_main)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for block in self.blocks:
            x = x + block(x)
        return self.norm(x)


class TabRv2Retrieval(nn.Module):
    def __init__(self, d_main: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.key_proj = nn.Linear(d_main, d_main, bias=False)
        self.label_emb = nn.Embedding(num_classes, d_main)
        self.value_mlp = nn.Sequential(
            nn.Linear(d_main, d_main * 2, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_main * 2, d_main, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,
        ctx_z: torch.Tensor,
        ctx_y: torch.Tensor,
        faiss_index,
        k_neighbors: int,
    ) -> torch.Tensor:
        q = self.key_proj(z)
        q_np = q.detach().cpu().numpy().astype("float32")
        dist, idx = faiss_index.search(q_np, min(k_neighbors, len(ctx_y)))

        dist = torch.tensor(dist, device=z.device, dtype=z.dtype)
        idx = torch.tensor(idx, device=z.device, dtype=torch.long)

        ctx_k = self.key_proj(ctx_z)
        neigh_k = ctx_k[idx]
        neigh_y = self.label_emb(ctx_y[idx])

        sim = -dist / math.sqrt(q.shape[1])
        weights = F.softmax(sim, dim=1)
        weights = self.dropout(weights)

        delta = q.unsqueeze(1) - neigh_k
        values = neigh_y + self.value_mlp(delta)
        return torch.sum(weights.unsqueeze(-1) * values, dim=1)


class LitTabRv2(pl.LightningModule):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 2,
        d_main: int = 256,
        context_size: int = 96,
        n_blocks: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        context_dropout: float = 0.0,
        freeze_context_epoch: int = 5,
        class_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = TabRv2Backbone(in_dim, d_main, n_blocks=n_blocks, dropout=dropout)
        self.retrieval = TabRv2Retrieval(d_main, num_classes, dropout=dropout)
        self.predictor = nn.Sequential(
            nn.LayerNorm(d_main),
            nn.Linear(d_main, d_main * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_main * 2, d_main),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(d_main, num_classes)

        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.class_weights = None
            self.loss_fn = nn.CrossEntropyLoss()

        self.ctx_frozen = False
        self.ctx_z = None
        self.ctx_y = None
        self.faiss_index = None

        self.imputer = None
        self.scaler = None
        self.feature_cols = None
        self.default_context = None

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def _sample_context(self, ctx_x: torch.Tensor, ctx_y: torch.Tensor):
        if self.hparams.context_dropout <= 0:
            return ctx_x, ctx_y
        keep_n = max(2, int(len(ctx_y) * (1.0 - self.hparams.context_dropout)))
        idx = torch.randperm(len(ctx_y), device=ctx_y.device)[:keep_n]
        return ctx_x[idx], ctx_y[idx]

    def _build_context(self, ctx_x: torch.Tensor, ctx_y: torch.Tensor):
        ctx_x = ctx_x.to(self.get_device())
        ctx_y = ctx_y.to(self.get_device())
        ctx_x, ctx_y = self._sample_context(ctx_x, ctx_y)
        with torch.no_grad():
            self.backbone.eval()
            self.ctx_z = self.backbone(ctx_x)
            self.ctx_y = ctx_y
        index = faiss.IndexFlatL2(self.ctx_z.shape[1])
        index.add(self.ctx_z.detach().cpu().numpy().astype("float32"))
        self.faiss_index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.faiss_index is None or self.ctx_z is None or self.ctx_y is None:
            raise RuntimeError("Context not initialized")
        z = self.backbone(x)
        r = self.retrieval(
            z,
            self.ctx_z,
            self.ctx_y,
            self.faiss_index,
            self.hparams.context_size,
        )
        h = z + r
        h = h + self.predictor(h)
        return self.head(h)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def on_train_epoch_start(self):
        if not self.ctx_frozen and self.current_epoch >= self.hparams.freeze_context_epoch:
            self.freeze_context(self.trainer.datamodule.ctx_train_x, self.trainer.datamodule.ctx_train_y)

    def freeze_context(self, ctx_x: torch.Tensor, ctx_y: torch.Tensor):
        self.ctx_frozen = True
        self._build_context(ctx_x, ctx_y)
        print("TabRv2 context frozen & FAISS built")

    def on_validation_epoch_start(self):
        self._build_context(self.trainer.datamodule.ctx_train_x, self.trainer.datamodule.ctx_train_y)
        print("TabRv2 validation context rebuilt from train context")

    def set_context(self, ctx_x: torch.Tensor, ctx_y: torch.Tensor):
        self._build_context(ctx_x, ctx_y)

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
        assert self.imputer is not None
        assert self.scaler is not None
        assert self.feature_cols is not None

        self.eval()
        x_np = self.scaler.transform(self.imputer.transform(df[self.feature_cols]))
        x = torch.tensor(x_np, dtype=torch.float32, device=self.get_device())

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
