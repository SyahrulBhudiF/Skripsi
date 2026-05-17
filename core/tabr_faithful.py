import math

import faiss
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    def __init__(self, d_block: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_block)
        self.linear1 = nn.Linear(d_block, d_block * 2)
        self.linear2 = nn.Linear(d_block * 2, d_block)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        z = self.linear1(z)
        z = F.relu(z)
        z = self.dropout(z)
        z = self.linear2(z)
        z = self.dropout(z)
        return x + z


class TabRFaithfulBackbone(nn.Module):
    def __init__(self, in_dim: int, d_main: int, n_blocks: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(in_dim, d_main)
        self.blocks = nn.ModuleList([
            _ResidualBlock(d_main, dropout) for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_main)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class TabRFaithfulRetrieval(nn.Module):
    def __init__(self, d_main: int, num_classes: int, dropout: float):
        super().__init__()
        self.W_k = nn.Linear(d_main, d_main, bias=False)
        self.W_v = nn.Linear(d_main, d_main, bias=False)
        self.W_y = nn.Embedding(num_classes, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_main * 2, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_main * 2, d_main, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, context_z, context_y, faiss_index, n_neighbors):
        q = self.W_k(z)
        q_np = q.detach().cpu().numpy().astype('float32')
        topk = min(n_neighbors, len(context_y))
        dist, idx = faiss_index.search(q_np, topk)
        dist = torch.tensor(dist, device=z.device, dtype=z.dtype)
        idx = torch.tensor(idx, device=z.device, dtype=torch.long)
        context_k = self.W_k(context_z)
        context_v = self.W_v(context_z)
        neigh_k = context_k[idx]
        neigh_v = context_v[idx]
        neigh_y = self.W_y(context_y[idx])
        logits = -dist / math.sqrt(q.shape[-1])
        alpha = F.softmax(logits, dim=1)
        alpha = self.dropout(alpha)
        delta = q.unsqueeze(1) - neigh_k
        neighbor_repr = neigh_v + neigh_y + self.T(delta)
        return torch.sum(alpha.unsqueeze(-1) * neighbor_repr, dim=1)


class LitTabRFaithful(pl.LightningModule):
    def __init__(self, in_dim, num_classes=2, d_main=256, n_blocks=3, n_neighbors=32, dropout=0.1, lr=3e-4, weight_decay=1e-5, freeze_context_epoch=16, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = TabRFaithfulBackbone(in_dim, d_main, n_blocks, dropout)
        self.retrieval = TabRFaithfulRetrieval(d_main, num_classes, dropout)
        self.head_norm = nn.LayerNorm(d_main)
        self.head = nn.Linear(d_main, num_classes)
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.class_weights = None
            self.loss_fn = nn.CrossEntropyLoss()
        self.context_z = None
        self.context_y = None
        self.faiss_index = None
        self.ctx_frozen = False
        self.imputer = None
        self.scaler = None
        self.feature_cols = None
        self.default_context = None

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def _fit_context(self, ctx_x, ctx_y):
        device = self.get_device()
        ctx_x = ctx_x.to(device)
        ctx_y = ctx_y.to(device)
        with torch.no_grad():
            self.backbone.eval()
            context_z = self.backbone(ctx_x)
        self.context_z = context_z
        self.context_y = ctx_y
        index = faiss.IndexFlatL2(context_z.shape[1])
        index.add(context_z.detach().cpu().numpy().astype('float32'))
        self.faiss_index = index

    def forward(self, x):
        if self.context_z is None or self.context_y is None or self.faiss_index is None:
            raise RuntimeError('Context not initialized')
        z = self.backbone(x)
        r = self.retrieval(z, self.context_z, self.context_y, self.faiss_index, self.hparams.n_neighbors)
        h = self.head_norm(z + r)
        return self.head(h)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

    def on_train_epoch_start(self):
        if not self.ctx_frozen and self.current_epoch >= self.hparams.freeze_context_epoch:
            self.freeze_context(self.trainer.datamodule.ctx_train_x, self.trainer.datamodule.ctx_train_y)

    def freeze_context(self, ctx_x, ctx_y):
        self.ctx_frozen = True
        self._fit_context(ctx_x, ctx_y)

    def on_validation_epoch_start(self):
        self._fit_context(self.trainer.datamodule.ctx_train_x, self.trainer.datamodule.ctx_train_y)

    def set_context(self, ctx_x, ctx_y):
        self._fit_context(ctx_x, ctx_y)

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
            self.set_context(*self.default_context)
        with torch.no_grad():
            logits = self(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = (prob > threshold).long()
        return pred.cpu().numpy(), prob.cpu().numpy()

    def predict_from_csv(self, csv_path, **kwargs):
        df = pd.read_csv(csv_path)
        return self.predict_from_df(df, **kwargs)
