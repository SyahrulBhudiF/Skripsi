import importlib.util
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import Tensor


_OFFICIAL_ROOT = Path(__file__).resolve().parent / 'tabular-dl-tabr-official'
if str(_OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_ROOT))
os.environ.setdefault('PROJECT_DIR', str(_OFFICIAL_ROOT))
_TABR_PATH = _OFFICIAL_ROOT / 'bin' / 'tabr.py'
spec = importlib.util.spec_from_file_location('official_tabr_module', _TABR_PATH)
official_tabr = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(official_tabr)
OfficialModel = official_tabr.Model


class OfficialTabRClassifier(pl.LightningModule):
    def __init__(
        self,
        in_dim: int,
        num_classes: int = 2,
        context_size: int = 8,
        d_main: int = 256,
        d_multiplier: float = 2.0,
        encoder_n_blocks: int = 0,
        predictor_n_blocks: int = 1,
        mixer_normalization='auto',
        context_dropout: float = 0.0,
        dropout0: float = 0.1,
        dropout1='dropout0',
        normalization: str = 'LayerNorm',
        activation: str = 'ReLU',
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = OfficialModel(
            n_num_features=in_dim,
            n_bin_features=0,
            cat_cardinalities=[],
            n_classes=num_classes,
            num_embeddings=None,
            d_main=d_main,
            d_multiplier=d_multiplier,
            encoder_n_blocks=encoder_n_blocks,
            predictor_n_blocks=predictor_n_blocks,
            mixer_normalization=mixer_normalization,
            context_dropout=context_dropout,
            dropout0=dropout0,
            dropout1=dropout1,
            normalization=normalization,
            activation=activation,
            memory_efficient=False,
            candidate_encoding_batch_size=None,
        )
        self.context_size = context_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.context_x = None
        self.context_y = None
        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.feature_cols = None
        self.default_context = None

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'label_encoder' in name or name.endswith('bias') or 'norm' in name or 'normalization' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return torch.optim.AdamW(
            [
                {'params': decay, 'weight_decay': self.weight_decay},
                {'params': no_decay, 'weight_decay': 0.0},
            ],
            lr=self.lr,
        )

    def _x_dict(self, x: Tensor):
        return {'num': x}

    def set_context(self, ctx_x: Tensor, ctx_y: Tensor):
        self.context_x = ctx_x.to(self.get_device())
        self.context_y = ctx_y.to(self.get_device())

    def set_default_context(self, ctx_x: Tensor, ctx_y: Tensor):
        self.default_context = (ctx_x, ctx_y)
        self.set_context(ctx_x, ctx_y)

    def on_validation_epoch_start(self):
        if self.default_context is not None:
            self.set_context(*self.default_context)

    def forward_with_indices(self, x: Tensor, y: Tensor | None, idx: Tensor | None, is_train: bool):
        if self.context_x is None or self.context_y is None:
            raise RuntimeError('Context not initialized')
        if is_train:
            assert y is not None and idx is not None
            train_indices = torch.arange(len(self.context_y), device=self.get_device())
            candidate_indices = train_indices[~torch.isin(train_indices, idx)]
            candidate_x = self.context_x[candidate_indices]
            candidate_y = self.context_y[candidate_indices]
        else:
            candidate_x = self.context_x
            candidate_y = self.context_y
        logits = self.model(
            x_=self._x_dict(x),
            y=y if is_train else None,
            candidate_x_=self._x_dict(candidate_x),
            candidate_y=candidate_y,
            context_size=self.context_size,
            is_train=is_train,
        ).squeeze(-1)
        return logits if logits.ndim == 2 else torch.stack([-logits, logits], dim=1)

    def forward(self, x: Tensor):
        return self.forward_with_indices(x, None, None, False)

    def predict_logits_tensor(self, x: Tensor):
        return self.forward_with_indices(x, None, None, False)

    def set_preprocessor(self, imputer, scaler, feature_cols):
        self.imputer = imputer
        self.scaler = scaler
        self.feature_cols = feature_cols

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
            logits = self.predict_logits_tensor(x)
            prob = F.softmax(logits, dim=1)[:, 1]
            pred = (prob > threshold).long()
        return pred.cpu().numpy(), prob.cpu().numpy()
