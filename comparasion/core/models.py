from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ArrayLike = np.ndarray


@dataclass(slots=True)
class ModelResult:
    model_name: str
    model: Any
    classes_: np.ndarray | None = None


@dataclass(slots=True)
class SequenceTransform:
    scaler: StandardScaler
    pca: PCA | None = None


@dataclass(slots=True)
class SequenceItem:
    emotion: str
    subject: str
    clip: str
    x: np.ndarray
    y: int

    @property
    def clip_id(self) -> str:
        return f"{self.emotion}__{self.subject}__{self.clip}"


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden[-1])


def build_svm(*, kernel: str = "rbf", c: float = 1.0, gamma: str | float = "scale", probability: bool = True, class_weight: str | dict | None = "balanced") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, C=c, gamma=gamma, probability=probability, class_weight=class_weight)),
    ])


def build_knn(*, n_neighbors: int = 5, weights: str = "uniform") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
    ])


def build_pca_svm(
    *,
    n_components: int | float | None = None,
    kernel: str = "rbf",
    c: float = 1.0,
    gamma: str | float = "scale",
    probability: bool = True,
    class_weight: str | dict | None = "balanced",
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components)))
    steps.append(("svm", SVC(kernel=kernel, C=c, gamma=gamma, probability=probability, class_weight=class_weight)))
    return Pipeline(steps)


def build_pca_knn(*, n_components: int | float | None = None, n_neighbors: int = 5, weights: str = "uniform") -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components)))
    steps.append(("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)))
    return Pipeline(steps)


def fit_model(model: Any, x_train: ArrayLike, y_train: ArrayLike, model_name: str) -> ModelResult:
    fitted = model.fit(x_train, y_train)
    classes_ = getattr(fitted, "classes_", None)
    return ModelResult(model_name=model_name, model=fitted, classes_=classes_)


def predict(model: Any, x: ArrayLike) -> np.ndarray:
    return np.asarray(model.predict(x))


def predict_proba(model: Any, x: ArrayLike) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba")
    return np.asarray(model.predict_proba(x))


def fit_sequence_transform(sequences: list[np.ndarray], n_components: int | None = None) -> SequenceTransform:
    scaler = StandardScaler()
    stacked = np.concatenate(sequences, axis=0)
    stacked = scaler.fit_transform(stacked)
    if n_components is None:
        return SequenceTransform(scaler=scaler)
    pca = PCA(n_components=n_components)
    pca.fit(stacked)
    return SequenceTransform(scaler=scaler, pca=pca)


def transform_sequences(sequences: list[np.ndarray], transform: SequenceTransform) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for sequence in sequences:
        transformed = transform.scaler.transform(sequence)
        if transform.pca is not None:
            transformed = transform.pca.transform(transformed)
        out.append(np.asarray(transformed, dtype=np.float32))
    return out


def pad_sequence_batch(sequences: list[np.ndarray], labels: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("No sequences provided")
    max_len = max(len(sequence) for sequence in sequences)
    feat_dim = sequences[0].shape[1]
    x = np.zeros((len(sequences), max_len, feat_dim), dtype=np.float32)
    lengths = np.zeros(len(sequences), dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    for i, sequence in enumerate(sequences):
        seq_len = len(sequence)
        x[i, :seq_len] = sequence
        lengths[i] = seq_len
    return x, lengths, y


def make_sequence_loader(
    x: np.ndarray,
    lengths: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_lstm_model(
    model: LSTMClassifier,
    train_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> LSTMClassifier:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, lb, yb in train_loader:
            xb = xb.to(device)
            lb = lb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, lb), yb)
            loss.backward()
            optimizer.step()
    return model


def predict_lstm(model: LSTMClassifier, loader: DataLoader, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for xb, lb, yb in loader:
            xb = xb.to(device)
            lb = lb.to(device)
            logits = model(xb, lb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(yb.numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def build_clip_sequences(
    df,
    *,
    feature_columns: list[str],
    label_column: str = "label_enc",
) -> list[SequenceItem]:
    sequences: list[SequenceItem] = []
    for keys, group in df.sort_values("frame").groupby(["emotion", "subject", "clip"]):
        emotion, subject, clip = keys
        frame_table = group.sort_values("frame")
        x = frame_table[feature_columns].to_numpy(dtype=np.float32)
        y = int(frame_table[label_column].iloc[0])
        sequences.append(SequenceItem(emotion=emotion, subject=subject, clip=clip, x=x, y=y))
    return sequences


def split_external_clips_by_emotion(
    sequences: list[SequenceItem],
    *,
    n_per_emotion: int = 1,
    seed: int = 42,
) -> tuple[list[SequenceItem], list[SequenceItem]]:
    rng = np.random.RandomState(seed)
    seq_ext: list[SequenceItem] = []
    seq_main: list[SequenceItem] = []
    for emotion in sorted({item.emotion for item in sequences}):
        emotion_items = [item for item in sequences if item.emotion == emotion]
        chosen_idx = set(rng.choice(np.arange(len(emotion_items)), size=min(n_per_emotion, len(emotion_items)), replace=False).tolist())
        for idx, item in enumerate(emotion_items):
            if idx in chosen_idx:
                seq_ext.append(item)
            else:
                seq_main.append(item)
    return seq_main, seq_ext


def split_main_sequences(
    sequence_items: list[SequenceItem],
    *,
    test_size: float,
    seed: int,
) -> tuple[list[SequenceItem], list[SequenceItem], np.ndarray]:
    labels = np.asarray([item.y for item in sequence_items], dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    keep_classes = classes[counts >= 2]
    dropped_classes = classes[counts < 2]
    keep_set = set(keep_classes.tolist())
    filtered_items = [item for item in sequence_items if item.y in keep_set]
    filtered_labels = np.asarray([item.y for item in filtered_items], dtype=np.int64)
    if len(np.unique(filtered_labels)) < 2:
        raise ValueError("Need at least 2 classes in main sequences after filtering")
    idx_train, idx_val = train_test_split(
        np.arange(len(filtered_items)),
        test_size=test_size,
        stratify=filtered_labels,
        random_state=seed,
    )
    return [filtered_items[i] for i in idx_train], [filtered_items[i] for i in idx_val], dropped_classes


def fit_lstm_split(
    train_items: list[SequenceItem],
    val_items: list[SequenceItem],
    *,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
    num_classes: int,
    n_components: int | None = None,
) -> tuple[LSTMClassifier, SequenceTransform, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = [item.x for item in train_items]
    val_x = [item.x for item in val_items]
    train_y = np.asarray([item.y for item in train_items], dtype=np.int64)
    val_y = np.asarray([item.y for item in val_items], dtype=np.int64)
    transform = fit_sequence_transform(train_x, n_components=n_components)
    train_x = transform_sequences(train_x, transform)
    val_x = transform_sequences(val_x, transform)
    x_tr, len_tr, y_tr = pad_sequence_batch(train_x, train_y)
    x_val, len_val, y_val = pad_sequence_batch(val_x, val_y)
    train_loader = make_sequence_loader(x_tr, len_tr, y_tr, batch_size, True)
    val_loader = make_sequence_loader(x_val, len_val, y_val, batch_size, False)
    model = LSTMClassifier(
        input_size=x_tr.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )
    model = train_lstm_model(model, train_loader, epochs=epochs, lr=lr, device=device)
    y_tr_true, y_tr_pred = predict_lstm(model, train_loader, device=device)
    y_val_true, y_val_pred = predict_lstm(model, val_loader, device=device)
    return model, transform, y_tr_true, y_tr_pred, y_val_true, y_val_pred


def evaluate_lstm_external(
    model: LSTMClassifier,
    transform: SequenceTransform,
    external_items: list[SequenceItem],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    ext_x = transform_sequences([item.x for item in external_items], transform)
    ext_y = np.asarray([item.y for item in external_items], dtype=np.int64)
    x_ext, len_ext, y_ext = pad_sequence_batch(ext_x, ext_y)
    ext_loader = make_sequence_loader(x_ext, len_ext, y_ext, batch_size, False)
    return predict_lstm(model, ext_loader, device=device)