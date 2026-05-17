from pathlib import Path

import numpy as np
import pandas as pd



def load_feature_excel(path: str | Path) -> pd.DataFrame:
    """Load a feature Excel file (flatten or quadran_sum)."""
    return pd.read_excel(Path(path))


def load_feature_vector(path: str | Path) -> np.ndarray:
    """Legacy: load a .npy feature and flatten it."""
    feature = np.load(Path(path))
    return np.asarray(feature, dtype=float).reshape(-1)


def stack_feature_vectors(paths: list[str | Path]) -> np.ndarray:
    vectors = [load_feature_vector(path) for path in paths]
    if not vectors:
        raise ValueError("No feature paths provided")
    return np.stack(vectors, axis=0)


def concat_region_features(feature_map: dict[str, np.ndarray], region_order: list[str] | None = None) -> np.ndarray:
    ordered_regions = region_order or sorted(feature_map)
    vectors = [np.asarray(feature_map[region], dtype=float).reshape(-1) for region in ordered_regions]
    if not vectors:
        raise ValueError("No features to concatenate")
    return np.concatenate(vectors, axis=0)
