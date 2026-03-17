from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class SpottingConfig:
    dataset_root: Path
    output_root: Path
    predictor_path: Path
    video_extensions: set[str]
    overwrite: bool = False
    block_size: int = 7
    padding_x: int = 6
    padding_y: int = 8
    min_phase_duration: int = 2
    top_k_apex: int = 5
    regions: dict[str, list[int]] | None = None
    target_size: dict[str, tuple[int, int]] | None = None
    distance_threshold: int = 1
    prominence_threshold: float = 0.005
    cutoff_ratio: float = 0.3

    def __post_init__(self) -> None:
        if self.regions is None:
            self.regions = {
                "mata_kanan": list(range(42, 48)),
                "mata_kiri": list(range(36, 42)),
                "alis_kanan": list(range(22, 27)),
                "alis_kiri": list(range(17, 22)),
                "mulut": list(range(48, 68)),
            }
        if self.target_size is None:
            self.target_size = {
                "mata_kanan": (48, 32),
                "mata_kiri": (48, 32),
                "alis_kanan": (48, 20),
                "alis_kiri": (48, 20),
                "mulut": (70, 35),
            }

    @property
    def output_dataset_root(self) -> Path:
        return self.output_root / "dataset"

    @property
    def metadata_path(self) -> Path:
        return self.output_root / "metadata.xlsx"


def run_spotting(_config: SpottingConfig) -> pd.DataFrame:
    raise NotImplementedError("Run logic lives in process_all_datasets_flexible.ipynb")
