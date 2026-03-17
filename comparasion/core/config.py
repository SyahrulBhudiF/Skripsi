from dataclasses import dataclass, field
from pathlib import Path


REGIONS = {
    "mulut": list(range(48, 68)),
    "mata_kiri": list(range(17, 22)) + list(range(36, 42)),
    "mata_kanan": list(range(22, 27)) + list(range(42, 48)),
    "alis_kiri": list(range(17, 22)),
    "alis_kanan": list(range(22, 27)),
}
TARGET_SIZE = {
    "mulut": (70, 35),
    "mata_kiri": (48, 32),
    "mata_kanan": (48, 32),
    "alis_kiri": (64, 24),
    "alis_kanan": (64, 24),
}


@dataclass(slots=True)
class ComparisonConfig:
    predictor_path: Path = Path("preprocess-anxiety/models/shape_predictor_68_face_landmarks.dat")
    output_root: Path = Path("comparasion/output_casme2")
    regions: dict[str, list[int]] = field(default_factory=lambda: dict(REGIONS))
    target_size: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(TARGET_SIZE))
    padding_x: int = 6
    padding_y: int = 8
    block_size: int = 7
    feature_method: str = "poc_abs"
    include_quadrant_counts: bool = True

    @property
    def roi_output_dir(self) -> Path:
        return self.output_root / "roi"

    @property
    def feature_output_dir(self) -> Path:
        return self.output_root / "features"
