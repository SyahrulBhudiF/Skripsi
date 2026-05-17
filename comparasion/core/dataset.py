from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(slots=True)
class FrameSample:
    sample_id: str
    frames: list[np.ndarray]
    frame_paths: list[Path]


@dataclass(slots=True)
class ClipInfo:
    clip_dir: Path
    emotion: str
    subject: str
    clip_name: str

    @property
    def sample_id(self) -> str:
        return f"{self.emotion}__{self.subject}__{self.clip_name}"


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def _detect_depth(root: Path) -> int:
    for emotion_dir in root.iterdir():
        if not emotion_dir.is_dir():
            continue
        for child in emotion_dir.iterdir():
            if not child.is_dir():
                continue
            return 2 if any(f.suffix.lower() in IMAGE_EXTS for f in child.iterdir() if f.is_file()) else 4
    return 4




def _discover_casme2(root: Path) -> list[ClipInfo]:
    """Walk CASME2: emotion/clip_dir (images inside) → list of ClipInfo."""
    clips: list[ClipInfo] = []
    for emotion_dir in sorted(root.iterdir()):
        if not emotion_dir.is_dir():
            continue
        for clip_dir in sorted(emotion_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            if any(f.suffix.lower() in IMAGE_EXTS for f in clip_dir.iterdir() if f.is_file()):
                # e.g. "03_EP19_08" → subject "03"
                parts = clip_dir.name.split("_", 1)
                subject = parts[0] if len(parts) > 1 else "unknown"
                clips.append(ClipInfo(
                    clip_dir=clip_dir,
                    emotion=emotion_dir.name,
                    subject=subject,
                    clip_name=clip_dir.name,
                ))
    return clips


def _discover_casme3(root: Path) -> list[ClipInfo]:
    """Walk CASME3: emotion/subject/char/clip_range → list of ClipInfo."""
    clips: list[ClipInfo] = []
    for emotion_dir in sorted(root.iterdir()):
        if not emotion_dir.is_dir():
            continue
        for subject_dir in sorted(emotion_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            for char_dir in sorted(subject_dir.iterdir()):
                if not char_dir.is_dir():
                    continue
                for clip_dir in sorted(char_dir.iterdir()):
                    if not clip_dir.is_dir():
                        continue
                    if any(f.suffix.lower() in IMAGE_EXTS for f in clip_dir.iterdir() if f.is_file()):
                        clips.append(ClipInfo(
                            clip_dir=clip_dir,
                            emotion=emotion_dir.name,
                            subject=subject_dir.name,
                            clip_name=clip_dir.name,
                        ))
    return clips


def discover_clips(dataset_root: str | Path) -> list[ClipInfo]:
    """Auto-detect dataset structure (CASME2 vs CASME3) and discover clips."""
    root = Path(dataset_root)
    depth = _detect_depth(root)
    if depth == 2:
        return _discover_casme2(root)
    return _discover_casme3(root)


def load_clip(clip: ClipInfo) -> FrameSample:
    """Load a ClipInfo into a FrameSample (sorted frames)."""
    frame_paths = sorted(
        p for p in clip.clip_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if len(frame_paths) < 2:
        raise ValueError(f"Need >= 2 frames: {clip.clip_dir}")
    return FrameSample(
        sample_id=clip.sample_id,
        frames=[_read_image(p) for p in frame_paths],
        frame_paths=frame_paths,
    )


def load_frame_pair(current_path: str | Path, reference_path: str | Path, sample_id: str | None = None) -> FrameSample:
    current = Path(current_path)
    reference = Path(reference_path)
    resolved_sample_id = sample_id or current.stem
    return FrameSample(
        sample_id=resolved_sample_id,
        frames=[_read_image(reference), _read_image(current)],
        frame_paths=[reference, current],
    )


def load_frame_sequence(sequence_dir: str | Path, sample_id: str | None = None) -> FrameSample:
    sequence_path = Path(sequence_dir)
    frame_paths = sorted(
        path for path in sequence_path.iterdir()
        if path.suffix.lower() in IMAGE_EXTS
    )
    if len(frame_paths) < 2:
        raise ValueError(f"Sequence must contain at least 2 frames: {sequence_path}")
    resolved_sample_id = sample_id or sequence_path.name
    frames = [_read_image(path) for path in frame_paths]
    return FrameSample(sample_id=resolved_sample_id, frames=frames, frame_paths=frame_paths)
