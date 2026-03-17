from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd

from comparasion.core.config import ComparisonConfig, REGIONS
from comparasion.core.dataset import ClipInfo, load_clip
from comparasion.core.roi import ROIExtractor

from features_extraction.poc import POC
from features_extraction.quadran import Quadran
from features_extraction.vektor import Vektor

COMPONENTS = list(REGIONS.keys())
QUADRANS = ["Q1", "Q2", "Q3", "Q4"]


def _extract_roi_per_frame(
    image: np.ndarray,
    extractor: ROIExtractor,
) -> dict[str, np.ndarray] | None:
    """Extract ROIs for a single frame. Returns None if face not detected."""
    try:
        return extractor.extract_rois(image)
    except ValueError:
        return None


def _to_gray(roi: np.ndarray) -> np.ndarray:
    if roi.ndim == 2:
        return roi
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def process_clip(
    clip: ClipInfo,
    extractor: ROIExtractor,
    config: ComparisonConfig,
) -> tuple[list[dict], list[dict]]:
    """
    Process a single clip:
    1. Extract ROIs per frame (save ALL ROI images)
    2. Baseline = frame[0] grayscale ROI per component
    3. frame[1..N] → POC → Vektor → Quadran per component
    4. Return (rows_flat, rows_quad) — caller accumulates

    Returns rows instead of saving per-clip. Caller collects all rows
    then saves 2 combined Excel files at the end.
    """
    sample = load_clip(clip)
    emotion = clip.emotion
    subject = clip.subject
    clip_name = clip.clip_name
    sample_id = clip.sample_id

    if len(sample.frames) < 2:
        raise ValueError(f"Need >= 2 frames: {clip.clip_dir}")

    # Extract ROIs per frame and save all
    rois_per_frame: list[dict[str, np.ndarray] | None] = []
    roi_dir = config.roi_output_dir / emotion / sample_id

    for frame_idx, frame in enumerate(sample.frames):
        rois = _extract_roi_per_frame(frame, extractor)
        rois_per_frame.append(rois)

        if rois is None:
            continue
        frame_name = sample.frame_paths[frame_idx].stem
        for region_name, roi_img in rois.items():
            region_dir = roi_dir / region_name
            region_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(region_dir / f"{frame_name}.jpg"), roi_img)

    # Baseline = frame[0] ROIs (grayscale)
    baseline_rois = rois_per_frame[0]
    if baseline_rois is None:
        raise ValueError(f"No face detected in baseline frame: {sample_id}")

    baseline_gray = {comp: _to_gray(baseline_rois[comp]) for comp in COMPONENTS}

    # Process each subsequent frame
    rows_flat: list[dict] = []
    rows_quad: list[dict] = []

    for idx in range(1, len(sample.frames)):
        frame_no = idx + 1
        rois = rois_per_frame[idx]
        if rois is None:
            continue

        flat_row: dict = {
            "emotion": emotion,
            "subject": subject,
            "clip": clip_name,
            "frame": frame_no,
        }
        quad_row: dict = flat_row.copy()

        all_ok = True
        for comp in COMPONENTS:
            gray = _to_gray(rois[comp])

            try:
                poc = POC(baseline_gray[comp], gray, config.block_size)
                vec = Vektor(poc.getPOC(), config.block_size)
                quad = Quadran(vec.getVektor()).getQuadran()
            except Exception:
                all_ok = False
                break

            # Flatten features: per-block x, y, theta, magnitude
            for b_id, qd in enumerate(quad, start=1):
                flat_row[f"{comp}_x{b_id}"] = qd[1]
                flat_row[f"{comp}_y{b_id}"] = qd[2]
                flat_row[f"{comp}_t{b_id}"] = qd[3]
                flat_row[f"{comp}_m{b_id}"] = qd[4]
            # 4QMV features: sum per quadrant with cos/sin decomposition for theta
            for q in QUADRANS:
                q_blocks = [b for b in quad if b[5] == q]
                quad_row[f"{comp}_{q}_x"] = sum(b[1] for b in q_blocks)
                quad_row[f"{comp}_{q}_y"] = sum(b[2] for b in q_blocks)
                quad_row[f"{comp}_{q}_m"] = sum(b[4] for b in q_blocks)

                # Circular data: use cos/sin decomposition instead of raw theta sum
                sum_cos_t = sum(np.cos(b[3]) for b in q_blocks)
                sum_sin_t = sum(np.sin(b[3]) for b in q_blocks)
                quad_row[f"{comp}_{q}_cos_t"] = sum_cos_t
                quad_row[f"{comp}_{q}_sin_t"] = sum_sin_t

                # Reconstruct mean angle from cos/sin components
                mean_theta = np.arctan2(sum_sin_t, sum_cos_t)
                quad_row[f"{comp}_{q}_mean_theta"] = mean_theta
                if config.include_quadrant_counts:
                    quad_row[f"{comp}_{q}_count"] = len(q_blocks)

        if not all_ok:
            continue

        flat_row["label"] = emotion
        quad_row["label"] = emotion
        rows_flat.append(flat_row)
        rows_quad.append(quad_row)

    return rows_flat, rows_quad



def run_dataset(
    clips: list[ClipInfo],
    config: ComparisonConfig,
) -> tuple[int, list[tuple[str, str]]]:
    """Process all clips sequentially. Saves 2 combined Excel files."""
    extractor = ROIExtractor(config)
    all_flat: list[dict] = []
    all_quad: list[dict] = []
    processed = 0
    errors: list[tuple[str, str]] = []

    for clip in clips:
        try:
            rows_flat, rows_quad = process_clip(clip, extractor, config)
            all_flat.extend(rows_flat)
            all_quad.extend(rows_quad)
            processed += 1
        except Exception as exc:
            errors.append((clip.sample_id, str(exc)))

    _save_combined(all_flat, all_quad, config)
    return processed, errors



def _worker_init(config_dict: dict) -> None:
    """Initialise per-worker globals (detector + predictor are not picklable)."""
    import comparasion.core.pipeline as _mod
    _mod._worker_config = ComparisonConfig(**config_dict)
    _mod._worker_extractor = ROIExtractor(_mod._worker_config)


def _worker_fn(clip: ClipInfo) -> tuple[list[dict], list[dict], str | None]:
    """Process one clip inside a worker. Returns (flat_rows, quad_rows, error_or_None)."""
    import comparasion.core.pipeline as _mod
    try:
        rows_flat, rows_quad = process_clip(clip, _mod._worker_extractor, _mod._worker_config)
        return rows_flat, rows_quad, None
    except Exception as exc:
        return [], [], f"{clip.sample_id}: {exc}"


def run_dataset_parallel(
    clips: list[ClipInfo],
    config: ComparisonConfig,
    n_proc: int | None = None,
) -> tuple[int, list[tuple[str, str]]]:
    """Process all clips in parallel. Saves 2 combined Excel files."""
    n_proc = n_proc or max(1, cpu_count() - 2)

    config_dict = {
        "predictor_path": config.predictor_path,
        "output_root": config.output_root,
        "regions": dict(config.regions),
        "target_size": dict(config.target_size),
        "padding_x": config.padding_x,
        "padding_y": config.padding_y,
        "block_size": config.block_size,
        "feature_method": config.feature_method,
    }

    all_flat: list[dict] = []
    all_quad: list[dict] = []
    processed = 0
    errors: list[tuple[str, str]] = []

    with Pool(n_proc, initializer=_worker_init, initargs=(config_dict,)) as pool:
        for rows_flat, rows_quad, err in pool.imap_unordered(_worker_fn, clips):
            if err is not None:
                sid, _, msg = err.partition(": ")
                errors.append((sid, msg))
            else:
                all_flat.extend(rows_flat)
                all_quad.extend(rows_quad)
                processed += 1

    _save_combined(all_flat, all_quad, config)
    return processed, errors



def _save_combined(
    all_flat: list[dict],
    all_quad: list[dict],
    config: ComparisonConfig,
) -> None:
    """Save all rows into 2 combined Excel files at output root."""
    config.feature_output_dir.mkdir(parents=True, exist_ok=True)

    if all_flat:
        pd.DataFrame(all_flat).to_excel(
            config.feature_output_dir / "poc_abs_flatten.xlsx", index=False
        )
    if all_quad:
        pd.DataFrame(all_quad).to_excel(
            config.feature_output_dir / "poc_abs_quadran_sum.xlsx", index=False
        )
