"""Helper CAS(ME)^2: ground truth, resolusi path, IoU, dan metrik F1 spotting.

Dipakai bersama oleh keempat notebook di folder ini supaya definisi ground truth
dan aturan matching persis sama di semua tahap.

Catatan penomoran frame
-----------------------
Ground truth CAS(ME)^2 memakai NOMOR IMG (`img001.jpg` -> 1), sedangkan hasil
spotting bekerja pada INDEKS LIST. Seluruh modul ini hanya memakai nomor img.
Konversinya dilakukan satu kali lewat `signal_index_to_img_number()`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GT_COLUMNS = [
    "subj_code",
    "clip_name",
    "onset",
    "apex",
    "offset",
    "au",
    "est_emotion",
    "expr_type",
    "self_emotion",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Landmark rigid untuk face alignment: batang hidung, dasar hidung, sudut mata.
# Mulut (48-67) dan alis (17-26) SENGAJA tidak dipakai -- justru bagian itu yang
# bergerak saat ekspresi; kalau ikut dipakai, gerakannya ikut ternetralkan.
RIGID_LANDMARKS = [27, 28, 29, 30, 31, 33, 35, 36, 39, 42, 45]

REGIONS = {
    "mulut": list(range(48, 68)),
    "mata_kiri": list(range(17, 22)) + list(range(36, 42)),
    "mata_kanan": list(range(22, 27)) + list(range(42, 48)),
}

TARGET_SIZE = {
    "mulut": (70, 35),
    "mata_kiri": (48, 32),
    "mata_kanan": (48, 32),
}

EMOTION_CLASSES = ["negative", "others", "positive"]
LABEL_MAP = {name: i for i, name in enumerate(EMOTION_CLASSES)}


@dataclass(slots=True)
class Casme2Config:
    """Konfigurasi CAS(ME)^2. Parameter spotting sengaja identik dengan
    `preprocess-anxiety/spotting.py` agar algoritma tidak berubah sama sekali."""

    project_root: Path
    dataset_root: Path = None
    output_root: Path = None
    predictor_path: Path = None

    # --- parameter spotting: JANGAN diubah, harus sama dengan pipeline anxiety ---
    block_size: int = 7
    padding_x: int = 6
    padding_y: int = 8
    min_phase_duration: int = 2
    top_k_apex: int = 0
    distance_threshold: int = 1
    prominence_threshold: float = 0.005
    cutoff_ratio: float = 0.3
    alpha: float = 1.0  # height = mean + alpha * std

    regions: dict[str, list[int]] = field(default_factory=lambda: dict(REGIONS))
    target_size: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(TARGET_SIZE))

    iou_threshold: float = 0.5
    overwrite: bool = False

    # Face alignment: setiap frame diregistrasi ke landmark referensi (similarity
    # transform) lalu ROI dipotong dari kotak referensi yang TETAP. Tanpa ini,
    # kotak ROI dihitung ulang tiap frame sehingga jitter landmark terbaca
    # sebagai gerakan. Bandingkan RMES 2023 Tabel V.
    # False = perilaku lama, dipakai sebagai baseline ablation.
    face_alignment: bool = True

    def __post_init__(self) -> None:
        root = Path(self.project_root)
        if self.dataset_root is None:
            self.dataset_root = root / "dataset" / "CAS-ME2"
        if self.output_root is None:
            self.output_root = root / "output" / "casme2"
        if self.predictor_path is None:
            self.predictor_path = root / "preprocess-anxiety" / "models" / "shape_predictor_68_face_landmarks.dat"

    # --- path turunan ---
    def _variant(self, name: str) -> str:
        """Hasil dengan dan tanpa face alignment disimpan terpisah.

        Nama lama (tanpa sufiks) tetap milik varian tanpa alignment, sehingga
        run yang sudah terlanjur jalan tidak perlu dihitung ulang dan bisa
        langsung dipakai sebagai baseline ablation.
        """
        return f"{name}_aligned" if self.face_alignment else name

    @property
    def rawpic_root(self) -> Path:
        return self.dataset_root / "rawpic"

    @property
    def selectedpic_root(self) -> Path:
        return self.dataset_root / "selectedpic"

    @property
    def coding_path(self) -> Path:
        return self.dataset_root / "CAS%28ME%29%5E2code_final.xlsx"

    @property
    def signal_root(self) -> Path:
        return self.output_root / self._variant("signals")

    @property
    def clip_root(self) -> Path:
        return self.output_root / self._variant("dataset")

    @property
    def feature_root(self) -> Path:
        return self.output_root / self._variant("features")

    @property
    def eval_root(self) -> Path:
        return self.output_root / self._variant("eval")

    @property
    def spotting_metadata_path(self) -> Path:
        return self.output_root / f'{self._variant("metadata_spotting")}.xlsx'

    @property
    def ground_truth_path(self) -> Path:
        return self.output_root / "ground_truth_micro.xlsx"


def natural_sort_key(value: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(value))]


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def signal_index_to_img_number(signal_index: int) -> int:
    """Indeks sinyal -> nomor img.

    `magnitudes[i]` dihitung dari pasangan (frames[i], frames[i+1]).
    `frames[j]` adalah `img{j+1}.jpg`, jadi nomor img = i + 2.
    """
    return int(signal_index) + 2


def map_emotion(self_emotion: str) -> str:
    """Rule pemetaan emosi (ditentukan pengguna).

    happiness                              -> positive
    disgust / fear / sadness / repression  -> negative
    selebihnya (surprise, anger, pain, ...) -> others
    """
    key = str(self_emotion).strip().lower()
    if key == "happiness":
        return "positive"
    if key in {"disgust", "fear", "sadness", "repression"}:
        return "negative"
    return "others"


def _subject_lookup(coding_path: Path) -> dict[int, str]:
    """naming rule1: kode subjek (1..22) -> nama folder (s15..s40)."""
    df = pd.read_excel(coding_path, sheet_name="naming rule1", header=None)
    df.columns = ["subject_no", "folder", "subj_code"]
    return {int(r.subj_code): str(r.folder) for r in df.itertuples(index=False)}


def _video_lookup(coding_path: Path) -> dict[str, str]:
    """naming rule2: prefix klip (`disgust2`) -> kode video (`0102`)."""
    df = pd.read_excel(coding_path, sheet_name="naming rule2", header=None)
    return {str(name): str(code).zfill(4) for code, name in zip(df[0], df[1])}


def clip_prefix(clip_name: str) -> str:
    """`disgust2_3` -> `disgust2`."""
    return re.sub(r"_\d+$", "", str(clip_name))


def resolve_video_dir(rawpic_root: Path, folder: str, video_code: str) -> Path | None:
    """`s15` + `0102` -> `rawpic/s15/15_0102eatingworms`."""
    subject_dir = rawpic_root / folder
    if not subject_dir.exists():
        return None
    stem = f"{folder[1:]}_{video_code}"
    for child in sorted(subject_dir.iterdir(), key=lambda p: natural_sort_key(p.name)):
        if child.is_dir() and child.name.startswith(stem):
            return child
    return None


def load_ground_truth(config: Casme2Config, micro_only: bool = True) -> pd.DataFrame:
    """Ground truth CAS(ME)^2 lengkap dengan path rawpic dan label emosi.

    File coding tidak punya baris header, jadi kolomnya di-assign manual.
    """
    df = pd.read_excel(config.coding_path, sheet_name="CASFEcode_final", header=None)
    df.columns = GT_COLUMNS

    if micro_only:
        df = df[df["expr_type"] == "micro-expression"].copy()
    df = df.reset_index(drop=True)

    subjects = _subject_lookup(config.coding_path)
    videos = _video_lookup(config.coding_path)

    rows = []
    for row in df.itertuples(index=False):
        folder = subjects.get(int(row.subj_code))
        prefix = clip_prefix(row.clip_name)
        video_code = videos.get(prefix)
        video_dir = resolve_video_dir(config.rawpic_root, folder, video_code) if (folder and video_code) else None
        selected_dir = config.selectedpic_root / folder / str(row.clip_name) if folder else None

        rows.append({
            "subject": folder,
            "subj_code": int(row.subj_code),
            "video_code": video_code,
            "video": video_dir.name if video_dir is not None else None,
            "video_dir": str(video_dir) if video_dir is not None else None,
            "video_key": f"{folder}/{video_dir.name}" if video_dir is not None else None,
            "clip_name": str(row.clip_name),
            "selected_dir": str(selected_dir) if selected_dir is not None else None,
            "selected_exists": bool(selected_dir.exists()) if selected_dir is not None else False,
            "onset": int(row.onset),
            "apex": int(row.apex),
            "offset": int(row.offset),
            "duration": int(row.offset) - int(row.onset) + 1 if int(row.offset) > 0 else None,
            "au": row.au,
            "expr_type": row.expr_type,
            "est_emotion": row.est_emotion,
            "self_emotion": row.self_emotion,
            "emotion": map_emotion(row.self_emotion),
        })

    return pd.DataFrame(rows)


def micro_video_keys(gt: pd.DataFrame) -> list[str]:
    """Daftar `subject/video` yang mengandung minimal satu micro-expression."""
    return sorted(gt["video_key"].dropna().unique().tolist(), key=natural_sort_key)


def load_frame_paths(video_dir: Path) -> list[Path]:
    """Frame satu video, diurutkan natural (`img999` sebelum `img1000`)."""
    return sorted(
        [p for p in Path(video_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: natural_sort_key(p.name),
    )


# ----------------------------------------------------------------------------
# Face alignment
# ----------------------------------------------------------------------------

class FaceAligner:
    """Deteksi landmark, registrasi ke frame referensi, lalu potong ROI.

    Tanpa alignment, kotak ROI dihitung ulang dari landmark setiap frame. Landmark
    dlib bergetar ~1 px dan ukuran kotaknya berayun sampai 12 px; karena tiap ROI
    di-`resize` ke ukuran tetap, peregangan itu terbaca POC sebagai gerakan. Efeknya
    magnitude punya noise floor tinggi dan tidak pernah mendekati nol.

    Dengan alignment, satu frame dijadikan referensi. Frame lain diputar/digeser/
    diskalakan agar landmark rigid-nya berimpit dengan referensi, lalu ROI dipotong
    dari kotak referensi yang tetap untuk seluruh urutan.

    Pemakaian::

        aligner = FaceAligner(config)
        aligner.reset()                      # per video (spotting) atau per klip (fitur)
        rois = aligner.rois(gray_or_bgr)     # frame valid pertama jadi referensi
    """

    def __init__(self, config: "Casme2Config") -> None:
        import dlib

        self.config = config
        self.enabled = config.face_alignment
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(config.predictor_path))
        self.reset()

    def reset(self) -> None:
        """Lupakan referensi. Wajib dipanggil saat pindah video/klip."""
        self._ref_rigid = None
        self._ref_boxes = None

    @property
    def has_reference(self) -> bool:
        return self._ref_rigid is not None

    def landmarks(self, gray: np.ndarray):
        faces = self._detector(gray)
        if not len(faces):
            return None
        face = max(faces, key=lambda r: r.width() * r.height())
        return self._predictor(gray, face)

    @staticmethod
    def _points(landmarks, indices: list[int]) -> np.ndarray:
        return np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in indices],
                        dtype=np.float32)

    def _box(self, points: np.ndarray, shape) -> tuple[int, int, int, int]:
        xs, ys = points[:, 0], points[:, 1]
        return (
            int(max(0, xs.min() - self.config.padding_x)),
            int(max(0, ys.min() - self.config.padding_y)),
            int(min(shape[1], xs.max() + self.config.padding_x)),
            int(min(shape[0], ys.max() + self.config.padding_y)),
        )

    def _crop(self, image: np.ndarray, boxes: dict) -> dict[str, np.ndarray] | None:
        import cv2

        out = {}
        for name, box in boxes.items():
            left, top, right, bottom = box
            roi = image[top:bottom, left:right]
            if roi.size == 0:
                return None
            out[name] = cv2.resize(roi, tuple(self.config.target_size[name]))
        return out

    def rois(self, image: np.ndarray) -> dict[str, np.ndarray] | None:
        """ROI ter-align satu frame. None kalau wajah tidak terdeteksi."""
        import cv2

        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmarks = self.landmarks(gray)
        if landmarks is None:
            return None

        boxes = {name: self._box(self._points(landmarks, indices), gray.shape)
                 for name, indices in self.config.regions.items()}

        if not self.enabled:
            # Perilaku lama: kotak dihitung ulang tiap frame, tanpa registrasi.
            return self._crop(gray, boxes)

        if self._ref_rigid is None:
            # Frame valid pertama jadi referensi untuk seluruh urutan.
            self._ref_rigid = self._points(landmarks, RIGID_LANDMARKS)
            self._ref_boxes = boxes
            return self._crop(gray, self._ref_boxes)

        matrix, _ = cv2.estimateAffinePartial2D(
            self._points(landmarks, RIGID_LANDMARKS), self._ref_rigid, method=cv2.LMEDS
        )
        if matrix is None:
            return None
        warped = cv2.warpAffine(gray, matrix, (gray.shape[1], gray.shape[0]),
                                flags=cv2.INTER_LINEAR)
        return self._crop(warped, self._ref_boxes)


def roi_magnitude(prev_rois: dict, curr_rois: dict, block_size: int) -> float:
    """Rata-rata magnitude POC antar dua himpunan ROI."""
    from features_extraction.poc import POC
    from features_extraction.quadran import Quadran
    from features_extraction.vektor import Vektor

    values = []
    for name in prev_rois:
        quad = Quadran(Vektor(POC(prev_rois[name], curr_rois[name], block_size).getPOC(),
                              block_size).getVektor()).getQuadran()
        values.append(float(np.mean([float(b[4]) for b in quad])) if len(quad) else 0.0)
    return float(np.mean(values)) if values else 0.0


# ----------------------------------------------------------------------------
# IoU dan metrik F1 spotting (protokol MEGC, dipakai di paper RMES 2023)
# ----------------------------------------------------------------------------

def compute_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """IoU dua interval frame, batas inklusif di kedua ujung."""
    a0, a1 = int(a[0]), int(a[1])
    b0, b1 = int(b[0]), int(b[1])
    intersection = max(0, min(a1, b1) - max(a0, b0) + 1)
    if intersection == 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0) + 1
    return intersection / union


def match_intervals(
    predicted: list[tuple[int, int]],
    ground_truth: list[tuple[int, int]],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Matching greedy 1-ke-1 pada IoU menurun.

    Satu interval GT hanya boleh diklaim oleh satu prediksi, dan sebaliknya.
    Mengembalikan (pasangan, indeks prediksi terpakai, indeks GT terpakai).
    """
    candidates = sorted(
        (
            (compute_iou(p, g), pi, gi)
            for pi, p in enumerate(predicted)
            for gi, g in enumerate(ground_truth)
        ),
        key=lambda x: (-x[0], x[1], x[2]),
    )

    pairs: list[tuple[int, int, float]] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for iou_value, pi, gi in candidates:
        if iou_value <= iou_threshold:
            break
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        pairs.append((pi, gi, iou_value))
    return pairs, used_pred, used_gt


def spotting_f1(n_true_positive: int, n_ground_truth: int, n_predicted: int) -> dict[str, float]:
    """Metrik F1 spotting MEGC.

    R  = sum(a_i) / sum(X_i)   X_i = jumlah GT micro di video i
    P  = sum(a_i) / sum(Y_i)   Y_i = jumlah interval hasil spotting di video i
    F1 = 2PR / (P + R)
    """
    recall = n_true_positive / n_ground_truth if n_ground_truth else 0.0
    precision = n_true_positive / n_predicted if n_predicted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "TP": int(n_true_positive),
        "FP": int(n_predicted - n_true_positive),
        "FN": int(n_ground_truth - n_true_positive),
        "n_ground_truth": int(n_ground_truth),
        "n_predicted": int(n_predicted),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# F1 spotting micro-expression CAS(ME)^2 dari literatur (paper RMES, Tabel II)
SOTA_F1_CASME2 = {
    "Duque et al., 2018": 0.0806,
    "Liong et al., 2021": 0.1173,
    "Wang et al., 2021 (MESNet)": 0.0360,
    "Yu et al., 2021 (LSSNet)": 0.0420,
    "Yang et al., 2021": 0.0153,
    "Liong et al., 2022": 0.0808,
    "Yap et al., 2022 (3D-CNN)": 0.0714,
    "Liong et al., 2023 (MTSN)": 0.1214,
    "RMES et al., 2023": 0.1489,
}
