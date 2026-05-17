from dataclasses import dataclass, field
from typing import Any
import cv2
import dlib
import numpy as np
from comparasion.core.config import ComparisonConfig


@dataclass
class ROIExtractor:
    config: ComparisonConfig
    _detector: Any = field(init=False, repr=False)
    _predictor: Any = field(init=False, repr=False)
    def __post_init__(self) -> None:
        if not self.config.predictor_path.exists():
            raise FileNotFoundError(f"Predictor tidak ditemukan: {self.config.predictor_path}")
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(self.config.predictor_path))

    def extract_rois(self, image: np.ndarray) -> dict[str, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._detector(gray)
        if len(faces) == 0:
            raise ValueError("No face detected")

        face = max(faces, key=lambda rect: rect.width() * rect.height())
        landmarks = self._predictor(gray, face)
        output: dict[str, np.ndarray] = {}
        for region_name, indices in self.config.regions.items():
            roi = self._extract_region(image=image, landmarks=landmarks, indices=indices)
            if roi.size == 0:
                raise ValueError(f"Empty ROI for region: {region_name}")
            output[region_name] = cv2.resize(roi, self.config.target_size[region_name])
        return output

    def _extract_region(self, image: np.ndarray, landmarks, indices: list[int]) -> np.ndarray:
        pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]
        xs, ys = zip(*pts)

        left = max(0, min(xs) - self.config.padding_x)
        top = max(0, min(ys) - self.config.padding_y)
        right = min(image.shape[1], max(xs) + self.config.padding_x)
        bottom = min(image.shape[0], max(ys) + self.config.padding_y)

        return image[top:bottom, left:right]
