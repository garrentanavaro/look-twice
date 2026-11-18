from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
DEFAULT_YUNET_PATH = Path("weights/face_detection_yunet_2023mar.onnx")


@dataclass
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float

    @property
    def area(self) -> float:
        return self.w * self.h


def ensure_yunet(path: str | Path = DEFAULT_YUNET_PATH) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading YuNet -> {path}")
        urllib.request.urlretrieve(YUNET_URL, path)
    return path


class YuNetDetector:
    def __init__(self, model_path: str | Path = DEFAULT_YUNET_PATH, score_threshold: float = 0.7,
                 nms_threshold: float = 0.3, max_side: int = 640):
        self.max_side = max_side
        self.net = cv2.FaceDetectorYN.create(
            str(ensure_yunet(model_path)), "", (320, 320), score_threshold, nms_threshold, 5000
        )

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        h, w = img_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(img_bgr, (round(w * scale), round(h * scale))) if scale < 1 else img_bgr
        self.net.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.net.detect(small)
        if faces is None:
            return []
        return [Detection(float(f[0]) / scale, float(f[1]) / scale, float(f[2]) / scale, float(f[3]) / scale,
                          float(f[14]))
                for f in faces]


def build_detector(name: str = "yunet", **kwargs):
    if name == "yunet":
        return YuNetDetector(**kwargs)
    if ":" in name:
        import importlib
        module, cls = name.split(":", 1)
        return getattr(importlib.import_module(module), cls)(**kwargs)
    raise ValueError(f"unknown detector: {name}")


def pick_main_face(dets: list[Detection]) -> Detection | None:
    return max(dets, key=lambda d: d.area, default=None)


def square_crop(img: np.ndarray, det: Detection, scale: float = 1.2, out_size: int = 256) -> np.ndarray:
    cx, cy = det.x + det.w / 2, det.y + det.h / 2
    side = max(det.w, det.h) * scale
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + int(round(side)), y0 + int(round(side))

    h, w = img.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
