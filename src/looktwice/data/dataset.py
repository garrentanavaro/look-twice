from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def build_transforms(train: bool, size: int, mean, std, aug: dict | None = None) -> A.Compose:
    ops: list = [A.Resize(size, size, interpolation=cv2.INTER_AREA)]
    if train and aug and aug.get("enabled", True):
        q_lo, q_hi = aug.get("jpeg_quality", [40, 80])
        ops += [
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_range=(q_lo, q_hi), p=aug.get("jpeg_p", 0.5)),
            A.GaussianBlur(blur_limit=tuple(aug.get("blur_limit", [3, 7])), p=aug.get("blur_p", 0.2)),
            A.RandomBrightnessContrast(brightness_limit=aug.get("brightness", 0.2),
                                       contrast_limit=aug.get("contrast", 0.2), p=aug.get("bc_p", 0.5)),
        ]
    ops += [A.Normalize(mean=mean, std=std), ToTensorV2()]
    return A.Compose(ops)


def build_eval_degradation(kind: str, level) -> A.BasicTransform | None:
    if kind == "none":
        return None
    if kind == "jpeg":
        return A.ImageCompression(quality_range=(int(level), int(level)), p=1.0)
    if kind == "blur":
        k = int(level) | 1
        return A.GaussianBlur(blur_limit=(k, k), p=1.0)
    if kind == "downscale":
        return A.Downscale(scale_range=(float(level), float(level)), p=1.0)
    raise ValueError(f"unknown degradation: {kind}")


class FaceCropDataset(Dataset):
    def __init__(self, index: pd.DataFrame, root: str | Path, transform: A.Compose,
                 degradation: A.BasicTransform | None = None):
        self.df = index.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.degradation = degradation

    @classmethod
    def from_index(cls, index_csv: str | Path, split: str, transform, root: str | Path | None = None,
                   max_items: int | None = None, seed: int = 0, **kw):
        df = pd.read_csv(index_csv, dtype={"target": str, "donor": str})
        df = df[df.split == split]
        if max_items and len(df) > max_items:
            df = pd.concat([g.sample(min(len(g), max_items // 2), random_state=seed)
                            for _, g in df.groupby("label")])
        return cls(df, root or Path(index_csv).parent, transform, **kw)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> np.ndarray:
        return self.df.label.to_numpy()

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = cv2.imread(str(self.root / row.path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.root / row.path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.degradation is not None:
            img = self.degradation(image=img)["image"]
        x = self.transform(image=img)["image"]
        return x, torch.tensor(float(row.label)), i
