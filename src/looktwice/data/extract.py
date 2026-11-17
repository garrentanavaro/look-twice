from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .faces import build_detector, pick_main_face, square_crop
from .metadata import LABEL_NAMES, REAL


@dataclass
class ExtractConfig:
    frames_real: int = 18
    frames_fake: int = 2
    edge_margin: float = 0.05
    retries: int = 3
    retry_step: int = 6
    bbox_scale: float = 1.2
    save_size: int = 256
    jpeg_quality: int = 95
    min_score: float = 0.7
    detector: str = "yunet"
    detector_path: str = "weights/face_detection_yunet_2023mar.onnx"


def sample_indices(n_frames: int, k: int, margin: float = 0.05) -> list[int]:
    if n_frames <= 0 or k <= 0:
        return []
    if k >= n_frames:
        return list(range(n_frames))
    lo, hi = int(n_frames * margin), max(int(n_frames * (1 - margin)) - 1, 0)
    if hi <= lo:
        lo, hi = 0, n_frames - 1
    idx = np.linspace(lo, hi, num=k + 2)[1:-1] if k < (hi - lo + 1) else np.arange(lo, hi + 1)
    return sorted(set(int(round(i)) for i in idx))


def _read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


_DETECTOR = None


def _init_worker(cfg: ExtractConfig):
    global _DETECTOR
    kwargs = {"model_path": cfg.detector_path, "score_threshold": cfg.min_score} if cfg.detector == "yunet" else {}
    _DETECTOR = build_detector(cfg.detector, **kwargs)


def process_video(job: tuple[dict, str, str, ExtractConfig], detector=None) -> tuple[list[dict], list[dict]]:
    row, video_root, out_dir, cfg = job
    det = detector or _DETECTOR
    label_name = LABEL_NAMES[row["label"]]
    k = cfg.frames_real if row["label"] == REAL else cfg.frames_fake
    save_dir = Path(out_dir) / "crops" / row["split"] / label_name
    save_dir.mkdir(parents=True, exist_ok=True)

    saved, failed = [], []
    cap = cv2.VideoCapture(str(Path(video_root) / row["path"]))
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or n_frames <= 0:
            return [], [{"video_id": row["video_id"], "frame": -1, "reason": "cannot_open"}]
        used: set[int] = set()
        for idx in sample_indices(n_frames, k, cfg.edge_margin):
            face, frame, fidx = None, None, idx
            for attempt in range(cfg.retries + 1):
                fidx = min(idx + attempt * cfg.retry_step, n_frames - 1)
                if fidx in used:
                    continue
                frame = _read_frame(cap, fidx)
                if frame is None:
                    continue
                face = pick_main_face(det.detect(frame))
                if face is not None:
                    break
            if face is None:
                failed.append({"video_id": row["video_id"], "frame": idx, "reason": "no_face"})
                continue
            used.add(fidx)
            crop = square_crop(frame, face, cfg.bbox_scale, cfg.save_size)
            fname = f"{row['video_id']}_f{fidx:05d}.jpg"
            cv2.imwrite(str(save_dir / fname), crop, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
            saved.append({
                "path": f"crops/{row['split']}/{label_name}/{fname}",
                "label": row["label"],
                "split": row["split"],
                "video_id": row["video_id"],
                "target": row["target"],
                "donor": row["donor"],
                "scene": row["scene"],
                "frame": fidx,
                "face_score": round(face.score, 4),
                "face_w": round(face.w, 1),
                "frame_w": frame.shape[1],
            })
    finally:
        cap.release()
    return saved, failed


def run_extract(videos: pd.DataFrame, video_root: str | Path, out_dir: str | Path,
                cfg: ExtractConfig, workers: int = 4, limit: int | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = videos[videos.split.isin(["train", "val", "test"])]
    if limit:
        todo = todo.groupby("label", group_keys=False).head(limit)
    jobs = [(r, str(video_root), str(out_dir), cfg) for r in todo.to_dict("records")]

    _init_worker(cfg)

    saved, failed = [], []
    if workers <= 1:
        results = map(process_video, jobs)
        for s, f in tqdm(results, total=len(jobs), desc="videos"):
            saved += s
            failed += f
    else:
        with Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for s, f in tqdm(pool.imap_unordered(process_video, jobs, chunksize=4),
                             total=len(jobs), desc="videos"):
                saved += s
                failed += f

    index = pd.DataFrame(saved).sort_values("path").reset_index(drop=True) if saved else pd.DataFrame()
    index.to_csv(out_dir / "crops_index.csv", index=False)
    pd.DataFrame(failed, columns=["video_id", "frame", "reason"]).to_csv(out_dir / "failures.csv", index=False)
    return index
