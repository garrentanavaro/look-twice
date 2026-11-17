from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from looktwice.data.faces import build_detector, pick_main_face, square_crop
from looktwice.thresholds import badge as badge_for

REPO = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 2**20
VIDEO_EXTS = {".mp4", ".webm", ".ogg", ".mov"}


def _preprocess_cfg() -> dict:
    p = REPO / "configs/preprocess.yaml"
    ext = (yaml.safe_load(p.read_text()) or {}).get("extract", {}) if p.exists() else {}
    return {"bbox_scale": float(ext.get("bbox_scale", 1.2)), "crop_size": int(ext.get("save_size", 256)),
            "min_score": float(ext.get("min_score", 0.7))}


class Predictor:
    def __init__(self, model_path: str | Path, detector, bbox_scale: float = 1.2, crop_size: int = 256,
                 threads: int | None = None, detector_name: str = "custom"):
        model_path = Path(model_path)
        info_path = model_path.with_suffix(".json")
        self.info = json.loads(info_path.read_text()) if info_path.exists() else {}
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(model_path), so, providers=providers)
        self.size = int(self.info.get("input_size") or self.session.get_inputs()[0].shape[-1])
        thr = self.info.get("thresholds") or {}
        self.t, self.lo, self.hi = (float(thr.get(k, 0.5)) for k in ("t", "lo", "hi"))
        self.detector, self.detector_name = detector, detector_name
        self.bbox_scale, self.crop_size = bbox_scale, crop_size
        self._lock = threading.Lock()

    def predict(self, jpeg: bytes) -> dict:
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image")
        t1 = time.perf_counter()
        with self._lock:
            face = pick_main_face(self.detector.detect(img))
        t2 = time.perf_counter()
        out = {"face_found": face is not None, "prob": None, "badge": "no_face", "box": None,
               "thresholds": {"t": self.t, "lo": self.lo, "hi": self.hi}}
        if face is not None:
            crop = square_crop(img, face, self.bbox_scale, self.crop_size)
            crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
            x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            prob = float(self.session.run(["prob"], {"image": x})[0][0])
            h, w = img.shape[:2]
            out.update(prob=prob, badge=str(badge_for([prob], self.lo, self.hi)[0]),
                       box=[round(face.x / w, 4), round(face.y / h, 4), round(face.w / w, 4), round(face.h / h, 4)],
                       face_score=round(face.score, 3))
        t3 = time.perf_counter()
        out["latency_ms"] = {"decode": (t1 - t0) * 1e3, "detect": (t2 - t1) * 1e3,
                             "infer": (t3 - t2) * 1e3 if face is not None else 0.0, "total": (t3 - t0) * 1e3}
        out["frame_size"] = [int(img.shape[1]), int(img.shape[0])]
        return out


def predictor_from_env() -> Predictor:
    pc = _preprocess_cfg()
    name = os.environ.get("LOOKTWICE_DETECTOR", "yunet")
    kwargs = {"model_path": os.environ.get("LOOKTWICE_YUNET", str(REPO / "weights/face_detection_yunet_2023mar.onnx")),
              "score_threshold": pc["min_score"]} if name == "yunet" else {}
    threads = int(os.environ["LOOKTWICE_THREADS"]) if os.environ.get("LOOKTWICE_THREADS") else None
    model = Path(os.environ.get("LOOKTWICE_MODEL", REPO / "weights/looktwice.onnx"))
    if not model.exists():
        raise SystemExit(f"model not found: {model} (run looktwice.export_onnx first or set LOOKTWICE_MODEL)")
    return Predictor(model, build_detector(name, **kwargs), pc["bbox_scale"], pc["crop_size"], threads, name)


def _pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def create_app(predictor: Predictor | None = None, clips_dir: str | Path | None = None) -> FastAPI:
    predictor = predictor or predictor_from_env()
    clips_dir = Path(clips_dir or os.environ.get("LOOKTWICE_CLIPS", REPO / "demo/clips"))
    app = FastAPI(title="LookTwice", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
    history: deque = deque(maxlen=2000)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": predictor.info.get("model_name"), "source_run": predictor.info.get("source_run"),
                "input_size": predictor.size, "thresholds": {"t": predictor.t, "lo": predictor.lo, "hi": predictor.hi},
                "providers": predictor.session.get_providers(), "detector": predictor.detector_name}

    @app.post("/predict")
    async def predict(request: Request):
        body = await request.body()
        if not body:
            raise HTTPException(400, "empty body: POST the frame as image/jpeg bytes")
        if len(body) > MAX_BYTES:
            raise HTTPException(413, "frame too large")
        try:
            res = predictor.predict(body)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        history.append({**res["latency_ms"], "face": res["face_found"]})
        return res

    @app.get("/stats")
    def stats():
        h = list(history)
        out = {"n": len(h), "no_face_rate": (sum(not r["face"] for r in h) / len(h)) if h else None}
        for k in ("decode", "detect", "infer", "total"):
            vals = [r[k] for r in h if k != "infer" or r["face"]]
            out[k] = {"mean": float(np.mean(vals)) if vals else None, "p50": _pct(vals, 50), "p95": _pct(vals, 95)}
        return out

    @app.post("/stats/reset")
    def reset_stats():
        history.clear()
        return {"ok": True}

    @app.get("/api/clips")
    def clips(set: str = ""):
        folder = (clips_dir / set) if set else clips_dir
        if not folder.resolve().is_relative_to(clips_dir.resolve()) or not folder.is_dir():
            return {"set": set, "clips": []}
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        return {"set": set, "clips": [{"name": p.stem, "url": "/clips/" + p.relative_to(clips_dir).as_posix()}
                                      for p in files]}

    @app.get("/")
    def root():
        return RedirectResponse("/demo/")

    if clips_dir.is_dir():
        app.mount("/clips", StaticFiles(directory=clips_dir), name="clips")
    app.mount("/demo", StaticFiles(directory=REPO / "demo", html=True), name="demo")
    return app
