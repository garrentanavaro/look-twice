from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import Predictor, create_app
from looktwice.export_onnx import export
from looktwice.models import build_model
from test_data_pipeline import DummyDetector, NoFaceDetector


def make_tiny_onnx(tmp: Path) -> Path:
    run = tmp / "run"
    run.mkdir(parents=True)
    model = build_model("resnet18", pretrained=False)
    torch.save({"model": model.state_dict(), "model_name": "resnet18", "timm_name": "resnet18", "input_size": 64,
                "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225), "epoch": 1,
                "val_metrics": {"val_auc": 0.5}}, run / "best.pt")
    pd.DataFrame({"label": [0, 0, 1, 1], "prob": [0.1, 0.3, 0.7, 0.9]}).to_csv(run / "val_preds.csv", index=False)
    path, _ = export(run, tmp / "w" / "tiny.onnx")
    return path


def jpeg(w=320, h=240, seed=0) -> bytes:
    img = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.fixture(scope="module")
def onnx_path(tmp_path_factory):
    return make_tiny_onnx(tmp_path_factory.mktemp("backend"))


@pytest.fixture()
def client(onnx_path, tmp_path):
    clips = tmp_path / "clips"
    (clips / "A").mkdir(parents=True)
    for n in ("clip02", "clip01"):
        (clips / "A" / f"{n}.mp4").write_bytes(b"\x00")
    (clips / "A" / "notes.txt").write_text("not a video")
    app = create_app(Predictor(onnx_path, DummyDetector(), crop_size=96, detector_name="dummy"), clips)
    return TestClient(app)


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["input_size"] == 64 and h["detector"] == "dummy"
    assert h["thresholds"]["lo"] <= h["thresholds"]["t"] <= h["thresholds"]["hi"]


def test_predict_with_face(client):
    r = client.post("/predict", content=jpeg(), headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200
    res = r.json()
    assert res["face_found"] and 0.0 <= res["prob"] <= 1.0
    assert res["badge"] in {"authentic", "uncertain", "manipulated"}
    assert len(res["box"]) == 4 and all(0 <= v <= 1 for v in res["box"])
    assert res["frame_size"] == [320, 240] and res["latency_ms"]["total"] > 0


def test_predict_matches_direct_onnx(onnx_path):
    import onnxruntime as ort
    from looktwice.data.faces import square_crop
    p = Predictor(onnx_path, DummyDetector(), crop_size=96)
    data = jpeg(seed=3)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    crop = cv2.resize(square_crop(img, DummyDetector().detect(img)[0], 1.2, 96), (64, 64), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255
    ref = ort.InferenceSession(str(onnx_path)).run(["prob"], {"image": x})[0][0]
    assert p.predict(data)["prob"] == pytest.approx(float(ref), abs=1e-6)


def test_predict_no_face(onnx_path):
    c = TestClient(create_app(Predictor(onnx_path, NoFaceDetector()), Path("does-not-exist")))
    res = c.post("/predict", content=jpeg()).json()
    assert res["face_found"] is False and res["prob"] is None and res["badge"] == "no_face"


def test_predict_bad_input(client):
    assert client.post("/predict", content=b"").status_code == 400
    assert client.post("/predict", content=b"not a jpeg").status_code == 400


def test_stats(client):
    for i in range(5):
        client.post("/predict", content=jpeg(seed=i))
    s = client.get("/stats").json()
    assert s["n"] == 5 and s["total"]["p95"] >= s["total"]["p50"] > 0 and s["no_face_rate"] == 0
    client.post("/stats/reset")
    assert client.get("/stats").json()["n"] == 0


def test_clips_and_demo(client):
    c = client.get("/api/clips", params={"set": "A"}).json()["clips"]
    assert [x["name"] for x in c] == ["clip01", "clip02"] and c[0]["url"] == "/clips/A/clip01.mp4"
    assert client.get("/api/clips", params={"set": "../.."}).json()["clips"] == []
    assert client.get("/clips/A/clip01.mp4").status_code == 200
    assert client.get("/demo/").status_code == 200
