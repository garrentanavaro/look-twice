import json

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest
import torch

from looktwice import evaluate, export_onnx
from looktwice.gradcam import gradcam
from looktwice.models import build_model
from looktwice.thresholds import badge, best_f1_threshold, choose_thresholds, select_band
from looktwice.train import train
from test_training import make_crops, tiny_cfg


def test_best_f1_threshold_separable():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    t = best_f1_threshold(y, p)
    assert t == pytest.approx(0.5)


def test_band_widens_until_confident_predictions_are_accurate():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(500), np.ones(500)].astype(int)
    p = np.clip(np.r_[rng.normal(0.3, 0.15, 500), rng.normal(0.7, 0.15, 500)], 0, 1)
    narrow = select_band(y, p, 0.5, target_acc=0.80)
    wide = select_band(y, p, 0.5, target_acc=0.97)
    assert narrow["reached"] and wide["width"] > narrow["width"]
    assert wide["coverage"] < narrow["coverage"]
    assert wide["confident_acc"] >= 0.97 or not wide["reached"]


def test_badge():
    assert badge([0.1, 0.5, 0.9], 0.4, 0.6).tolist() == ["authentic", "uncertain", "manipulated"]
    assert badge([0.6], 0.4, 0.6).tolist() == ["manipulated"]


def test_choose_thresholds_single_class_is_safe():
    r = choose_thresholds([1, 1, 1], [0.2, 0.6, 0.9])
    assert r["t"] == 0.5


@pytest.mark.parametrize("name,frozen", [("resnet18", False), ("resnet18", True),
                                         ("xception", False), ("efficientnet_b4", False)])
def test_gradcam_shapes(name, frozen):
    m = build_model(name, pretrained=False, freeze_backbone=frozen)
    cams, probs = gradcam(m, torch.randn(2, 3, 96, 96))
    assert cams.shape == (2, 96, 96) and probs.shape == (2,)
    assert np.all((cams >= 0) & (cams <= 1))


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("evalrun")
    index = make_crops(tmp / "data", per_split=(48, 16, 16))
    df = pd.read_csv(index)
    rng = np.random.default_rng(1)
    df["scene"] = np.where(df.frame % 2, "outside", "kitchen")
    df["face_score"] = rng.uniform(0.7, 1.0, len(df))
    df["face_w"] = rng.uniform(40, 200, len(df))
    df["frame_w"] = 1920
    df.to_csv(index, index=False)
    train(tiny_cfg(index, tmp / "runs", **{"train.epochs": 3}))
    return tmp, index, tmp / "runs" / "tiny"


def test_evaluate_cli(tiny_run):
    tmp, index, run = tiny_run
    out = tmp / "eval"
    evaluate.main(["--run", str(run), "--index", str(index), "--out", str(out), "--robustness", "--errors",
                   "--gradcam", "2", "--num-workers", "0", "--batch-size", "8", "--device", "cpu"])
    r = out / "tiny"
    for f in ["thresholds.json", "test_preds.csv", "test_metrics.json", "confusion_frame.png", "confusion_video.png",
              "history.png", "robustness.csv", "robustness.png", "errors_by_scene.csv", "errors_by_face_size.csv",
              "errors_by_detector_conf.csv", "gradcam.png"]:
        assert (r / f).exists(), f
    assert (out / "roc_test.png").exists() and (out / "comparison_test.csv").exists()
    m = json.loads((r / "test_metrics.json").read_text())
    assert m["frame@t"]["n"] == 16 and m["n_videos"] == 8
    rob = pd.read_csv(r / "robustness.csv")
    assert len(rob) == len(evaluate.DEGRADATIONS) and rob.auc.between(0, 1).all()
    preds = pd.read_csv(r / "test_preds.csv")
    assert set(preds.badge) <= {"authentic", "uncertain", "manipulated"}


def test_export_verify_and_bench(tiny_run):
    tmp, _, run = tiny_run
    onnx_path, lat = tmp / "w" / "tiny.onnx", tmp / "latency.csv"
    export_onnx.main(["--run", str(run), "--out", str(onnx_path), "--bench", "--iters", "3",
                      "--latency-csv", str(lat)])
    info = json.loads(onnx_path.with_suffix(".json").read_text())
    assert info["input_size"] == 64 and info["thresholds"]["lo"] <= info["thresholds"]["hi"]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(["prob"], {"image": np.random.rand(5, 3, 64, 64).astype(np.float32)})[0]
    assert out.shape == (5,) and np.all((out >= 0) & (out <= 1))
    df = pd.read_csv(lat)
    assert {"pytorch", "onnxruntime"} <= set(df.engine) and (df.mean_ms > 0).all()
    export_onnx.main(["--run", str(run), "--out", str(onnx_path), "--bench", "--iters", "2",
                      "--latency-csv", str(lat)])
    assert len(pd.read_csv(lat)) == 2 * len(df)
