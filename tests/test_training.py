import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from looktwice.config import load_config, parse_override
from looktwice.data.dataset import FaceCropDataset, build_eval_degradation, build_transforms
from looktwice.metrics import binary_metrics, video_level
from looktwice.models import build_model, count_params
from looktwice.run_experiments import collect, expand_stage
from looktwice.train import SmoothedBCE, train

ROOT = Path(__file__).resolve().parents[1]


def make_crops(out: Path, per_split=(24, 8, 8), size=48, seed=0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for split, n in zip(("train", "val", "test"), per_split):
        for i in range(n):
            label = i % 2
            img = rng.integers(0, 80, (size, size, 3), dtype=np.uint8)
            img[..., 2 if label else 0] += 120
            rel = f"crops/{split}/{label}/v{i // 4}_{label}_{i}.jpg"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / rel), img)
            rows.append({"path": rel, "label": label, "split": split, "video_id": f"{split}_v{i // 4}_{label}",
                         "target": "01", "donor": "02" if label else None, "frame": i})
    pd.DataFrame(rows).to_csv(out / "crops_index.csv", index=False)
    return out / "crops_index.csv"


def tiny_cfg(index: Path, out_dir: Path, **over) -> dict:
    files = [ROOT / "configs/train/base.yaml", ROOT / "configs/train/recipe_B.yaml"]
    base = {"data.index": str(index), "data.num_workers": 0, "out_dir": str(out_dir), "device": "cpu",
            "model.name": "resnet18", "model.pretrained": False, "input_size": 64,
            "optim.batch_size": 8, "optim.lr": 1.0e-3, "train.epochs": 3, "run_name": "tiny"}
    return load_config(files, {**base, **over})


def test_config_merge_and_overrides():
    cfg = load_config([ROOT / "configs/train/base.yaml", ROOT / "configs/train/recipe_C.yaml"],
                      ["optim.lr=1e-2", "model.name=xception", "aug.enabled=false"])
    assert cfg["optim"]["name"] == "sgd" and cfg["optim"]["lr"] == 0.01
    assert cfg["model"]["name"] == "xception" and cfg["aug"]["enabled"] is False
    assert cfg["loss"]["label_smoothing"] == 0.05
    assert isinstance(cfg["optim"]["weight_decay"], float)
    assert parse_override("a.b=[1, 2]") == ("a.b", [1, 2])


def test_recipe_b_cosine_matches_epochs():
    cfg = load_config([ROOT / "configs/train/base.yaml", ROOT / "configs/train/recipe_B.yaml"])
    assert cfg["sched"]["t_max"] in (None, cfg["train"]["epochs"])


def test_expand_stages():
    import yaml
    plan = yaml.safe_load((ROOT / "configs/train/experiments.yaml").read_text())
    s1 = expand_stage(plan, "stage1")
    assert len(s1) == 9 and s1[0][0] == "s1_resnet50_A"
    with pytest.raises(SystemExit):
        expand_stage(plan, "stage2")
    plan["stages"]["stage2"].update(model="xception", recipe="B")
    s2 = expand_stage(plan, "stage2")
    assert {r[0] for r in s2} >= {"s2_xception_B_E1_no_aug", "s2_xception_B_E2_frozen"}
    plan["stages"]["stage3"].update(model="xception", recipe="B")
    assert [r[2]["seed"] for r in expand_stage(plan, "stage3")] == [1, 2]


def test_smoothed_bce():
    logits, y = torch.zeros(4), torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert torch.isclose(SmoothedBCE(0.0)(logits, y), torch.log(torch.tensor(2.0)))
    big = torch.tensor([-20.0, 20.0, -20.0, 20.0])
    assert SmoothedBCE(0.1)(big, y) > SmoothedBCE(0.0)(big, y)


def test_freeze_backbone_only_trains_head():
    m = build_model("resnet18", pretrained=False, freeze_backbone=True)
    total, trainable = count_params(m)
    assert trainable == 512 + 1 and total > 1_000_000


def test_transforms_and_dataset(tmp_path):
    index = make_crops(tmp_path)
    mean, std = (0.5,) * 3, (0.5,) * 3
    ds = FaceCropDataset.from_index(index, "train", build_transforms(True, 64, mean, std, {"enabled": True}))
    x, y, i = ds[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32 and y.item() in (0.0, 1.0)
    small = FaceCropDataset.from_index(index, "train", build_transforms(False, 64, mean, std), max_items=10)
    assert len(small) == 10 and sorted(small.df.label.value_counts().tolist()) == [5, 5]
    for kind, level in [("jpeg", 30), ("blur", 5), ("downscale", 0.5)]:
        ds_d = FaceCropDataset.from_index(index, "val", build_transforms(False, 64, mean, std),
                                          degradation=build_eval_degradation(kind, level))
        assert ds_d[0][0].shape == (3, 64, 64)


def test_metrics():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 0)
    assert m["recall"] == 1.0 and m["auc"] == 1.0
    assert np.isnan(binary_metrics([1, 1], [0.2, 0.9])["auc"])
    v = video_level(pd.DataFrame({"video_id": ["a", "a", "b"], "label": [1, 1, 0], "prob": [0.2, 0.8, 0.1]}))
    assert v.set_index("video_id").loc["a", "prob"] == pytest.approx(0.5)


@pytest.mark.parametrize("recipe_over", [
    {},
    {"optim.name": "sgd", "sched.name": "plateau", "train.monitor": "val_loss"},
    {"optim.name": "adam", "sched.name": "constant", "loss.pos_weight": "auto", "model.freeze_backbone": True},
])
def test_train_end_to_end(tmp_path, recipe_over):
    index = make_crops(tmp_path / "data")
    summary = train(tiny_cfg(index, tmp_path / "runs", **recipe_over))
    run = tmp_path / "runs" / "tiny"
    for f in ("config.yaml", "history.csv", "best.pt", "val_preds.csv", "summary.json"):
        assert (run / f).exists(), f
    hist = pd.read_csv(run / "history.csv")
    assert 1 <= len(hist) <= 3 and {"train_loss", "val_loss", "val_auc", "val_video_auc", "lr"} <= set(hist.columns)
    assert 1 <= summary["best_epoch"] <= len(hist)
    ckpt = torch.load(run / "best.pt", weights_only=False)
    assert ckpt["mean"] and ckpt["input_size"] == 64
    assert len(pd.read_csv(run / "val_preds.csv")) == 8
    assert json.loads((run / "summary.json").read_text())["run"] == "tiny"
    assert len(collect(tmp_path / "runs")) == 1


def test_model_learns_synthetic_signal(tmp_path):
    index = make_crops(tmp_path / "data", per_split=(64, 16, 16))
    summary = train(tiny_cfg(index, tmp_path / "runs", **{"train.epochs": 6, "loss.label_smoothing": 0.0}))
    assert summary["val_auc"] > 0.9
