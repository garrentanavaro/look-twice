from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from looktwice.data.dataset import FaceCropDataset, build_eval_degradation, build_transforms
from looktwice.gradcam import denormalize, gradcam, overlay
from looktwice.metrics import binary_metrics, video_level
from looktwice.models import count_params, load_checkpoint
from looktwice.plotting import image_grid, plot_confusion, plot_history, plot_robustness, plot_roc
from looktwice.thresholds import badge, choose_thresholds

DEGRADATIONS = [("jpeg", 90), ("jpeg", 70), ("jpeg", 50), ("jpeg", 30), ("jpeg", 15),
                ("blur", 3), ("blur", 7), ("blur", 11),
                ("downscale", 0.75), ("downscale", 0.5), ("downscale", 0.25)]
META_COLS = ["path", "video_id", "label", "target", "donor", "scene", "frame", "face_score", "face_w", "frame_w"]


@torch.no_grad()
def predict_probs(model, ds: FaceCropDataset, device, batch_size=64, num_workers=4) -> np.ndarray:
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=device.type == "cuda")
    probs = np.empty(len(ds), dtype=np.float32)
    for x, _, idx in dl:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x.to(device)).squeeze(1).float())
        probs[idx.numpy()] = p.cpu().numpy()
    return probs


def metrics_block(preds: pd.DataFrame, t: float) -> dict:
    vid = video_level(preds)
    return {
        "frame@0.5": binary_metrics(preds.label, preds.prob, 0.5),
        "frame@t": binary_metrics(preds.label, preds.prob, t),
        "video@0.5": binary_metrics(vid.label, vid.prob, 0.5),
        "video@t": binary_metrics(vid.label, vid.prob, t),
        "n_videos": int(len(vid)),
    }


def error_tables(preds: pd.DataFrame, t: float, out: Path) -> None:
    df = preds.assign(wrong=((preds.prob >= t).astype(int) != preds.label).astype(int),
                      error_type=np.where(preds.label == 1, "FN (missed fake)", "FP (false alarm)"))
    groups = {"scene": "scene", "target_actor": "target"}
    if "face_w" in df and df.face_w.notna().any():
        df["face_rel_size"] = pd.qcut(df.face_w / df.frame_w, 4, duplicates="drop").astype(str)
        groups["face_size"] = "face_rel_size"
    if "face_score" in df and df.face_score.notna().any():
        df["face_score_bin"] = pd.qcut(df.face_score, 4, duplicates="drop").astype(str)
        groups["detector_conf"] = "face_score_bin"
    for name, col in groups.items():
        t_ = (df.groupby([col, "label"]).wrong.agg(["mean", "size"]).unstack("label")
                .rename(columns={0: "real", 1: "fake"}))
        t_.columns = [f"{'error_rate' if a == 'mean' else 'n'}_{b}" for a, b in t_.columns]
        t_.sort_values(t_.columns[0], ascending=False).to_csv(out / f"errors_by_{name}.csv")


def error_montages(preds: pd.DataFrame, t: float, root: Path, out: Path, k: int = 16) -> None:
    import cv2
    for name, mask, asc in [("false_positives", (preds.label == 0) & (preds.prob >= t), False),
                            ("false_negatives", (preds.label == 1) & (preds.prob < t), True)]:
        worst = preds[mask].sort_values("prob", ascending=asc).head(k)
        imgs = [cv2.cvtColor(cv2.imread(str(root / p)), cv2.COLOR_BGR2RGB) for p in worst.path]
        caps = [f"p(fake)={p:.2f}\n{v[:18]}" for p, v in zip(worst.prob, worst.video_id)]
        title = {"false_positives": "Real faces flagged as fake (most confident first)",
                 "false_negatives": "Missed fakes (most confident first)"}[name]
        image_grid(imgs, caps, out / f"{name}.png", f"{title} — {mask.sum()} total")


def gradcam_figure(model, ds: FaceCropDataset, preds: pd.DataFrame, t: float, meta: dict,
                   device, out: Path, n: int) -> None:
    pred = (preds.prob >= t).astype(int)
    groups = [("TP fake", (preds.label == 1) & (pred == 1)), ("TN real", (preds.label == 0) & (pred == 0)),
              ("FP", (preds.label == 0) & (pred == 1)), ("FN", (preds.label == 1) & (pred == 0))]
    imgs, caps = [], []
    for name, mask in groups:
        rows = preds.index[mask][:n].tolist()
        if not rows:
            continue
        x = torch.stack([ds[i][0] for i in rows]).to(device)
        cams, probs = gradcam(model, x)
        for xi, cam, p in zip(x, cams, probs):
            imgs.append(overlay(denormalize(xi, meta["mean"], meta["std"]), cam))
            caps.append(f"{name}  p={p:.2f}")
    image_grid(imgs, caps, out / "gradcam.png", "Grad-CAM: regions driving the prediction", ncols=n)


def evaluate_run(run_dir: Path, index: str, split: str, out_root: Path, device, args) -> dict:
    out = out_root / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    model, meta = load_checkpoint(run_dir / "best.pt", device)
    size, mean, std = meta["input_size"], meta["mean"], meta["std"]

    val = pd.read_csv(run_dir / "val_preds.csv")
    thr = choose_thresholds(val.label, val.prob, args.target_acc)
    (out / "thresholds.json").write_text(json.dumps(thr, indent=2))
    t = thr["t"]

    tf = build_transforms(False, size, mean, std)
    ds = FaceCropDataset.from_index(index, split, tf)
    preds = ds.df[[c for c in META_COLS if c in ds.df.columns]].copy()
    preds["prob"] = predict_probs(model, ds, device, args.batch_size, args.num_workers)
    preds["badge"] = badge(preds.prob, thr["lo"], thr["hi"])
    preds.to_csv(out / f"{split}_preds.csv", index=False)

    res = metrics_block(preds, t)
    conf = preds.badge != "uncertain"
    correct = (preds.badge == "manipulated").astype(int) == preds.label
    res["badges"] = {**preds.badge.value_counts(normalize=True).round(4).to_dict(),
                     "confident_acc": float(correct[conf].mean()) if conf.any() else None}
    res["thresholds"] = thr
    n_params, _ = count_params(model)
    summary_path = run_dir / "summary.json"
    train_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    (out / f"{split}_metrics.json").write_text(json.dumps(res, indent=2, default=float))

    fm, vm = res["frame@t"], res["video@t"]
    plot_confusion(fm["tn"], fm["fp"], fm["fn"], fm["tp"], out / "confusion_frame.png",
                   f"{run_dir.name}: frames (t={t:.2f})")
    plot_confusion(vm["tn"], vm["fp"], vm["fn"], vm["tp"], out / "confusion_video.png",
                   f"{run_dir.name}: videos (t={t:.2f})")
    if (run_dir / "history.csv").exists():
        plot_history(pd.read_csv(run_dir / "history.csv"), run_dir.name, out / "history.png")

    if args.robustness:
        rows = []
        for kind, level in DEGRADATIONS:
            ds_d = FaceCropDataset.from_index(index, split, tf, degradation=build_eval_degradation(kind, level))
            p = predict_probs(model, ds_d, device, args.batch_size, args.num_workers)
            m = binary_metrics(preds.label, p, t)
            v = video_level(preds[["video_id", "label"]].assign(prob=p))
            rows.append({"kind": kind, "level": level, "auc": m["auc"], "f1": m["f1"], "recall": m["recall"],
                         "precision": m["precision"], "video_auc": binary_metrics(v.label, v.prob, t)["auc"]})
        rob = pd.DataFrame(rows)
        rob.to_csv(out / "robustness.csv", index=False)
        plot_robustness(rob, res["frame@t"], out / "robustness.png", f"{run_dir.name}: robustness ({split})")

    if args.errors:
        error_tables(preds, t, out)
        error_montages(preds, t, ds.root, out)
    if args.gradcam:
        gradcam_figure(model, ds, preds, t, meta, device, out, args.gradcam)

    fpr, tpr, _ = roc_curve(preds.label, preds.prob) if preds.label.nunique() == 2 else ([0, 1], [0, 1], None)
    return {
        "run": run_dir.name, "model": meta["model_name"], "params_M": round(n_params / 1e6, 2),
        "threshold": round(t, 4), "uncertain_band": f"[{thr['lo']:.2f}, {thr['hi']:.2f})",
        **{f"frame_{k}": fm[k] for k in ("acc", "precision", "recall", "f1", "auc")},
        **{f"video_{k}": vm[k] for k in ("acc", "precision", "recall", "f1", "auc")},
        "uncertain_frac": res["badges"].get("uncertain", 0.0), "confident_acc": res["badges"]["confident_acc"],
        "val_auc": train_summary.get("val_auc"), "sec_per_epoch": train_summary.get("sec_per_epoch"),
        "_roc": (np.asarray(fpr), np.asarray(tpr), fm["auc"]),
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="looktwice.evaluate")
    p.add_argument("--run", type=Path, action="append", required=True, help="run folder with best.pt")
    p.add_argument("--index", required=True, help="crops_index.csv")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--out", type=Path, default=Path("results/eval"))
    p.add_argument("--robustness", action="store_true")
    p.add_argument("--errors", action="store_true")
    p.add_argument("--gradcam", type=int, default=0, metavar="N", help="N examples per TP/TN/FP/FN")
    p.add_argument("--target-acc", type=float, default=0.95, help="accuracy required outside the Uncertain band")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    rows = []
    for run in args.run:
        print(f"evaluating {run.name} on {args.split} ...")
        r = evaluate_run(run, args.index, args.split, args.out, device, args)
        print(f"  frame AUC={r['frame_auc']:.4f} F1={r['frame_f1']:.4f} recall={r['frame_recall']:.4f} | "
              f"video AUC={r['video_auc']:.4f} F1={r['video_f1']:.4f} | uncertain={r['uncertain_frac']:.1%}")
        rows.append(r)

    for i in range(0, len(rows), 3):
        chunk = rows[i:i + 3]
        suffix = "" if len(rows) <= 3 else f"_{i // 3 + 1}"
        plot_roc({r["run"]: r["_roc"] for r in chunk}, args.out / f"roc_{args.split}{suffix}.png",
                 f"ROC — {args.split} set (frame level)")
    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    table.to_csv(args.out / f"comparison_{args.split}.csv", index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
