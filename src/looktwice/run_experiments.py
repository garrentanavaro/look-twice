from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd
import yaml

from looktwice.config import load_config, parse_override
from looktwice.train import train


def expand_stage(plan: dict, stage: str) -> list[tuple[str, list[str], dict]]:
    st = plan["stages"][stage]
    base, recipes = plan["base"], plan["recipes"]
    runs = []
    if stage == "stage1":
        for m in st["models"]:
            for r in st["recipes"]:
                runs.append((f"s1_{m}_{r}", [*base, recipes[r]], {"model.name": m}))
    elif stage == "stage2":
        _require(st, stage)
        for key, ov in st["runs"].items():
            runs.append((f"s2_{st['model']}_{st['recipe']}_{key}", [*base, recipes[st["recipe"]]],
                         {"model.name": st["model"], **ov}))
    elif stage == "stage3":
        _require(st, stage)
        for s in st["seeds"]:
            runs.append((f"s3_{st['model']}_{st['recipe']}_seed{s}", [*base, recipes[st["recipe"]]],
                         {"model.name": st["model"], "seed": s, **(st.get("overrides") or {})}))
    else:
        raise ValueError(f"unknown stage {stage}")
    return runs


def _require(st: dict, stage: str) -> None:
    if not st.get("model") or not st.get("recipe"):
        raise SystemExit(f"{stage}: isi dulu `model` dan `recipe` di experiments.yaml (dari hasil stage1)")


def collect(out_dir: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*/summary.json"))]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = ["run", "model", "recipe", "seed", "best_epoch", "epochs_run", "val_auc", "val_f1", "val_precision",
            "val_recall", "val_acc", "val_video_auc", "val_video_f1", "final_train_loss", "final_val_loss",
            "params_total", "params_trainable", "checkpoint_mb", "sec_per_epoch", "train_time_s", "peak_gpu_mem_mb"]
    return df[[c for c in cols if c in df.columns]]


def main(argv=None):
    p = argparse.ArgumentParser(prog="looktwice.run_experiments")
    p.add_argument("--plan", type=Path, default=Path("configs/train/experiments.yaml"))
    p.add_argument("--stage", choices=["stage1", "stage2", "stage3"])
    p.add_argument("--only", action="append", default=[], help="run-name substring filter")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override for every run")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rerun", action="store_true", help="ignore existing summary.json")
    p.add_argument("--collect", action="store_true", help="write runs table to results/runs_summary.csv")
    args = p.parse_args(argv)

    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    global_ov = dict(map(parse_override, args.set))
    out_dir = Path(global_ov.get("out_dir", "runs"))

    if args.stage:
        runs = [r for r in expand_stage(plan, args.stage) if not args.only or any(o in r[0] for o in args.only)]
        for i, (name, files, ov) in enumerate(runs, 1):
            done = (out_dir / name / "summary.json").exists()
            status = "skip (done)" if done and not args.rerun else "run"
            print(f"[{i}/{len(runs)}] {name}: {status}  overrides={ov}")
            if args.dry_run or (done and not args.rerun):
                continue
            cfg = load_config(files, {**global_ov, **ov, "run_name": name})
            try:
                train(cfg)
            except Exception:
                traceback.print_exc()

    if args.collect or args.stage and not args.dry_run:
        table = collect(out_dir)
        if len(table):
            Path("results").mkdir(exist_ok=True)
            table.to_csv("results/runs_summary.csv", index=False)
            print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
