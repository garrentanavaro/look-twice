from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import pandas as pd
import yaml

from looktwice.data.extract import ExtractConfig, run_extract
from looktwice.data.metadata import scan_videos, summarize
from looktwice.data.split import (apply_split, assign_actors, list_actors, search_seeds,
                                  split_counts, split_report)

DEFAULT_CONFIG = Path("configs/preprocess.yaml")


def load_config(path: Path | None) -> dict:
    path = path or DEFAULT_CONFIG
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    print(f"[warn] config {path} not found, using defaults")
    return {}


def read_videos(meta: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(meta / name, dtype={"target": str, "donor": str})


def cmd_inspect(args, cfg):
    scan = scan_videos(args.root)
    print(summarize(scan))
    args.out.mkdir(parents=True, exist_ok=True)
    scan.videos.to_csv(args.out / "videos.csv", index=False)
    (args.out / "inspect_issues.json").write_text(json.dumps({
        "unparsed": scan.unparsed, "unlabeled": scan.unlabeled,
        "duplicates": scan.duplicates, "warnings": scan.warnings}, indent=2))
    print(f"\nwrote {args.out / 'videos.csv'}")


def cmd_split(args, cfg):
    sc = cfg.get("split", {})
    n_val, n_test = sc.get("n_val", 4), sc.get("n_test", 4)
    seed = args.seed if args.seed is not None else sc.get("seed", 42)
    policy = args.policy or sc.get("policy", "strict")
    videos = read_videos(args.out, "videos.csv")

    if args.search_seeds:
        res = search_seeds(videos, n_val, n_test, args.search_seeds, policy)
        print(f"top seeds (policy={policy}) ranked by min fake videos in val/test:")
        print(res.head(10).to_string(index=False))
        res.to_csv(args.out / "seed_search.csv", index=False)
        return

    actor_split = assign_actors(list_actors(videos), n_val, n_test, seed)
    print(f"seed={seed}\n" + split_report(videos, actor_split))
    df = apply_split(videos, actor_split, policy)
    df.to_csv(args.out / "videos_split.csv", index=False)
    (args.out / "actor_split.json").write_text(json.dumps(
        {"seed": seed, "policy": policy, "actors": actor_split}, indent=2))
    print(f"\nused policy={policy}; wrote {args.out / 'videos_split.csv'}")


def cmd_extract(args, cfg):
    known = {f.name for f in fields(ExtractConfig)}
    ecfg = ExtractConfig(**{k: v for k, v in cfg.get("extract", {}).items() if k in known})
    videos = read_videos(args.meta, "videos_split.csv")
    index = run_extract(videos, args.root, args.out, ecfg, workers=args.workers, limit=args.limit)
    print("\ncrops per split:")
    print(split_counts(index) if len(index) else "no crops saved")
    fails = pd.read_csv(args.out / "failures.csv")
    print(f"failed frames: {len(fails)}  ({fails.reason.value_counts().to_dict() if len(fails) else {}})")


def main(argv=None):
    p = argparse.ArgumentParser(prog="looktwice.preprocess")
    p.add_argument("--config", type=Path, default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inspect", help="scan videos, parse actor ids, write videos.csv")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--out", type=Path, default=Path("data/meta"))

    s = sub.add_parser("split", help="assign actors to train/val/test, write videos_split.csv")
    s.add_argument("--out", type=Path, default=Path("data/meta"))
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--policy", choices=["strict", "target"], default=None)
    s.add_argument("--search-seeds", type=int, default=0, metavar="N")

    s = sub.add_parser("extract", help="sample frames, crop faces, write crops_index.csv")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--meta", type=Path, default=Path("data/meta"))
    s.add_argument("--out", type=Path, default=Path("data/faces"))
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--limit", type=int, default=None, help="only N videos per label (smoke test)")

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    {"inspect": cmd_inspect, "split": cmd_split, "extract": cmd_extract}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
