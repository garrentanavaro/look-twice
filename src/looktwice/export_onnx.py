from __future__ import annotations

import argparse
import inspect
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from looktwice.models import load_checkpoint
from looktwice.thresholds import choose_thresholds


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model((image - self.mean) / self.std)).squeeze(1)


def export(run_dir: Path, out: Path, opset: int = 17, target_acc: float = 0.95) -> tuple[Path, dict]:
    model, meta = load_checkpoint(run_dir / "best.pt", "cpu")
    wrapper = ExportWrapper(model, meta["mean"], meta["std"]).eval()
    size = int(meta["input_size"])
    dummy = torch.rand(1, 3, size, size)
    out.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(input_names=["image"], output_names=["prob"], opset_version=opset,
                  dynamic_axes={"image": {0: "batch"}, "prob": {0: "batch"}})
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(wrapper, (dummy,), str(out), **kwargs)

    thr = None
    if (run_dir / "val_preds.csv").exists():
        val = pd.read_csv(run_dir / "val_preds.csv")
        thr = choose_thresholds(val.label, val.prob, target_acc)
    info = {
        "model_name": meta["model_name"], "timm_name": meta.get("timm_name"), "input_size": size,
        "input": "float32 [N,3,S,S] RGB in [0,1]", "output": "float32 [N] P(fake)",
        "mean": list(meta["mean"]), "std": list(meta["std"]), "thresholds": thr,
        "source_run": run_dir.name, "best_epoch": meta.get("epoch"),
        "val_auc": (meta.get("val_metrics") or {}).get("val_auc"),
        "onnx_mb": round(out.stat().st_size / 2**20, 2), "opset": opset,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.with_suffix(".json").write_text(json.dumps(info, indent=2, default=float))
    return out, info


def verify(run_dir: Path, onnx_path: Path, n: int = 4, atol: float = 1e-4) -> float:
    import onnxruntime as ort
    model, meta = load_checkpoint(run_dir / "best.pt", "cpu")
    wrapper = ExportWrapper(model, meta["mean"], meta["std"]).eval()
    x = torch.rand(n, 3, meta["input_size"], meta["input_size"])
    with torch.no_grad():
        ref = wrapper(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(["prob"], {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    if diff > atol:
        raise AssertionError(f"ONNX output differs from PyTorch by {diff:.2e} (> {atol})")
    return diff


def _timeit(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.asarray(ts)
    return {"mean_ms": ts.mean(), "p50_ms": np.percentile(ts, 50), "p95_ms": np.percentile(ts, 95)}


def benchmark(run_dir: Path, onnx_path: Path, warmup: int = 10, iters: int = 50,
              threads: int | None = None) -> pd.DataFrame:
    import onnxruntime as ort
    model, meta = load_checkpoint(run_dir / "best.pt", "cpu")
    wrapper = ExportWrapper(model, meta["mean"], meta["std"]).eval()
    size = meta["input_size"]
    x = torch.rand(1, 3, size, size)
    if threads:
        torch.set_num_threads(threads)
    cpu = platform.processor() or platform.machine()
    rows = []

    with torch.inference_mode():
        rows.append({"engine": "pytorch", "device": f"cpu ({cpu})", **_timeit(lambda: wrapper(x), warmup, iters)})
        if torch.cuda.is_available():
            w, xg = wrapper.cuda(), x.cuda()

            def run_gpu():
                w(xg)
                torch.cuda.synchronize()
            rows.append({"engine": "pytorch", "device": torch.cuda.get_device_name(),
                         **_timeit(run_gpu, warmup, iters)})

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    for provider in ["CPUExecutionProvider", "CUDAExecutionProvider"]:
        if provider not in ort.get_available_providers():
            continue
        sess = ort.InferenceSession(str(onnx_path), so, providers=[provider])
        xn = x.numpy()
        rows.append({"engine": "onnxruntime", "device": provider.replace("ExecutionProvider", "").lower()
                     + (f" ({cpu})" if provider.startswith("CPU") else ""),
                     **_timeit(lambda: sess.run(["prob"], {"image": xn}), warmup, iters)})

    df = pd.DataFrame(rows).assign(run=run_dir.name, model=meta["model_name"], input_size=size,
                                   threads=threads or torch.get_num_threads(), batch=1)
    df["under_200ms"] = df.p95_ms < 200
    return df


def main(argv=None):
    p = argparse.ArgumentParser(prog="looktwice.export_onnx")
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, default=None, help="only with a single --run (default weights/<run>.onnx)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--latency-csv", type=Path, default=Path("results/latency.csv"))
    args = p.parse_args(argv)
    if args.out and len(args.run) > 1:
        p.error("--out only works with a single --run")

    tables = []
    for run in args.run:
        path, info = export(run, args.out or Path("weights") / f"{run.name}.onnx", args.opset)
        diff = verify(run, path)
        print(f"{run.name}: wrote {path} ({info['onnx_mb']} MB), max |onnx - torch| = {diff:.1e}")
        if args.bench:
            tables.append(benchmark(run, path, iters=args.iters, threads=args.threads))
            print(tables[-1][["engine", "device", "mean_ms", "p50_ms", "p95_ms", "under_200ms"]]
                  .to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    if tables:
        new = pd.concat(tables, ignore_index=True)
        args.latency_csv.parent.mkdir(parents=True, exist_ok=True)
        if args.latency_csv.exists():
            new = pd.concat([pd.read_csv(args.latency_csv), new], ignore_index=True)
        new.to_csv(args.latency_csv, index=False)


if __name__ == "__main__":
    main()
