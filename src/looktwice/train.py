from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from looktwice.config import load_config
from looktwice.data.dataset import FaceCropDataset, build_transforms
from looktwice.metrics import binary_metrics, prefixed, video_level
from looktwice.models import MODEL_ZOO, build_model, count_params, model_data_config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


def run_name_from(cfg: dict) -> str:
    if cfg.get("run_name"):
        return cfg["run_name"]
    return f"{cfg['model']['name']}_{cfg.get('recipe', 'custom')}_s{cfg.get('seed', 0)}"


class SmoothedBCE(nn.Module):
    def __init__(self, eps: float = 0.0, pos_weight: float | None = None):
        super().__init__()
        self.eps = eps
        self.register_buffer("pos_weight", torch.tensor(pos_weight) if pos_weight else None)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y * (1 - self.eps) + 0.5 * self.eps
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)


def build_optimizer(params, o: dict) -> torch.optim.Optimizer:
    lr = float(o["lr"]) * float(o.get("lr_scale", 1.0))
    wd = float(o.get("weight_decay", 0.0))
    name = o["name"].lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=float(o.get("momentum", 0.9)),
                               weight_decay=wd, nesterov=bool(o.get("nesterov", False)))
    raise ValueError(f"unknown optimizer {name}")


def build_scheduler(opt, s: dict, epochs: int, monitor_mode: str):
    name = s.get("name", "constant").lower()
    if name == "constant":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(s.get("t_max") or epochs),
                                                          eta_min=float(s.get("eta_min", 0.0)))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode=monitor_mode,
                                                          factor=float(s.get("factor", 0.5)),
                                                          patience=int(s.get("patience", 3)))
    raise ValueError(f"unknown scheduler {name}")


def train_one_epoch(model, loader, criterion, opt, scaler, device, amp: bool, grad_clip):
    model.train()
    tot_loss, n, correct, norms = 0.0, 0, 0, []
    params = [p for p in model.parameters() if p.requires_grad]
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x).squeeze(1)
            loss = criterion(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(params, float(grad_clip) if grad_clip else float("inf"))
        scaler.step(opt)
        scaler.update()
        if torch.isfinite(norm):
            norms.append(norm.item())
        tot_loss += loss.item() * len(y)
        correct += ((logits.detach() > 0).float() == y).sum().item()
        n += len(y)
    return {"train_loss": tot_loss / max(n, 1), "train_acc": correct / max(n, 1),
            "grad_norm": float(np.mean(norms)) if norms else float("nan")}


@torch.no_grad()
def predict(model, loader, criterion, device, amp: bool):
    model.eval()
    tot_loss, n, probs, idxs = 0.0, 0, [], []
    for x, y, i in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x).squeeze(1).float()
        tot_loss += criterion(logits, y).item() * len(y)
        n += len(y)
        probs.append(torch.sigmoid(logits).cpu())
        idxs.append(i)
    return tot_loss / max(n, 1), torch.cat(probs).numpy(), torch.cat(idxs).numpy()


def train(cfg: dict) -> dict:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    tcfg, dcfg, ocfg = cfg["train"], cfg["data"], cfg["optim"]
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    epochs = int(tcfg["epochs"])
    monitor = tcfg.get("monitor", "val_auc")
    mode = "min" if monitor.endswith("loss") else "max"

    run_dir = Path(cfg.get("out_dir", "runs")) / run_name_from(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    m = cfg["model"]
    model = build_model(m["name"], pretrained=bool(m.get("pretrained", True)),
                        drop_rate=float(m.get("drop_rate", 0.0)),
                        freeze_backbone=bool(m.get("freeze_backbone", False))).to(device)
    dc = model_data_config(model)
    size = int(cfg.get("input_size", 224))
    mean, std = tuple(dc["mean"]), tuple(dc["std"])

    index, root = dcfg["index"], dcfg.get("root")
    train_ds = FaceCropDataset.from_index(index, "train", build_transforms(True, size, mean, std, cfg.get("aug")),
                                          root=root, max_items=dcfg.get("max_train"), seed=seed)
    val_ds = FaceCropDataset.from_index(index, "val", build_transforms(False, size, mean, std),
                                        root=root, max_items=dcfg.get("max_val"), seed=seed)
    bs, nw = int(ocfg["batch_size"]), int(dcfg.get("num_workers", 4))
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=len(train_ds) > bs,
                          pin_memory=device.type == "cuda", worker_init_fn=_worker_init, generator=g,
                          persistent_workers=nw > 0)
    val_dl = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, num_workers=nw,
                        pin_memory=device.type == "cuda", persistent_workers=nw > 0)

    lcfg = cfg.get("loss", {})
    pos_weight = lcfg.get("pos_weight")
    if pos_weight == "auto":
        pos_weight = float((train_ds.labels == 0).sum() / max((train_ds.labels == 1).sum(), 1))
    criterion = SmoothedBCE(float(lcfg.get("label_smoothing", 0.0)), pos_weight).to(device)
    opt = build_optimizer([p for p in model.parameters() if p.requires_grad], ocfg)
    sched = build_scheduler(opt, cfg.get("sched", {}), epochs, mode)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    n_params, n_trainable = count_params(model)
    print(f"[{run_dir.name}] device={device} amp={amp} train={len(train_ds)} val={len(val_ds)} "
          f"params={n_params/1e6:.1f}M trainable={n_trainable/1e6:.1f}M mean={mean} std={std}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, best, best_epoch, bad_epochs = [], -math.inf if mode == "max" else math.inf, -1, 0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_now = opt.param_groups[0]["lr"]
        row = {"epoch": epoch, "lr": lr_now,
               **train_one_epoch(model, train_dl, criterion, opt, scaler, device, amp, tcfg.get("grad_clip"))}
        val_loss, probs, idxs = predict(model, val_dl, criterion, device, amp)
        preds = val_ds.df.iloc[idxs][["path", "video_id", "label"]].assign(prob=probs)
        fm = binary_metrics(preds.label, preds.prob)
        vm = binary_metrics(*video_level(preds)[["label", "prob"]].T.values)
        row.update({"val_loss": val_loss, **prefixed(fm, "val_"), **prefixed(vm, "val_video_", ("acc", "f1", "auc")),
                    "epoch_time_s": time.time() - t0})
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        score = row[monitor]
        if isinstance(score, float) and math.isnan(score):
            score = -val_loss if mode == "max" else val_loss
        improved = score > best if mode == "max" else score < best
        print(f"  ep{epoch:02d} lr={lr_now:.2e} train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
              f"val_auc={row['val_auc']:.4f} val_f1={row['val_f1']:.4f} video_auc={row['val_video_auc']:.4f} "
              f"({row['epoch_time_s']:.0f}s){' *' if improved else ''}")
        if improved:
            best, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"model": model.state_dict(), "model_name": m["name"],
                        "timm_name": MODEL_ZOO.get(m["name"], m["name"]), "input_size": size,
                        "mean": mean, "std": std, "epoch": epoch, "val_metrics": row}, run_dir / "best.pt")
            preds.to_csv(run_dir / "val_preds.csv", index=False)
        else:
            bad_epochs += 1

        if sched is not None:
            sched.step(score) if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau) else sched.step()
        if bad_epochs >= int(tcfg.get("early_stop_patience", epochs)):
            print(f"  early stop at epoch {epoch} (best {monitor}={best:.4f} @ {best_epoch})")
            break

    hist = pd.DataFrame(history)
    best_row = hist[hist.epoch == best_epoch].iloc[0].to_dict()
    summary = {
        "run": run_dir.name, "model": m["name"], "recipe": cfg.get("recipe"), "seed": seed,
        "best_epoch": best_epoch, "epochs_run": len(hist), "monitor": monitor,
        **{k: best_row[k] for k in best_row if k.startswith("val_")},
        "final_train_loss": float(hist.train_loss.iloc[-1]), "final_val_loss": float(hist.val_loss.iloc[-1]),
        "params_total": n_params, "params_trainable": n_trainable,
        "checkpoint_mb": (run_dir / "best.pt").stat().st_size / 2**20,
        "train_time_s": time.time() - t_start, "sec_per_epoch": float(hist.epoch_time_s.mean()),
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor() or "cpu",
        "n_train": len(train_ds), "n_val": len(val_ds), "pos_weight": pos_weight,
        "torch": torch.__version__,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(prog="looktwice.train")
    p.add_argument("--config", action="append", required=True, help="YAML file(s), merged in order")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. optim.lr=3e-4")
    args = p.parse_args(argv)
    summary = train(load_config(args.config, args.set))
    print(json.dumps({k: summary[k] for k in ("run", "best_epoch", "val_auc", "val_f1", "val_video_auc")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
