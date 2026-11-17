from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
REF = "#8a8984"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("looktwice_blue", BLUE_RAMP)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": TEXT, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.edgecolor": GRID,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 6,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_history(history: pd.DataFrame, title: str, path: str | Path) -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    fig.suptitle(title, color=TEXT, fontweight="bold")
    ax[0].plot(history.epoch, history.train_loss, color=SERIES[0], label="train")
    ax[0].plot(history.epoch, history.val_loss, color=SERIES[1], label="validation")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[1].plot(history.epoch, history.train_acc, color=SERIES[0], label="train accuracy")
    ax[1].plot(history.epoch, history.val_acc, color=SERIES[1], label="val accuracy")
    ax[1].plot(history.epoch, history.val_auc, color=SERIES[2], label="val AUC")
    ax[1].set(title="Metric", xlabel="epoch", ylim=(0, 1.02))
    ax[2].plot(history.epoch, history.lr, color=SERIES[0])
    ax[2].set(title="Learning rate", xlabel="epoch", yscale="log")
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="lower right")
    for a in ax:
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return _save(fig, path)


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], path: str | Path,
             title: str = "ROC — test set") -> Path:
    if len(curves) > 3:
        raise ValueError("max 3 ROC curves per figure; split into several figures")
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=REF, lw=1, ls="--", label="chance")
    for (name, (fpr, tpr, auc)), c in zip(curves.items(), SERIES):
        ax.plot(fpr, tpr, color=c, label=f"{name} (AUC {auc:.3f})")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title=title,
           xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_confusion(tn: int, fp: int, fn: int, tp: int, path: str | Path, title: str) -> Path:
    m = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(m, cmap=SEQ_CMAP, vmin=0, vmax=max(m.max(), 1))
    ax.grid(False)
    for (i, j), v in np.ndenumerate(m):
        dark_cell = v > 0.55 * m.max()
        pct = v / max(m[i].sum(), 1)
        ax.text(j, i, f"{v}\n{pct:.0%} of row", ha="center", va="center",
                color=SURFACE if dark_cell else TEXT, fontsize=10)
    ax.set_xticks([0, 1], ["pred real", "pred fake"])
    ax.set_yticks([0, 1], ["true real", "true fake"])
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    return _save(fig, path)


def plot_robustness(df: pd.DataFrame, clean: dict, path: str | Path, title: str) -> Path:
    kinds = [k for k in ("jpeg", "blur", "downscale") if k in set(df.kind)]
    xlabels = {"jpeg": "JPEG quality (lower = stronger)", "blur": "Gaussian kernel (larger = stronger)",
               "downscale": "scale factor (lower = stronger)"}
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.4 * len(kinds), 3.5), sharey=True, squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        d = df[df.kind == kind].sort_values("level", ascending=kind != "blur")
        x = np.arange(len(d))
        ax.plot(x, d.auc, color=SERIES[0], marker="o", label="AUC")
        ax.plot(x, d.f1, color=SERIES[1], marker="o", label="F1")
        ax.axhline(clean["auc"], color=REF, lw=1, ls="--", label="clean AUC")
        ax.set_xticks(x, [f"{v:g}" for v in d.level])
        ax.set(title=kind, xlabel=xlabels[kind], ylim=(0, 1.02))
    axes[0][0].set_ylabel("score")
    axes[0][-1].legend(loc="lower left")
    fig.suptitle(title, color=TEXT, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def image_grid(images: list[np.ndarray], captions: list[str], path: str | Path, title: str,
               ncols: int = 8) -> Path:
    if not images:
        return Path(path)
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 2.15 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, cap in zip(axes.flat, images, captions):
        ax.imshow(img)
        ax.set_title(cap, fontsize=7, color=TEXT_2)
    fig.suptitle(title, color=TEXT, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)
