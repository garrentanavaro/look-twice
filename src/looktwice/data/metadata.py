from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REAL, FAKE = 0, 1
LABEL_NAMES = {REAL: "real", FAKE: "fake"}

FNAME_RE = re.compile(
    r"^(?P<target>\d+)"
    r"(?:_(?P<donor>\d+))?"
    r"__(?P<scene>.+?)"
    r"(?:__(?P<uid>[A-Za-z0-9]+))?$"
)

VIDEO_COLUMNS = ["path", "video_id", "label", "target", "donor", "scene", "uid"]


def parse_name(stem: str) -> dict | None:
    m = FNAME_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    return {
        "target": d["target"].zfill(2),
        "donor": d["donor"].zfill(2) if d["donor"] else None,
        "scene": d["scene"],
        "uid": d["uid"],
    }


def label_from_path(
    path: Path,
    real_keywords: tuple[str, ...] = ("original",),
    fake_keywords: tuple[str, ...] = ("manipulated",),
) -> int | None:
    parts = [p.lower() for p in path.parts[:-1]]
    is_fake = any(k in p for p in parts for k in fake_keywords)
    is_real = any(k in p for p in parts for k in real_keywords)
    if is_fake and not is_real:
        return FAKE
    if is_real and not is_fake:
        return REAL
    return None


@dataclass
class ScanResult:
    videos: pd.DataFrame
    unparsed: list[str] = field(default_factory=list)
    unlabeled: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def scan_videos(root: str | Path, exts: tuple[str, ...] = (".mp4",)) -> ScanResult:
    root = Path(root)
    rows, unparsed, unlabeled, warnings = [], [], [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(root)
        label = label_from_path(rel)
        if label is None:
            unlabeled.append(rel.as_posix())
            continue
        info = parse_name(p.stem)
        if info is None:
            unparsed.append(rel.as_posix())
            continue
        if label == FAKE and info["donor"] is None:
            warnings.append(f"fake video without donor id: {rel}")
        if label == REAL and info["donor"] is not None:
            warnings.append(f"real video with donor id: {rel}")
        rows.append({"path": rel.as_posix(), "video_id": p.stem, "label": label, **info})

    df = pd.DataFrame(rows, columns=VIDEO_COLUMNS)
    dup_mask = df.duplicated(subset=["label", "video_id"], keep="first")
    duplicates = df.loc[dup_mask, "path"].tolist()
    df = df.loc[~dup_mask].reset_index(drop=True)
    return ScanResult(df, unparsed, unlabeled, duplicates, warnings)


def summarize(scan: ScanResult, n_examples: int = 3) -> str:
    df = scan.videos
    lines = [f"videos parsed : {len(df)}"]
    for lab, name in LABEL_NAMES.items():
        sub = df[df.label == lab]
        lines.append(f"  {name:<4}: {len(sub)}  e.g. {sub.video_id.head(n_examples).tolist()}")
    actors = sorted(set(df.target) | set(df.donor.dropna()))
    lines.append(f"actors        : {len(actors)} -> {actors}")
    real_per_actor = df[df.label == REAL].groupby("target").size()
    fake_per_actor = df[df.label == FAKE].groupby("target").size()
    lines.append(f"real/actor    : min={real_per_actor.min()} max={real_per_actor.max()}" if len(real_per_actor) else "real/actor    : -")
    lines.append(f"fake/actor    : min={fake_per_actor.min()} max={fake_per_actor.max()}" if len(fake_per_actor) else "fake/actor    : -")
    for name, items in [("unparsed", scan.unparsed), ("unlabeled", scan.unlabeled),
                        ("duplicates", scan.duplicates), ("warnings", scan.warnings)]:
        lines.append(f"{name:<14}: {len(items)}" + (f"  e.g. {items[:n_examples]}" if items else ""))
    return "\n".join(lines)
