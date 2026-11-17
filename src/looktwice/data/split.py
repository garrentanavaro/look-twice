from __future__ import annotations

import numpy as np
import pandas as pd

from .metadata import FAKE, LABEL_NAMES

SPLITS = ("train", "val", "test")
POLICIES = ("target", "strict")


def list_actors(videos: pd.DataFrame) -> list[str]:
    return sorted(set(videos.target) | set(videos.donor.dropna()))


def assign_actors(actors: list[str], n_val: int, n_test: int, seed: int) -> dict[str, str]:
    if n_val + n_test >= len(actors):
        raise ValueError(f"n_val + n_test ({n_val + n_test}) must be < number of actors ({len(actors)})")
    rng = np.random.default_rng(seed)
    order = [str(a) for a in rng.permutation(sorted(actors))]
    test, val, train = order[:n_test], order[n_test:n_test + n_val], order[n_test + n_val:]
    return {**{a: "train" for a in train}, **{a: "val" for a in val}, **{a: "test" for a in test}}


def apply_split(videos: pd.DataFrame, actor_split: dict[str, str], policy: str) -> pd.DataFrame:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    df = videos.copy()
    df["split"] = df.target.map(actor_split)
    df["donor_split"] = df.donor.map(actor_split)
    if policy == "strict":
        leak = (df.label == FAKE) & df.donor_split.notna() & (df.donor_split != df.split)
        df.loc[leak, "split"] = "drop"
    if df.split.isna().any():
        raise ValueError("some target actors are missing from actor_split")
    return df


def split_counts(df: pd.DataFrame) -> pd.DataFrame:
    t = (df.assign(label=df.label.map(LABEL_NAMES))
           .groupby(["split", "label"]).size().unstack(fill_value=0))
    order = [s for s in (*SPLITS, "drop") if s in t.index]
    return t.reindex(order)


def split_report(videos: pd.DataFrame, actor_split: dict[str, str]) -> str:
    out = []
    for split in SPLITS:
        out.append(f"{split:<5} actors: {sorted(a for a, s in actor_split.items() if s == split)}")
    for policy in POLICIES:
        out.append(f"\n[policy={policy}] videos per split")
        out.append(split_counts(apply_split(videos, actor_split, policy)).to_string())
    return "\n".join(out)


def search_seeds(videos: pd.DataFrame, n_val: int, n_test: int, n_seeds: int,
                 policy: str = "strict") -> pd.DataFrame:
    actors = list_actors(videos)
    rows = []
    for seed in range(n_seeds):
        df = apply_split(videos, assign_actors(actors, n_val, n_test, seed), policy)
        fakes = df[df.label == FAKE].split.value_counts()
        rows.append({
            "seed": seed,
            "fake_train": int(fakes.get("train", 0)),
            "fake_val": int(fakes.get("val", 0)),
            "fake_test": int(fakes.get("test", 0)),
            "fake_dropped": int(fakes.get("drop", 0)),
        })
    res = pd.DataFrame(rows)
    res["min_val_test"] = res[["fake_val", "fake_test"]].min(axis=1)
    return res.sort_values(["min_val_test", "fake_train"], ascending=False).reset_index(drop=True)
