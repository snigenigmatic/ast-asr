#!/usr/bin/env python3
"""
make_svarah_split.py
Create a deterministic, stratified 70/30 train/eval split of Svarah by language_family.

Usage:
    python scripts/make_svarah_split.py               # create split
    python scripts/make_svarah_split.py --verify       # print counts & check disjointness
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ast-asr"))
from data_loader import load_svarah

SPLIT_DIR = Path("data/svarah_split")
SEED = 42
TRAIN_FRAC = 0.7


def make_split():
    df = load_svarah(max_samples=None, cache_dir="cache")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_uids, eval_uids = [], []
    rng = random.Random(SEED)

    for family, group in df.groupby("language_family"):
        uids = group["uid"].tolist()
        rng.shuffle(uids)
        cutoff = int(len(uids) * TRAIN_FRAC)
        train_uids.extend(uids[:cutoff])
        eval_uids.extend(uids[cutoff:])
        print(f"  {family:20s}  train={cutoff}  eval={len(uids) - cutoff}  total={len(uids)}")

    (SPLIT_DIR / "train_uids.txt").write_text("\n".join(str(u) for u in sorted(train_uids)) + "\n")
    (SPLIT_DIR / "eval_uids.txt").write_text("\n".join(str(u) for u in sorted(eval_uids)) + "\n")

    print(f"\nTotal: train={len(train_uids)}  eval={len(eval_uids)}")
    print(f"Written to {SPLIT_DIR}/")


def verify():
    train_uids = set((SPLIT_DIR / "train_uids.txt").read_text().strip().split("\n"))
    eval_uids = set((SPLIT_DIR / "eval_uids.txt").read_text().strip().split("\n"))
    overlap = train_uids & eval_uids
    print(f"Train UIDs: {len(train_uids)}")
    print(f"Eval UIDs:  {len(eval_uids)}")
    print(f"Overlap:    {len(overlap)}")
    assert len(overlap) == 0, f"CONTAMINATION: {len(overlap)} overlapping UIDs!"
    print("OK: splits are disjoint.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify()
    else:
        make_split()
