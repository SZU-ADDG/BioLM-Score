#!/usr/bin/env python3
"""Export the exact random PDBbind train/validation IDs used by train_model.py.

This reproduces PDBbindDataset.train_and_test_split in BioLM_Score/data/data2.py:
NumPy's legacy global RNG is seeded, validation indices are sampled without
replacement, and training indices are the sorted set difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_ids(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=True)
    if loaded.ndim == 1:
        ids = loaded
    elif loaded.ndim == 2 and loaded.shape[0] in (1, 2):
        ids = loaded[0]
    else:
        raise ValueError(f"Expected an IDs vector or [IDs, labels] array; got shape {loaded.shape}")
    ids = np.asarray(ids).astype(str)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Prepared PDBbind ID array contains duplicates")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-ids", required=True, type=Path,
                        help="The exact *_ids.npy file consumed by train_model.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, required=True,
                        help="Explicit run randomization values supplied by the caller")
    parser.add_argument("--valnum", type=int, default=1500)
    args = parser.parse_args()

    ids_path = args.prepared_ids.resolve()
    ids = extract_ids(ids_path)
    if args.valnum <= 0 or args.valnum >= len(ids):
        raise ValueError(f"valnum must be between 1 and {len(ids) - 1}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_records = []
    for seed in args.seeds:
        np.random.seed(seed)
        validation_indices = np.random.choice(np.arange(len(ids)), args.valnum, replace=False)
        train_indices = np.setdiff1d(np.arange(len(ids)), validation_indices)
        train_ids = ids[train_indices]
        validation_ids = ids[validation_indices]
        train_path = args.output_dir / f"seed{seed}_train_ids.txt"
        validation_path = args.output_dir / f"seed{seed}_val_ids.txt"
        train_path.write_text("".join(f"{value}\n" for value in train_ids), encoding="utf-8")
        validation_path.write_text("".join(f"{value}\n" for value in validation_ids), encoding="utf-8")
        seed_records.append({
            "seed": seed,
            "train_count": len(train_ids),
            "validation_count": len(validation_ids),
            "train_ids_file": train_path.name,
            "train_ids_sha256": sha256(train_path),
            "validation_ids_file": validation_path.name,
            "validation_ids_sha256": sha256(validation_path),
        })

    manifest = {
        "prepared_ids_file": str(ids_path),
        "prepared_ids_sha256": sha256(ids_path),
        "prepared_complex_count": len(ids),
        "validation_count": args.valnum,
        "split_implementation": "BioLM_Score/data/data2.py:PDBbindDataset.train_and_test_split",
        "randomness": "numpy.random.seed(seed); numpy.random.choice(arange(N), valnum, replace=False); train=np.setdiff1d(arange(N), validation)",
        "seed_records": seed_records,
    }
    manifest_path = args.output_dir / "random_split_ids_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
