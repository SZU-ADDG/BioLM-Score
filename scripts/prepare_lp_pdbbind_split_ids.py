#!/usr/bin/env python3
"""Regenerate the LP-PDBBind ID lists used for model-compatible retraining.

The filters match ``prepare_lp_pdbbind_dataset.py``: non-covalent CL1 records
from the official training split and non-covalent CL2 records from validation
and test. The downloaded CSV is not copied into this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED_COUNTS = {"train": 7393, "validation": 958, "test": 2171}
SOURCE_CSV_SHA256 = "1f18a0a341be0332d5a8190ca883c5020debe147816ecb920f5219cc4b3d2c55"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select(csv_path: Path):
    frame = pd.read_csv(csv_path, index_col=0)
    frame.index = frame.index.astype(str).str.lower()
    non_covalent = ~frame["covalent"].astype(bool)
    subsets = {
        "train": frame[(frame["new_split"] == "train") & frame["CL1"].astype(bool) & non_covalent],
        "validation": frame[(frame["new_split"] == "val") & frame["CL2"].astype(bool) & non_covalent],
        "test": frame[(frame["new_split"] == "test") & frame["CL2"].astype(bool) & non_covalent],
    }
    observed = {name: len(part) for name, part in subsets.items()}
    if observed != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected official LP-PDBBind counts: {observed}")
    for name, part in subsets.items():
        if part.index.duplicated().any():
            raise ValueError(f"Duplicate PDB identifiers in LP-PDBBind {name} subset")
    if subsets["train"].index.intersection(subsets["validation"].index).size:
        raise ValueError("LP-PDBBind training and validation IDs overlap")
    return subsets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lp-csv", required=True, type=Path, help="Path to official LP_PDBBind.csv")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-different-source-hash",
        action="store_true",
        help="Permit a source CSV whose SHA256 differs from the version recorded for this study",
    )
    args = parser.parse_args()
    csv_path = args.lp_csv.resolve()
    digest = sha256(csv_path)
    if digest != SOURCE_CSV_SHA256 and not args.allow_different_source_hash:
        raise ValueError(
            f"LP-PDBBind source CSV hash changed: {digest}; expected {SOURCE_CSV_SHA256}. "
            "Review the upstream version before using --allow-different-source-hash."
        )
    selected = select(csv_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    output_names = {"train": "train", "validation": "val", "test": "test"}
    for name, frame in selected.items():
        target = args.output_dir / f"LP_PDBBind_model_compatible_{output_names[name]}_ids.txt"
        target.write_text("".join(f"{identifier}\n" for identifier in frame.index), encoding="utf-8")
        outputs[name] = {"count": len(frame), "file": target.name, "sha256": sha256(target)}
    manifest = {
        "source_csv": str(csv_path),
        "source_csv_sha256": digest,
        "filters": {
            "train": "new_split=train AND CL1=true AND covalent=false",
            "validation": "new_split=val AND CL2=true AND covalent=false",
            "test": "new_split=test AND CL2=true AND covalent=false",
        },
        "records": outputs,
        "note": "This is the study's model-compatible subset of the upstream LP-PDBBind release, not a claim that the 7,393/958 lists are upstream's standalone official split files.",
    }
    manifest_path = args.output_dir / "lp_pdbbind_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
