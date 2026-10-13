#!/usr/bin/env python3
"""Validate the small identifier-only manifests distributed with this code."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IDENTIFIERS = ROOT / "reproducibility" / "identifiers"
TIME_SPLIT = ROOT / "BioLM_Score" / "data"
MANIFEST = ROOT / "reproducibility" / "manifests" / "identifier_manifest.json"


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def check_count(label: str, values: list[str], expected: int, expected_hash: str, path: Path) -> None:
    if len(values) != expected or len(set(values)) != expected:
        raise SystemExit(
            f"{label}: expected {expected} unique IDs, got {len(values)} rows / {len(set(values))} unique"
        )
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise SystemExit(f"{label}: SHA256 mismatch: {actual_hash} != {expected_hash}")
    print(f"{label}: n={len(values)} sha256={actual_hash}")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    specifications = {
        "time-split train": "time_split_train",
        "time-split validation": "time_split_validation",
        "CASF exclusion": "time_split_casf_exclusion",
        "LP-PDBBind train": "lp_pdbbind_model_compatible_train",
        "LP-PDBBind validation": "lp_pdbbind_model_compatible_validation",
        "BDB2020+": "bdb2020plus_pdb_ids",
        "DEKOIS 2.0 targets": "dekois2_target_ids",
    }
    collections = {}
    for label, key in specifications.items():
        spec = manifest["files"][key]
        path = ROOT / spec["path"]
        values = read_ids(path)
        check_count(label, values, spec["count"], spec["sha256"], path)
        collections[label] = values
    lp_train = set(collections["LP-PDBBind train"])
    lp_val = set(collections["LP-PDBBind validation"])
    if lp_train & lp_val:
        raise SystemExit("LP-PDBBind train/validation overlap")
    dekois_targets = set(collections["DEKOIS 2.0 targets"])
    with (IDENTIFIERS / "DEKOIS2.0_target_chain_map.tsv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    mapped_targets = {row["dekois_target"] for row in rows}
    dekois_map_spec = manifest["files"]["dekois2_target_chain_map"]
    map_path = ROOT / dekois_map_spec["path"]
    if len(rows) != dekois_map_spec["rows"] or mapped_targets != dekois_targets:
        raise SystemExit(
            f"DEKOIS chain map mismatch: {len(rows)} rows, {len(mapped_targets)} unique targets"
        )
    if sha256(map_path) != dekois_map_spec["sha256"]:
        raise SystemExit("DEKOIS chain map SHA256 mismatch")
    print(f"DEKOIS target-chain mappings: n={len(rows)} rows / {len(mapped_targets)} targets")

    group_spec = manifest["files"]["pdbbind_mmseqs_target_groups"]
    group_path = ROOT / group_spec["path"]
    group_rows = []
    with group_path.open(encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row:
                continue
            if len(row) != 2 or not row[0].strip() or not row[1].strip():
                raise SystemExit(f"Invalid PDBbind target-group row at line {line_number}")
            group_rows.append((row[0].strip().lower(), row[1].strip().lower()))
    members = [member for _, member in group_rows]
    representatives = {representative for representative, _ in group_rows}
    if len(group_rows) != group_spec["rows"] or len(set(members)) != group_spec["unique_members"]:
        raise SystemExit(
            f"PDBbind target-group mapping mismatch: {len(group_rows)} rows / {len(set(members))} unique members"
        )
    if not representatives.issubset(set(members)):
        raise SystemExit("PDBbind target-group representatives must also occur as members")
    if sha256(group_path) != group_spec["sha256"]:
        raise SystemExit("PDBbind target-group mapping SHA256 mismatch")
    print(
        f"PDBbind MMseqs2 target groups: n={len(group_rows)} rows / "
        f"{len(representatives)} representatives / {len(set(members))} members"
    )
    print("All identifier-only manifest checks passed.")


if __name__ == "__main__":
    main()
