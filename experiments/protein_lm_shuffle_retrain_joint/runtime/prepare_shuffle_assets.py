#!/usr/bin/env python3
"""Build the memory-mapped input bank used by the shuffle-control trainer.

This script consumes locally obtained PDBbind v2020 graphs and per-complex
protein/ligand embeddings. It writes only derived caches beneath ``--output-dir``;
no benchmark structures, labels, embeddings, or model weights are distributed
by this repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_COMPLEXES = 19_015
EXPECTED_CASF_EXCLUSIONS = 285
EXPECTED_DIMENSIONS = {"protein": 1_152, "ligand": 1_024}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_ids_and_labels(path: Path) -> tuple[list[str], np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if payload.ndim != 2 or payload.shape[0] < 2:
        raise ValueError(f"Expected a two-row IDs/labels array, got shape {payload.shape}")
    ids = [str(value).strip().lower() for value in payload[0].tolist()]
    labels = np.asarray(payload[1], dtype=np.float64)
    if len(ids) != len(labels) or not np.isfinite(labels).all():
        raise ValueError("PDBbind IDs/labels have inconsistent lengths or non-finite values")
    if len(ids) != EXPECTED_COMPLEXES or len(set(ids)) != EXPECTED_COMPLEXES:
        raise ValueError(f"Expected {EXPECTED_COMPLEXES} unique standard-split IDs; got {len(ids)}")
    return ids, labels


def read_target_groups(path: Path) -> dict[str, str]:
    groups: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or len(row) < 2:
                continue
            representative, member = row[0].strip().lower(), row[1].strip().lower()
            if not representative or not member:
                raise ValueError(f"Empty target-group ID at {path}:{line_number}")
            if member in groups and groups[member] != representative:
                raise ValueError(f"Conflicting target group for {member}")
            groups[member] = representative
    if len(groups) != EXPECTED_COMPLEXES:
        raise ValueError(f"Expected {EXPECTED_COMPLEXES} target-group members, got {len(groups)}")
    return groups


def load_graphs(path: Path, expected_count: int, label: str):
    import torch

    graphs = torch.load(path, map_location="cpu")
    if len(graphs) != expected_count:
        raise ValueError(f"{label} graph coverage mismatch: {len(graphs)} != {expected_count}")
    return graphs


def build_bank(
    kind: str,
    ids: list[str],
    graphs,
    embedding_dir: Path,
    output_dir: Path,
) -> dict:
    expected_dim = EXPECTED_DIMENSIONS[kind]
    counts: list[int] = []
    source_hashes: list[str] = []
    source_paths: list[Path] = []
    # Both collections are checked against the complete ID count before this
    # function is called. Keep Python 3.9 compatibility (zip(strict=...) is 3.10+).
    for pdb_id, graph in zip(ids, graphs):
        path = embedding_dir / f"{pdb_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2 or len(array) != len(graph.x):
            raise ValueError(f"{kind} graph/embedding row mismatch for {pdb_id}: {array.shape}")
        if array.shape[1] != expected_dim or not np.isfinite(array).all():
            raise ValueError(f"Invalid {kind} embedding shape or values for {pdb_id}: {array.shape}")
        counts.append(int(len(array)))
        source_hashes.append(sha256(path))
        source_paths.append(path)

    offsets = np.cumsum([0, *counts], dtype=np.int64)
    destination = output_dir / f"{kind}.npy"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    bank = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(int(offsets[-1]), expected_dim)
    )
    for index, path in enumerate(source_paths):
        source = np.load(path, allow_pickle=False)
        bank[offsets[index] : offsets[index + 1]] = source
        if not np.array_equal(bank[offsets[index] : offsets[index + 1]], source):
            raise ValueError(f"Embedding precision changed while building {kind} bank: {path.name}")
    bank.flush()
    del bank
    temporary.replace(destination)
    return {
        "dimension": expected_dim,
        "counts": counts,
        "offsets": offsets.tolist(),
        "pool_sha256": sha256(destination),
        "source_sha256": source_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT,
                        help="Root containing data/train_data and local embedding directories")
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--protein-graphs", type=Path)
    parser.add_argument("--ligand-graphs", type=Path)
    parser.add_argument("--protein-embeddings-dir", type=Path)
    parser.add_argument("--ligand-embeddings-dir", type=Path)
    parser.add_argument("--casf-exclusion-file", type=Path,
                        default=REPOSITORY_ROOT / "BioLM_Score/data/casf_excluded_ids.txt")
    parser.add_argument("--target-group-map", type=Path,
                        default=REPOSITORY_ROOT / "reproducibility/identifiers/PDBbind_v2020_MMseqs2_100_80_target_groups.tsv")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to <project-root>/data/embedding_bank")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_root = project_root / "data"
    train_root = data_root / "train_data"
    ids_path = (args.ids_file or train_root / "v2020_train_ids.npy").resolve()
    protein_graph_path = (args.protein_graphs or train_root / "v2020_train_prot.pt").resolve()
    ligand_graph_path = (args.ligand_graphs or train_root / "v2020_train_lig.pt").resolve()
    protein_dir = (args.protein_embeddings_dir or data_root / "protein_embeddings").resolve()
    ligand_dir = (args.ligand_embeddings_dir or data_root / "ligand_embeddings").resolve()
    exclusion_path = args.casf_exclusion_file.resolve()
    group_path = args.target_group_map.resolve()
    output_dir = (args.output_dir or data_root / "embedding_bank").resolve()

    ids, labels = read_ids_and_labels(ids_path)
    excluded = {
        line.strip().lower()
        for line in exclusion_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if len(excluded) != EXPECTED_CASF_EXCLUSIONS or set(ids) & excluded:
        raise ValueError("CASF exclusion list must contain 285 IDs disjoint from the standard training IDs")
    target_map = read_target_groups(group_path)
    if set(ids) != set(target_map):
        missing = sorted(set(ids) - set(target_map))[:10]
        extra = sorted(set(target_map) - set(ids))[:10]
        raise ValueError(f"Target-group map does not match training IDs (missing={missing}, extra={extra})")

    protein_graphs = load_graphs(protein_graph_path, len(ids), "protein")
    ligand_graphs = load_graphs(ligand_graph_path, len(ids), "ligand")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "ids": ids,
        "labels": labels.tolist(),
        "protein": build_bank("protein", ids, protein_graphs, protein_dir, output_dir),
        "ligand": build_bank("ligand", ids, ligand_graphs, ligand_dir, output_dir),
        "target_groups": [target_map[pdb_id] for pdb_id in ids],
        "target_definition": "MMseqs2 100% identity and bidirectional coverage >=80%; not a protein-family holdout",
        "casf_excluded": sorted(excluded),
        "source_sha256": {
            "ids": sha256(ids_path),
            "protein_graphs": sha256(protein_graph_path),
            "ligand_graphs": sha256(ligand_graph_path),
            "casf_exclusion": sha256(exclusion_path),
            "target_group_map": sha256(group_path),
        },
    }
    atomic_json(output_dir / "metadata.json", metadata)
    print(json.dumps({
        "complete": True,
        "complexes": len(ids),
        "casf_exclusions": len(excluded),
        "protein_embedding_rows": int(metadata["protein"]["offsets"][-1]),
        "ligand_embedding_rows": int(metadata["ligand"]["offsets"][-1]),
        "target_groups": len(set(metadata["target_groups"])),
        "metadata_sha256": sha256(output_dir / "metadata.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
