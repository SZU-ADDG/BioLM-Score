#!/usr/bin/env python3
"""Prepare model-independent DEKOIS compound, pose-flow, and sequence metadata."""

from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets", nargs="+")
    return parser.parse_args()


def atomic_gzip_tsv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def canonical_properties(mol):
    # Some DEKOIS records pass the SDF supplier but still cannot be
    # kekulized during RemoveHs/MolToSmiles.  Treat those records as invalid
    # metadata instead of aborting the other ~95k compounds.
    base = Chem.RemoveHs(mol)
    canonical = Chem.MolToSmiles(base, canonical=True, isomericSmiles=True)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(base)
    scaffold = (
        Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=True)
        if scaffold_mol.GetNumAtoms()
        else ""
    )
    return (
        canonical,
        scaffold,
        int(Descriptors.HeavyAtomCount(base)),
        float(Descriptors.MolWt(base)),
        int(Lipinski.NumRotatableBonds(base)),
    )


def sdf_supplier(path: Path):
    if path.suffix == ".gz":
        handle = gzip.open(path, "rb")
        return handle, Chem.ForwardSDMolSupplier(handle, sanitize=True, removeHs=False)
    handle = path.open("rb")
    return handle, Chem.ForwardSDMolSupplier(handle, sanitize=True, removeHs=False)


def prepared_compound_id(pose_name: str) -> str:
    pieces = pose_name.rsplit("-", 1)
    return pieces[0] if len(pieces) == 2 and pieces[1].isdigit() else pose_name


def read_original(path: Path, target: str, label: int):
    rows = []
    invalid = 0
    handle, supplier = sdf_supplier(path)
    try:
        for row_index, mol in enumerate(supplier):
            if mol is None:
                invalid += 1
                continue
            name = (
                mol.GetProp("Name")
                if mol.HasProp("Name")
                else mol.GetProp("_Name") if mol.HasProp("_Name") else str(row_index)
            )
            try:
                canonical, scaffold, heavy, mw, rot = canonical_properties(mol)
            except Exception:
                invalid += 1
                continue
            rows.append(
                (target, str(name), label, canonical, scaffold, heavy, mw, rot)
            )
    finally:
        handle.close()
    return rows, invalid


def read_prepared(path: Path, target: str, label: int):
    compounds: OrderedDict[str, dict] = OrderedDict()
    invalid = 0
    handle, supplier = sdf_supplier(path)
    try:
        for row_index, mol in enumerate(supplier):
            if mol is None:
                invalid += 1
                continue
            pose_name = (
                mol.GetProp("_Name") if mol.HasProp("_Name") else str(row_index)
            )
            compound = prepared_compound_id(str(pose_name))
            if compound not in compounds:
                try:
                    canonical, scaffold, heavy, mw, rot = canonical_properties(mol)
                except Exception:
                    invalid += 1
                    continue
                compounds[compound] = {
                    "canonical": canonical,
                    "scaffold": scaffold,
                    "heavy": heavy,
                    "mw": mw,
                    "rot": rot,
                    "poses": 0,
                }
            compounds[compound]["poses"] += 1
    finally:
        handle.close()
    rows = [
        (
            target,
            compound,
            label,
            values["canonical"],
            values["scaffold"],
            values["heavy"],
            values["mw"],
            values["rot"],
            values["poses"],
        )
        for compound, values in compounds.items()
    ]
    return rows, invalid


def read_glide_csv(path: Path) -> tuple[int, set[str]]:
    frame = pd.read_csv(path, compression="bz2")
    ids = set(map(str, frame["id"].tolist()))
    return len(frame), ids


def extract_sequences(pdb_path: Path):
    chains: dict[str, OrderedDict[tuple[str, str], str]] = {}
    pdb_id = None
    title = []
    with pdb_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("REMARK   4") and pdb_id is None:
                fields = line.split()
                if len(fields) >= 3 and len(fields[2]) == 4:
                    pdb_id = fields[2].upper()
            if line.startswith("TITLE"):
                title.append(line[10:].strip())
            if not line.startswith("ATOM"):
                continue
            residue = line[17:20].strip().upper()
            aa = AA3_TO_1.get(residue)
            if aa is None:
                continue
            chain = line[21].strip() or "_"
            residue_key = (line[22:26].strip(), line[26].strip())
            chains.setdefault(chain, OrderedDict()).setdefault(residue_key, aa)
    sequences = {
        chain: "".join(residues.values())
        for chain, residues in chains.items()
        if len(residues) >= 20
    }
    return pdb_id, " ".join(title), sequences


def process_target(data_root: Path, output_root: Path, target: str) -> None:
    base = data_root / target
    original_rows = []
    prepared_rows = []
    flow = {
        "target": target,
        "classes": {},
    }
    for class_name, plural, label in (
        ("active", "actives", 1),
        ("decoy", "decoys", 0),
    ):
        original_path = base / f"{target}_{plural}.sdf.gz"
        prepared_dir = base / f"{target}_{plural}_SP"
        prepared_path = prepared_dir / f"{target}_{plural}_SP.sdf"
        glide_csv = prepared_dir / f"{target}_{plural}_SP.csv.bz2"
        rows_original, invalid_original = read_original(
            original_path, target, label
        )
        rows_prepared, invalid_prepared = read_prepared(
            prepared_path, target, label
        )
        csv_count, csv_ids = read_glide_csv(glide_csv)
        prepared_ids = {row[1] for row in rows_prepared}
        original_rows.extend(rows_original)
        prepared_rows.extend(rows_prepared)
        pose_counts = np.asarray([row[-1] for row in rows_prepared], dtype=float)
        flow["classes"][class_name] = {
            "original_valid_compounds": len(rows_original),
            "original_invalid_records": invalid_original,
            "glide_csv_compounds": csv_count,
            "prepared_valid_compounds": len(rows_prepared),
            "prepared_invalid_pose_records": invalid_prepared,
            "prepared_pose_count": int(pose_counts.sum()),
            "pose_count_mean": float(pose_counts.mean()),
            "pose_count_median": float(np.median(pose_counts)),
            "pose_count_min": int(pose_counts.min()),
            "pose_count_max": int(pose_counts.max()),
            "glide_csv_without_valid_pose_count": len(csv_ids - prepared_ids),
            "glide_csv_without_valid_pose_ids": sorted(csv_ids - prepared_ids),
        }

    atomic_gzip_tsv(
        output_root / "original_compounds" / f"{target}.tsv.gz",
        (
            "target", "compound_id", "label", "canonical_smiles",
            "murcko_scaffold", "heavy_atoms", "molecular_weight",
            "rotatable_bonds",
        ),
        original_rows,
    )
    atomic_gzip_tsv(
        output_root / "prepared_compounds" / f"{target}.tsv.gz",
        (
            "target", "compound_id", "label", "canonical_smiles",
            "murcko_scaffold", "heavy_atoms", "molecular_weight",
            "rotatable_bonds", "pose_count",
        ),
        prepared_rows,
    )
    protein_path = base / f"{target}_prot" / f"{target}_p.pdb"
    pdb_id, title, sequences = extract_sequences(protein_path)
    flow["template_pdb_id"] = pdb_id
    flow["template_title"] = title
    flow["protein_chains"] = {
        chain: {"length": len(sequence), "sequence": sequence}
        for chain, sequence in sequences.items()
    }
    atomic_json(output_root / "flow" / f"{target}.json", flow)
    fasta_path = output_root / "target_sequences" / f"{target}.fasta"
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(
        "".join(
            f">{target}|{chain}|{pdb_id or 'unknown'}\n{sequence}\n"
            for chain, sequence in sequences.items()
        )
    )


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    targets = args.targets or [
        path.name
        for path in sorted(args.data_root.iterdir())
        if path.is_dir()
    ]
    for index, target in enumerate(targets, 1):
        print(f"metadata target={index}/{len(targets)} {target}", flush=True)
        process_target(args.data_root, args.output_root, target)


if __name__ == "__main__":
    main()
