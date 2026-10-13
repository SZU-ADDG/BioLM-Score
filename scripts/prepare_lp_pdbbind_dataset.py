#!/usr/bin/env python3
"""Prepare the exact LP-PDBBind paper training/validation dataset.

The paper uses non-covalent CL1 records from ``new_split=train`` for
training and non-covalent CL2 records from ``new_split=val`` for model
selection.  Existing PDBBind-v2020 graphs/embeddings are reused only after
shape validation; missing assets are generated with the project's original
10 A graph and canonical Chemformer atom-order conventions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import AllChem

from BioLM_Score.feats.mol2graph_rdmda_res import load_mol, mol_to_graph, prot_to_graph


AA_CODES = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "LYS": "K", "ILE": "I", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TYR": "Y", "TRP": "W",
    "SEC": "U", "ORN": "O",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_numpy(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def selected_frames(csv_path: Path):
    frame = pd.read_csv(csv_path, index_col=0)
    frame.index = frame.index.astype(str).str.lower()
    covalent = frame["covalent"].astype(bool)
    train = frame[
        (frame["new_split"] == "train") & frame["CL1"].astype(bool) & ~covalent
    ].copy()
    validation = frame[
        (frame["new_split"] == "val") & frame["CL2"].astype(bool) & ~covalent
    ].copy()
    test = frame[
        (frame["new_split"] == "test") & frame["CL2"].astype(bool) & ~covalent
    ].copy()
    if (len(train), len(validation), len(test)) != (7393, 958, 2171):
        raise AssertionError(
            f"Unexpected official LP-PDBBind counts: "
            f"{len(train)}/{len(validation)}/{len(test)}"
        )
    if set(train.index) & set(validation.index):
        raise AssertionError("LP train/validation overlap")
    for name, selected in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        values = pd.to_numeric(selected["value"], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Non-finite affinity in {name}")
    return frame, train, validation, test


def residue_records(path: Path):
    records = []
    seen = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip()
        # Waters are not graph nodes.  Other non-amino-acid residues,
        # including metal ions such as NA, are retained by prot_to_graph and
        # therefore must occupy a matching placeholder row in the ESM tensor;
        # AA_CODES.get(..., "X") supplies that placeholder.
        if resname == "HOH":
            continue
        chain = line[21:22].strip()
        resid_text = line[22:26].strip()
        if not resid_text:
            continue
        key = (chain, int(resid_text), line[26:27].strip(), resname)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "chain": key[0],
                "resid": key[1],
                "icode": key[2],
                "resname": resname,
                "aa": AA_CODES.get(resname, "X"),
            }
        )
    return records


def graph_residue_records(path: Path):
    """Return residues exactly as retained by the deployed graph parser."""
    molecule = load_mol(str(path), explicit_H=False, use_chirality=True)
    if molecule is None:
        raise ValueError(f"Could not parse graph pocket: {path}")
    universe = mda.Universe(molecule)
    records = []
    for residue in universe.residues:
        atom_index = int(residue.atoms.indices[0])
        atom = molecule.GetAtomWithIdx(atom_index)
        info = atom.GetPDBResidueInfo()
        if info is None:
            raise ValueError(f"Pocket atom lacks PDB residue metadata: {path}")
        resname = info.GetResidueName().strip()
        key = (
            info.GetChainId().strip(),
            int(info.GetResidueNumber()),
            info.GetInsertionCode().strip(),
            resname,
        )
        records.append(
            {
                "chain": key[0],
                "resid": key[1],
                "icode": key[2],
                "resname": resname,
                "aa": AA_CODES.get(resname, "X"),
            }
        )
    return records


def map_pocket_to_full(full_records, pocket_records):
    buckets: dict[tuple, list[int]] = {}
    for index, record in enumerate(full_records):
        key = (
            record["chain"],
            record["resid"],
            record["icode"],
            record["resname"],
        )
        buckets.setdefault(key, []).append(index)
    used: dict[tuple, int] = {}
    indices = []
    for record in pocket_records:
        key = (
            record["chain"],
            record["resid"],
            record["icode"],
            record["resname"],
        )
        occurrence = used.get(key, 0)
        choices = buckets.get(key, [])
        if occurrence >= len(choices):
            raise ValueError(f"Pocket residue absent from full protein: {key}")
        indices.append(choices[occurrence])
        used[key] = occurrence + 1
    return indices


def canonical_ligand(path: Path, template_smiles: Optional[str] = None):
    mol = Chem.MolFromMolFile(str(path), removeHs=True)
    if mol is None:
        raise ValueError(f"Could not parse ligand: {path}")
    bond_order_repaired = False
    if any(
        bond.GetBondType()
        not in {
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC,
        }
        for bond in mol.GetBonds()
    ):
        if template_smiles:
            template = Chem.MolFromSmiles(str(template_smiles))
            if template is None:
                raise ValueError(f"Invalid official SMILES template for {path}")
            if template.GetNumAtoms() != mol.GetNumAtoms():
                raise ValueError(
                    f"Official SMILES/coordinate atom-count mismatch for {path}: "
                    f"{template.GetNumAtoms()} vs {mol.GetNumAtoms()}"
                )
            # RDKit transfers bond orders and formal charges while retaining
            # the coordinate molecule's conformer.  This is the same
            # template-based correction used by the official LP-PDBBind BDB
            # preparation notebook.
            mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
            # A small number of official SMILES use the generic ``~`` bond
            # (for example, a boronate coordinate bond in 4hze).  The
            # project's graph featurizer and RDKit CIP implementation require
            # an integer bond order.  Preserve connectivity and coordinates
            # and encode any remaining generic bond as SINGLE, the only
            # representation supported by the deployed GNN edge vocabulary.
            for bond in mol.GetBonds():
                if bond.GetBondType() not in {
                    Chem.BondType.SINGLE,
                    Chem.BondType.DOUBLE,
                    Chem.BondType.TRIPLE,
                    Chem.BondType.AROMATIC,
                }:
                    bond.SetBondType(Chem.BondType.SINGLE)
            bond_order_repaired = True
    smiles = Chem.MolToSmiles(mol, canonical=True)
    order = list(
        mol.GetPropsAsDict(includePrivate=True, includeComputed=True)[
            "_smilesAtomOutputOrder"
        ]
    )
    if sorted(order) != list(range(mol.GetNumAtoms())):
        raise AssertionError(f"Non-bijective canonical atom order: {path}")
    return mol, smiles, order, bond_order_repaired


def paths_for_id(structures: list[Path], pdb_id: str):
    directory = next(
        (root / pdb_id for root in structures if (root / pdb_id).is_dir()),
        None,
    )
    if directory is None:
        raise FileNotFoundError(
            f"No structure directory for {pdb_id} in "
            f"{[str(root) for root in structures]}"
        )
    return {
        "full": directory / f"{pdb_id}_protein.pdb",
        "pocket": directory / f"{pdb_id}_pocket_chain_10A.pdb",
        "ligand": directory / f"{pdb_id}_ligand_sort.sdf",
    }


def create_embedding_metadata(args, train, validation) -> dict:
    selected_ids = list(train.index) + list(validation.index)
    original_ids_array = np.load(
        args.original_data / "v2020_train_ids.npy", allow_pickle=True
    )
    original_graph_ids = {
        str(value).lower() for value in original_ids_array[0]
    }
    missing_graphs = [
        pdb_id for pdb_id in selected_ids if pdb_id not in original_graph_ids
    ]
    missing_protein = [
        pdb_id
        for pdb_id in selected_ids
        if not (args.base_protein_embeddings / f"{pdb_id}.npy").exists()
    ]
    missing_ligand = [
        pdb_id
        for pdb_id in selected_ids
        if not (args.base_ligand_embeddings / f"{pdb_id}.npy").exists()
    ]
    # Any sample whose graph must be generated from the current LP structure
    # receives fresh graph-aligned embeddings as well.  Existing embeddings
    # for such records may have been derived from a different pocket residue
    # selection (2ylc is a verified 53-vs-52 example).
    missing_union = sorted(
        set(missing_protein) | set(missing_ligand) | set(missing_graphs)
    )
    metadata_dir = args.output / "features" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for pdb_id in missing_union:
        paths = paths_for_id(args.structures, pdb_id)
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)
        full_residues = residue_records(paths["full"])
        # The raw PDB may contain residues that RDKit drops during the exact
        # load_mol -> MDAnalysis graph path.  Derive pocket rows from that same
        # parsed molecule so ESM rows and graph nodes are guaranteed to agree.
        pocket_residues = graph_residue_records(paths["pocket"])
        kept = map_pocket_to_full(full_residues, pocket_residues)
        mol, smiles, order, _ = canonical_ligand(paths["ligand"])
        metadata = {
            "pdb_id": pdb_id,
            "protein_path": str(paths["full"]),
            "pocket_path": str(paths["pocket"]),
            "ligand_path": str(paths["ligand"]),
            "full_sequence": "".join(record["aa"] for record in full_residues),
            "full_sequence_length": len(full_residues),
            "pocket_residue_count": len(pocket_residues),
            "pocket_full_sequence_indices": kept,
            "canonical_smiles": smiles,
            "canonical_atom_order": order,
            "ligand_atom_count": mol.GetNumAtoms(),
            "needs_protein_embedding": (
                pdb_id in missing_protein or pdb_id in missing_graphs
            ),
            "needs_ligand_embedding": (
                pdb_id in missing_ligand or pdb_id in missing_graphs
            ),
            "source_sha256": {key: sha256(path) for key, path in paths.items()},
        }
        atomic_json(metadata_dir / f"{pdb_id}.json", metadata)
    manifest = {
        "complete": True,
        "official_selection": {
            "train": "new_split=train AND CL1 AND non-covalent",
            "validation": "new_split=val AND CL2 AND non-covalent",
            "test": "new_split=test AND CL2 AND non-covalent",
        },
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": 2171,
            "missing_protein_embeddings": len(missing_protein),
            "missing_ligand_embeddings": len(missing_ligand),
            "missing_original_graphs": len(missing_graphs),
            "missing_embedding_union": len(missing_union),
        },
        "missing_original_graphs": missing_graphs,
        "missing_protein_embeddings": missing_protein,
        "missing_ligand_embeddings": missing_ligand,
        "metadata_ids": missing_union,
        "lp_csv": str(args.lp_csv),
        "lp_csv_sha256": sha256(args.lp_csv),
    }
    atomic_json(args.output / "audit" / "lp_selection_and_missing_assets.json", manifest)
    return manifest


def ensure_symlink(destination: Path, source: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        if sha256(destination) == sha256(source):
            return
        raise FileExistsError(f"Conflicting embedding output: {destination}")
    os.symlink(source.resolve(), destination)


def build_new_graphs(pdb_id: str, structures: Path, official_smiles: str):
    paths = paths_for_id(structures, pdb_id)
    protein_mol = load_mol(str(paths["pocket"]), explicit_H=False, use_chirality=True)
    ligand_mol, _, order, bond_order_repaired = canonical_ligand(
        paths["ligand"], template_smiles=official_smiles
    )
    if protein_mol is None or ligand_mol is None:
        raise ValueError(f"Structure parsing failed: {pdb_id}")
    protein_graph = prot_to_graph(protein_mol, 10.0)
    canonical_mol = Chem.RenumberAtoms(ligand_mol, order)
    ligand_graph = mol_to_graph(
        canonical_mol, explicit_H=False, use_chirality=True
    )
    return protein_graph, ligand_graph, bond_order_repaired


def validate_embedding_pair(
    *,
    pdb_id: str,
    protein_nodes: int,
    ligand_nodes: int,
    graph_source: str,
    generated_protein: Path,
    generated_ligand: Path,
    output_protein: Path,
    output_ligand: Path,
    base_protein_embeddings: Path,
    base_ligand_embeddings: Path,
):
    """Validate and link one embedding pair.

    This is deliberately independent per complex so the expensive file reads,
    finite-value scans, and provenance hashes can run concurrently without
    changing graph construction, sample ordering, or any scientific input.
    """
    protein_source = (
        generated_protein / f"{pdb_id}.npy"
        if (generated_protein / f"{pdb_id}.npy").exists()
        else base_protein_embeddings / f"{pdb_id}.npy"
    )
    ligand_source = (
        generated_ligand / f"{pdb_id}.npy"
        if (generated_ligand / f"{pdb_id}.npy").exists()
        else base_ligand_embeddings / f"{pdb_id}.npy"
    )
    for path in (protein_source, ligand_source):
        if not path.exists():
            raise FileNotFoundError(path)
    protein_embedding = np.load(protein_source, mmap_mode="r")
    ligand_embedding = np.load(ligand_source, mmap_mode="r")
    expected_protein = (protein_nodes, 1152)
    expected_ligand = (ligand_nodes, 1024)
    if protein_embedding.shape != expected_protein:
        raise AssertionError(
            f"{pdb_id}: protein graph/embedding mismatch "
            f"{expected_protein} vs {protein_embedding.shape}"
        )
    if ligand_embedding.shape != expected_ligand:
        raise AssertionError(
            f"{pdb_id}: ligand graph/embedding mismatch "
            f"{expected_ligand} vs {ligand_embedding.shape}"
        )
    if not np.isfinite(protein_embedding).all() or not np.isfinite(ligand_embedding).all():
        raise FloatingPointError(f"{pdb_id}: non-finite embedding")
    ensure_symlink(output_protein / f"{pdb_id}.npy", protein_source)
    ensure_symlink(output_ligand / f"{pdb_id}.npy", ligand_source)
    return {
        "pdb_id": pdb_id,
        "graph_source": graph_source,
        "protein_nodes": protein_nodes,
        "ligand_nodes": ligand_nodes,
        "protein_embedding": str(protein_source),
        "ligand_embedding": str(ligand_source),
        "protein_embedding_sha256": sha256(protein_source),
        "ligand_embedding_sha256": sha256(ligand_source),
    }


def finalize_dataset(args, train, validation) -> None:
    generated_protein = args.output / "features" / "protein_embeddings"
    generated_ligand = args.output / "features" / "ligand_embeddings"
    output_protein = args.output / "data" / "protein_embeddings"
    output_ligand = args.output / "data" / "ligand_embeddings"
    selected = pd.concat([train, validation])
    selected_ids = list(selected.index)

    original_ids_array = np.load(args.original_data / "v2020_train_ids.npy", allow_pickle=True)
    original_ids = [str(value).lower() for value in original_ids_array[0]]
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("Duplicate IDs in original graph tensors")
    original_lookup = {pdb_id: index for index, pdb_id in enumerate(original_ids)}
    original_ligands = torch.load(args.original_data / "v2020_train_lig.pt")
    original_proteins = torch.load(args.original_data / "v2020_train_prot.pt")
    if not (
        len(original_ids) == len(original_ligands) == len(original_proteins)
    ):
        raise AssertionError("Original tensor lengths disagree")

    protein_graphs = []
    ligand_graphs = []
    validation_jobs = []
    generated_graph_ids = []
    bond_order_repaired_ids = []
    for number, pdb_id in enumerate(selected_ids, start=1):
        if pdb_id in original_lookup:
            index = original_lookup[pdb_id]
            protein_graph = original_proteins[index]
            ligand_graph = original_ligands[index]
            graph_source = "reused_original"
        else:
            print(f"generating missing graph {pdb_id}", flush=True)
            try:
                protein_graph, ligand_graph, bond_order_repaired = build_new_graphs(
                    pdb_id,
                    args.structures,
                    str(selected.loc[pdb_id, "smiles"]),
                )
            except Exception as error:
                raise RuntimeError(
                    f"Failed to generate missing graph for {pdb_id}"
                ) from error
            generated_graph_ids.append(pdb_id)
            if bond_order_repaired:
                bond_order_repaired_ids.append(pdb_id)
            graph_source = "generated_project_original_10A"
        protein_graphs.append(protein_graph)
        ligand_graphs.append(ligand_graph)
        validation_jobs.append(
            {
                "pdb_id": pdb_id,
                "protein_nodes": int(protein_graph.x.shape[0]),
                "ligand_nodes": int(ligand_graph.x.shape[0]),
                "graph_source": graph_source,
            }
        )
        if number % 500 == 0 or number == len(selected_ids):
            print(
                f"assembled graphs {number:04d}/{len(selected_ids):04d}",
                flush=True,
            )

    workers = max(1, int(os.environ.get("LP_ASSET_WORKERS", "16")))
    print(
        f"validating {len(validation_jobs)} embedding pairs with {workers} threads",
        flush=True,
    )
    validation_rows = [None] * len(validation_jobs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                validate_embedding_pair,
                **job,
                generated_protein=generated_protein,
                generated_ligand=generated_ligand,
                output_protein=output_protein,
                output_ligand=output_ligand,
                base_protein_embeddings=args.base_protein_embeddings,
                base_ligand_embeddings=args.base_ligand_embeddings,
            ): index
            for index, job in enumerate(validation_jobs)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            validation_rows[index] = future.result()
            if completed % 100 == 0 or completed == len(validation_jobs):
                print(
                    f"validated embeddings {completed:04d}/{len(validation_jobs):04d}",
                    flush=True,
                )

    data_dir = args.output / "data" / "train_data"
    labels = pd.to_numeric(selected["value"]).to_numpy(dtype=np.float64)
    ids_and_labels = np.asarray(
        [selected_ids, [repr(float(value)) for value in labels]], dtype=str
    )
    atomic_numpy(data_dir / "v2020_train_ids.npy", ids_and_labels)
    atomic_torch(data_dir / "v2020_train_prot.pt", protein_graphs)
    atomic_torch(data_dir / "v2020_train_lig.pt", ligand_graphs)
    atomic_text(args.output / "data" / "train_ids.txt", "\n".join(train.index) + "\n")
    atomic_text(
        args.output / "data" / "val_ids.txt", "\n".join(validation.index) + "\n"
    )
    del original_ligands, original_proteins

    audit = {
        "complete": True,
        "counts": {
            "dataset": len(selected_ids),
            "train": len(train),
            "validation": len(validation),
            "reused_graphs": len(selected_ids) - len(generated_graph_ids),
            "generated_graphs": len(generated_graph_ids),
        },
        "generated_graph_ids": generated_graph_ids,
        "official_smiles_bond_order_repaired_ids": bond_order_repaired_ids,
        "data_files": {
            path.name: {"path": str(path), "sha256": sha256(path)}
            for path in (
                data_dir / "v2020_train_ids.npy",
                data_dir / "v2020_train_prot.pt",
                data_dir / "v2020_train_lig.pt",
                args.output / "data" / "train_ids.txt",
                args.output / "data" / "val_ids.txt",
            )
        },
        "embedding_validation": validation_rows,
    }
    atomic_json(args.output / "audit" / "lp_training_assets_complete.json", audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("metadata", "finalize"), required=True)
    parser.add_argument("--lp-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--structures", required=True, type=Path, nargs="+")
    parser.add_argument("--original-data", required=True, type=Path)
    parser.add_argument("--base-protein-embeddings", required=True, type=Path)
    parser.add_argument("--base-ligand-embeddings", required=True, type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.lp_csv = args.lp_csv.resolve()
    args.structures = [path.resolve() for path in args.structures]
    args.original_data = args.original_data.resolve()
    args.base_protein_embeddings = args.base_protein_embeddings.resolve()
    args.base_ligand_embeddings = args.base_ligand_embeddings.resolve()
    for directory in (
        args.output / "audit",
        args.output / "features" / "metadata",
        args.output / "features" / "protein_embeddings",
        args.output / "features" / "ligand_embeddings",
        args.output / "data",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _, train, validation, _ = selected_frames(args.lp_csv)
    if args.phase == "metadata":
        create_embedding_metadata(args, train, validation)
    else:
        finalize_dataset(args, train, validation)


if __name__ == "__main__":
    main()
