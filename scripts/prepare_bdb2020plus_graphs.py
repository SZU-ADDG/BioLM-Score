#!/usr/bin/env python3
"""Prepare BDB2020+ chain pockets and graphs with the BioLM-Score code path."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.spatial import cKDTree, distance_matrix
from torch_geometric.data import Data

import BioLM_Score.feats.mol2graph_rdmda_res as graph_features
from BioLM_Score.feats.mol2graph_rdmda_res import (
    load_mol,
    mol_to_graph,
    prot_to_graph,
)


AA_CODES = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "LYS": "K",
    "ILE": "I",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TYR": "Y",
    "TRP": "W",
    "SEC": "U",
    "ORN": "O",
}

_SPATIAL_RESIDUES = None
_SPATIAL_CUTOFF = None


def evaluate_spatial_residue_pair(pair):
    residue_i, residue_j = map(int, pair)
    distances = graph_features.calc_dist(
        _SPATIAL_RESIDUES[residue_i], _SPATIAL_RESIDUES[residue_j]
    )
    minimum = float(distances.min())
    if minimum > _SPATIAL_CUTOFF:
        return None
    return (
        residue_i,
        residue_j,
        minimum * 0.1,
        float(distances.max()) * 0.1,
    )


def evaluate_residue_feature(index):
    return graph_features.calc_res_features(_SPATIAL_RESIDUES[index])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def ligand_coordinates(path: Path) -> np.ndarray:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mol = next((item for item in supplier if item is not None), None)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Could not read 3D ligand: {path}")
    return np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)


def protein_atom_records(path: Path):
    records = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            xyz = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float64,
            )
        except ValueError:
            continue
        records.append((line, line[21:22], xyz))
    if not records:
        raise ValueError(f"No ATOM records in {path}")
    return records


def select_contact_chains(protein_path: Path, ligand_path: Path, cutoff: float):
    records = protein_atom_records(protein_path)
    lig_xyz = ligand_coordinates(ligand_path)
    chain_min_sq: dict[str, float] = {}
    block = 2048
    for start in range(0, len(records), block):
        xyz = np.stack([item[2] for item in records[start : start + block]])
        min_sq = ((xyz[:, None, :] - lig_xyz[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        for item, distance_sq in zip(records[start : start + block], min_sq):
            chain = item[1]
            chain_min_sq[chain] = min(chain_min_sq.get(chain, np.inf), float(distance_sq))
    selected = [chain for chain, dist_sq in chain_min_sq.items() if dist_sq <= cutoff**2]
    fallback = False
    if not selected:
        selected = [min(chain_min_sq, key=chain_min_sq.get)]
        fallback = True
    ordered = list(OrderedDict((item[1], None) for item in records))
    selected = [chain for chain in ordered if chain in selected]
    return records, selected, fallback, {
        chain if chain.strip() else "<blank>": float(np.sqrt(chain_min_sq[chain]))
        for chain in ordered
    }


def write_chain_pocket(
    protein_path: Path, ligand_path: Path, output_path: Path, cutoff: float
):
    records, selected, fallback, chain_minimum_distances = select_contact_chains(
        protein_path, ligand_path, cutoff
    )
    selected_set = set(selected)
    output_lines = [line for line, chain, _ in records if chain in selected_set]
    output_lines.extend(["TER", "END", ""])
    atomic_write_text(output_path, "\n".join(output_lines))
    return selected, fallback, chain_minimum_distances


def residue_records(mol):
    universe = mda.Universe(mol)
    records = []
    for residue in universe.residues:
        resname = str(residue.resname).strip()
        if resname in {"HOH", "NA"}:
            continue
        chain = str(getattr(residue, "segid", "")).strip()
        resid = int(residue.resid)
        icode = str(getattr(residue, "icode", "")).strip()
        records.append(
            {
                "chain": chain,
                "resid": resid,
                "icode": icode,
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


def canonical_ligand_graph(path: Path):
    mol = load_mol(str(path), explicit_H=False, use_chirality=True)
    if mol is None:
        raise ValueError(f"Could not load ligand: {path}")
    smiles = Chem.MolToSmiles(mol, canonical=True)
    order = list(
        mol.GetPropsAsDict(includePrivate=True, includeComputed=True)[
            "_smilesAtomOutputOrder"
        ]
    )
    reordered = Chem.RenumberAtoms(mol, order)
    graph = mol_to_graph(reordered, explicit_H=False, use_chirality=True)
    return graph, smiles, order


def prot_to_graph_exact_spatial(prot, cutoff: float, workers: int = 1):
    """Exact original residue graph with a spatial candidate prefilter.

    The original implementation evaluates every ordered residue pair. This
    implementation first finds residue pairs having at least one atom pair
    within the same cutoff, then computes all original edge attributes for
    those pairs in the original lexicographic edge order.
    """
    universe = mda.Universe(prot)
    residues = universe.residues
    n_residues = len(residues)
    global _SPATIAL_RESIDUES, _SPATIAL_CUTOFF
    _SPATIAL_RESIDUES = residues
    _SPATIAL_CUTOFF = cutoff
    if workers > 1:
        with mp.get_context("fork").Pool(processes=workers) as pool:
            residue_features = np.asarray(
                pool.map(evaluate_residue_feature, range(n_residues), chunksize=32)
            )
    else:
        residue_features = np.asarray(
            [graph_features.calc_res_features(residue) for residue in residues]
        )

    residue_atom_positions = [
        np.asarray(residue.atoms.positions, dtype=np.float64)
        for residue in residues
    ]
    geometric_centers = np.asarray(
        [positions.mean(axis=0) for positions in residue_atom_positions]
    )
    radii = np.asarray(
        [
            np.linalg.norm(positions - center, axis=1).max()
            for positions, center in zip(residue_atom_positions, geometric_centers)
        ]
    )
    coarse_pairs = cKDTree(geometric_centers).query_pairs(
        cutoff + 2.0 * float(radii.max()), output_type="ndarray"
    )
    neighbors = [set() for _ in range(n_residues)]
    eligible_pairs = []
    for residue_i, residue_j in coarse_pairs:
        center_distance = np.linalg.norm(
            geometric_centers[residue_i] - geometric_centers[residue_j]
        )
        if center_distance > cutoff + radii[residue_i] + radii[residue_j]:
            continue
        eligible_pairs.append((int(residue_i), int(residue_j)))

    if workers > 1:
        with mp.get_context("fork").Pool(processes=workers) as pool:
            evaluated_pairs = pool.map(
                evaluate_spatial_residue_pair, eligible_pairs, chunksize=256
            )
    else:
        evaluated_pairs = [
            evaluate_spatial_residue_pair(pair) for pair in eligible_pairs
        ]
    extrema_by_pair = {}
    for result in evaluated_pairs:
        if result is None:
            continue
        residue_i, residue_j, minimum, maximum = result
        extrema_by_pair[(residue_i, residue_j)] = (minimum, maximum)
        neighbors[residue_i].add(residue_j)
        neighbors[residue_j].add(residue_i)
    edge_ids = [
        (residue_i, residue_j)
        for residue_i in range(n_residues)
        for residue_j in sorted(neighbors[residue_i])
    ]
    if not edge_ids:
        raise ValueError("Protein residue graph has no edges")

    distance_extrema = []
    for residue_i, residue_j in edge_ids:
        pair = (min(residue_i, residue_j), max(residue_i, residue_j))
        distance_extrema.append(extrema_by_pair[pair])
    distance_extrema = np.asarray(distance_extrema)

    ca_positions = torch.tensor(
        np.asarray([graph_features.obtain_ca_pos(residue) for residue in residues])
    )
    center_positions = torch.tensor(
        universe.atoms.center_of_mass(compound="residues")
    )
    ca_matrix = distance_matrix(ca_positions, ca_positions)
    center_matrix = distance_matrix(center_positions, center_positions)
    ca_distance = torch.tensor(
        [ca_matrix[i, j] for i, j in edge_ids]
    ) * 0.1
    center_distance = torch.tensor(
        [center_matrix[i, j] for i, j in edge_ids]
    ) * 0.1
    # check_connect is equivalent to testing whether exactly one bond crosses
    # an adjacent-residue boundary. Precomputing the crossing counts avoids
    # repeatedly querying the full bond topology for both directed edges.
    crossing_bonds = {}
    atom_residue_indices = np.asarray(universe.atoms.resindices, dtype=np.int64)
    for atom_i, atom_j in np.asarray(universe.bonds.indices, dtype=np.int64):
        residue_i = int(atom_residue_indices[atom_i])
        residue_j = int(atom_residue_indices[atom_j])
        if abs(residue_i - residue_j) != 1:
            continue
        pair = (min(residue_i, residue_j), max(residue_i, residue_j))
        crossing_bonds[pair] = crossing_bonds.get(pair, 0) + 1
    connectivity_values = np.asarray(
        [
            int(
                abs(residue_i - residue_j) == 1
                and crossing_bonds.get(
                    (min(residue_i, residue_j), max(residue_i, residue_j)), 0
                )
                == 1
            )
            for residue_i, residue_j in edge_ids
        ],
        dtype=np.int64,
    )
    connectivity = torch.tensor(connectivity_values)
    edge_features = torch.cat(
        [
            connectivity.view(-1, 1),
            ca_distance.view(-1, 1),
            center_distance.view(-1, 1),
            torch.tensor(distance_extrema),
        ],
        dim=1,
    )
    residue_coordinates = torch.tensor(
        np.asarray(
            [
                np.concatenate(
                    [
                        residue.atoms.positions,
                        np.full(
                            (
                                graph_features.RES_MAX_NATOMS
                                - len(residue.atoms),
                                3,
                            ),
                            np.nan,
                        ),
                    ],
                    axis=0,
                )
                for residue in residues
            ]
        )
    )
    source, destination = zip(*edge_ids)
    return Data(
        x=torch.tensor(residue_features, dtype=torch.float),
        edge_index=torch.tensor([source, destination]),
        pos=residue_coordinates,
        edge_attr=torch.tensor(np.asarray(edge_features), dtype=torch.float),
    )


def summary_row(metadata):
    return {
        key: metadata[key]
        for key in (
            "pdb_id",
            "pKa",
            "delta_g_kcal_mol",
            "selected_chains",
            "chain_selection_fallback",
            "full_sequence_length",
            "pocket_residue_count",
            "ligand_atom_count",
        )
    }


def finalize(output: Path, cutoff: float) -> None:
    metadata_paths = sorted((output / "features" / "metadata").glob("*.json"))
    if len(metadata_paths) != 115:
        raise AssertionError(
            f"Cannot finalize incomplete metadata coverage: {len(metadata_paths)}/115"
        )
    summary = [
        summary_row(json.loads(path.read_text(encoding="utf-8")))
        for path in metadata_paths
    ]
    targets_path = output / "features" / "targets.json"
    atomic_write_text(targets_path, json.dumps(summary, indent=2) + "\n")
    complete = {
        "complete": True,
        "n_targets": len(summary),
        "cutoff_angstrom": cutoff,
        "chain_definition": (
            "all protein ATOM records in chains with any atom within 10 A of ligand"
        ),
        "canonical_ligand_atom_renumbering": True,
        "targets_sha256": sha256(targets_path),
    }
    atomic_write_text(
        output / "audit" / "structure_graph_preparation_complete.json",
        json.dumps(complete, indent=2) + "\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdb-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")

    output = args.output.resolve()
    pocket_dir = output / "features" / "pockets"
    graph_dir = output / "features" / "graphs"
    metadata_dir = output / "features" / "metadata"
    for directory in (pocket_dir, graph_dir, metadata_dir, output / "audit"):
        directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.bdb_root / "BDB2020+.csv")
    if "accurate" in frame:
        accurate = frame["accurate"].astype(str).str.lower().isin({"true", "1"})
        frame = frame[accurate].copy()
    frame["pdbid"] = frame["pdbid"].astype(str).str.upper()
    frame = frame.sort_values("pdbid").reset_index(drop=True)
    if len(frame) != 115:
        raise ValueError(f"Expected 115 accurate BDB2020+ entries, found {len(frame)}")
    if args.finalize_only:
        finalize(output, args.cutoff)
        return

    summary = []
    shard_frame = frame.iloc[args.shard_index :: args.shard_count]
    for row in shard_frame.itertuples(index=False):
        pdb_id = row.pdbid
        complex_dir = args.bdb_root / "dataset" / pdb_id
        protein_path = complex_dir / "protein.pdb"
        ligand_path = complex_dir / "ligand.sdf"
        pocket_path = pocket_dir / f"{pdb_id}_pocket_chain_10A.pdb"
        protein_graph_path = graph_dir / f"{pdb_id}_prot.pt"
        ligand_graph_path = graph_dir / f"{pdb_id}_lig.pt"
        metadata_path = metadata_dir / f"{pdb_id}.json"
        if (
            pocket_path.exists()
            and protein_graph_path.exists()
            and ligand_graph_path.exists()
            and metadata_path.exists()
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            summary.append(summary_row(metadata))
            print(
                f"{len(summary):03d}/{len(shard_frame):03d} {pdb_id} reused",
                flush=True,
            )
            continue

        selected_chains, fallback, chain_minimum_distances = write_chain_pocket(
            protein_path, ligand_path, pocket_path, args.cutoff
        )
        full_mol = load_mol(str(protein_path), explicit_H=False, use_chirality=True)
        pocket_mol = load_mol(str(pocket_path), explicit_H=False, use_chirality=True)
        if full_mol is None or pocket_mol is None:
            raise ValueError(f"Protein parsing failed for {pdb_id}")
        full_residues = residue_records(full_mol)
        pocket_residues = residue_records(pocket_mol)
        kept_indices = map_pocket_to_full(full_residues, pocket_residues)
        full_sequence = "".join(record["aa"] for record in full_residues)

        if len(pocket_residues) > 1000:
            protein_graph = prot_to_graph_exact_spatial(
                pocket_mol, args.cutoff, workers=16
            )
        else:
            protein_graph = prot_to_graph(pocket_mol, args.cutoff)
        ligand_graph, canonical_smiles, atom_order = canonical_ligand_graph(ligand_path)
        if protein_graph.x.shape[0] != len(pocket_residues):
            raise AssertionError(
                f"{pdb_id}: protein graph/residue mismatch "
                f"{protein_graph.x.shape[0]} != {len(pocket_residues)}"
            )
        if ligand_graph.x.shape[0] != len(atom_order):
            raise AssertionError(
                f"{pdb_id}: ligand graph/order mismatch "
                f"{ligand_graph.x.shape[0]} != {len(atom_order)}"
            )
        atomic_torch_save(protein_graph_path, protein_graph)
        atomic_torch_save(ligand_graph_path, ligand_graph)

        metadata = {
            "pdb_id": pdb_id,
            "pKa": float(row.pKa),
            "delta_g_kcal_mol": float(-1.987204e-3 * 298 * np.log(10) * row.pKa),
            "protein_path": str(protein_path),
            "ligand_path": str(ligand_path),
            "pocket_path": str(pocket_path),
            "protein_graph_path": str(protein_graph_path),
            "ligand_graph_path": str(ligand_graph_path),
            "selected_chains": selected_chains,
            "chain_selection_fallback": fallback,
            "chain_minimum_distances_angstrom": chain_minimum_distances,
            "full_sequence": full_sequence,
            "full_sequence_length": len(full_sequence),
            "pocket_residue_count": len(pocket_residues),
            "pocket_full_sequence_indices": kept_indices,
            "canonical_smiles": canonical_smiles,
            "canonical_atom_order": atom_order,
            "ligand_atom_count": int(ligand_graph.x.shape[0]),
            "source_sha256": {
                "protein": sha256(protein_path),
                "ligand": sha256(ligand_path),
                "pocket": sha256(pocket_path),
            },
        }
        atomic_write_text(metadata_path, json.dumps(metadata, indent=2) + "\n")
        summary.append(summary_row(metadata))
        print(
            f"{len(summary):03d}/{len(shard_frame):03d} {pdb_id} "
            f"sequence={len(full_sequence)} pocket={len(pocket_residues)} "
            f"ligand={ligand_graph.x.shape[0]}",
            flush=True,
        )

    shard_status = {
        "complete": True,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "n_targets": len(summary),
        "pdb_ids": [item["pdb_id"] for item in summary],
    }
    atomic_write_text(
        output / "audit" / f"structure_graph_shard_{args.shard_index}.json",
        json.dumps(shard_status, indent=2) + "\n",
    )
    if args.shard_count == 1:
        finalize(output, args.cutoff)


if __name__ == "__main__":
    main()
