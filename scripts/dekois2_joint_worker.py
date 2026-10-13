#!/usr/bin/env python3
"""Score prepared DEKOIS 2.0x Glide-SP poses with joint GatedGCN checkpoints.

This worker preserves the project's original DEKOIS graph construction and MDN
probability-sum score.  Operational changes are limited to deterministic target
sharding, single-threaded execution, atomic compressed outputs, and sequential
checkpoint loading so several variants can reuse one parsed target.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch as th
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch_geometric.graphgym.register import layer_dict
from torch_scatter import scatter_add


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    model_type: str
    ablation: str
    checkpoint: Path


ABLATION_CONFIGS = {
    "full": {"use_protein_lm": True, "use_ligand_lm": True, "use_gnn": True},
    "no_protein_lm": {
        "use_protein_lm": False,
        "use_ligand_lm": True,
        "use_gnn": True,
    },
    "no_ligand_lm": {
        "use_protein_lm": True,
        "use_ligand_lm": False,
        "use_gnn": True,
    },
    "no_gnn": {"use_protein_lm": True, "use_ligand_lm": True, "use_gnn": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=("active", "decoy"),
        default=("active", "decoy"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--graph-workers",
        type=int,
        default=4,
        help="Bounded replacement for data_dekois.py's n_jobs=-1 graph builder.",
    )
    parser.add_argument("--score-cutoff", type=float, default=5.0)
    parser.add_argument("--graph-cutoff", type=float, default=10.0)
    parser.add_argument("--limit-poses", type=int)
    parser.add_argument("--write-descriptors", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_specs(path: Path) -> list[ModelSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for item in payload["models"]:
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = path.resolve().parent / checkpoint
        specs.append(
            ModelSpec(
                variant=item["variant"],
                model_type=item["model_type"],
                ablation=item["ablation"],
                checkpoint=checkpoint.resolve(),
            )
        )
    variants = [spec.variant for spec in specs]
    if len(variants) != len(set(variants)):
        raise ValueError("duplicate variant in model manifest")
    for spec in specs:
        if spec.model_type not in {"biolm", "genscore"}:
            raise ValueError(spec)
        if spec.ablation not in ABLATION_CONFIGS:
            raise ValueError(spec)
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(spec.checkpoint)
    return specs


def compound_id(pose_id: str) -> str:
    # Prepared titles are e.g. BDB13530-37-0000 and VSDataset appends "-<row>".
    # Removing the final two fields recovers the compound ID BDB13530-37.
    parts = pose_id.rsplit("-", 2)
    if len(parts) != 3:
        raise ValueError(f"unexpected DEKOIS pose id: {pose_id}")
    return parts[0]


def install_project_imports(project_root: Path):
    sys.path.insert(0, str(project_root.resolve()))
    # Import the exact BioLM_Score/data/data_dekois.py supplied by the user.
    # Its unused gen_pocket=True helper imports a broken OpenBabel binding in
    # this environment.  The paper's prepared-DEKOIS path always supplies a
    # pre-extracted pocket (gen_pocket=False), so provide only an explicit stub
    # for that unreachable helper instead of switching to a different loader.
    pocket_module = "BioLM_Score.feats.extract_pocket_prody"
    if pocket_module not in sys.modules:
        stub = types.ModuleType(pocket_module)

        def extract_pocket_unavailable(*args, **kwargs):
            raise RuntimeError(
                "On-the-fly pocket extraction is disabled; prepared 10A pocket required"
            )

        stub.extract_pocket = extract_pocket_unavailable
        sys.modules[pocket_module] = stub

    from BioLM_Score.data.data_dekois import VSDataset as PaperVSDataset
    from GenScore.model.model4 import (
        GenScore as PaperBioLMScore,
        GatedGCN as PaperBioGatedGCN,
    )
    # GenScore and BioLM_Score vendor the same GraphGym registration key.
    # Their layer classes are already imported above; clearing only the global
    # registry entry permits importing the ablation-capable extension too.
    layer_dict.pop("gatedgcnconv", None)
    from BioLM_Score.model.model4 import BioLLMScore, GatedGCN as BioGatedGCN
    from BioLM_Score.model.utils2 import collate_fn

    class BoundedVSDataset(PaperVSDataset):
        """Paper loader with only n_jobs=-1 replaced by an explicit bound."""

        def __init__(self, *args, graph_workers=4, **kwargs):
            self.graph_workers = max(int(graph_workers), 1)
            super().__init__(*args, **kwargs)

        def _mol_to_graph(self):
            if self.parallel:
                return Parallel(
                    n_jobs=self.graph_workers,
                    backend="threading",
                )(delayed(self._mol_to_graph0)(lig) for lig in self.ligs)
            return [self._mol_to_graph0(lig) for lig in self.ligs]

    return (
        BoundedVSDataset,
        PaperBioLMScore,
        PaperBioGatedGCN,
        BioLLMScore,
        BioGatedGCN,
        collate_fn,
    )


def gatedgcn_backbones(gatedgcn):
    common = dict(
        num_hidden_channels=128,
        residual=True,
        dropout_rate=0.15,
        equivstable_pe=False,
        num_layers=6,
    )
    return (
        gatedgcn(in_channels=41, edge_features=10, **common),
        gatedgcn(in_channels=41, edge_features=5, **common),
    )


def build_model(
    spec,
    device,
    PaperBioLMScore,
    PaperBioGatedGCN,
    BioLLMScore,
    BioGatedGCN,
):
    if spec.model_type == "biolm":
        if spec.ablation == "full":
            ligand, protein = gatedgcn_backbones(PaperBioGatedGCN)
            model = PaperBioLMScore(
                ligand,
                protein,
                in_channels=128,
                hidden_dim=128,
                n_gaussians=10,
                dropout_rate=0.15,
                dist_threhold=5.0,
            )
        else:
            config = ABLATION_CONFIGS[spec.ablation]
            if config["use_gnn"]:
                ligand, protein = gatedgcn_backbones(BioGatedGCN)
            else:
                ligand, protein = None, None
            model = BioLLMScore(
                ligand,
                protein,
                in_channels=128,
                hidden_dim=128,
                n_gaussians=10,
                dropout_rate=0.15,
                dist_threhold=5.0,
                **config,
            )
    else:
        if spec.ablation != "full":
            raise ValueError("GenScore supports only ablation=full")
        layer_dict.pop("gatedgcnconv", None)
        from GenScore.model.model_original import GatedGCN, GenScore

        ligand, protein = gatedgcn_backbones(GatedGCN)
        model = GenScore(ligand, protein, 128, 128, 10, 0.15, 5.0)
    checkpoint = th.load(spec.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def class_paths(data_root: Path, target: str, class_name: str) -> dict[str, Path]:
    plural = "actives" if class_name == "active" else "decoys"
    base = data_root / target
    prefix = f"{target}_{plural}_SP"
    return {
        "ligand": base / prefix / f"{prefix}.sdf",
        "ligand_embedding": base / prefix / f"{target}_ligand_features.npy",
        "pocket": base / f"{target}_prot" / f"{target}_pocket_chain_10A.pdb",
        "protein_embedding": base / f"{target}_prot" / f"{target}_p.npy",
    }


def make_dataset(VSDataset, paths, graph_cutoff: float, graph_workers: int):
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return VSDataset(
        ligs=str(paths["ligand"]),
        prot=str(paths["pocket"]),
        protein_emb=str(paths["protein_embedding"]),
        ligand_emb=str(paths["ligand_embedding"]),
        cutoff=graph_cutoff,
        explicit_H=False,
        use_chirality=True,
        parallel=True,
        graph_workers=graph_workers,
    )


def subset_dataset(dataset, limit: Optional[int]):
    if limit is None or len(dataset) <= limit:
        return dataset
    # VSDataset stores a batched graph; slicing through Dataset.get preserves its
    # original ID/embedding alignment without rebuilding graphs.
    from torch.utils.data import Subset

    return Subset(dataset, range(limit))


def probability_score(pi, sigma, mu, distance, batch_index, batch_size, cutoff):
    normal = Normal(mu, sigma)
    probability = (normal.log_prob(distance.expand_as(normal.loc)) + th.log(pi)).exp()
    probability = probability.sum(1)
    probability = probability.masked_fill(distance.squeeze(1) > cutoff, 0.0)
    return scatter_add(
        probability, batch_index, dim=0, dim_size=batch_size
    ), distance, batch_index


def node_counts(batch) -> list[int]:
    ptr = getattr(batch, "ptr", None)
    values = th.bincount(batch.batch) if ptr is None else ptr[1:] - ptr[:-1]
    return [int(value) for value in values.detach().cpu().tolist()]


def score_dataset(
    dataset,
    model,
    spec,
    device,
    batch_size,
    cutoff,
    collate_fn,
    collect_geometry,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    ids: list[str] = []
    scores: list[float] = []
    geometry: list[tuple[str, int, int, int]] = []
    with th.no_grad():
        for batch_data in loader:
            (
                batch_ids,
                protein_graph,
                ligand_graph,
                protein_embeddings,
                ligand_embeddings,
                _,
            ) = batch_data
            batch_ids = list(map(str, batch_ids))
            protein_graph = protein_graph.to(device)
            ligand_graph = ligand_graph.to(device)
            protein_embeddings = protein_embeddings.to(device)
            ligand_embeddings = ligand_embeddings.to(device)
            protein_input = protein_graph.clone()
            ligand_input = ligand_graph.clone()
            if spec.model_type == "biolm":
                result = model(
                    ligand_input,
                    protein_input,
                    protein_embeddings,
                    ligand_embeddings,
                )
            else:
                result = model(ligand_input, protein_input)
            pi, sigma, mu, distance, _, _, batch_index = result
            batch_scores, distance, batch_index = probability_score(
                pi,
                sigma,
                mu,
                distance,
                batch_index,
                len(batch_ids),
                cutoff,
            )
            ids.extend(batch_ids)
            scores.extend(map(float, batch_scores.detach().cpu().numpy()))
            if collect_geometry:
                contact_mask = (distance.squeeze(1) <= cutoff).to(th.int64)
                contacts = scatter_add(
                    contact_mask,
                    batch_index,
                    dim=0,
                    dim_size=len(batch_ids),
                )
                for item_id, heavy, pocket, contact_count in zip(
                    batch_ids,
                    node_counts(ligand_graph),
                    node_counts(protein_graph),
                    contacts.detach().cpu().tolist(),
                ):
                    geometry.append(
                        (item_id, heavy, pocket, int(contact_count))
                    )
            del (
                protein_graph,
                ligand_graph,
                protein_embeddings,
                ligand_embeddings,
                protein_input,
                ligand_input,
                result,
            )
    return ids, scores, geometry


def atomic_gzip_table(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
    temporary.replace(path)


def gzip_data_rows(path: Path) -> int:
    if not path.is_file():
        return -1
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except (OSError, EOFError):
        return -1


def molecule_descriptors(dataset, label: int):
    seen: set[str] = set()
    rows = []
    invalid_poses = 0
    for pose_id, mol in zip(dataset.ids, dataset.ligs):
        cid = compound_id(str(pose_id))
        if cid in seen:
            continue
        if mol is None:
            # RDKit can return None for a malformed pose while the original
            # VSDataset still has other valid poses for the same compound.
            # Descriptor sidecars are analysis-only, so leave the compound
            # unseen and allow a later valid pose to provide its descriptors.
            invalid_poses += 1
            continue
        try:
            row = (
                cid,
                label,
                int(Descriptors.HeavyAtomCount(mol)),
                float(Descriptors.MolWt(mol)),
                int(Lipinski.NumRotatableBonds(mol)),
            )
        except (AttributeError, RuntimeError, ValueError):
            invalid_poses += 1
            continue
        seen.add(cid)
        rows.append(row)
    if invalid_poses:
        print(
            f"descriptor sidecar skipped malformed poses={invalid_poses}",
            flush=True,
        )
    return rows


def run_target(
    args,
    target,
    specs,
    VSDataset,
    PaperBioLMScore,
    PaperBioGatedGCN,
    BioLLMScore,
    BioGatedGCN,
    collate_fn,
    device,
):
    started = time.time()
    datasets = {}
    compound_rows = []
    expected_pose_count = 0
    for class_name in args.classes:
        paths = class_paths(args.data_root, target, class_name)
        full_dataset = make_dataset(
            VSDataset,
            paths,
            args.graph_cutoff,
            args.graph_workers,
        )
        datasets[class_name] = subset_dataset(full_dataset, args.limit_poses)
        expected_pose_count += len(datasets[class_name])
        if args.write_descriptors and args.limit_poses is None:
            compound_rows.extend(
                molecule_descriptors(
                    full_dataset, 1 if class_name == "active" else 0
                )
            )

    pose_geometry: dict[str, tuple[int, int, int]] = {}
    output_counts = {}
    for spec_position, spec in enumerate(specs):
        output = args.output_root / "scores" / spec.variant / f"{target}.tsv.gz"
        if gzip_data_rows(output) == expected_pose_count:
            output_counts[spec.variant] = expected_pose_count
            print(
                f"target={target} model={spec_position + 1}/{len(specs)} "
                f"variant={spec.variant} reused poses={expected_pose_count}",
                flush=True,
            )
            continue
        model = build_model(
            spec,
            device,
            PaperBioLMScore,
            PaperBioGatedGCN,
            BioLLMScore,
            BioGatedGCN,
        )
        rows = []
        # Geometry is model-independent.  Collect it for both active and
        # decoy datasets during the first model that actually runs.
        collect_geometry_this_model = args.write_descriptors and not pose_geometry
        for class_name in args.classes:
            label = 1 if class_name == "active" else 0
            ids, scores, geometry = score_dataset(
                datasets[class_name],
                model,
                spec,
                device,
                args.batch_size,
                args.score_cutoff,
                collate_fn,
                collect_geometry_this_model,
            )
            rows.extend(
                (item_id, compound_id(item_id), label, score)
                for item_id, score in zip(ids, scores)
            )
            for item_id, heavy, pocket, contacts in geometry:
                pose_geometry[item_id] = (heavy, pocket, contacts)
        atomic_gzip_table(
            output,
            ("pose_id", "compound_id", "label", "score"),
            rows,
        )
        output_counts[spec.variant] = len(rows)
        del model
        gc.collect()
        th.cuda.empty_cache()
        print(
            f"target={target} model={spec_position + 1}/{len(specs)} "
            f"variant={spec.variant} poses={len(rows)}",
            flush=True,
        )

    if args.write_descriptors and args.limit_poses is None:
        compound_output = (
            args.output_root / "descriptors" / f"{target}.compounds.tsv.gz"
        )
        pose_output = args.output_root / "descriptors" / f"{target}.poses.tsv.gz"
        if not pose_geometry and gzip_data_rows(pose_output) != expected_pose_count:
            # This can occur when every score file survived an interrupted
            # attempt but the shared descriptor sidecar did not.  Reuse the
            # first checkpoint only to recover model-independent geometry.
            spec = specs[0]
            model = build_model(
                spec,
                device,
                PaperBioLMScore,
                PaperBioGatedGCN,
                BioLLMScore,
                BioGatedGCN,
            )
            for class_name in args.classes:
                _, _, geometry = score_dataset(
                    datasets[class_name],
                    model,
                    spec,
                    device,
                    args.batch_size,
                    args.score_cutoff,
                    collate_fn,
                    True,
                )
                for item_id, heavy, pocket, contacts in geometry:
                    pose_geometry[item_id] = (heavy, pocket, contacts)
            del model
            gc.collect()
            th.cuda.empty_cache()
        atomic_gzip_table(
            compound_output,
            ("compound_id", "label", "heavy_atoms", "molecular_weight", "rotatable_bonds"),
            compound_rows,
        )
        atomic_gzip_table(
            pose_output,
            ("pose_id", "heavy_atoms", "pocket_residues", "contact_pairs_5A"),
            (
                (pose_id, values[0], values[1], values[2])
                for pose_id, values in pose_geometry.items()
            ),
        )

    status = {
        "complete": True,
        "target": target,
        "classes": list(args.classes),
        "expected_pose_count": expected_pose_count,
        "output_counts": output_counts,
        "models": {
            spec.variant: {
                "checkpoint": str(spec.checkpoint),
                "sha256": sha256(spec.checkpoint),
            }
            for spec in specs
        },
        "write_descriptors": args.write_descriptors,
        "elapsed_seconds": time.time() - started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    status_path = args.output_root / "status" / f"{target}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = status_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(status_path)


def main() -> None:
    args = parse_args()
    th.set_num_threads(1)
    for name in ("project_root", "data_root", "output_root", "model_manifest"):
        setattr(args, name, getattr(args, name).resolve())
    if not th.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = th.device("cuda")
    specs = load_specs(args.model_manifest)
    (
        VSDataset,
        PaperBioLMScore,
        PaperBioGatedGCN,
        BioLLMScore,
        BioGatedGCN,
        collate_fn,
    ) = install_project_imports(args.project_root)
    for position, target in enumerate(args.targets, 1):
        print(f"starting target={position}/{len(args.targets)} {target}", flush=True)
        run_target(
            args,
            target,
            specs,
            VSDataset,
            PaperBioLMScore,
            PaperBioGatedGCN,
            BioLLMScore,
            BioGatedGCN,
            collate_fn,
            device,
        )


if __name__ == "__main__":
    main()
