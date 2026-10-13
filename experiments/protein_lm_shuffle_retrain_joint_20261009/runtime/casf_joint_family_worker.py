#!/usr/bin/env python3
"""Evaluate protein-LM-only shuffle controls with the original CASF logic.

The benchmark data loading, MDN probability sum, scoring, docking, and screening
definitions intentionally match the archived CASF evaluator.  The extensions are
operational only:

Only the pocket-residue protein LM tensor is perturbed. Protein graphs,
coordinates, ligand inputs, pair masks, and MDN scoring remain unchanged.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch as th
from torch.distributions import Normal
from torch.utils.data import DataLoader, Subset
from torch_geometric.graphgym.register import layer_dict
from torch_scatter import scatter_add

from protein_lm_shuffle import CONTROLS, ProteinLMShuffleBank


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
    parser.add_argument(
        "--model-spec",
        action="append",
        nargs=4,
        metavar=("VARIANT", "MODEL_TYPE", "ABLATION", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("scoring", "docking", "screening"),
        default=("scoring", "docking", "screening"),
    )
    parser.add_argument("--cutoffs", type=float, nargs="+", default=(5.0,))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scoring-data-dir", type=Path, required=True)
    parser.add_argument("--genscore-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit-items", type=int, help="Smoke test only")
    parser.add_argument("--limit-ligands", type=int, help="Smoke test only")
    parser.add_argument("--controls", nargs="+", choices=CONTROLS, default=CONTROLS)
    parser.add_argument("--shuffle-seed", type=int, default=20260811)
    return parser.parse_args()


def read_casf_table(path: Path, header_prefix: str) -> pd.DataFrame:
    header = None
    rows: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(header_prefix):
            header = line
        elif not line.startswith("#"):
            rows.append(line)
    if header is None:
        raise ValueError(f"header {header_prefix!r} not found in {path}")
    return pd.read_csv(
        StringIO("\n".join([header] + rows)), sep=r"\s+", engine="python"
    )


def probability_score(
    pi,
    sigma,
    mu,
    distance,
    batch_index,
    batch_size: int,
    cutoff: float,
):
    """The exact MDN probability sum used by the archived evaluator."""
    normal = Normal(mu, sigma)
    probability = (normal.log_prob(distance.expand_as(normal.loc)) + th.log(pi)).exp()
    probability = probability.sum(1)
    probability = probability.masked_fill(distance.squeeze(1) > cutoff, 0.0)
    return scatter_add(probability, batch_index, dim=0, dim_size=batch_size)


def cutoff_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def output_variant(control: str, spec: ModelSpec, cutoff: float, cutoff_count: int) -> str:
    base = f"{control}_{spec.variant}"
    if cutoff_count == 1:
        return base
    return f"{base}__cutoff_{cutoff_token(cutoff)}A"


def atomic_table(path: Path, columns: tuple[str, ...], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(repr(value) if isinstance(value, float) else str(value) for value in row))
            handle.write("\n")
    temporary.replace(path)


def complex_paths(data_root: Path, pdb_id: str) -> dict[str, Path]:
    processed = data_root / "pdbbind" / "PDBbind_v2020_processed" / pdb_id
    return {
        "pocket": processed / f"{pdb_id}_pocket_chain_10A.pdb",
        "native": processed / f"{pdb_id}_ligand_sort.sdf",
        "protein_embedding": (
            data_root
            / "protein_embs"
            / "pdbbindv2020_embeddings"
            / "pocket_embeddings"
            / f"{pdb_id}.npy"
        ),
        "native_ligand_embedding": data_root / "ligand_embs" / f"{pdb_id}.npy",
    }


def install_project_imports(project_root: Path):
    project_root = project_root.resolve()
    sys.path.insert(0, str(project_root))
    from BioLM_Score.data.data2 import PDBbindDataset, VSDataset
    from BioLM_Score.model.model4 import BioLLMScore, GatedGCN as BioGatedGCN
    from BioLM_Score.model.utils2 import collate_fn

    return PDBbindDataset, VSDataset, BioLLMScore, BioGatedGCN, collate_fn


def make_dataset(
    VSDataset,
    ligand: Path,
    pocket: Path,
    protein_embedding: Path,
    ligand_embedding: Path,
):
    for path in (ligand, pocket, protein_embedding, ligand_embedding):
        if not path.exists():
            raise FileNotFoundError(path)
    return VSDataset(
        ligs=str(ligand),
        prot=str(pocket),
        protein_emb=str(protein_embedding),
        ligand_emb=str(ligand_embedding),
        cutoff=10.0,
        explicit_H=False,
        use_chirality=True,
        parallel=False,
    )


def gatedgcn_backbones(gatedgcn):
    common = dict(
        num_hidden_channels=128,
        residual=True,
        dropout_rate=0.15,
        equivstable_pe=False,
        num_layers=6,
    )
    ligand = gatedgcn(in_channels=41, edge_features=10, **common)
    protein = gatedgcn(in_channels=41, edge_features=5, **common)
    return ligand, protein


def load_models(
    args: argparse.Namespace,
    specs: list[ModelSpec],
    device: th.device,
    BioLLMScore,
    BioGatedGCN,
):
    models: dict[str, tuple[ModelSpec, th.nn.Module]] = {}
    genscore_classes = None
    for spec in specs:
        if spec.variant in models:
            raise ValueError(f"duplicate variant: {spec.variant}")
        if spec.model_type == "biolm":
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
                dist_threhold=7.0,
                **config,
            )
        elif spec.model_type == "genscore":
            if spec.ablation != "full":
                raise ValueError("GenScore only supports ablation=full")
            if genscore_classes is None:
                sys.path.insert(0, str(args.genscore_root.parent))
                layer_dict.pop("gatedgcnconv", None)
                from GenScore.model.model_original import GatedGCN, GenScore

                genscore_classes = (GatedGCN, GenScore)
            GenGatedGCN, GenScore = genscore_classes
            ligand, protein = gatedgcn_backbones(GenGatedGCN)
            model = GenScore(ligand, protein, 128, 128, 10, 0.15, 7.0)
        else:
            raise ValueError(f"unsupported model type: {spec.model_type}")

        checkpoint = th.load(spec.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        models[spec.variant] = (spec, model.to(device).eval())
    return models


def node_counts(batch) -> list[int]:
    ptr = getattr(batch, "ptr", None)
    if ptr is None:
        values = th.bincount(batch.batch)
    else:
        values = ptr[1:] - ptr[:-1]
    return [int(value) for value in values.detach().cpu().tolist()]


def predict_many(
    dataset,
    models,
    device,
    batch_size: int,
    cutoffs: tuple[float, ...],
    collate_fn,
    shuffle_bank: ProteinLMShuffleBank,
    controls: tuple[str, ...],
    protein_key: Optional[str] = None,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    ids: list[str] = []
    score_keys = [
        (control, variant, cutoff)
        for control in controls
        for variant in models
        for cutoff in cutoffs
    ]
    scores: dict[tuple[str, str, float], list[float]] = {key: [] for key in score_keys}
    descriptors: list[tuple[str, int, int, int]] = []
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
            ids.extend(batch_ids)
            protein_graph = protein_graph.to(device)
            ligand_graph = ligand_graph.to(device)
            ligand_embeddings = ligand_embeddings.to(device)
            ligand_counts = node_counts(ligand_graph)
            protein_counts = node_counts(protein_graph)
            protein_keys = (
                batch_ids
                if protein_key is None
                else [str(protein_key)] * len(batch_ids)
            )
            descriptor_distance = None
            descriptor_batch = None

            for control in controls:
                shuffled_protein_embeddings = shuffle_bank.apply_batch(
                    protein_embeddings,
                    protein_counts,
                    protein_keys,
                    control,
                ).to(device)
                for variant, (spec, model) in models.items():
                    if spec.model_type != "biolm" or spec.ablation != "full":
                        raise ValueError("protein LM shuffle requires full BioLM-Score")
                    # GatedGCN mutates graph feature tensors in place. Each
                    # forward therefore receives independent graph clones.
                    ligand_input = ligand_graph.clone()
                    protein_input = protein_graph.clone()
                    result = model(
                        ligand_input,
                        protein_input,
                        shuffled_protein_embeddings,
                        ligand_embeddings,
                    )
                    pi, sigma, mu, distance, _, _, batch_index = result
                    if descriptor_distance is None:
                        descriptor_distance = distance
                        descriptor_batch = batch_index
                    for cutoff in cutoffs:
                        batch_scores = probability_score(
                            pi,
                            sigma,
                            mu,
                            distance,
                            batch_index,
                            len(batch_ids),
                            cutoff,
                        )
                        scores[(control, variant, cutoff)].extend(
                            map(float, batch_scores.detach().cpu().numpy())
                        )
                    del ligand_input, protein_input, result
                del shuffled_protein_embeddings

            assert descriptor_distance is not None and descriptor_batch is not None
            contact_mask = (descriptor_distance.squeeze(1) <= 5.0).to(th.int64)
            contact_counts = scatter_add(
                contact_mask,
                descriptor_batch,
                dim=0,
                dim_size=len(batch_ids),
            )
            for item_id, heavy, pocket, contacts in zip(
                batch_ids,
                ligand_counts,
                protein_counts,
                contact_counts.detach().cpu().tolist(),
            ):
                descriptors.append((item_id, heavy, pocket, int(contacts)))
    return ids, scores, descriptors


def output_map(
    args,
    specs: list[ModelSpec],
    controls: tuple[str, ...],
    cutoffs: tuple[float, ...],
    phase: str,
    name: str,
):
    mapping = {}
    for control in controls:
        for spec in specs:
            for cutoff in cutoffs:
                variant = output_variant(control, spec, cutoff, len(cutoffs))
                mapping[(control, spec.variant, cutoff)] = (
                    args.output_root / "score_files" / variant / phase / name
                )
    return mapping


def descriptors_for_key(
    descriptors: list[tuple[str, int, int, int]],
    normalized_ids: Optional[List[str]] = None,
):
    rows = descriptors
    if normalized_ids is not None:
        rows = [
            (normalized, heavy, pocket, contacts)
            for normalized, (_, heavy, pocket, contacts) in zip(
                normalized_ids, descriptors
            )
        ]
    return rows


def run_scoring(
    args, specs, models, device, controls, cutoffs, PDBbindDataset, collate_fn, shuffle_bank
):
    data = PDBbindDataset(
        ids=str(args.scoring_data_dir / "v2020_casf_ids.npy"),
        ligs=str(args.scoring_data_dir / "v2020_casf_lig.pt"),
        prots=str(args.scoring_data_dir / "v2020_casf_prot.pt"),
        protein_embs=str(
            args.data_root
            / "protein_embs"
            / "pdbbindv2020_embeddings"
            / "pocket_embeddings"
        ),
        ligand_embs=str(args.data_root / "ligand_embs"),
    )
    selected = (
        data
        if args.limit_items is None
        else Subset(data, range(min(args.limit_items, len(data))))
    )
    ids, scores, descriptors = predict_many(
        selected,
        models,
        device,
        args.batch_size,
        cutoffs,
        collate_fn,
        shuffle_bank,
        controls,
    )
    outputs = output_map(args, specs, controls, cutoffs, "scoring", "all_scores.dat")
    descriptor_outputs = output_map(
        args, specs, controls, cutoffs, "scoring", "all_descriptors.dat"
    )
    for key, values in scores.items():
        atomic_table(outputs[key], ("#code", "score"), zip(ids, values))
        atomic_table(
            descriptor_outputs[key],
            ("#code", "heavy_atoms", "pocket_residues", "contact_pairs_5A"),
            descriptors,
        )
    print(f"scoring complete complexes={len(ids)}", flush=True)


def target_ids(args, phase: str) -> list[str]:
    table = read_casf_table(
        args.data_root / "CASF-2016" / f"power_{phase}" / "CoreSet.dat",
        "#code",
    )
    values = table["#code"].astype(str).tolist()
    return values if args.limit_items is None else values[: args.limit_items]


def run_docking(
    args, specs, models, device, controls, cutoffs, VSDataset, collate_fn, shuffle_bank
):
    docking_root = args.data_root / "CASF-2016" / "decoys_docking"
    ids = target_ids(args, "docking")
    for position, pdb_id in enumerate(ids, 1):
        outputs = output_map(
            args, specs, controls, cutoffs, "docking", f"{pdb_id}_score.dat"
        )
        descriptor_outputs = output_map(
            args, specs, controls, cutoffs, "docking", f"{pdb_id}_descriptors.dat"
        )
        expected = list(outputs.values()) + list(descriptor_outputs.values())
        if all(path.exists() and path.stat().st_size > 20 for path in expected):
            continue
        paths = complex_paths(args.data_root, pdb_id)
        dataset = make_dataset(
            VSDataset,
            docking_root / f"{pdb_id}_decoys.sdf",
            paths["pocket"],
            paths["protein_embedding"],
            paths["native_ligand_embedding"],
        )
        pose_ids, scores, descriptors = predict_many(
            dataset,
            models,
            device,
            args.batch_size,
            cutoffs,
            collate_fn,
            shuffle_bank,
            controls,
            protein_key=pdb_id,
        )
        normalized_ids = [str(pose).split("-", 1)[0] for pose in pose_ids]
        normalized_descriptors = descriptors_for_key(descriptors, normalized_ids)
        for key, values in scores.items():
            atomic_table(outputs[key], ("#code", "score"), zip(normalized_ids, values))
            atomic_table(
                descriptor_outputs[key],
                ("#code", "heavy_atoms", "pocket_residues", "contact_pairs_5A"),
                normalized_descriptors,
            )
        print(f"docking {position}/{len(ids)} {pdb_id}", flush=True)


def run_screening(
    args, specs, models, device, controls, cutoffs, VSDataset, collate_fn, shuffle_bank
):
    screening_root = args.data_root / "CASF-2016" / "decoys_screening"
    processed_root = args.data_root / "pdbbind" / "PDBbind_v2020_processed"
    targets = sorted(
        path.name
        for path in screening_root.iterdir()
        if path.is_dir() and (processed_root / path.name).is_dir()
    )
    if args.limit_items is not None:
        targets = targets[: args.limit_items]
    ligand_ids = sorted(
        path.name
        for path in (args.data_root / "CASF-2016" / "coreset").iterdir()
        if path.is_dir()
    )
    if args.limit_ligands is not None:
        ligand_ids = ligand_ids[: args.limit_ligands]

    for target_position, target in enumerate(targets, 1):
        outputs = output_map(
            args, specs, controls, cutoffs, "screening", f"{target}_score.dat"
        )
        descriptor_outputs = output_map(
            args, specs, controls, cutoffs, "screening", f"{target}_descriptors.dat"
        )
        expected = list(outputs.values()) + list(descriptor_outputs.values())
        if all(path.exists() and path.stat().st_size > 20 for path in expected):
            continue
        paths = complex_paths(args.data_root, target)
        rows = {key: [] for key in outputs}
        descriptor_rows: list[tuple[str, int, int, int]] = []
        for ligand_position, ligand_id in enumerate(ligand_ids, 1):
            dataset = make_dataset(
                VSDataset,
                screening_root / target / f"{target}_{ligand_id}.sdf",
                paths["pocket"],
                paths["protein_embedding"],
                args.data_root / "casf2016_ligand_embs" / f"{ligand_id}.npy",
            )
            pose_ids, scores, descriptors = predict_many(
                dataset,
                models,
                device,
                args.batch_size,
                cutoffs,
                collate_fn,
                shuffle_bank,
                controls,
                protein_key=target,
            )
            normalized_ids = [str(pose).split("-", 1)[0] for pose in pose_ids]
            for key, values in scores.items():
                rows[key].extend(zip(normalized_ids, values))
            descriptor_rows.extend(descriptors_for_key(descriptors, normalized_ids))
            if ligand_position % 20 == 0:
                print(
                    f"screening target={target_position}/{len(targets)} "
                    f"ligand={ligand_position}/{len(ligand_ids)}",
                    flush=True,
                )
        for key in outputs:
            atomic_table(outputs[key], ("#code_ligand_num", "score"), rows[key])
            atomic_table(
                descriptor_outputs[key],
                (
                    "#code_ligand_num",
                    "heavy_atoms",
                    "pocket_residues",
                    "contact_pairs_5A",
                ),
                descriptor_rows,
            )


def main() -> None:
    args = parse_args()
    th.set_num_threads(1)
    if not th.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for field in (
        "project_root",
        "data_root",
        "scoring_data_dir",
        "genscore_root",
        "output_root",
    ):
        setattr(args, field, getattr(args, field).resolve())
    specs = [
        ModelSpec(
            variant=variant,
            model_type=model_type,
            ablation=ablation,
            checkpoint=Path(checkpoint).resolve(),
        )
        for variant, model_type, ablation, checkpoint in args.model_spec
    ]
    for spec in specs:
        if spec.model_type not in ("biolm", "genscore"):
            raise ValueError(spec.model_type)
        if spec.ablation not in ABLATION_CONFIGS:
            raise ValueError(spec.ablation)
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(spec.checkpoint)
    # Cross-process family locks allow an auxiliary free-GPU scheduler to run
    # later seeds without racing the original sequential coordinator.  flock
    # is released automatically if a worker exits or is killed; lock files are
    # intentionally persistent and contain only diagnostic ownership text.
    lock_root = args.output_root / "locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    family_lock_handles = []
    for spec in sorted(specs, key=lambda item: item.variant):
        handle = (lock_root / f"{spec.variant}.lock").open("a+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} variant={spec.variant}\n")
        handle.flush()
        family_lock_handles.append(handle)
    cutoffs = tuple(sorted(set(map(float, args.cutoffs))))
    if not cutoffs or any(value <= 0 for value in cutoffs):
        raise ValueError(f"invalid cutoffs: {cutoffs}")
    controls = tuple(dict.fromkeys(map(str, args.controls)))

    (
        PDBbindDataset,
        VSDataset,
        BioLLMScore,
        BioGatedGCN,
        collate_fn,
    ) = install_project_imports(args.project_root)
    device = th.device("cuda:0")
    casf_ids_raw = np.load(
        args.scoring_data_dir / "v2020_casf_ids.npy", allow_pickle=False
    )
    casf_id_values = casf_ids_raw[0] if casf_ids_raw.ndim == 2 else casf_ids_raw
    casf_ids = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.asarray(casf_id_values).reshape(-1).tolist()
    ]
    embedding_paths = {
        pdb_id: complex_paths(args.data_root, pdb_id)["protein_embedding"]
        for pdb_id in sorted(set(casf_ids))
    }
    shuffle_bank = ProteinLMShuffleBank(embedding_paths, seed=args.shuffle_seed)
    shuffle_bank.write_audit(
        args.output_root / "audit" / "casf_protein_lm_shuffle_mapping.json"
    )
    models = load_models(
        args, specs, device, BioLLMScore=BioLLMScore, BioGatedGCN=BioGatedGCN
    )
    started = time.time()
    for task in args.tasks:
        if task == "scoring":
            run_scoring(
                args,
                specs,
                models,
                device,
                controls,
                cutoffs,
                PDBbindDataset,
                collate_fn,
                shuffle_bank,
            )
        elif task == "docking":
            run_docking(
                args,
                specs,
                models,
                device,
                controls,
                cutoffs,
                VSDataset,
                collate_fn,
                shuffle_bank,
            )
        else:
            run_screening(
                args,
                specs,
                models,
                device,
                controls,
                cutoffs,
                VSDataset,
                collate_fn,
                shuffle_bank,
            )

    outputs = [
        output_variant(control, spec, cutoff, len(cutoffs))
        for control in controls
        for spec in specs
        for cutoff in cutoffs
    ]
    status = {
        "complete": True,
        "models": [
            {
                "variant": spec.variant,
                "model_type": spec.model_type,
                "ablation": spec.ablation,
                "checkpoint": str(spec.checkpoint),
            }
            for spec in specs
        ],
        "cutoffs": cutoffs,
        "controls": controls,
        "shuffle_seed": args.shuffle_seed,
        "outputs": outputs,
        "tasks": list(args.tasks),
        "elapsed_seconds": time.time() - started,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    path = args.output_root / "worker_status" / f"{outputs[0]}__family.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
