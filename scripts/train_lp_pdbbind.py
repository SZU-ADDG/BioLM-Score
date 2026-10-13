"""Reproducible base, fine-tuning, and joint training entry point.

The affinity objective used by the original utilities is ``corr * weight``.
Therefore paper alpha >= 0 is passed to the training loop as ``-alpha`` so
that it is equivalent to L_aff = -Corr(score, affinity).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ABLATION_CONFIGS = {
    "full": {"use_protein_lm": True, "use_ligand_lm": True, "use_gnn": True},
    "no_protein_lm": {"use_protein_lm": False, "use_ligand_lm": True, "use_gnn": True},
    "no_ligand_lm": {"use_protein_lm": True, "use_ligand_lm": False, "use_gnn": True},
    "no_gnn": {"use_protein_lm": True, "use_ligand_lm": True, "use_gnn": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=("biolm", "genscore"), required=True)
    parser.add_argument("--ablation", choices=tuple(ABLATION_CONFIGS), default="full")
    parser.add_argument("--stage", choices=("base", "ft", "joint"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--encoder", choices=("gt", "gatedgcn"), default="gt")
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--aux-weight", type=float, default=1e-3)
    parser.add_argument("--val-size", type=int, default=1500)
    parser.add_argument("--train-ids-file", type=Path)
    parser.add_argument("--val-ids-file", type=Path)
    parser.add_argument(
        "--exclude-ids-file",
        type=Path,
        help="IDs forbidden in either explicit split (for example CASF-2016 core IDs)",
    )
    parser.add_argument("--max-samples", type=int, help="Optional smoke-test subset; omit for full training")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-gaussians", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--train-distance-cutoff", type=float, default=7.0)
    parser.add_argument("--score-distance-cutoff", type=float, default=5.0)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "train_data")
    parser.add_argument(
        "--protein-embeddings",
        type=Path,
        default=REPO_ROOT / "data" / "protein_embs" / "pdbbindv2020_embeddings" / "pocket_embeddings",
    )
    parser.add_argument("--ligand-embeddings", type=Path, default=REPO_ROOT / "data" / "ligand_embs")
    parser.add_argument("--tensorboard-dir", type=Path)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = (th.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def normalize_id(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    normalized = str(value).strip().lower()
    if not re.fullmatch(r"[a-z0-9]{4}", normalized):
        raise ValueError(f"invalid complex ID: {value!r}")
    return normalized


def read_id_file(path: Path) -> tuple[list[str], str]:
    path = path.resolve()
    raw = path.read_bytes()
    values = [normalize_id(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate IDs in {path}: {duplicates[:20]}")
    if not values:
        raise ValueError(f"no IDs found in {path}")
    return values, hashlib.sha256(raw).hexdigest()


def explicit_split_indices(dataset_ids, args: argparse.Namespace) -> Optional[tuple[np.ndarray, np.ndarray, dict]]:
    requested = (args.train_ids_file is not None, args.val_ids_file is not None)
    if any(requested) and not all(requested):
        raise ValueError("--train-ids-file and --val-ids-file must be supplied together")
    if not all(requested):
        if args.exclude_ids_file is not None:
            raise ValueError("--exclude-ids-file requires explicit train and validation ID files")
        return None

    train_raw, train_sha256 = read_id_file(args.train_ids_file)
    val_raw, val_sha256 = read_id_file(args.val_ids_file)
    raw_overlap = sorted(set(train_raw).intersection(val_raw))
    if raw_overlap:
        raise ValueError(f"train/validation overlap: {raw_overlap[:20]}")

    excluded: list[str] = []
    excluded_sha256 = None
    if args.exclude_ids_file is not None:
        excluded, excluded_sha256 = read_id_file(args.exclude_ids_file)
    excluded_set = set(excluded)
    train_filtered = [value for value in train_raw if value not in excluded_set]
    val_filtered = [value for value in val_raw if value not in excluded_set]
    if not train_filtered or not val_filtered:
        raise ValueError("CASF exclusion left an empty train or validation split")
    leakage = sorted((set(train_filtered) | set(val_filtered)).intersection(excluded_set))
    if leakage:
        raise AssertionError(f"excluded IDs remain in split: {leakage[:20]}")

    normalized_dataset_ids = [normalize_id(value) for value in dataset_ids]
    if len(normalized_dataset_ids) != len(set(normalized_dataset_ids)):
        raise ValueError("duplicate complex IDs in v2020_train_ids.npy")
    lookup = {value: index for index, value in enumerate(normalized_dataset_ids)}
    missing_train = sorted(set(train_filtered).difference(lookup))
    missing_val = sorted(set(val_filtered).difference(lookup))
    if missing_train or missing_val:
        raise ValueError(
            f"explicit split IDs missing from training tensors: "
            f"train={missing_train[:20]} val={missing_val[:20]}"
        )

    if args.max_samples is not None:
        if args.max_samples < 2:
            raise ValueError("--max-samples must be at least 2")
        train_filtered = train_filtered[: args.max_samples]
        val_filtered = val_filtered[: args.max_samples]
    train_indices = np.asarray([lookup[value] for value in train_filtered], dtype=np.int64)
    val_indices = np.asarray([lookup[value] for value in val_filtered], dtype=np.int64)
    fingerprint_input = json.dumps(
        {"train": train_filtered, "val": val_filtered, "excluded": sorted(excluded_set)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    audit = {
        "mode": "explicit_ids",
        "train_ids_file": str(args.train_ids_file.resolve()),
        "val_ids_file": str(args.val_ids_file.resolve()),
        "exclude_ids_file": str(args.exclude_ids_file.resolve()) if args.exclude_ids_file else None,
        "train_file_sha256": train_sha256,
        "val_file_sha256": val_sha256,
        "exclude_file_sha256": excluded_sha256,
        "train_raw_count": len(train_raw),
        "val_raw_count": len(val_raw),
        "excluded_reference_count": len(excluded_set),
        "train_excluded_count": len(set(train_raw).intersection(excluded_set)),
        "val_excluded_count": len(set(val_raw).intersection(excluded_set)),
        "train_final_count": len(train_filtered),
        "val_final_count": len(val_filtered),
        "train_val_overlap_count": 0,
        "excluded_overlap_after_filter_count": 0,
        "dataset_id_count": len(normalized_dataset_ids),
        "missing_train_count": 0,
        "missing_val_count": 0,
        "split_fingerprint": hashlib.sha256(fingerprint_input).hexdigest(),
    }
    return train_indices, val_indices, audit


def build_backbones(graph_transformer, gated_gcn, args: argparse.Namespace):
    common = dict(
        num_hidden_channels=args.hidden_dim,
        dropout_rate=args.dropout,
        num_layers=6,
    )
    if args.encoder == "gt":
        extra = dict(
            activ_fn=th.nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply="batch",
        )
        ligand = graph_transformer(in_channels=41, edge_features=10, **common, **extra)
        protein = graph_transformer(in_channels=41, edge_features=5, **common, **extra)
    else:
        extra = dict(residual=True, equivstable_pe=False)
        ligand = gated_gcn(in_channels=41, edge_features=10, **common, **extra)
        protein = gated_gcn(in_channels=41, edge_features=5, **common, **extra)
    return ligand, protein


def main() -> None:
    args = parse_args()
    if not th.cuda.is_available():
        raise RuntimeError("CUDA is required; launch with CUDA_VISIBLE_DEVICES set to an available GPU")
    if args.stage == "ft" and args.init_checkpoint is None:
        raise ValueError("--init-checkpoint is required for stage=ft")
    if args.stage == "base" and args.alpha != 1.0:
        print("stage=base ignores --alpha and uses affinity weight 0", flush=True)
    if args.model_type == "genscore" and args.ablation != "full":
        raise ValueError("--ablation applies only to model-type=biolm")

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = (args.tensorboard_dir or args.output.parent / "tensorboard" / args.output.stem).resolve()
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    th.set_num_threads(1)
    th.multiprocessing.set_sharing_strategy("file_system")
    device = th.device("cuda:0")
    generator = th.Generator().manual_seed(args.seed)

    def split_indices(size: int) -> tuple[np.ndarray, np.ndarray]:
        usable = size if args.max_samples is None else min(size, args.max_samples)
        if usable < 2:
            raise ValueError("at least two samples are required")
        val_size = min(args.val_size, usable - 1)
        candidates = np.arange(usable)
        rng = np.random.RandomState(args.seed)
        val_indices = rng.choice(candidates, val_size, replace=False)
        train_indices = np.setdiff1d(candidates, val_indices)
        return train_indices, val_indices

    ids_path = args.data_dir / "v2020_train_ids.npy"
    ligands_path = args.data_dir / "v2020_train_lig.pt"
    proteins_path = args.data_dir / "v2020_train_prot.pt"
    for required in (ids_path, ligands_path, proteins_path):
        if not required.exists():
            raise FileNotFoundError(required)
    loaded_ids = np.asarray(np.load(ids_path, allow_pickle=True))
    dataset_ids = loaded_ids[0].reshape(-1) if loaded_ids.ndim == 2 else loaded_ids.reshape(-1)
    explicit_split = explicit_split_indices(dataset_ids, args)
    if explicit_split is None:
        train_indices, val_indices = split_indices(len(dataset_ids))
        split_audit = {
            "mode": "random",
            "train_final_count": int(len(train_indices)),
            "val_final_count": int(len(val_indices)),
            "seed": args.seed,
            "split_fingerprint": hashlib.sha256(
                np.concatenate((train_indices, np.asarray([-1]), val_indices)).tobytes()
            ).hexdigest(),
        }
    else:
        train_indices, val_indices, split_audit = explicit_split

    if args.model_type == "biolm":
        from torch.utils.data import DataLoader

        from BioLM_Score.data.data2 import PDBbindDataset
        from BioLM_Score.model.model4 import BioLLMScore, GatedGCN, GraphTransformer
        from BioLM_Score.model.utils2 import EarlyStopping, collate_fn, run_a_train_epoch, run_an_eval_epoch

        data = PDBbindDataset(
            ids=str(ids_path),
            ligs=str(ligands_path),
            prots=str(proteins_path),
            protein_embs=str(args.protein_embeddings),
            ligand_embs=str(args.ligand_embeddings),
        )
        train_data = PDBbindDataset(
            ids=data.pdbids[train_indices],
            ligs=data.gls[train_indices],
            prots=data.gps[train_indices],
            labels=data.labels[train_indices],
            protein_embs=[data.protein_embs[i] for i in train_indices],
            ligand_embs=[data.ligand_embs[i] for i in train_indices],
        )
        val_data = PDBbindDataset(
            ids=data.pdbids[val_indices],
            ligs=data.gls[val_indices],
            prots=data.gps[val_indices],
            labels=data.labels[val_indices],
            protein_embs=[data.protein_embs[i] for i in val_indices],
            ligand_embs=[data.ligand_embs[i] for i in val_indices],
        )
        ablation_config = ABLATION_CONFIGS[args.ablation]
        if ablation_config["use_gnn"]:
            ligand_model, protein_model = build_backbones(GraphTransformer, GatedGCN, args)
        else:
            ligand_model, protein_model = None, None
        model = BioLLMScore(
            ligand_model,
            protein_model,
            in_channels=args.hidden_dim,
            hidden_dim=args.hidden_dim,
            n_gaussians=args.n_gaussians,
            dropout_rate=args.dropout,
            dist_threhold=args.train_distance_cutoff,
            **ablation_config,
        ).to(device)
        loader_extra = {"collate_fn": collate_fn}
    else:
        from torch_geometric.loader import DataLoader

        from GenScore.data.data_original import PDBbindDataset
        from GenScore.model.model_original import GatedGCN, GenScore, GraphTransformer
        from GenScore.model.utils_original import EarlyStopping, run_a_train_epoch, run_an_eval_epoch

        data = PDBbindDataset(ids=str(ids_path), ligs=str(ligands_path), prots=str(proteins_path))
        train_data = PDBbindDataset(
            ids=data.pdbids[train_indices],
            ligs=data.gls[train_indices],
            prots=data.gps[train_indices],
            labels=data.labels[train_indices],
        )
        val_data = PDBbindDataset(
            ids=data.pdbids[val_indices],
            ligs=data.gls[val_indices],
            prots=data.gps[val_indices],
            labels=data.labels[val_indices],
        )
        ligand_model, protein_model = build_backbones(GraphTransformer, GatedGCN, args)
        model = GenScore(
            ligand_model,
            protein_model,
            in_channels=args.hidden_dim,
            hidden_dim=args.hidden_dim,
            n_gaussians=args.n_gaussians,
            dropout_rate=args.dropout,
            dist_threhold=args.train_distance_cutoff,
        ).to(device)
        loader_extra = {}

    if args.stage == "ft":
        metadata_path = args.init_checkpoint.with_suffix(".json")
        if metadata_path.exists():
            init_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {"model_type": args.model_type, "ablation": args.ablation, "stage": "base"}
            mismatches = {
                key: (init_metadata.get(key, "full" if key == "ablation" else None), value)
                for key, value in expected.items()
                if init_metadata.get(key, "full" if key == "ablation" else None) != value
            }
            if mismatches:
                raise ValueError(f"init checkpoint metadata mismatch: {mismatches}")
            init_fingerprint = init_metadata.get("split_audit", {}).get("split_fingerprint")
            if init_fingerprint is None:
                # Base checkpoints produced before split auditing was added do
                # not contain a fingerprint.  They are still safe FT inputs
                # when both stages use the legacy deterministic random split
                # with exactly the same seed and split-defining arguments.
                legacy_args = init_metadata.get("args", {})
                legacy_random_split = all(
                    legacy_args.get(key) in (None, "None")
                    for key in ("train_ids_file", "val_ids_file", "exclude_ids_file")
                )
                legacy_data_dir = legacy_args.get("data_dir")
                legacy_matches = {
                    "random_split": legacy_random_split and split_audit.get("mode") == "random",
                    "seed": init_metadata.get("seed") == args.seed,
                    "val_size": legacy_args.get("val_size") == args.val_size,
                    "max_samples": legacy_args.get("max_samples") == args.max_samples,
                    "data_dir": (
                        legacy_data_dir is not None
                        and Path(legacy_data_dir).resolve() == args.data_dir.resolve()
                    ),
                }
                if not all(legacy_matches.values()):
                    raise ValueError(
                        "legacy init checkpoint lacks split fingerprint and "
                        f"split-defining arguments differ: {legacy_matches}"
                    )
                init_fingerprint = split_audit["split_fingerprint"]
                print(
                    json.dumps(
                        {
                            "event": "legacy_split_fingerprint_accepted",
                            "checkpoint": str(args.init_checkpoint),
                            "split_fingerprint": init_fingerprint,
                            "checks": legacy_matches,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if init_fingerprint != split_audit["split_fingerprint"]:
                raise ValueError(
                    "init checkpoint split mismatch: "
                    f"{init_fingerprint!r} != {split_audit['split_fingerprint']!r}"
                )
        checkpoint = th.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        **loader_extra,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        **loader_extra,
    )

    affinity_weight = 0.0 if args.stage == "base" else -float(args.alpha)
    stopper = EarlyStopping(patience=args.patience, mode="lower", filename=str(args.output))
    writer = SummaryWriter(str(tensorboard_dir))
    start = time.time()
    best_epoch = 0
    print(
        json.dumps(
            {
                "event": "start",
                "model_type": args.model_type,
                "ablation": args.ablation,
                "stage": args.stage,
                "seed": args.seed,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "train_size": len(train_data),
                "val_size": len(val_data),
                "affinity_weight_in_code": affinity_weight,
                "paper_alpha": 0.0 if args.stage == "base" else args.alpha,
                "split_audit": split_audit,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(args.num_epochs):
        train_metrics = run_a_train_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            affi_weight=affinity_weight,
            aux_weight=args.aux_weight,
            dist_threhold=args.score_distance_cutoff,
            device=device,
        )
        val_metrics = run_an_eval_epoch(
            model,
            val_loader,
            dist_threhold=args.score_distance_cutoff,
            affi_weight=affinity_weight,
            aux_weight=args.aux_weight,
            device=device,
        )
        if not np.isfinite(train_metrics[1]) or not np.isfinite(val_metrics[1]):
            raise FloatingPointError(f"non-finite MDN loss at epoch {epoch + 1}")
        should_stop = stopper.step(val_metrics[0], model)
        if stopper.best_score == val_metrics[0]:
            best_epoch = epoch + 1
        names = ("total", "mdn", "corr", "atom", "bond")
        for name, value in zip(names, train_metrics):
            writer.add_scalar(f"train/{name}", value, epoch + 1)
        for name, value in zip(names, val_metrics):
            writer.add_scalar(f"val/{name}", value, epoch + 1)
        print(
            f"epoch={epoch + 1} train_total={train_metrics[0]:.6f} "
            f"val_total={val_metrics[0]:.6f} val_mdn={val_metrics[1]:.6f} "
            f"val_corr={val_metrics[2]:.6f} best={stopper.best_score:.6f}",
            flush=True,
        )
        if should_stop:
            break

    writer.close()
    stopper.load_checkpoint(model)
    final_val = run_an_eval_epoch(
        model,
        val_loader,
        dist_threhold=args.score_distance_cutoff,
        affi_weight=affinity_weight,
        aux_weight=args.aux_weight,
        device=device,
    )
    metadata = {
        "model_type": args.model_type,
        "ablation": args.ablation,
        "stage": args.stage,
        "seed": args.seed,
        "paper_alpha": 0.0 if args.stage == "base" else args.alpha,
        "affinity_weight_in_code": affinity_weight,
        "best_epoch": best_epoch,
        "epochs_ran": epoch + 1,
        "elapsed_seconds": time.time() - start,
        "checkpoint": str(args.output),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "split_audit": split_audit,
        "final_validation": dict(zip(("total", "mdn", "corr", "atom", "bond"), map(float, final_val))),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "complete", **metadata}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
