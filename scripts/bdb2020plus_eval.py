#!/usr/bin/env python3
"""Score BDB2020+ and each original validation split with six official weights."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from GenScore.model.model4 import GatedGCN as BioGatedGCN
from GenScore.model.model4 import GenScore as BioModel
from GenScore.model.model_original import GatedGCN as OriginalGatedGCN
from GenScore.model.model_original import GenScore as OriginalModel


def load_model_manifest(path: Path):
    """Read explicit model/seed/checkpoint mappings instead of local paths."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = []
    for item in payload["models"]:
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = path.parent / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model_name = str(item["model"])
        seed = int(item["seed"])
        if model_name not in {"BioLM-Score", "GenScore"}:
            raise ValueError(f"unsupported model name: {model_name}")
        models.append((model_name, seed, checkpoint))
    expected = {(model, seed) for model in ("BioLM-Score", "GenScore") for seed in (23, 24, 8)}
    observed = {(model, seed) for model, seed, _ in models}
    if len(models) != 6 or observed != expected:
        raise ValueError(f"expected one checkpoint per model/seed pair; got {observed}")
    return models


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def build_model(model_name: str, device: torch.device):
    cls_gnn = BioGatedGCN if model_name == "BioLM-Score" else OriginalGatedGCN
    cls_model = BioModel if model_name == "BioLM-Score" else OriginalModel
    ligand_model = cls_gnn(
        in_channels=41,
        edge_features=10,
        num_hidden_channels=128,
        residual=True,
        dropout_rate=0.15,
        equivstable_pe=False,
        num_layers=6,
    )
    protein_model = cls_gnn(
        in_channels=41,
        edge_features=5,
        num_hidden_channels=128,
        residual=True,
        dropout_rate=0.15,
        equivstable_pe=False,
        num_layers=6,
    )
    return cls_model(
        ligand_model,
        protein_model,
        in_channels=128,
        hidden_dim=128,
        n_gaussians=10,
        dropout_rate=0.15,
        dist_threhold=7.0,
    ).to(device)


def mdn_probability(pi, sigma, mu, distance):
    standardized = (distance.expand_as(mu) - mu) / sigma
    density = (
        torch.exp(-0.5 * standardized**2)
        / (sigma * np.sqrt(2.0 * np.pi))
    )
    return (pi * density).sum(dim=1)


class GraphEmbeddingDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def collate(items):
    pdb_ids, protein_graphs, ligand_graphs, protein_embeddings, ligand_embeddings, labels = zip(
        *items
    )
    return (
        pdb_ids,
        Batch.from_data_list(protein_graphs),
        Batch.from_data_list(ligand_graphs),
        pad_sequence(protein_embeddings, batch_first=True),
        pad_sequence(ligand_embeddings, batch_first=True),
        torch.tensor(labels, dtype=torch.float32),
    )


@torch.no_grad()
def predict_loader(model, model_name, loader, device):
    model.eval()
    output = []
    for batch_number, batch in enumerate(loader, start=1):
        pdb_ids, protein_graph, ligand_graph, protein_emb, ligand_emb, labels = batch
        protein_graph = protein_graph.to(device)
        ligand_graph = ligand_graph.to(device)
        if model_name == "BioLM-Score":
            values = model(
                ligand_graph,
                protein_graph,
                protein_emb.to(device),
                ligand_emb.to(device),
            )
        else:
            values = model(ligand_graph, protein_graph)
        pi, sigma, mu, distance, _, _, pair_batch = values
        probability = mdn_probability(pi, sigma, mu, distance)
        probability[distance[:, 0] > 5.0] = 0.0
        pair_batch = pair_batch.to(device)
        scores = torch.zeros(
            len(pdb_ids), device=device, dtype=probability.dtype
        )
        scores.scatter_add_(0, pair_batch, probability)
        for pdb_id, score, label in zip(pdb_ids, scores.cpu().numpy(), labels.numpy()):
            output.append(
                {
                    "pdb_id": str(pdb_id),
                    "raw_score": float(score),
                    "pKa": float(label),
                }
            )
        if batch_number % 20 == 0:
            print(f"  batches={batch_number}/{len(loader)}")
    return output


def bdb_records(experiment: Path):
    records = []
    for metadata_path in sorted((experiment / "features" / "metadata").glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pdb_id = metadata["pdb_id"]
        protein_graph = torch.load(metadata["protein_graph_path"], map_location="cpu")
        ligand_graph = torch.load(metadata["ligand_graph_path"], map_location="cpu")
        protein_embedding = torch.tensor(
            np.load(experiment / "features" / "protein_embeddings" / f"{pdb_id}.npy"),
            dtype=torch.float32,
        )
        ligand_embedding = torch.tensor(
            np.load(experiment / "features" / "ligand_embeddings" / f"{pdb_id}.npy"),
            dtype=torch.float32,
        )
        if protein_graph.x.shape[0] != protein_embedding.shape[0]:
            raise AssertionError(f"{pdb_id}: protein graph/embedding mismatch")
        if ligand_graph.x.shape[0] != ligand_embedding.shape[0]:
            raise AssertionError(f"{pdb_id}: ligand graph/embedding mismatch")
        records.append(
            (
                pdb_id,
                protein_graph,
                ligand_graph,
                protein_embedding,
                ligand_embedding,
                float(metadata["pKa"]),
            )
        )
    if len(records) != 115:
        raise AssertionError(f"BDB graph coverage incomplete: {len(records)}/115")
    return records


def load_training_graphs(data_root: Path):
    ids_labels = np.load(data_root / "v2020_train_ids.npy", allow_pickle=True)
    pdb_ids = ids_labels[0].astype(str)
    labels = ids_labels[1].astype(float)
    ligand_graphs = torch.load(data_root / "v2020_train_lig.pt", map_location="cpu")
    protein_graphs = torch.load(data_root / "v2020_train_prot.pt", map_location="cpu")
    if not (
        len(pdb_ids) == len(labels) == len(ligand_graphs) == len(protein_graphs) == 19015
    ):
        raise AssertionError("Unexpected original PDBBind training data coverage")
    return pdb_ids, labels, ligand_graphs, protein_graphs


def validation_records(
    seed,
    pdb_ids,
    labels,
    ligand_graphs,
    protein_graphs,
    protein_embedding_dir: Path,
    ligand_embedding_dir: Path,
):
    np.random.seed(seed)
    validation_indices = np.random.choice(np.arange(len(pdb_ids)), 1500, replace=False)
    records = []
    for index in validation_indices:
        pdb_id = pdb_ids[index]
        protein_embedding = torch.tensor(
            np.load(protein_embedding_dir / f"{pdb_id}.npy"), dtype=torch.float32
        )
        ligand_embedding = torch.tensor(
            np.load(ligand_embedding_dir / f"{pdb_id}.npy"), dtype=torch.float32
        )
        if protein_graphs[index].x.shape[0] != protein_embedding.shape[0]:
            raise AssertionError(f"{pdb_id}: validation protein embedding mismatch")
        if ligand_graphs[index].x.shape[0] != ligand_embedding.shape[0]:
            raise AssertionError(f"{pdb_id}: validation ligand embedding mismatch")
        records.append(
            (
                pdb_id,
                protein_graphs[index],
                ligand_graphs[index],
                protein_embedding,
                ligand_embedding,
                float(labels[index]),
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--training-data", required=True, type=Path)
    parser.add_argument(
        "--training-protein-embeddings",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--training-ligand-embeddings",
        required=True,
        type=Path,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    models = load_model_manifest(args.model_manifest.resolve())

    experiment = args.experiment.resolve()
    raw_dir = experiment / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    bdb = bdb_records(experiment)
    pdb_ids, labels, ligand_graphs, protein_graphs = load_training_graphs(
        args.training_data
    )

    all_bdb = []
    all_validation = []
    checkpoint_manifest = []
    cached_validation = {}
    for task_number, (model_name, seed, checkpoint) in enumerate(models, start=1):
        task_key = f"{model_name.replace('-', '').lower()}_seed{seed}"
        bdb_path = raw_dir / f"{task_key}_bdb.csv"
        validation_path = raw_dir / f"{task_key}_validation.csv"
        if bdb_path.exists() and validation_path.exists():
            bdb_frame = pd.read_csv(bdb_path)
            validation_frame = pd.read_csv(validation_path)
            if len(bdb_frame) == 115 and len(validation_frame) == 1500:
                all_bdb.append(bdb_frame)
                all_validation.append(validation_frame)
                print(f"{task_number}/6 {task_key} reused")
                continue

        if seed not in cached_validation:
            print(f"loading original validation split seed={seed}")
            cached_validation[seed] = validation_records(
                seed,
                pdb_ids,
                labels,
                ligand_graphs,
                protein_graphs,
                args.training_protein_embeddings,
                args.training_ligand_embeddings,
            )
        validation_loader = DataLoader(
            GraphEmbeddingDataset(cached_validation[seed]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )
        bdb_loader = DataLoader(
            GraphEmbeddingDataset(bdb),
            batch_size=max(1, args.batch_size // 2),
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )

        print(f"{task_number}/6 loading {task_key}")
        model = build_model(model_name, device)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        validation_output = predict_loader(
            model, model_name, validation_loader, device
        )
        bdb_output = predict_loader(model, model_name, bdb_loader, device)
        validation_frame = pd.DataFrame(validation_output)
        bdb_frame = pd.DataFrame(bdb_output)
        for frame in (validation_frame, bdb_frame):
            frame.insert(0, "seed", seed)
            frame.insert(0, "model", model_name)
            frame["checkpoint"] = str(checkpoint)
        atomic_csv(validation_path, validation_frame)
        atomic_csv(bdb_path, bdb_frame)
        all_validation.append(validation_frame)
        all_bdb.append(bdb_frame)
        checkpoint_manifest.append(
            {
                "model": model_name,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
            }
        )
        del model, validation_loader, bdb_loader
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{task_number}/6 {task_key} complete")

    combined_validation = pd.concat(all_validation, ignore_index=True)
    combined_bdb = pd.concat(all_bdb, ignore_index=True)
    if len(combined_validation) != 9000 or len(combined_bdb) != 690:
        raise AssertionError(
            f"Unexpected coverage validation={len(combined_validation)}, "
            f"BDB={len(combined_bdb)}"
        )
    atomic_csv(raw_dir / "validation_predictions.csv", combined_validation)
    atomic_csv(raw_dir / "bdb_predictions.csv", combined_bdb)
    if not checkpoint_manifest:
        checkpoint_manifest = [
            {
                "model": model_name,
                "seed": seed,
                "checkpoint": checkpoint,
                "checkpoint_sha256": sha256(Path(checkpoint)),
            }
            for model_name, seed, checkpoint in models
        ]
    atomic_json(
        experiment / "audit" / "raw_inference_complete.json",
        {
            "complete": True,
            "validation_rows": len(combined_validation),
            "bdb_rows": len(combined_bdb),
            "models": ["BioLM-Score", "GenScore"],
            "seeds": [23, 24, 8],
            "validation_size_per_seed_model": 1500,
            "bdb_size_per_seed_model": 115,
            "score_definition": "sum of MDN pair probability densities for real distances <=5 A",
            "checkpoints": checkpoint_manifest,
            "validation_predictions_sha256": sha256(
                raw_dir / "validation_predictions.csv"
            ),
            "bdb_predictions_sha256": sha256(raw_dir / "bdb_predictions.csv"),
        },
    )


if __name__ == "__main__":
    main()
