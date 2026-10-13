#!/usr/bin/env python3
"""Score the LP validation set and BDB2020+ with one retrained checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from GenScore.model.model4 import GatedGCN as BioGatedGCN
from GenScore.model.model4 import GenScore as BioModel
from GenScore.model.model_original import GatedGCN as OriginalGatedGCN
from GenScore.model.model_original import GenScore as OriginalModel


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
    gnn_class = BioGatedGCN if model_name == "BioLM-Score" else OriginalGatedGCN
    model_class = BioModel if model_name == "BioLM-Score" else OriginalModel
    ligand_model = gnn_class(
        in_channels=41,
        edge_features=10,
        num_hidden_channels=128,
        residual=True,
        dropout_rate=0.15,
        equivstable_pe=False,
        num_layers=6,
    )
    protein_model = gnn_class(
        in_channels=41,
        edge_features=5,
        num_hidden_channels=128,
        residual=True,
        dropout_rate=0.15,
        equivstable_pe=False,
        num_layers=6,
    )
    return model_class(
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
    density = torch.exp(-0.5 * standardized**2) / (
        sigma * np.sqrt(2.0 * np.pi)
    )
    return (pi * density).sum(dim=1)


class Records(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


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
def predict(model, model_name, records, batch_size, device):
    loader = DataLoader(
        Records(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    model.eval()
    rows = []
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
        scores = torch.zeros(
            len(pdb_ids), device=device, dtype=probability.dtype
        )
        scores.scatter_add_(0, pair_batch.to(device), probability)
        for pdb_id, score, label in zip(pdb_ids, scores.cpu().numpy(), labels.numpy()):
            rows.append(
                {
                    "pdb_id": str(pdb_id),
                    "raw_score": float(score),
                    "pKa": float(label),
                }
            )
        if batch_number % 20 == 0:
            print(f"batches={batch_number}/{len(loader)}", flush=True)
    del loader
    gc.collect()
    return rows


def lp_validation_records(training_experiment: Path):
    data = training_experiment / "data"
    tensor_root = data / "train_data"
    ids_labels = np.load(tensor_root / "v2020_train_ids.npy", allow_pickle=False)
    pdb_ids = ids_labels[0].astype(str)
    labels = ids_labels[1].astype(float)
    ligand_graphs = torch.load(
        tensor_root / "v2020_train_lig.pt", map_location="cpu"
    )
    protein_graphs = torch.load(
        tensor_root / "v2020_train_prot.pt", map_location="cpu"
    )
    validation_ids = {
        line.strip()
        for line in (data / "val_ids.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    records = []
    for index, pdb_id in enumerate(pdb_ids):
        if pdb_id not in validation_ids:
            continue
        protein_embedding = torch.tensor(
            np.load(data / "protein_embeddings" / f"{pdb_id}.npy"),
            dtype=torch.float32,
        )
        ligand_embedding = torch.tensor(
            np.load(data / "ligand_embeddings" / f"{pdb_id}.npy"),
            dtype=torch.float32,
        )
        if protein_graphs[index].x.shape[0] != protein_embedding.shape[0]:
            raise AssertionError(f"{pdb_id}: validation protein mismatch")
        if ligand_graphs[index].x.shape[0] != ligand_embedding.shape[0]:
            raise AssertionError(f"{pdb_id}: validation ligand mismatch")
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
    observed_ids = {record[0] for record in records}
    if len(records) != 958 or observed_ids != validation_ids:
        raise AssertionError(
            f"LP validation coverage mismatch: rows={len(records)}, "
            f"IDs={len(observed_ids)}, expected={len(validation_ids)}"
        )
    return records


def bdb_records(bdb_experiment: Path):
    records = []
    for metadata_path in sorted(
        (bdb_experiment / "features" / "metadata").glob("*.json")
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pdb_id = metadata["pdb_id"]
        protein_graph = torch.load(metadata["protein_graph_path"], map_location="cpu")
        ligand_graph = torch.load(metadata["ligand_graph_path"], map_location="cpu")
        protein_embedding = torch.tensor(
            np.load(
                bdb_experiment
                / "features"
                / "protein_embeddings"
                / f"{pdb_id}.npy"
            ),
            dtype=torch.float32,
        )
        ligand_embedding = torch.tensor(
            np.load(
                bdb_experiment
                / "features"
                / "ligand_embeddings"
                / f"{pdb_id}.npy"
            ),
            dtype=torch.float32,
        )
        if protein_graph.x.shape[0] != protein_embedding.shape[0]:
            raise AssertionError(f"{pdb_id}: BDB protein mismatch")
        if ligand_graph.x.shape[0] != ligand_embedding.shape[0]:
            raise AssertionError(f"{pdb_id}: BDB ligand mismatch")
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
        raise AssertionError(f"BDB coverage mismatch: {len(records)}/115")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-experiment", required=True, type=Path)
    parser.add_argument("--bdb-experiment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=["BioLM-Score", "GenScore"])
    parser.add_argument("--seed", required=True, type=int, choices=[8, 23, 24])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    task_key = f"{args.model.replace('-', '').lower()}_seed{args.seed}"
    validation_path = args.output / f"{task_key}_validation.csv"
    bdb_path = args.output / f"{task_key}_bdb.csv"
    status_path = args.output / f"{task_key}_complete.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            status.get("complete")
            and len(pd.read_csv(validation_path)) == 958
            and len(pd.read_csv(bdb_path)) == 115
            and status.get("checkpoint_sha256") == sha256(args.checkpoint)
        ):
            print(f"{task_key} already complete", flush=True)
            return

    checkpoint_companion = args.checkpoint.with_suffix(".json")
    if not checkpoint_companion.exists():
        raise FileNotFoundError(f"Missing checkpoint companion: {checkpoint_companion}")

    device = torch.device(args.device)
    validation = lp_validation_records(args.training_experiment)
    bdb = bdb_records(args.bdb_experiment)
    model = build_model(args.model, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    validation_rows = predict(
        model, args.model, validation, args.batch_size, device
    )
    bdb_rows = predict(
        model, args.model, bdb, max(1, args.batch_size // 2), device
    )
    for frame in (validation_rows, bdb_rows):
        for row in frame:
            row["model"] = args.model
            row["seed"] = args.seed
            row["checkpoint"] = str(args.checkpoint)
    validation_frame = pd.DataFrame(validation_rows)
    bdb_frame = pd.DataFrame(bdb_rows)
    atomic_csv(validation_path, validation_frame)
    atomic_csv(bdb_path, bdb_frame)
    atomic_json(
        status_path,
        {
            "complete": True,
            "model": args.model,
            "seed": args.seed,
            "validation_rows": len(validation_frame),
            "bdb_rows": len(bdb_frame),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "checkpoint_companion_sha256": sha256(checkpoint_companion),
            "validation_sha256": sha256(validation_path),
            "bdb_sha256": sha256(bdb_path),
        },
    )


if __name__ == "__main__":
    main()
