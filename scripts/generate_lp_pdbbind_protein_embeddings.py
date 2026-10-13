#!/usr/bin/env python3
"""Generate missing ESM-C residue embeddings for the LP-PDBBind paper split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers


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


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, value)
    temp.replace(path)


def encode_sequence(model, sequence: str) -> torch.Tensor:
    protein = ESMProtein(sequence=sequence)
    tokens = model.encode(protein)
    with torch.no_grad():
        output = model.logits(
            tokens, LogitsConfig(sequence=True, return_embeddings=True)
        )
    embeddings = output.embeddings.squeeze(0)
    if embeddings.shape[0] != len(sequence) + 2:
        raise AssertionError(
            f"ESM-C length mismatch: {embeddings.shape[0]} vs {len(sequence) + 2}"
        )
    return embeddings[1:-1].float().cpu()


def encode_long_sequence(
    model, sequence: str, chunk_size: int, overlap: int
) -> tuple[torch.Tensor, list[dict]]:
    if len(sequence) <= chunk_size:
        return encode_sequence(model, sequence), [
            {"start": 0, "end": len(sequence), "length": len(sequence)}
        ]
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")
    total = torch.zeros((len(sequence), 1152), dtype=torch.float32)
    counts = torch.zeros((len(sequence), 1), dtype=torch.float32)
    chunks = []
    start = 0
    while start < len(sequence):
        end = min(len(sequence), start + chunk_size)
        chunk_embedding = encode_sequence(model, sequence[start:end])
        total[start:end] += chunk_embedding
        counts[start:end] += 1
        chunks.append({"start": start, "end": end, "length": end - start})
        if end == len(sequence):
            break
        start = end - overlap
    if torch.any(counts == 0):
        raise AssertionError("Uncovered residues in chunked ESM-C inference")
    return total / counts, chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=1022)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()

    experiment = args.experiment.resolve()
    metadata_dir = experiment / "features" / "metadata"
    output_dir = experiment / "features" / "protein_embeddings"
    status_dir = experiment / "features" / "esm_status"
    output_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    with torch.device(args.device):
        model = ESMC(
            d_model=1152,
            n_heads=18,
            n_layers=36,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=False,
        ).eval()
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state_dict)
    model = model.to(args.device).eval()
    statuses = []
    metadata_paths = [
        path
        for path in sorted(metadata_dir.glob("*.json"))
        if json.loads(path.read_text(encoding="utf-8")).get(
            "needs_protein_embedding", False
        )
    ]
    for number, metadata_path in enumerate(metadata_paths, start=1):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pdb_id = metadata["pdb_id"]
        output_path = output_dir / f"{pdb_id}.npy"
        status_path = status_dir / f"{pdb_id}.json"
        expected_rows = int(metadata["pocket_residue_count"])
        if output_path.exists() and status_path.exists():
            existing = np.load(output_path, mmap_mode="r")
            if existing.shape == (expected_rows, 1152):
                status = json.loads(status_path.read_text(encoding="utf-8"))
                statuses.append(status)
                print(f"{number:03d}/{len(metadata_paths):03d} {pdb_id} reused")
                continue

        full, chunks = encode_long_sequence(
            model,
            metadata["full_sequence"],
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        kept = np.asarray(metadata["pocket_full_sequence_indices"], dtype=np.int64)
        pocket = full[kept].numpy().astype(np.float32, copy=False)
        if pocket.shape != (expected_rows, 1152):
            raise AssertionError(
                f"{pdb_id}: pocket embedding shape {pocket.shape}, "
                f"expected {(expected_rows, 1152)}"
            )
        if not np.isfinite(pocket).all():
            raise AssertionError(f"{pdb_id}: non-finite ESM-C embedding")
        atomic_npy(output_path, pocket)
        status = {
            "pdb_id": pdb_id,
            "complete": True,
            "full_sequence_length": len(metadata["full_sequence"]),
            "pocket_residue_count": expected_rows,
            "embedding_shape": list(pocket.shape),
            "mode": "full_sequence" if len(chunks) == 1 else "overlap_chunked",
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "chunks": chunks,
            "output": str(output_path),
            "sha256": sha256(output_path),
        }
        atomic_json(status_path, status)
        statuses.append(status)
        print(
            f"{number:03d}/{len(metadata_paths):03d} {pdb_id} "
            f"sequence={len(metadata['full_sequence'])} mode={status['mode']}"
        )
        torch.cuda.empty_cache()

    if len(statuses) != len(metadata_paths) or not all(
        item["complete"] for item in statuses
    ):
        raise AssertionError(
            f"ESM-C coverage incomplete: {len(statuses)}/{len(metadata_paths)}"
        )
    complete = {
        "complete": True,
        "n_targets": len(statuses),
        "n_full_sequence": sum(item["mode"] == "full_sequence" for item in statuses),
        "n_overlap_chunked": sum(item["mode"] == "overlap_chunked" for item in statuses),
        "embedding_dimension": 1152,
        "model": "ESM-C 600M",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "statuses": statuses,
    }
    atomic_json(
        experiment / "audit" / "esm_embedding_complete.json",
        complete,
    )


if __name__ == "__main__":
    main()
