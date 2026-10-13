#!/usr/bin/env python3
"""Generate missing atom-aligned Chemformer embeddings for LP-PDBBind."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch.nn.functional import pad


def install_lightning_load_stub() -> None:
    """Provide the small Lightning API surface needed to restore Chemformer."""

    module = types.ModuleType("pytorch_lightning")

    class LightningModule(torch.nn.Module):
        def save_hyperparameters(self, *args, **kwargs):
            return None

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            hyperparameters = dict(
                checkpoint.get("hyper_parameters")
                or checkpoint.get("hparams")
                or {}
            )
            hyperparameters.update(kwargs)
            model = cls(*args, **hyperparameters)
            state = checkpoint.get("state_dict", checkpoint)
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Chemformer checkpoint mismatch: "
                    f"missing={incompatible.missing_keys[:20]}, "
                    f"unexpected={incompatible.unexpected_keys[:20]}"
                )
            return model

    module.LightningModule = LightningModule
    module.LightningDataModule = object
    module.__path__ = []
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    callbacks.__path__ = []
    model_checkpoint = types.ModuleType(
        "pytorch_lightning.callbacks.model_checkpoint"
    )

    class ModelCheckpoint:
        pass

    ModelCheckpoint.__module__ = "pytorch_lightning.callbacks.model_checkpoint"
    model_checkpoint.ModelCheckpoint = ModelCheckpoint
    callbacks.ModelCheckpoint = ModelCheckpoint
    module.callbacks = callbacks
    sys.modules["pytorch_lightning"] = module
    sys.modules["pytorch_lightning.callbacks"] = callbacks
    sys.modules[
        "pytorch_lightning.callbacks.model_checkpoint"
    ] = model_checkpoint
    omegaconf = types.ModuleType("omegaconf")

    class DictConfig(dict):
        pass

    class ListConfig(list):
        pass

    class OmegaConf:
        @staticmethod
        def to_container(value, *args, **kwargs):
            return value

    omegaconf.DictConfig = DictConfig
    omegaconf.ListConfig = ListConfig
    omegaconf.OmegaConf = OmegaConf
    sys.modules["omegaconf"] = omegaconf


install_lightning_load_stub()

from molbart.models.transformer_models import BARTModel  # noqa: E402
from molbart.utils.samplers.beam_search_samplers import DecodeSampler  # noqa: E402
from molbart.utils.tokenizers.tokenizers import ChemformerTokenizer  # noqa: E402


NON_ATOMIC_TOKENS = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "(",
    ")",
    ".",
    "=",
    "#",
    "-",
    "/",
    "\\",
    ":",
    "~",
    "@",
    "?",
    ">",
    "*",
    "$",
    "^",
    "&",
    "<PAD>",
    "<MASK>",
    "<SEP>",
] + [f"<UNUSED_{index}>" for index in range(200)] + [
    f"%{index}" for index in range(10, 100)
]


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


def canonical_smiles_and_atom_count(ligand_path: Path):
    mol = Chem.MolFromMolFile(str(ligand_path), removeHs=True)
    if mol is None:
        raise ValueError(f"Could not parse ligand: {ligand_path}")
    smiles = Chem.MolToSmiles(mol, canonical=True)
    order = list(
        mol.GetPropsAsDict(includePrivate=True, includeComputed=True)[
            "_smilesAtomOutputOrder"
        ]
    )
    return smiles, len(order), order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--vocabulary", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    experiment = args.experiment.resolve()
    output_dir = experiment / "features" / "ligand_embeddings"
    status_dir = experiment / "features" / "chemformer_status"
    output_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    metadata_paths = [
        path
        for path in sorted((experiment / "features" / "metadata").glob("*.json"))
        if json.loads(path.read_text(encoding="utf-8")).get(
            "needs_ligand_embedding", False
        )
    ]

    tokenizer = ChemformerTokenizer(filename=str(args.vocabulary))
    sampler = DecodeSampler(tokenizer, args.max_seq_len)
    model = BARTModel.load_from_checkpoint(
        str(args.checkpoint),
        decode_sampler=sampler,
        vocabulary_size=len(tokenizer),
    )
    model = model.to(args.device).eval()

    pending = []
    statuses = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pdb_id = metadata["pdb_id"]
        output_path = output_dir / f"{pdb_id}.npy"
        status_path = status_dir / f"{pdb_id}.json"
        expected_atoms = int(metadata["ligand_atom_count"])
        if output_path.exists() and status_path.exists():
            existing = np.load(output_path, mmap_mode="r")
            if existing.shape == (expected_atoms, 1024):
                statuses.append(json.loads(status_path.read_text(encoding="utf-8")))
                continue
        smiles, atom_count, order = canonical_smiles_and_atom_count(
            Path(metadata["ligand_path"])
        )
        if smiles != metadata["canonical_smiles"] or order != metadata["canonical_atom_order"]:
            raise AssertionError(f"{pdb_id}: canonical ligand order is not reproducible")
        if atom_count != expected_atoms:
            raise AssertionError(f"{pdb_id}: ligand atom count changed")
        pending.append((metadata, smiles, output_path, status_path))

    for start in range(0, len(pending), args.batch_size):
        batch_items = pending[start : start + args.batch_size]
        batch_smiles = [item[1] for item in batch_items]
        encoded = tokenizer.encode(batch_smiles)
        token_lists = tokenizer.tokenize(batch_smiles)
        inputs = []
        masks = []
        atom_keeps = []
        for tokens, values in zip(token_lists, encoded):
            if values.shape[0] > args.max_seq_len:
                raise ValueError(
                    f"SMILES exceeds Chemformer context: {values.shape[0]} > "
                    f"{args.max_seq_len}"
                )
            atom_keeps.append([token not in NON_ATOMIC_TOKENS for token in tokens])
            inputs.append(
                pad(
                    values,
                    pad=(0, args.max_seq_len - values.shape[0]),
                    value=tokenizer.vocabulary[tokenizer.special_tokens["pad"]],
                ).to(args.device)
            )
            masks.append(
                pad(
                    torch.zeros(values.shape, dtype=torch.bool),
                    pad=(0, args.max_seq_len - values.shape[0]),
                    value=True,
                ).to(args.device)
            )
        batch = {
            "encoder_input": torch.stack(inputs).T,
            "encoder_pad_mask": torch.stack(masks).T,
        }
        with torch.no_grad():
            outputs = model.encode(batch).permute(1, 0, 2)
        valid_masks = ~batch["encoder_pad_mask"].T
        for item, output, valid_mask, atom_keep in zip(
            batch_items, outputs, valid_masks, atom_keeps
        ):
            metadata, smiles, output_path, status_path = item
            pdb_id = metadata["pdb_id"]
            valid_output = output[valid_mask]
            if len(valid_output) != len(atom_keep):
                raise AssertionError(
                    f"{pdb_id}: token/encoder mismatch "
                    f"{len(valid_output)} != {len(atom_keep)}"
                )
            atom_features = (
                valid_output[atom_keep].float().cpu().numpy().astype(np.float32)
            )
            expected_shape = (int(metadata["ligand_atom_count"]), 1024)
            if atom_features.shape != expected_shape:
                raise AssertionError(
                    f"{pdb_id}: Chemformer/atom alignment mismatch "
                    f"{atom_features.shape} != {expected_shape}"
                )
            if not np.isfinite(atom_features).all():
                raise AssertionError(f"{pdb_id}: non-finite Chemformer embedding")
            atomic_npy(output_path, atom_features)
            status = {
                "pdb_id": pdb_id,
                "complete": True,
                "canonical_smiles": smiles,
                "atom_count": expected_shape[0],
                "embedding_shape": list(atom_features.shape),
                "output": str(output_path),
                "sha256": sha256(output_path),
            }
            atomic_json(status_path, status)
            statuses.append(status)
            print(
                f"{len(statuses):03d}/{len(metadata_paths):03d} {pdb_id} "
                f"atoms={expected_shape[0]}"
            )
        torch.cuda.empty_cache()

    if len(statuses) != len(metadata_paths) or not all(
        item["complete"] for item in statuses
    ):
        raise AssertionError(
            f"Chemformer coverage incomplete: {len(statuses)}/{len(metadata_paths)}"
        )
    statuses.sort(key=lambda item: item["pdb_id"])
    atomic_json(
        experiment / "audit" / "chemformer_embedding_complete.json",
        {
            "complete": True,
            "n_targets": len(statuses),
            "embedding_dimension": 1024,
            "model": "Chemformer combined-large step=1000000",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "vocabulary": str(args.vocabulary),
            "vocabulary_sha256": sha256(args.vocabulary),
            "statuses": statuses,
        },
    )


if __name__ == "__main__":
    main()
