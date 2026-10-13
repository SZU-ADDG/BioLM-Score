#!/usr/bin/env python3
"""Deterministic protein-LM-only perturbations with unchanged graph geometry.

The bank operates on the pocket-residue ESM-C arrays consumed by the existing
BioLM-Score datasets.  It never changes protein graphs, coordinates, residue
counts, ligand inputs, or pair masks.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


CONTROLS = ("within_pocket_lm_shuffle", "cross_target_lm_shuffle")


def _stable_seed(base_seed: int, token: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _sha256_array(array: np.ndarray) -> str:
    view = np.ascontiguousarray(array).view(np.uint8)
    return hashlib.sha256(view).hexdigest()


class ProteinLMShuffleBank:
    """Precompute within-pocket and cross-target pocket-embedding shuffles.

    Cross-target assignment is a global, without-replacement residue-vector
    derangement.  Target blocks are deterministically shuffled, their rows are
    independently permuted, and the concatenated source pool is circularly
    shifted by the largest pocket size.  Because no pocket contains more than
    half of all residues, this gives every recipient slot a vector from a
    different target while using every source vector exactly once.
    """

    def __init__(self, embedding_paths: dict[str, Path], seed: int = 20260811):
        if len(embedding_paths) < 2:
            raise ValueError("at least two targets are required")
        self.seed = int(seed)
        self.paths = {str(k): Path(v).resolve() for k, v in embedding_paths.items()}
        self.original: dict[str, np.ndarray] = {}
        dimensions = set()
        for key, path in self.paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            array = np.load(path, allow_pickle=False)
            if array.ndim != 2 or array.shape[0] < 2:
                raise ValueError(f"invalid protein embedding shape for {key}: {array.shape}")
            if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
                raise ValueError(f"invalid protein embedding values for {key}")
            array = np.asarray(array, dtype=np.float32)
            dimensions.add(int(array.shape[1]))
            self.original[key] = array
        if len(dimensions) != 1:
            raise ValueError(f"inconsistent protein embedding dimensions: {dimensions}")
        self.embedding_dim = dimensions.pop()

        self.within: dict[str, np.ndarray] = {}
        self.within_permutation: dict[str, np.ndarray] = {}
        for key, array in self.original.items():
            rng = np.random.default_rng(_stable_seed(self.seed, f"within:{key}"))
            identity = np.arange(len(array))
            for _ in range(10000):
                permutation = rng.permutation(len(array))
                if not np.any(permutation == identity):
                    break
            else:
                raise RuntimeError(f"could not create within-pocket derangement: {key}")
            self.within_permutation[key] = permutation
            self.within[key] = np.ascontiguousarray(array[permutation])

        order = sorted(self.original)
        order_rng = np.random.default_rng(_stable_seed(self.seed, "cross:block-order"))
        order_rng.shuffle(order)
        blocks = []
        owners = []
        slices: dict[str, slice] = {}
        cursor = 0
        for key in order:
            array = self.original[key]
            row_rng = np.random.default_rng(_stable_seed(self.seed, f"cross:rows:{key}"))
            row_order = row_rng.permutation(len(array))
            block = array[row_order]
            blocks.append(block)
            owners.extend([key] * len(block))
            slices[key] = slice(cursor, cursor + len(block))
            cursor += len(block)
        pool = np.concatenate(blocks, axis=0)
        owner_array = np.asarray(owners, dtype=object)
        self.cross_offset = max(len(value) for value in self.original.values())
        if self.cross_offset * 2 > len(pool):
            raise ValueError("largest target is too large for circular cross-target derangement")
        shifted_pool = np.roll(pool, -self.cross_offset, axis=0)
        shifted_owners = np.roll(owner_array, -self.cross_offset)
        self.cross: dict[str, np.ndarray] = {}
        self.cross_owner_counts: dict[str, dict[str, int]] = {}
        for key, target_slice in slices.items():
            donor_owners = shifted_owners[target_slice]
            if np.any(donor_owners == key):
                raise RuntimeError(f"cross-target derangement contains self donor: {key}")
            self.cross[key] = np.ascontiguousarray(shifted_pool[target_slice])
            self.cross_owner_counts[key] = dict(Counter(map(str, donor_owners)))
        assigned_rows = np.concatenate([self.cross[key] for key in order], axis=0)
        if not np.array_equal(shifted_pool, assigned_rows):
            raise RuntimeError("cross-target assignment does not preserve the global vector pool")
        self._validated_input_targets: set[str] = set()

    def transformed_array(self, target: str, control: str) -> np.ndarray:
        target = str(target)
        if control == "within_pocket_lm_shuffle":
            return self.within[target]
        if control == "cross_target_lm_shuffle":
            return self.cross[target]
        raise ValueError(f"unknown control: {control}")

    def apply_batch(
        self,
        embeddings: torch.Tensor,
        node_counts: list[int],
        targets: list[str],
        control: str,
    ) -> torch.Tensor:
        if control not in CONTROLS:
            raise ValueError(control)
        if len(node_counts) != len(targets) or embeddings.shape[0] != len(targets):
            raise ValueError("batch metadata length mismatch")
        output = embeddings.clone()
        for index, (target, count) in enumerate(zip(map(str, targets), node_counts)):
            if target not in self.original:
                raise KeyError(f"protein LM shuffle target missing from bank: {target}")
            expected = self.original[target]
            if int(count) != len(expected):
                raise ValueError(
                    f"protein graph/embedding count mismatch for {target}: {count} vs {len(expected)}"
                )
            if target not in self._validated_input_targets:
                observed = embeddings[index, :count].detach().cpu().numpy()
                if not np.allclose(observed, expected, rtol=0.0, atol=1e-6):
                    difference = float(np.max(np.abs(observed - expected)))
                    raise ValueError(
                        f"dataset embedding order mismatch for {target}: max_abs={difference}"
                    )
                self._validated_input_targets.add(target)
            replacement = torch.as_tensor(
                self.transformed_array(target, control),
                dtype=output.dtype,
                device=output.device,
            )
            output[index, :count] = replacement
        return output

    def audit(self) -> dict:
        mapping_digest = hashlib.sha256()
        for key in sorted(self.original):
            mapping_digest.update(key.encode("utf-8"))
            mapping_digest.update(self.within_permutation[key].tobytes())
            mapping_digest.update(self.cross[key].tobytes())
        return {
            "schema_version": 1,
            "seed": self.seed,
            "controls": list(CONTROLS),
            "target_count": len(self.original),
            "total_residue_vectors": int(sum(map(len, self.original.values()))),
            "embedding_dimension": self.embedding_dim,
            "cross_assignment": "global without-replacement circular residue-vector derangement",
            "cross_offset": int(self.cross_offset),
            "cross_self_donor_slots": 0,
            "within_fixed_points": 0,
            "mapping_sha256": mapping_digest.hexdigest(),
            "targets": {
                key: {
                    "residue_count": len(self.original[key]),
                    "embedding_path": str(self.paths[key]),
                    "embedding_sha256": _sha256_array(self.original[key]),
                    "cross_donor_target_counts": self.cross_owner_counts[key],
                }
                for key in sorted(self.original)
            },
        }

    def write_audit(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.audit(), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
