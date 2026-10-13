#!/usr/bin/env python3
"""Summarize DEKOIS metrics for the two retrained protein-LM shuffle controls.

Raw pose scores are reduced to the maximum-scoring pose per compound before
AUROC, BEDROC80.5, and EF1% are calculated. Checkpoints remain independent:
the script reports each seed's target-mean metric and then the mean and sample
SD across seeds. It never averages predictions.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


DEFAULT_TARGETS = Path(__file__).resolve().parents[1] / "reproducibility/identifiers/DEKOIS2.0_target_ids.txt"
CONTROLS = ("within_pocket_lm_shuffle", "cross_target_lm_shuffle")
METRICS = ("auroc", "bedroc_80_5", "ef1_percent")


def read_targets(path: Path) -> list[str]:
    targets = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if len(targets) != 81 or len(set(targets)) != 81:
        raise ValueError(f"Expected 81 unique DEKOIS target IDs; got {len(targets)}")
    return targets


def infer_control_seed(variant: str) -> tuple[str, int]:
    match = re.fullmatch(r"(within_pocket_lm_shuffle|cross_target_lm_shuffle)_.+_seed(\d+)", variant)
    if not match:
        raise ValueError(f"Unexpected shuffle variant name: {variant}")
    return match.group(1), int(match.group(2))


def compound_metrics(scores: pd.DataFrame) -> dict[str, Union[float, int]]:
    required = {"pose_id", "compound_id", "label", "score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Pose-score table missing columns: {sorted(missing)}")
    if scores["pose_id"].duplicated().any():
        raise ValueError("Duplicate pose IDs in score table")
    if not np.isfinite(scores["score"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite pose score")
    labels_per_compound = scores.groupby("compound_id")["label"].nunique()
    if (labels_per_compound != 1).any():
        raise ValueError("A compound has inconsistent activity labels across poses")
    compounds = (
        scores.sort_values(["compound_id", "score", "pose_id"],
                           ascending=[True, False, True], kind="mergesort")
        .drop_duplicates("compound_id", keep="first")
        .sort_values(["score", "compound_id"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    labels = compounds["label"].astype(int).to_numpy()
    score_values = compounds["score"].astype(float).to_numpy()
    n = len(compounds)
    n_actives = int(labels.sum())
    n_decoys = n - n_actives
    if n == 0 or n_actives == 0 or n_decoys == 0:
        raise ValueError(f"Both classes are required; n={n}, actives={n_actives}, decoys={n_decoys}")
    n_top = max(1, int(n * 0.01))
    top_actives = int(labels[:n_top].sum())
    from rdkit.ML.Scoring.Scoring import CalcBEDROC
    from sklearn.metrics import roc_auc_score

    pairs = [[float(score), int(label)] for score, label in zip(score_values, labels)]
    return {
        "n_compounds": n,
        "n_actives": n_actives,
        "n_decoys": n_decoys,
        "auroc": float(roc_auc_score(labels, score_values)),
        "bedroc_80_5": float(CalcBEDROC(pairs, 1, 80.5)),
        "ef1_percent": float(top_actives / (n_actives * 0.01)),
    }


def bootstrap_mean_ci(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_tests(
    target_metrics: pd.DataFrame,
    baseline_path: Path,
    target_ids: list[str],
    seeds: tuple[int, ...],
    replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path)
    required = {"family", "target", "seed", *METRICS}
    if required - set(baseline.columns):
        raise ValueError(f"Baseline metric file missing columns: {sorted(required - set(baseline.columns))}")
    rows = []
    from scipy.stats import wilcoxon

    seed_metrics = target_metrics[target_metrics["seed"].isin(seeds)]
    for control in CONTROLS:
        control_rows = seed_metrics[seed_metrics["control"] == control]
        control_by_target = control_rows.groupby("target")[list(METRICS)].mean().reindex(target_ids)
        if control_by_target.isna().any().any():
            raise ValueError(f"Incomplete target coverage for {control}")
        for baseline_family in ("Full BioLM-Score", "Graph-only / GenScore"):
            reference = baseline[baseline["family"] == baseline_family]
            reference = reference[reference["seed"].isin(seeds)]
            if set(reference["seed"].unique()) != set(seeds):
                raise ValueError(f"{baseline_family} baseline does not contain all requested seeds")
            reference_by_target = reference.groupby("target")[list(METRICS)].mean().reindex(target_ids)
            if reference_by_target.isna().any().any():
                raise ValueError(f"Incomplete baseline target coverage for {baseline_family}")
            for metric_index, metric in enumerate(METRICS):
                delta = (control_by_target[metric] - reference_by_target[metric]).to_numpy(dtype=float)
                if not np.isfinite(delta).all():
                    raise ValueError(f"Non-finite paired differences for {control}, {baseline_family}, {metric}")
                p_value = 1.0 if not np.any(delta) else float(
                    wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="auto").pvalue
                )
                estimate, ci_low, ci_high = bootstrap_mean_ci(
                    delta, replicates, bootstrap_seed + metric_index
                )
                rows.append({
                    "control": control,
                    "baseline": baseline_family,
                    "metric": metric,
                    "target_units": len(delta),
                    "mean_difference": estimate,
                    "target_bootstrap_ci95_low": ci_low,
                    "target_bootstrap_ci95_high": ci_high,
                    "wins": int((delta > 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "losses": int((delta < 0).sum()),
                    "wilcoxon_two_sided_p": p_value,
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": bootstrap_seed + metric_index,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-root", required=True, type=Path,
                        help="DEKOIS worker output root containing scores/<variant>/<target>.tsv.gz")
    parser.add_argument("--targets-file", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--variants", nargs="+", required=True,
                        help="The six variant directory names, one per control and seed")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-metrics", type=Path,
                        help="Optional per-target/per-seed metrics CSV for Full and Graph-only paired tests")
    parser.add_argument("--seeds", nargs="+", type=int, required=True,
                        help="Explicit run randomization values supplied by the caller")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, required=True,
                        help="Explicit bootstrap randomization value supplied by the caller")
    args = parser.parse_args()

    seeds = tuple(sorted(set(args.seeds)))
    if len(seeds) != len(args.seeds) or not seeds:
        raise ValueError("Duplicate or empty seed list")
    variants = {}
    for variant in args.variants:
        control, seed = infer_control_seed(variant)
        key = (control, seed)
        if key in variants:
            raise ValueError(f"Duplicate variant for {key}: {variants[key]}, {variant}")
        variants[key] = variant
    expected = {(control, seed) for control in CONTROLS for seed in seeds}
    if set(variants) != expected:
        raise ValueError(f"Expected one variant per control/seed; missing={sorted(expected - set(variants))}, extra={sorted(set(variants) - expected)}")
    targets = read_targets(args.targets_file)
    rows = []
    pose_sets: dict[str, dict[str, set[str]]] = {}
    label_sets: dict[str, dict[str, dict[str, int]]] = {}
    for (control, seed), variant in sorted(variants.items()):
        pose_sets[variant] = {}
        label_sets[variant] = {}
        for target in targets:
            path = args.scores_root / "scores" / variant / f"{target}.tsv.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            scores = pd.read_csv(path, sep="\t", compression="gzip")
            metrics = compound_metrics(scores)
            pose_sets[variant][target] = set(scores["pose_id"].astype(str))
            label_sets[variant][target] = dict(zip(scores["compound_id"].astype(str), scores["label"].astype(int)))
            rows.append({"control": control, "variant": variant, "seed": seed, "target": target, **metrics})

    metric_frame = pd.DataFrame(rows)
    reference_variant = next(iter(variants.values()))
    for variant in variants.values():
        for target in targets:
            if pose_sets[variant][target] != pose_sets[reference_variant][target]:
                raise ValueError(f"Pose-set mismatch: {variant}, target={target}")
            if label_sets[variant][target] != label_sets[reference_variant][target]:
                raise ValueError(f"Compound-label mismatch: {variant}, target={target}")

    seed_metrics = metric_frame.groupby(["control", "variant", "seed"], as_index=False)[list(METRICS)].mean()
    summary_rows = []
    for control, group in seed_metrics.groupby("control", sort=True):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            summary_rows.append({
                "control": control,
                "metric": metric,
                "checkpoints": int(len(values)),
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
            })
    summary = pd.DataFrame(summary_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(args.output_dir / "per_target_per_seed_metrics.csv", index=False)
    seed_metrics.to_csv(args.output_dir / "seed_target_mean_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "control_mean_sample_sd.csv", index=False)

    paired = None
    if args.baseline_metrics:
        paired = paired_tests(metric_frame, args.baseline_metrics, targets, seeds,
                              args.bootstrap_replicates, args.bootstrap_seed)
        paired.to_csv(args.output_dir / "paired_target_tests.csv", index=False)
    audit = {
        "complete": True,
        "targets": len(targets),
        "controls": list(CONTROLS),
        "seeds": list(seeds),
        "variant_count": len(variants),
        "metrics": list(METRICS),
        "compound_score": "maximum pose score per compound",
        "ef1_percent_top_count": "max(1, int(number_of_compounds * 0.01))",
        "replication_summary": "per-checkpoint target means; control mean and sample SD (ddof=1); no prediction ensemble",
        "paired_unit": "DEKOIS target after averaging checkpoint metrics within each target",
        "paired_tests_included": bool(args.baseline_metrics),
    }
    (args.output_dir / "analysis_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
