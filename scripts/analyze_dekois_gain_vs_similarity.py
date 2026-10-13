#!/usr/bin/env python3
"""Relate paired DEKOIS target-level gains to nearest-training-protein identity.

The input metric table must contain one row per target, checkpoint variant,
and seed (as emitted by ``analyze_dekois2.py``). The user supplies the exact
three variants for each side so a primary comparison cannot be confused with
a shuffle or ablation control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


METRICS = ("auroc", "bedroc_80_5", "ef1_percent")


def paired_target_deltas(
    metrics: pd.DataFrame,
    variants_a: list[str],
    variants_b: list[str],
    metric: str,
) -> pd.DataFrame:
    required = {"target", "variant", "seed", metric}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metric file is missing columns: {sorted(missing)}")
    if len(variants_a) != len(set(variants_a)) or len(variants_b) != len(set(variants_b)):
        raise ValueError("Variant selectors contain duplicates")
    if len(variants_a) != len(variants_b):
        raise ValueError("Supply a one-to-one, seed-matched variant list for both models")

    def select(variants: list[str], label: str) -> tuple[pd.DataFrame, dict[str, int]]:
        selected = metrics[metrics["variant"].isin(variants)][["target", "variant", "seed", metric]].copy()
        absent = set(variants) - set(selected["variant"])
        if absent:
            raise ValueError(f"{label} variants not found: {sorted(absent)}")
        if selected.duplicated(["target", "variant", "seed"]).any():
            raise ValueError(f"Duplicate target/variant/seed rows for {label}")
        seeds_per_variant = selected.groupby("variant")["seed"].nunique()
        if (seeds_per_variant != 1).any() or len(seeds_per_variant) != len(variants):
            raise ValueError(f"Each {label} variant must identify exactly one checkpoint seed")
        seed_map = selected[["variant", "seed"]].drop_duplicates().set_index("variant")["seed"].to_dict()
        if len(set(seed_map.values())) != len(seed_map):
            raise ValueError(f"{label} variants do not map one-to-one to seeds")
        return selected, seed_map

    selected_a, seeds_a = select(variants_a, "model A")
    selected_b, seeds_b = select(variants_b, "model B")
    if set(seeds_a.values()) != set(seeds_b.values()):
        raise ValueError(f"Checkpoint seed sets differ: A={seeds_a}, B={seeds_b}")

    pair_rows = []
    for seed in sorted(seeds_a.values()):
        variant_a = next(name for name, value in seeds_a.items() if value == seed)
        variant_b = next(name for name, value in seeds_b.items() if value == seed)
        a = selected_a[selected_a["variant"] == variant_a][["target", "seed", metric]]
        b = selected_b[selected_b["variant"] == variant_b][["target", "seed", metric]]
        paired = a.merge(b, on=["target", "seed"], suffixes=("_a", "_b"), validate="one_to_one")
        if paired.empty:
            raise ValueError(f"No paired target rows for seed {seed}")
        paired["delta_a_minus_b"] = paired[f"{metric}_a"] - paired[f"{metric}_b"]
        pair_rows.append(paired[["target", "seed", f"{metric}_a", f"{metric}_b", "delta_a_minus_b"]])
    per_seed = pd.concat(pair_rows, ignore_index=True)
    target_rows = per_seed.groupby("target", as_index=False).agg(
        metric_a_mean=(f"{metric}_a", "mean"),
        metric_b_mean=(f"{metric}_b", "mean"),
        delta_a_minus_b_mean=("delta_a_minus_b", "mean"),
        seed_count=("seed", "nunique"),
    )
    if target_rows["seed_count"].nunique() != 1 or int(target_rows["seed_count"].iloc[0]) != len(variants_a):
        raise ValueError("Targets do not have complete, seed-matched metric coverage")
    return target_rows


def bootstrap_correlation_ci(x: np.ndarray, y: np.ndarray, method: str, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(x)
    values = []
    for _ in range(reps):
        indices = rng.integers(0, n, size=n)
        xx, yy = x[indices], y[indices]
        if np.std(xx) == 0 or np.std(yy) == 0:
            continue
        result = pearsonr(xx, yy)[0] if method == "pearson" else spearmanr(xx, yy)[0]
        if np.isfinite(result):
            values.append(float(result))
    if len(values) < reps * 0.9:
        raise ValueError(f"Too many degenerate bootstrap replicates: valid={len(values)}/{reps}")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path,
                        help="per_target_per_seed_metrics.csv from the DEKOIS analysis")
    parser.add_argument("--protein-similarity", required=True, type=Path,
                        help="protein_similarity_per_target.tsv")
    parser.add_argument("--variants-a", nargs="+", required=True,
                        help="Exact variant names for BioLM-Score, one per seed")
    parser.add_argument("--variants-b", nargs="+", required=True,
                        help="Exact seed-matched GenScore variant names")
    parser.add_argument("--metrics-to-test", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, required=True,
                        help="Explicit bootstrap randomization value supplied by the caller")
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")

    metrics = pd.read_csv(args.metrics)
    similarity = pd.read_csv(args.protein_similarity, sep="\t")
    if not {"target", "max_identity_percent"}.issubset(similarity.columns):
        raise ValueError("Similarity file must contain target and max_identity_percent columns")
    if similarity["target"].duplicated().any():
        raise ValueError("Protein similarity file contains duplicate targets")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for metric_index, metric in enumerate(args.metrics_to_test):
        deltas = paired_target_deltas(metrics, args.variants_a, args.variants_b, metric)
        joined = deltas.merge(
            similarity[["target", "max_identity_percent"]],
            on="target",
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not (joined["_merge"] == "both").all():
            missing = joined.loc[joined["_merge"] != "both", "target"].tolist()
            raise ValueError(f"Target coverage differs between metrics and similarity: {missing}")
        joined = joined.drop(columns="_merge").sort_values("target")
        x = joined["max_identity_percent"].to_numpy(dtype=float)
        y = joined["delta_a_minus_b_mean"].to_numpy(dtype=float)
        if len(x) < 3 or not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError(f"Insufficient or non-finite target-level data for {metric}")
        p_result = pearsonr(x, y)
        s_result = spearmanr(x, y)
        low_p, high_p = bootstrap_correlation_ci(
            x, y, "pearson", args.bootstrap_replicates, args.bootstrap_seed + metric_index
        )
        low_s, high_s = bootstrap_correlation_ci(
            x, y, "spearman", args.bootstrap_replicates, args.bootstrap_seed + 100 + metric_index
        )
        joined["metric"] = metric
        joined.to_csv(args.output_dir / f"target_deltas_{metric}.csv", index=False)
        summaries.extend([
            {"metric": metric, "method": "Pearson", "n_targets": len(joined),
             "correlation": float(p_result[0]), "p_value": float(p_result[1]),
             "target_bootstrap_ci95_low": low_p, "target_bootstrap_ci95_high": high_p},
            {"metric": metric, "method": "Spearman", "n_targets": len(joined),
             "correlation": float(s_result[0]), "p_value": float(s_result[1]),
             "target_bootstrap_ci95_low": low_s, "target_bootstrap_ci95_high": high_s},
        ])

    summary = {
        "comparison": {"model_a": args.variants_a, "model_b": args.variants_b,
                       "delta": "target-level mean across seed-matched (A - B) metrics"},
        "similarity": "maximum sequence identity to the supervised training-protein reference, in percent",
        "resampling_unit": "DEKOIS target, paired model predictions within each seed before averaging seed differences",
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "results": summaries,
        "warning": "Use only the primary BioLM-Score and seed-matched reproduced GenScore variants requested by the manuscript; do not substitute shuffled or ablation controls.",
    }
    (args.output_dir / "gain_vs_similarity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(summaries).to_csv(args.output_dir / "gain_vs_similarity_summary.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
