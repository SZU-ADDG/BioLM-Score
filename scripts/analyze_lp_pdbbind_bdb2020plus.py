#!/usr/bin/env python3
"""Combine six LP-retrained BDB2020+ workers and analyze validation-calibrated scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


R_KCAL = 1.987204e-3
TEMPERATURE_K = 298.0
PKA_TO_DG = -R_KCAL * TEMPERATURE_K * np.log(10.0)
MODELS = ["BioLM-Score", "GenScore"]
SEEDS = [8, 23, 24]
PAPER_ROWS = [
    {
        "method": "AutoDock Vina",
        "training": "original",
        "rmse_kcal_mol": 3.31,
        "pearson_r": 0.23,
    },
    {
        "method": "AutoDock Vina",
        "training": "LP-PDBBind retrained",
        "rmse_kcal_mol": 2.10,
        "rmse_sd": 0.00,
        "pearson_r": 0.29,
        "pearson_sd": 0.00,
    },
    {
        "method": "IGN",
        "training": "original",
        "rmse_kcal_mol": 1.62,
        "pearson_r": 0.41,
    },
    {
        "method": "IGN",
        "training": "LP-PDBBind retrained",
        "rmse_kcal_mol": 1.38,
        "rmse_sd": 0.09,
        "pearson_r": 0.54,
        "pearson_sd": 0.04,
    },
    {
        "method": "RF-Score",
        "training": "original",
        "rmse_kcal_mol": 1.80,
        "pearson_r": 0.36,
    },
    {
        "method": "RF-Score",
        "training": "LP-PDBBind retrained",
        "rmse_kcal_mol": 1.61,
        "rmse_sd": 0.008,
        "pearson_r": 0.51,
        "pearson_sd": 0.008,
    },
    {
        "method": "DeepDTA",
        "training": "original",
        "rmse_kcal_mol": 1.98,
        "pearson_r": 0.18,
    },
    {
        "method": "DeepDTA",
        "training": "LP-PDBBind retrained",
        "rmse_kcal_mol": 1.72,
        "rmse_sd": 0.10,
        "pearson_r": 0.26,
        "pearson_sd": 0.07,
    },
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


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def calibrated_metrics(frame: pd.DataFrame):
    truth_pka = frame["pKa"].to_numpy(float)
    pred_pka = frame["predicted_pKa"].to_numpy(float)
    truth_dg = PKA_TO_DG * truth_pka
    pred_dg = PKA_TO_DG * pred_pka
    return {
        "n": len(frame),
        "pearson_r": float(pearsonr(pred_pka, truth_pka).statistic),
        "spearman_rho": float(spearmanr(pred_pka, truth_pka).statistic),
        "rmse_pKa": float(np.sqrt(np.mean((pred_pka - truth_pka) ** 2))),
        "mae_pKa": float(np.mean(np.abs(pred_pka - truth_pka))),
        "rmse_kcal_mol": float(np.sqrt(np.mean((pred_dg - truth_dg) ** 2))),
        "mae_kcal_mol": float(np.mean(np.abs(pred_dg - truth_dg))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    args = parser.parse_args()
    args.analysis.mkdir(parents=True, exist_ok=True)

    validation_frames = []
    bdb_frames = []
    worker_status = []
    for model in MODELS:
        for seed in SEEDS:
            key = f"{model.replace('-', '').lower()}_seed{seed}"
            status_path = args.raw / f"{key}_complete.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if not status.get("complete"):
                raise AssertionError(f"Incomplete worker: {key}")
            validation = pd.read_csv(args.raw / f"{key}_validation.csv")
            bdb = pd.read_csv(args.raw / f"{key}_bdb.csv")
            if len(validation) != 958 or len(bdb) != 115:
                raise AssertionError(
                    f"{key}: validation={len(validation)}, BDB={len(bdb)}"
                )
            validation_frames.append(validation)
            bdb_frames.append(bdb)
            worker_status.append(status)

    validation = pd.concat(validation_frames, ignore_index=True)
    bdb = pd.concat(bdb_frames, ignore_index=True)
    atomic_csv(args.raw / "validation_predictions.csv", validation)
    atomic_csv(args.raw / "bdb_predictions.csv", bdb)

    calibrated_groups = []
    calibrators = []
    per_seed = []
    for (model, seed), validation_group in validation.groupby(["model", "seed"]):
        x = validation_group["raw_score"].to_numpy(float)
        y = validation_group["pKa"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, deg=1)
        validation_pred = intercept + slope * x
        calibrators.append(
            {
                "model": model,
                "seed": int(seed),
                "slope_raw_to_pKa": float(slope),
                "intercept_raw_to_pKa": float(intercept),
                "validation_n": len(validation_group),
                "validation_pearson_r": float(
                    pearsonr(validation_pred, y).statistic
                ),
                "validation_spearman_rho": float(
                    spearmanr(validation_pred, y).statistic
                ),
                "validation_rmse_pKa": float(
                    np.sqrt(np.mean((validation_pred - y) ** 2))
                ),
                "validation_mae_pKa": float(np.mean(np.abs(validation_pred - y))),
            }
        )
        group = bdb[(bdb["model"] == model) & (bdb["seed"] == seed)].copy()
        group["predicted_pKa"] = intercept + slope * group["raw_score"]
        group["experimental_delta_g_kcal_mol"] = PKA_TO_DG * group["pKa"]
        group["predicted_delta_g_kcal_mol"] = PKA_TO_DG * group["predicted_pKa"]
        calibrated_groups.append(group)
        per_seed.append(
            {
                "model": model,
                "seed": int(seed),
                "raw_score_pearson_r": float(
                    pearsonr(group["raw_score"], group["pKa"]).statistic
                ),
                "raw_score_spearman_rho": float(
                    spearmanr(group["raw_score"], group["pKa"]).statistic
                ),
                **calibrated_metrics(group),
            }
        )

    calibrated = pd.concat(calibrated_groups, ignore_index=True)
    calibrators_frame = pd.DataFrame(calibrators).sort_values(["model", "seed"])
    per_seed_frame = pd.DataFrame(per_seed).sort_values(["model", "seed"])
    atomic_csv(args.analysis / "calibrators_from_lp_validation.csv", calibrators_frame)
    atomic_csv(args.analysis / "bdb2020_per_complex_calibrated.csv", calibrated)
    atomic_csv(args.analysis / "per_seed_metrics.csv", per_seed_frame)

    summary_rows = []
    for model, group in per_seed_frame.groupby("model"):
        row = {"model": model, "n_seeds": len(group)}
        for column in (
            "raw_score_pearson_r",
            "raw_score_spearman_rho",
            "pearson_r",
            "spearman_rho",
            "rmse_pKa",
            "mae_pKa",
            "rmse_kcal_mol",
            "mae_kcal_mol",
        ):
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_sd"] = float(group[column].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("model")
    atomic_csv(args.analysis / "three_seed_mean_sd.csv", summary)

    paired_rows = []
    for seed in SEEDS:
        bio = per_seed_frame[
            (per_seed_frame["model"] == "BioLM-Score")
            & (per_seed_frame["seed"] == seed)
        ].iloc[0]
        gen = per_seed_frame[
            (per_seed_frame["model"] == "GenScore")
            & (per_seed_frame["seed"] == seed)
        ].iloc[0]
        paired_rows.append(
            {
                "seed": seed,
                "delta_pearson_biolm_minus_genscore": float(
                    bio["pearson_r"] - gen["pearson_r"]
                ),
                "delta_spearman_biolm_minus_genscore": float(
                    bio["spearman_rho"] - gen["spearman_rho"]
                ),
                "delta_rmse_kcal_mol_biolm_minus_genscore": float(
                    bio["rmse_kcal_mol"] - gen["rmse_kcal_mol"]
                ),
            }
        )
    paired = pd.DataFrame(paired_rows)
    atomic_csv(args.analysis / "paired_seed_comparison.csv", paired)

    paper = pd.DataFrame(PAPER_ROWS)
    atomic_csv(args.analysis / "paper_table2_reference.csv", paper)
    comparison_rows = [
        {"source": "LP-PDBBind arXiv v2 Table 2", **row} for row in PAPER_ROWS
    ]
    for row in summary_rows:
        comparison_rows.append(
            {
                "source": "This work; mean of LP-retrained seeds 8/23/24",
                "method": row["model"],
                "training": "LP-PDBBind CL1/CL2 non-covalent; validation-only affine calibration",
                "rmse_kcal_mol": row["rmse_kcal_mol_mean"],
                "rmse_sd": row["rmse_kcal_mol_sd"],
                "pearson_r": row["pearson_r_mean"],
                "pearson_sd": row["pearson_r_sd"],
            }
        )
    atomic_csv(args.analysis / "paper_comparison.csv", pd.DataFrame(comparison_rows))

    bio = summary[summary["model"] == "BioLM-Score"].iloc[0]
    gen = summary[summary["model"] == "GenScore"].iloc[0]
    result = {
        "complete": True,
        "dataset": "BDB2020+",
        "n_complexes": 115,
        "training_split": {
            "official_raw": {"train": 11513, "validation": 2422, "test": 4860},
            "paper_filtered_used": {"train": 7393, "validation": 958, "test": 2171},
            "train_rule": "new_split=train AND CL1 AND non-covalent",
            "validation_rule": "new_split=val AND CL2 AND non-covalent",
            "test_rule": "new_split=test AND CL2 AND non-covalent",
            "fingerprint": "c41265c4e53c583237aa4f60a44089c707d5b2ac912869c6fa1afff642b36053",
        },
        "models": MODELS,
        "seeds": SEEDS,
        "calibration": (
            "For each model and seed, ordinary least squares raw_score -> pKa "
            "was fitted only on the exact LP validation set (n=958), then frozen "
            "before BDB2020+ evaluation. BDB labels were never used for fitting."
        ),
        "score_definition": (
            "Sum of MDN pair probability densities for observed protein-ligand "
            "distances <=5 Angstrom."
        ),
        "pKa_to_delta_g_factor_kcal_mol": float(PKA_TO_DG),
        "three_seed_summary": summary_rows,
        "paired_seed_comparison": paired_rows,
        "paper_reference": PAPER_ROWS,
        "key_differences": {
            "pearson_mean_biolm_minus_genscore": float(
                bio["pearson_r_mean"] - gen["pearson_r_mean"]
            ),
            "rmse_mean_biolm_minus_genscore_kcal_mol": float(
                bio["rmse_kcal_mol_mean"] - gen["rmse_kcal_mol_mean"]
            ),
        },
        "worker_status": worker_status,
    }
    atomic_json(args.analysis / "summary.json", result)

    output_files = [
        path
        for path in args.analysis.iterdir()
        if path.is_file() and path.name != "validation_complete.json"
    ]
    atomic_json(
        args.analysis / "validation_complete.json",
        {
            "complete": True,
            "expected_validation_rows": 6 * 958,
            "observed_validation_rows": len(validation),
            "expected_bdb_rows": 6 * 115,
            "observed_bdb_rows": len(calibrated),
            "finite_numeric_outputs": bool(
                np.isfinite(
                    per_seed_frame.select_dtypes(include=[np.number]).to_numpy()
                ).all()
            ),
            "files_sha256": {
                path.name: sha256(path) for path in sorted(output_files)
            },
        },
    )


if __name__ == "__main__":
    main()
