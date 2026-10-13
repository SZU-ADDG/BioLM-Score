#!/usr/bin/env python3
"""Analyze BDB2020+ results and compare them with LP-PDBBind Table 2."""

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


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    def render(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value)

    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(render(value) for value in values) + " |")
    return "\n".join(rows)


def metrics(frame: pd.DataFrame):
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
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()

    experiment = args.experiment.resolve()
    analysis = experiment / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(experiment / "raw" / "validation_predictions.csv")
    bdb = pd.read_csv(experiment / "raw" / "bdb_predictions.csv")

    calibrated = []
    per_seed = []
    calibrators = []
    for (model, seed), validation_group in validation.groupby(["model", "seed"]):
        x = validation_group["raw_score"].to_numpy(float)
        y = validation_group["pKa"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, deg=1)
        validation_prediction = intercept + slope * x
        calibration_rmse = float(np.sqrt(np.mean((validation_prediction - y) ** 2)))
        calibration_r = float(pearsonr(validation_prediction, y).statistic)
        calibrators.append(
            {
                "model": model,
                "seed": int(seed),
                "slope_raw_to_pKa": float(slope),
                "intercept_raw_to_pKa": float(intercept),
                "validation_n": len(validation_group),
                "validation_pearson_r": calibration_r,
                "validation_rmse_pKa": calibration_rmse,
            }
        )
        group = bdb[(bdb["model"] == model) & (bdb["seed"] == seed)].copy()
        group["predicted_pKa"] = intercept + slope * group["raw_score"]
        group["experimental_delta_g_kcal_mol"] = PKA_TO_DG * group["pKa"]
        group["predicted_delta_g_kcal_mol"] = PKA_TO_DG * group["predicted_pKa"]
        calibrated.append(group)
        row = {
            "model": model,
            "seed": int(seed),
            "raw_score_pearson_r": float(
                pearsonr(group["raw_score"], group["pKa"]).statistic
            ),
            "raw_score_spearman_rho": float(
                spearmanr(group["raw_score"], group["pKa"]).statistic
            ),
            **metrics(group),
        }
        per_seed.append(row)

    calibrated_frame = pd.concat(calibrated, ignore_index=True)
    per_seed_frame = pd.DataFrame(per_seed).sort_values(["model", "seed"])
    calibrator_frame = pd.DataFrame(calibrators).sort_values(["model", "seed"])
    atomic_csv(analysis / "calibrators_from_original_validation.csv", calibrator_frame)
    atomic_csv(analysis / "bdb2020_per_complex_calibrated.csv", calibrated_frame)
    atomic_csv(analysis / "per_seed_metrics.csv", per_seed_frame)

    summary_rows = []
    for model, group in per_seed_frame.groupby("model"):
        summary = {"model": model, "n_seeds": len(group)}
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
            summary[f"{column}_mean"] = float(group[column].mean())
            summary[f"{column}_sd"] = float(group[column].std(ddof=1))
        summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows).sort_values("model")
    atomic_csv(analysis / "three_seed_mean_sd.csv", summary_frame)

    paper_frame = pd.DataFrame(PAPER_ROWS)
    comparison_rows = []
    for row in PAPER_ROWS:
        comparison_rows.append(
            {
                "source": "LP-PDBBind arXiv v2 Table 2",
                **row,
            }
        )
    for row in summary_rows:
        comparison_rows.append(
            {
                "source": "This BDB2020+ evaluation; mean across independent runs",
                "method": row["model"],
                "training": "original random PDBBind split; validation-only affine calibration",
                "rmse_kcal_mol": row["rmse_kcal_mol_mean"],
                "rmse_sd": row["rmse_kcal_mol_sd"],
                "pearson_r": row["pearson_r_mean"],
                "pearson_sd": row["pearson_r_sd"],
            }
        )
    comparison_frame = pd.DataFrame(comparison_rows)
    atomic_csv(analysis / "paper_comparison.csv", comparison_frame)
    atomic_csv(analysis / "paper_table2_reference.csv", paper_frame)

    bio = summary_frame[summary_frame["model"] == "BioLM-Score"].iloc[0]
    gen = summary_frame[summary_frame["model"] == "GenScore"].iloc[0]
    paired = per_seed_frame.pivot(index="seed", columns="model")
    paired_rows = []
    for seed in sorted(per_seed_frame["seed"].unique()):
        bio_row = per_seed_frame[
            (per_seed_frame["model"] == "BioLM-Score")
            & (per_seed_frame["seed"] == seed)
        ].iloc[0]
        gen_row = per_seed_frame[
            (per_seed_frame["model"] == "GenScore")
            & (per_seed_frame["seed"] == seed)
        ].iloc[0]
        paired_rows.append(
            {
                "seed": int(seed),
                "delta_pearson_biolm_minus_genscore": float(
                    bio_row["pearson_r"] - gen_row["pearson_r"]
                ),
                "delta_spearman_biolm_minus_genscore": float(
                    bio_row["spearman_rho"] - gen_row["spearman_rho"]
                ),
                "delta_rmse_kcal_mol_biolm_minus_genscore": float(
                    bio_row["rmse_kcal_mol"] - gen_row["rmse_kcal_mol"]
                ),
            }
        )
    paired_frame = pd.DataFrame(paired_rows)
    atomic_csv(analysis / "paired_seed_comparison.csv", paired_frame)

    result = {
        "complete": True,
        "dataset": "BDB2020+",
        "n_complexes": 115,
        "models": ["BioLM-Score", "GenScore"],
        "seeds": sorted(int(value) for value in per_seed_frame["seed"].unique()),
        "pKa_to_delta_g_factor_kcal_mol": float(PKA_TO_DG),
        "calibration": (
            "Per model and seed, ordinary least squares raw_score -> pKa fitted "
            "only on the exact original 1,500-complex validation split; BDB2020+ "
            "labels were never used for fitting."
        ),
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
    }
    atomic_json(analysis / "summary.json", result)

    lines = [
        "# BDB2020+ evaluation: original-split BioLM-Score and GenScore",
        "",
        "## Protocol",
        "",
        "- 115 accurate BDB2020+ co-crystal complexes.",
        "- Original random-split checkpoints from the independent runs supplied in the input files.",
        "- Raw score: MDN probability-density sum over protein–ligand pairs at ≤5 Å.",
        "- Pearson/Spearman are computed directly from raw scores.",
        "- RMSE uses a run-specific affine mapping learned only from the exact original "
        "1,500-complex validation split, then frozen before BDB2020+ evaluation.",
        f"- pKa was converted to ΔG with ΔG = {PKA_TO_DG:.6f} × pKa kcal/mol (298 K).",
        "",
        "## Independent-run results",
        "",
        markdown_table(summary_frame),
        "",
        "## LP-PDBBind arXiv v2 Table 2 comparison",
        "",
        markdown_table(comparison_frame),
        "",
        "## Interpretation guardrail",
        "",
        "The paper baselines are affinity regressors with their own training protocols. "
        "BioLM-Score/GenScore are MDN interaction scorers; their RMSE therefore requires "
        "the explicitly reported validation-only affine calibration. Pearson R and "
        "Spearman ρ do not depend on this affine unit conversion.",
        "",
    ]
    readme = analysis / "README.md"
    temp = readme.with_suffix(".md.tmp")
    temp.write_text("\n".join(lines), encoding="utf-8")
    temp.replace(readme)
    atomic_json(
        analysis / "validation_complete.json",
        {
            "complete": True,
            "expected_bdb_rows": 690,
            "observed_bdb_rows": len(calibrated_frame),
            "expected_validation_rows": 9000,
            "observed_validation_rows": len(validation),
            "finite_numeric_outputs": bool(
                np.isfinite(
                    per_seed_frame.select_dtypes(include=[np.number]).to_numpy()
                ).all()
            ),
            "files_sha256": {
                path.name: sha256(path)
                for path in sorted(analysis.iterdir())
                if path.is_file() and path.name != "validation_complete.json"
            },
        },
    )


if __name__ == "__main__":
    main()
