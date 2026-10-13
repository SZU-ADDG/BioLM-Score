# BioLM-Score

Research code for protein–ligand scoring and the experiments reported in the accompanying study. This repository provides model code, preprocessing and embedding-generation entry points, evaluation scripts, and lightweight reproducibility identifiers. It does **not** redistribute benchmark structures, poses, activity labels, per-complex embeddings, pretrained model weights, or trained checkpoints.

## Reproduce the reported results

Start with the [experiment guide](docs/REPRODUCIBILITY.md), which maps each main-text result to its scripts and gives path-based command templates. It covers standard PDBbind/CASF-2016 evaluation, DEKOIS 2.0, BDB2020+, LP-PDBBind, time-split evaluation, and the protein-LM shuffle controls.

Before running an experiment, validate the committed identifier lists and hashes:

```bash
python scripts/validate_reproducibility_manifests.py
```

Dataset access routes, study-specific filters, and the exact scope of the identifier files are listed in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md). The benchmark data and pretrained weights must be obtained from their respective providers; generated features and model outputs are written to user-selected local paths.

## Repository contents

- `BioLM_Score/`: datasets, graph/features, model, and core CASF evaluation code.
- `scripts/`: training, benchmark preparation/evaluation, analysis, and manifest validation entry points.
- `chemformer/`, `esm3/`: vendored source needed by the ligand/protein embedding-generation scripts; weights are not included.
- `reproducibility/identifiers/`: lightweight benchmark and split IDs, plus the protein-group membership map used for the shuffle control.
- `experiments/`: code and instructions for the protein-LM shuffle controls; no run outputs or checkpoints.

The detailed table-to-script map, environment notes, and reproduction commands are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Third-party code and model-weight notices are in [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).

## Environment

The root `environment.yml` describes the main training/evaluation environment. Embedding generation may require separate compatible environments for the vendored ESM-C and Chemformer code. See the experiment guide and third-party notices before installing or obtaining external components.
