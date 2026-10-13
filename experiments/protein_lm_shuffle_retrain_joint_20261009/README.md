# Protein-LM shuffle controls (joint GatedGCN)

This directory contains the training and evaluation code for the two training-stage protein-LM shuffle controls used in the study: `within_pocket_lm_shuffle` and `cross_target_lm_shuffle`. It contains source code and a lightweight identifier/group map only. It does not contain benchmark structures, affinity/activity labels, graph tensors, embeddings, checkpoints, or server logs.

The experiment workers are intended for Linux with CUDA; some use POSIX file locks and the CUDA/PyTorch-Geometric stack. GitHub Desktop on Windows can sync the source, but is not the supported runtime for model training or inference.

## Data and feature inputs

Obtain PDBbind v2020 and CASF-2016 from their providers, prepare the PDBbind training graphs with the main repository's documented graph-preparation procedure, and generate protein and ligand embeddings with the scripts in `esm3/` and `chemformer/`. The embedding arrays are local generated inputs; they are not GitHub artifacts. Keep the original trained ESM-C/Chemformer weights separate and follow their providers' terms. The PDBbind sample IDs and CASF exclusions are read from the prepared local `v2020_train_ids.npy` and `BioLM_Score/data/casf_excluded_ids.txt`.

The cross-target shuffle uses the committed identifier-only file `reproducibility/identifiers/PDBbind_v2020_MMseqs2_100_80_target_groups.tsv`. Its two columns are representative PDB ID and member PDB ID. This is the exact donor-exclusion map used by the experiment: 19,015 members, SHA-256 `67d37046dad63981fe4bd222c105246df27bec9cf36c71a51a1d1e4609069e20`. Groups were defined by MMseqs2 at 100% sequence identity with bidirectional sequence coverage of at least 80%; the file contains no sequence or structural data. The MMseqs2 executable version was not retained in the available run metadata, so use this frozen, hash-checked map rather than silently regenerating a potentially different one.

## Build local embedding banks

The expected inputs are 19,015 ordered training IDs/labels, corresponding protein and ligand graph files, one protein embedding `.npy` and ligand embedding `.npy` per complex, and the committed group map. Protein embeddings must have 1,152 columns and one row per pocket residue; ligand embeddings must have 1,024 columns and one row per ligand atom. The preparation script checks counts, graph/embedding row alignment, finite values, CASF exclusion, and group-map coverage before writing derived local caches and SHA-256 metadata.

```bash
python experiments/protein_lm_shuffle_retrain_joint_20261009/runtime/prepare_shuffle_assets.py \
  --project-root /path/to/local/project-with-data \
  --ids-file /path/to/v2020_train_ids.npy \
  --protein-graphs /path/to/v2020_train_prot.pt \
  --ligand-graphs /path/to/v2020_train_lig.pt \
  --protein-embeddings-dir /path/to/pdbbind_protein_embeddings \
  --ligand-embeddings-dir /path/to/pdbbind_ligand_embeddings \
  --output-dir /path/to/local/embedding_bank
```

## Train one control/checkpoint

The training setup is joint BioLM-Score GatedGCN training from scratch, with both protein and ligand LM features enabled, 17,515 training complexes and a seed-matched 1,500-complex validation set. It uses ordinary random mini-batches (batch size 64), learning rate `1e-3`, weight decay `1e-5`, dropout `0.15`, auxiliary-loss weight `1e-3`, 7 Å training cutoff, 5 Å scoring cutoff, maximum 2,000 epochs, and patience 70. Use the pre-specified independent run identifiers for the checkpoint set; their numeric values are intentionally not recorded in this public description. The shuffle RNG is a deterministic function of the model run identifier.

Run the command once for each model run identifier and control. For example:

```bash
python experiments/protein_lm_shuffle_retrain_joint_20261009/runtime/train_shuffle_joint.py \
  --control within_pocket_lm_shuffle --seed "$MODEL_SEED" --shuffle-seed "$SHUFFLE_BASE_SEED" \
  --train-data-dir /path/to/prepared/pdbbind/train_data \
  --embedding-bank-dir /path/to/local/embedding_bank \
  --output outputs/shuffle/within_${MODEL_RUN_ID}/best.pth
```

Repeat with `--control cross_target_lm_shuffle` and each remaining run identifier. The trainer combines the model run identifier with the base shuffle value, writes the exact train/validation ID lists and input/configuration hashes beside the checkpoint, and retains each checkpoint independently. Do not average predictions across checkpoints.

## Evaluate DEKOIS 2.0

Use the six saved checkpoints as separate, run-matched runs. The model manifest format is shown in `reproducibility/manifests/shuffle_control_model.template.json`; create a one-checkpoint manifest for each run identifier, set its checkpoint path, and run both controls with the corresponding shuffle RNG. The DEKOIS worker imports the released GenScore reference implementation as well as this repository, so make that separate upstream checkout importable in the environment.

```bash
python experiments/protein_lm_shuffle_retrain_joint_20261009/runtime/dekois_strict_controls_worker.py \
  --project-root /path/to/BioLM-Score \
  --data-root /path/to/prepared/DEKOIS_2.0x \
  --output-root outputs/shuffle/dekois \
  --model-manifest /path/to/model_run_A.json \
  --targets 11betahsd1 17betahsd1 \
  --controls within_pocket_lm_shuffle cross_target_lm_shuffle \
  --shuffle-seed "$SHUFFLE_SEED_A" --write-descriptors
```

Repeat for all 81 identifiers in `reproducibility/identifiers/DEKOIS2.0_target_ids.txt` and for each independent run, using the matching checkpoint and derived shuffle value. The worker emits pose-level raw scores for each control/checkpoint; `scripts/analyze_shuffle_dekois.py` reduces each compound to its maximum-scoring pose and reports AUROC, BEDROC80.5, and EF1%, along with checkpoint-level mean and sample SD. The per-checkpoint CSVs are intermediate audit data; for summary reporting, use the aggregate mean ± sample SD rather than listing all checkpoint values. Optional target-level paired tests require a per-target/per-run baseline metrics CSV for Full BioLM-Score and Graph-only/GenScore.

For the Main Table 3 zero-protein-feature diagnostic, use a full BioLM-Score checkpoint manifest with `--controls zero_protein_feature`. This is an inference-only strict-ligand intervention; it does not retrain a model and does not alter the graph, coordinates, ligand features, or pair mask.

```bash
python scripts/analyze_shuffle_dekois.py \
  --scores-root outputs/shuffle/dekois \
  --variants within_pocket_lm_shuffle_joint_biolm_run_A \
             within_pocket_lm_shuffle_joint_biolm_run_B \
             within_pocket_lm_shuffle_joint_biolm_run_C \
             cross_target_lm_shuffle_joint_biolm_run_A \
             cross_target_lm_shuffle_joint_biolm_run_B \
             cross_target_lm_shuffle_joint_biolm_run_C \
  --output-dir outputs/shuffle/dekois/analysis
```

## Checkpoint distribution

Checkpoints are intentionally excluded from Git. If the checkpoints are distributed, upload the six shuffle-control checkpoints as a versioned Zenodo record, publish the DOI, and record each filename, run identifier, control, byte size, and SHA-256 in the paper/repository release notes. The existing Zenodo record cited by the root README contains the earlier main BioLM-Score/GenScore weights; it must not be assumed to contain these later shuffle-control checkpoints.
