# Reproducing the experiments

The commands below are entry-point templates, not self-contained downloads: benchmark files, trained checkpoints, and pretrained ESM-C/Chemformer weights must be obtained separately; per-complex embeddings are generated locally by the documented scripts. For BDB2020+ and the strict DEKOIS worker, the upstream GenScore source checkout must also be obtained separately. No command writes into a fixed machine-specific directory; all locations are supplied by the user.

Run `python scripts/validate_reproducibility_manifests.py` first. It checks the committed identifier counts, duplicate IDs, LP train/validation separation, and the 88-to-81 DEKOIS chain-query mapping.

The checkpoint manifest examples use symbolic run labels such as `RUN_A` and are templates rather than ready-to-run manifests; replace those labels with the identifiers used by the local checkpoint files before evaluation.

## Main and supplementary table map

| Result | Data / IDs | Entry points |
|---|---|---|
| Main Table 1: standard PDBbind CASF-2016 scoring, ranking, docking, and screening | PDBbind v2020 processed complexes and CASF-2016; CASF benchmark data/labels are not bundled. | `scripts/casf2016_scoring_ranking.py`, `scripts/casf2016_docking.py`, `scripts/casf2016_screening.py`; then the CASF metric scripts supplied with the benchmark/code release. |
| Main Table 2 and DEKOIS target-level analyses | DEKOIS 2.0x local release; target IDs and chain-query map in `reproducibility/identifiers/`. | `scripts/prepare_dekois2_metadata.py`, `scripts/dekois2_joint_worker.py`, and `scripts/analyze_dekois2.py`. The analysis entry point also consumes the completed raw scores, similarity metadata, and a Surfdock summary as documented by its CLI. |
| Supplementary Table S5: BDB2020+ external evaluation | 115 PDB IDs in `reproducibility/identifiers/BDB2020plus_pdb_ids.txt`; structures/labels from the LP-PDBBind BDB2020+ archive. | `scripts/prepare_bdb2020plus_graphs.py`, `scripts/prepare_bdb2020plus_protein_embeddings.py`, `scripts/prepare_bdb2020plus_ligand_embeddings.py`, `scripts/bdb2020plus_eval.py`, and `scripts/analyze_bdb2020plus.py`. The evaluator requires a six-checkpoint JSON manifest; see `reproducibility/manifests/bdb2020plus_models.template.json`. |
| LP-PDBBind retraining and external evaluation | Upstream LP-PDBBind table filtered as specified in `DATA_SOURCES.md`; exact train/validation IDs are committed. | `scripts/prepare_lp_pdbbind_split_ids.py`, `scripts/prepare_lp_pdbbind_dataset.py`, `scripts/generate_lp_pdbbind_protein_embeddings.py`, `scripts/generate_lp_pdbbind_ligand_embeddings.py`, `scripts/run_lp_pdbbind_training.py`, `scripts/score_lp_pdbbind_bdb2020plus.py`, and `scripts/analyze_lp_pdbbind_bdb2020plus.py`. The actual trainer is `scripts/train_lp_pdbbind.py`, called by the coordinator. |
| Main Table 3: DEKOIS protein-information controls (graph-only, within-pocket shuffle, cross-target shuffle, zero-protein-feature) | Same DEKOIS 2.0x data as Main Table 2; the two shuffle controls use the independently trained checkpoints described below. | Graph-only: evaluate the reproduced GenScore checkpoint with `scripts/dekois2_joint_worker.py`. Shuffle controls: `experiments/protein_lm_shuffle_retrain_joint_20261009/runtime/dekois_strict_controls_worker.py --controls within_pocket_lm_shuffle cross_target_lm_shuffle`, summarized with `scripts/analyze_shuffle_dekois.py`. Zero-protein-feature: use the same worker with `--controls zero_protein_feature`; this is an inference-only strict-ligand intervention and does not require a separately trained checkpoint. |
| Main Table 4: score normalization and inference-cutoff sensitivity | Same CASF-2016 and DEKOIS 2.0 data as Main Tables 1–2; no additional training is required. | CASF cutoff sweeps use `experiments/protein_lm_shuffle_retrain_joint_20261009/runtime/casf_joint_family_worker.py --cutoffs`. DEKOIS score and graph cutoffs use `scripts/dekois2_joint_worker.py --score-cutoff` and `--graph-cutoff`; post-hoc normalization is summarized by `scripts/analyze_dekois2.py`. |
| Main Table 5: modality ablation (Full, Graph-only, W/O-protein-LM, W/O-ligand-LM, LM-only) | Same CASF-2016 and DEKOIS 2.0 data as Main Tables 1–2; each reported modality variant is evaluated from its corresponding checkpoint. | DEKOIS variants are selected through the model-manifest `ablation` field in `scripts/dekois2_joint_worker.py`; CASF variants use the corresponding model-manifest evaluator in `experiments/protein_lm_shuffle_retrain_joint_20261009/runtime/casf_joint_family_worker.py`. `scripts/train_model.py` is the standard full-model trainer and does not expose modality-ablation switches. |
| Protein-LM shuffle-control supplementary results | Six independently trained joint GatedGCN checkpoints: two shuffle controls (`within_pocket_lm_shuffle`, `cross_target_lm_shuffle`) × multiple independent runs. Training uses 17,515 complexes and a seed-matched 1,500-complex validation set; the shuffle RNG is a deterministic function of the model run identifier. The exact cross-target donor-exclusion map is committed as an identifier-only TSV and hash-checked. | Training and asset preparation: `experiments/protein_lm_shuffle_retrain_joint_20261009/runtime/{prepare_shuffle_assets.py,train_shuffle_joint.py}`. CASF: entry points above. DEKOIS: `runtime/dekois_strict_controls_worker.py`; summarize raw pose scores with `scripts/analyze_shuffle_dekois.py`. See the experiment README for per-run commands and the three-checkpoint mean/sample-SD convention. Do not ensemble predictions. |
| Time-split results | Exact `train_ids.txt`, `val_ids.txt`, and `casf_excluded_ids.txt` in `BioLM_Score/data/`. | `scripts/train_model.py` with the listed IDs used to prepare the data; use the model's existing CASF entry points for evaluation. |

## Standard PDBbind training split

For the random split, the prepared PDBbind training graph collection is passed to `scripts/train_model.py`. The script calls `PDBbindDataset.train_and_test_split(valnum=1500, seed=...)`; the held-out 1,500 complexes are sampled from that collection for validation, and the rest are used for training. This method sets NumPy's legacy RNG to the supplied seed, samples validation indices without replacement, and defines training indices with `numpy.setdiff1d`. The three-checkpoint comparison uses pre-specified independent run identifiers. To export and hash the exact membership lists from the same prepared ID array used by training:

```bash
python scripts/export_random_split_ids.py \
  --prepared-ids /path/to/train_data/v2020_train_ids.npy \
  --output-dir outputs/random_split_ids --seeds "$SEED_A" "$SEED_B" "$SEED_C" --valnum 1500
```

The generated JSON records the input ID-array hash, split implementation, each seed, counts, and hashes of the resulting ID lists. To keep model initialization and data-split randomness coupled as in the implementation, pass one seed per training run and preserve it in the checkpoint filename and run log.

Example (repeat once per independent run identifier, changing output names):

```bash
python scripts/train_model.py \
  --data_dir /data/pdbbind/prepared/train_data \
  --data_prefix v2020_train \
  --protein_embedding_dir /data/embeddings/esm_c/pdbbind/pocket_embeddings \
  --ligand_embedding_dir /data/embeddings/chemformer/pdbbind \
  --model_path outputs/mmgatedgcn_1.0_seed_A.pth \
  --log_dir outputs/logs --encoder gatedgcn --seeds "$SEED_A" \
  --batch_size 64 --num_epochs 2000 --valnum 1500 \
  --dist_threhold 7 --dist_threhold2 5 --patience 70
```

This script is the standard random-split trainer. It is not the LP-PDBBind trainer and does not generate the time-split IDs.

## CASF-2016 commands

The three entry points take explicit paths and do not set `CUDA_VISIBLE_DEVICES` internally; select the device externally (for example, with `CUDA_VISIBLE_DEVICES=0`). Example for scoring/ranking:

```bash
python scripts/casf2016_scoring_ranking.py \
  --model_path /path/to/checkpoint.pth --model_type biolm --encoder gatedgcn \
  --data-dir /path/to/prepared/casf_graphs --test-prefix v2020_casf \
  --protein-embeddings-dir /path/to/protein/pocket_embeddings \
  --ligand-embeddings-dir /path/to/ligand/casf2016 \
  --core-set-file /path/to/CASF-2016/power_scoring/CoreSet.dat \
  --output-dir outputs/casf/scoring --outprefix biolm_seed_A
```

For screening, supply `--casf-root` containing `coreset/` and `decoys_screening/`, the processed PDBbind root, the protein/ligand embedding directories, and an output directory. For docking, `--casf-root` must contain `coreset/` and `decoys_docking/`; the processed PDBbind root also supplies the native complex pose. See each script's `--help` for exact options. The original evaluation definitions and scoring functions are retained; only paths and device selection are externalized.

## DEKOIS 2.0 entry points

The light target/query lists are in `reproducibility/identifiers/`. Metadata generation takes the local DEKOIS data root and emits the per-target compound, pose-flow, and sequence metadata used by the analysis. To regenerate metadata:

```bash
python scripts/prepare_dekois2_metadata.py \
  --data-root /path/to/DEKOIS_2.0x --output-root outputs/dekois/metadata \
  --targets 11betahsd1 17betahsd1 a2a
```

The full target list can be supplied via a small shell loop or by omitting `--targets` to use subdirectories in the local data root. To score a target, provide a JSON model manifest and call:

```bash
python scripts/dekois2_joint_worker.py \
  --project-root . --data-root /path/to/DEKOIS_2.0x \
  --output-root outputs/dekois --model-manifest /path/to/dekois_models.json \
  --targets 11betahsd1 --write-descriptors
```

The manifest schema is illustrated in `reproducibility/manifests/dekois_joint_models.template.json`. The completed analysis is run with explicit raw/output roots and the Surfdock summary path:

```bash
python scripts/analyze_dekois2.py \
  --experiment-root outputs/dekois --output-dir outputs/dekois/analysis \
  --surfdock-summary /path/to/surfdock_target_metrics.csv
```

The association between target-level BioLM-Score gains and sequence similarity
can be calculated from the per-target/per-run metric
table and the same training-reference similarity table. Supply the three
run-matched primary model variants explicitly; the names below match the
template manifest. The script averages the paired BioLM-minus-GenScore metric
difference across seeds for each target and reports Pearson/Spearman
correlations with target-bootstrap 95% confidence intervals. It does not use
shuffle or modality-ablation controls as substitutes for the primary comparison.

```bash
python scripts/analyze_dekois_gain_vs_similarity.py \
  --metrics outputs/dekois/analysis/per_target_per_seed_metrics.csv \
  --protein-similarity outputs/dekois/metadata/protein_similarity_per_target.tsv \
  --variants-a full_biolm_run_A full_biolm_run_B full_biolm_run_C \
  --variants-b graph_only_genscore_run_A graph_only_genscore_run_B graph_only_genscore_run_C \
  --output-dir outputs/dekois/analysis/gain_vs_similarity
```

## BDB2020+ and LP-PDBBind

For BDB2020+, obtain the benchmark archive from the LP-PDBBind source, prepare graphs and features, and create a copy of the model template with paths to the local checkpoints. The reproduced GenScore model implementation is a separate dependency: clone [sc8668/GenScore](https://github.com/sc8668/GenScore) and make it importable in the active environment. The evaluator accepts explicit paths rather than repository-specific `/data*` paths:

```bash
python scripts/bdb2020plus_eval.py \
  --experiment outputs/bdb2020plus --model-manifest /path/to/bdb_models.json \
  --training-data /path/to/prepared/pdbbind_train \
  --training-protein-embeddings /path/to/protein/embeddings \
  --training-ligand-embeddings /path/to/ligand/embeddings
python scripts/analyze_bdb2020plus.py --experiment outputs/bdb2020plus
```

For LP-PDBBind, first regenerate and verify the IDs, then run the preparation, embedding, training, and external evaluation entry points listed in the table above. The training script requires explicit paths for the upstream structures, the study's original PDBbind data/features reused where available, and output artifacts. It records selected IDs and input hashes in its run metadata.

## Environment and version capture

Use `environment.yml` as the Python 3.9.19 training/evaluation environment, with CUDA-compatible versions for the GPU host. The vendored ESM-C feature-extraction sources currently use Python 3.10+ syntax and therefore require a separate compatible environment (or precomputed local embeddings) for feature generation. Record the commit hashes of this repository and external GenScore/ESM-C/Chemformer checkouts, the SHA-256 of downloaded archives/checkpoints, and the exact command line for every run. Per-complex embedding arrays and external checkpoint/model weights are not copied into this repository. If the six shuffle-control checkpoints are released, put them on a versioned Zenodo record and add its DOI plus per-file SHA-256 values; the existing Zenodo DOI is for earlier main-model weights only.

To rebuild graph and embedding inputs rather than use prepared local features, use the graph builder with explicit references and paths, then the corresponding feature generator. For example:

```bash
python BioLM_Score/feats/mol2graph_rdmda_res.py \
  --dir /path/to/PDBbind_v2020_processed --ref /path/to/pdbbind_affinity.csv \
  --filter-ids "" --cutoff 10 --outprefix /path/to/prepared/v2020_train

python chemformer/get_canonical_smiles_feat.py \
  --model-checkpoint /path/to/chemformer.ckpt --input-csv /path/to/canonical_smiles.csv \
  --output-dir /path/to/ligand_embeddings --vocabulary /path/to/bart_vocab.json

ESMC_600M_WEIGHTS=/path/to/esmc_600m.pth \
python esm3/get_pocket_embs_pipeline.py \
  --input-root /path/to/prepared_complexes --output-dir /path/to/protein_embeddings
```

These are feature-preparation examples; they do not download or redistribute PDBbind files or model weights. Confirm atom/residue alignment and missing-feature counts before training.
