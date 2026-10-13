# Dataset sources and redistributed files

This repository contains code and lightweight identifiers, not benchmark data. It does **not** redistribute protein structures, ligand structures or poses, affinity/activity labels, generated molecular embeddings, or trained checkpoints. Users obtain benchmark files and pretrained model weights from their providers, then use the repository's embedding-generation scripts to create local per-complex arrays. Counts and SHA-256 checks for the committed identifier/group lists are recorded in [`reproducibility/manifests/identifier_manifest.json`](../reproducibility/manifests/identifier_manifest.json) and checked by `scripts/validate_reproducibility_manifests.py`.

## Sources used in the experiments

| Resource | Source and access route | What is recorded here |
|---|---|---|
| PDBbind v2020 | [PDBbind](https://pdbbind-plus.org.cn/); access/download is subject to the provider's terms. The time-split train, validation, and CASF-exclusion IDs are in `BioLM_Score/data/`. Random-split training uses the prepared PDBbind training collection and the validation sampler documented in `REPRODUCIBILITY.md`. |
| LP-PDBBind | [THGLab/LP-PDBBind](https://github.com/THGLab/LP-PDBBind), including `dataset/LP_PDBBind.csv` and the published split metadata. | `reproducibility/identifiers/LP_PDBBind_model_compatible_{train,val}_ids.txt` records the exact non-covalent subset used by this study; `scripts/prepare_lp_pdbbind_split_ids.py` regenerates the lists from the upstream CSV. The source CSV SHA-256 recorded by the experiment is `1f18a0a341be0332d5a8190ca883c5020debe147816ecb920f5219cc4b3d2c55`. |
| BDB2020+ | The release used here is distributed in the [LP-PDBBind repository](https://github.com/THGLab/LP-PDBBind), under `dataset/BDB2020+.tgz`. | `reproducibility/identifiers/BDB2020plus_pdb_ids.txt` contains the 115 evaluated PDB IDs only. It does not contain the affinity labels or structures. |
| DEKOIS 2.0 | The benchmark is described in the [DEKOIS 2.0 paper](https://doi.org/10.1021/ci400115b); the original project site is [dekois.com](http://www.dekois.com/). The experiment used the locally obtained `DEKOIS_2.0x` release/layout. | `reproducibility/identifiers/DEKOIS2.0_target_ids.txt` records the 81 target directory IDs. `DEKOIS2.0_target_chain_map.tsv` records 88 sequence-query-to-target mappings (some targets have multiple chain queries); these are identifiers and sequence lengths, not protein sequences or structures. |
| CASF-2016 | [CASF benchmark resources](http://www.pdbbind.org.cn/casf.php), distributed with/alongside PDBbind under the applicable terms. | The repository provides evaluation code; benchmark structures, poses, labels, and precomputed embeddings are not included. |
| Main BioLM-Score / reproduced GenScore checkpoints | [Existing Zenodo record](https://doi.org/10.5281/zenodo.21878818). | This record contains the earlier main-model weights. It is not identified as hosting the later protein-LM shuffle-control checkpoints. If those are released, use a versioned Zenodo record and list each file's run identifier, control, byte size, and SHA-256. |
| Protein/ligand embedding files | Generated locally from provider-obtained sequences/structures and provider-obtained ESM-C/Chemformer weights. | Embedding-generation code is in `esm3/` and `chemformer/`; no per-complex embedding arrays or pretrained ESM-C/Chemformer weights are bundled. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for code-versus-weight and redistribution cautions. |
| Protein-LM shuffle target groups | Exact identifier-only mapping used in the 2026-10-09 cross-target shuffle experiment. | `reproducibility/identifiers/PDBbind_v2020_MMseqs2_100_80_target_groups.tsv` contains representative/member PDB IDs only; it does not include sequence or structure data. Its row count and SHA-256 are validated by the manifest checker. |

## LP-PDBBind subset definition

The model-compatible subset is a study-specific selection from the upstream LP-PDBBind table, not a claim that the 7,393/958 lists are standalone official split files. The training IDs satisfy `new_split=train`, `CL1=true`, and `covalent=false`; validation IDs satisfy `new_split=val`, `CL2=true`, and `covalent=false`. The source table version is pinned by the SHA-256 above. The corresponding counts are 7,393 training and 958 validation complexes. The upstream non-covalent CL2 test subset contains 2,171 IDs; the helper can regenerate that list too, but the training experiment used the train and validation subsets.

To regenerate the lists after downloading the upstream CSV:

```bash
python scripts/prepare_lp_pdbbind_split_ids.py \
  --lp-csv /path/to/LP-PDBBind/dataset/LP_PDBBind.csv \
  --output-dir reproducibility/identifiers/generated
```

If the upstream file hash differs, the script stops rather than silently using a different release. After checking the version change, `--allow-different-source-hash` permits regeneration, while the emitted JSON records the actual hash.

## Why the raw datasets are not mirrored here

The code repository is intentionally kept lightweight. Raw PDBbind/DEKOIS/BDB structures and benchmark labels remain with their providers; the repository supplies access links, study-specific identifiers, exact filters, preparation/evaluation entry points, and integrity checks. This avoids rehosting third-party benchmark materials and makes the required local data access explicit.
