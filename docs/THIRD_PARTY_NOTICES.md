# Third-party code and model notices

This repository includes vendored code from third-party projects. Each vendored directory carries its own LICENSE file. This repository does not currently provide a root-level project license for the authors' own code.

## Chemformer

- **Source:** https://github.com/MolecularAI/Chemformer
- **Vendored version:** 2.1.0 (per local `chemformer/pyproject.toml`)
- **License:** Apache License 2.0 (see `chemformer/LICENSE`)
- **Location in this repo:** `chemformer/`

The vendored code's license is determined by the upstream `LICENSE` file and `pyproject.toml`, both of which declare Apache 2.0. Note that the upstream `README.md` describes the software as MIT-licensed, which conflicts with the formal license file; this repository follows the `LICENSE` file and package metadata as the authoritative source.

The upstream `LICENSE` file does not fill in a specific copyright holder in its appendix. The `pyproject.toml` lists AstraZeneca affiliates as authors/maintainers, but this repository does not assert a copyright attribution beyond what the upstream sources explicitly state.

Redistribution of this code requires preserving the Apache 2.0 license and copyright notices, and marking any modified files.

**Model weights** are not included in this repository. Users must obtain Chemformer pretrained weights independently and comply with the terms provided by the weight distributor.

## ESM / ESM-C

- **Source:** https://github.com/evolutionaryscale/esm
- **Vendored version:** 3.1.5post1 (per local `esm3/esm/__init__.py`)
- **License:** EvolutionaryScale Cambrian Open License Agreement (see `esm3/LICENSE`)
- **Location in this repo:** `esm3/`
- **Full license text:** https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement

At version 3.1.5, the upstream `LICENSE.md` lists the GitHub source code under the **Cambrian Open License Agreement**. Note that the upstream `main` branch has since changed to MIT; the license stated here matches the vendored version.

Key redistribution requirements under the Cambrian Open License:
- Provide a copy of the license agreement to recipients
- Mark modified files with prominent notices
- Display "Built with ESM" where applicable
- Retain all copyright, patent, trademark, and attribution notices

**Model weights** are licensed separately and are not included in this repository:
- ESM-C 300M weights: Cambrian Open License Agreement
- ESM-C 600M weights: [Cambrian Non-Commercial License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement)
- ESM3 Open Model weights: Cambrian Non-Commercial License Agreement

Users must obtain model weights independently and comply with the applicable license terms from the provider.

### Additional copyright notices within ESM code

Some files within `esm3/esm/` contain code from other projects and retain their original copyright headers:

- `esm3/esm/layers/rotary.py` — Copyright 2022 EleutherAI and HuggingFace Inc. (Apache License 2.0)
- `esm3/esm/utils/residue_constants.py` — Copyright 2021 AlQuraishi Laboratory / DeepMind Technologies Limited (Apache License 2.0)

## Benchmarks and upstream implementations

PDBbind, CASF-2016, DEKOIS 2.0, and BDB2020+ structures, poses, and labels are not bundled. Obtain them from the providers/upstream releases and follow their access and redistribution conditions. Some comparison scripts require the separate GenScore source checkout; it is not vendored here and remains subject to its own terms.
