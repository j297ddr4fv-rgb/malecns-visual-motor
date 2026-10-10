# MaleCNS visual–motor simulation

Code and compact experimental data for **Connectome organization shapes visual direction responses in a six-legged virtual fly**, by Xvhang Liu (刘旭航), School of Life Sciences, Xiamen University.

The system links pose-dependent retinal sampling, a fitted FlyVis optic-lobe model, a MaleCNS-constrained spiking network, an explicit descending-neuron decoder and a six-legged FlyGym body controller. The decoder and gait controller are engineered components; the network provides locomotor power and steering signals.

## Included results

| Experiment | Completed runs | Public material |
|---|---:|---|
| Frozen-model behavioral evaluation | 302 | Per-run and grouped statistics; analysis input snapshot; reversal traces |
| Original vs one rewired network with body feedback | 18 | Neural/body time series, summaries and graph-intervention records |
| Identical-input original vs three rewired networks | 36 | Per-run neural traces, summaries, graph-intervention records and paired contrasts |

- `data/per_run_results.csv`, `data/group_results.csv`, `data/analysis_snapshot.json`: the 302-run behavior cohort.
- `data/mechanism/`: the 18-run body-feedback comparison. Large synchronized CSV files are losslessly gzip-compressed; use `gzip.open(path, "rt")` or decompress them to read.
- `data/common-input/`: the 36-run fixed-input neural comparison, including each neural trace.
- `data/development/`: baseline calibration and diagnostic records; these are not additional confirmatory runs.
- `figures/`: the manuscript's 13 saved figures.
- `scripts/`, `config/`, `tests/`: the project's source and configuration snapshot.
- `replay/code/`: the separate path-adapted fixed-input replay source used in the local review bundle.

The original network shows direction-dependent DNp15 contrasts. Global degree/strength-preserving target rewiring reduces those contrasts in the three sampled null networks. This is a result within the specified model and readout, not identification of a particular biological pathway.

## Analyze the supplied fixed-input results

Use a Python environment with NumPy and Matplotlib (recorded versions in `requirements-analysis.txt`). Run from the repository root:

```sh
python scripts/analyze_publication_common_input.py --root data/common-input --figures outputs/common-input
```

This command reconstructs each run's means from its neural trace, checks the recorded input identities across network arms and writes paired/group summaries and figures. It does not launch a simulation or require a GPU.

## Run new simulations

Main project entry points are `scripts/run_flygym_canonical_fullfield_optomotor_v3.py` and `scripts/run_publication_common_input.py`. The body experiments require FlyGym/MuJoCo plus the CUDA neural/vision environment; the fixed-input replay source requires FlyVis, PyTorch and Triton. Pinned environment records are supplied as `requirements-neural-reference.txt` and `requirements-body-reference.txt`; these are provenance records, not a universally portable CUDA installer.

Download external assets from their upstream providers using the links, pinned versions and checkpoint hashes in [MODEL_DOWNLOADS.md](MODEL_DOWNLOADS.md). SOURCE_ASSETS.md records model versions, source hashes, transformations and the selected FlyVis checkpoint. `replay/code/scripts/replay_common_input.py --help` lists explicit asset/path options. Pretrained weights, connectome CSR arrays, the T4/T5 target map, retinal input archives and full physics trajectories are not distributed in this public release. A fresh numerical replay therefore also requires these external assets; the public data are sufficient for reanalyzing the included outputs, not a one-command regeneration of all simulations.

The portable replay entry point has not been independently executed as a validation run. The supplied outputs were produced by the project execution scripts identified in SOURCE_ASSETS.md. Machine-local data paths have been replaced by relative run identifiers; numerical fields and recorded input digests are retained. The current source snapshot includes other project experiments; only the three cohorts listed above supply data for this paper.

## Licenses and citation

Original code: **MIT** (`LICENSE`). Original experimental data and figures: **CC BY 4.0** (`DATA_LICENSE.md`). Third-party terms and attribution are listed in `THIRD_PARTY.md`. Use `CITATION.cff` to cite this work; it is an unpublished manuscript and no journal DOI is assigned.

Computing hardware was personally funded and privately owned by the author. Contact: 21620242202487@stu.xmu.edu.cn.
