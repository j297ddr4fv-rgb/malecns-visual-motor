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
python scripts/render_complex_stimulus_figure.py --data data/per_run_results.csv --out outputs/Fig12_illusions.pdf
```

The first command reconstructs each run's means from its neural trace, checks the recorded input identities across network arms and writes paired/group summaries and figures. The second command regenerates the complex-stimulus panel. Neither command launches a simulation or requires a GPU.

## Run new simulations

Main project entry points are `scripts/run_flygym_canonical_fullfield_optomotor_v3.py` and `scripts/run_publication_common_input.py`. The body experiments require FlyGym/MuJoCo plus the CUDA neural/vision environment; the fixed-input replay source requires FlyVis, PyTorch and Triton. Pinned environment records are supplied as `requirements-neural-reference.txt` and `requirements-body-reference.txt`; these are provenance records, not a universally portable CUDA installer.

Download external assets from their upstream providers using the links, pinned versions and checkpoint hashes in [MODEL_DOWNLOADS.md](MODEL_DOWNLOADS.md). SOURCE_ASSETS.md records model versions, source hashes, transformations and the selected FlyVis checkpoint. `replay/code/scripts/replay_common_input.py --help` lists explicit asset/path options. Pretrained weights, connectome CSR arrays, the T4/T5 target map, retinal input archives and full physics trajectories are not distributed in this public release. A fresh numerical replay therefore also requires these external assets; the public data are sufficient for reanalyzing the included outputs, not a one-command regeneration of all simulations.

The portable replay entry point was executed for original-network positive60, seed 1729, using the separate reviewer reproduction archive and upstream FlyVis checkpoint. All 2,600 numeric trace rows, retinal-stream hashes and fitted-current-stream hashes match the saved reference. This representative check does not validate every network arm or a fresh body simulation. The supplied outputs were produced by the project execution scripts identified in SOURCE_ASSETS.md. Machine-local data paths have been replaced by relative run identifiers; numerical fields and recorded input digests are retained. The current source snapshot includes other project experiments; only the three cohorts listed above supply data for this paper.

## Licenses and citation

Original code: **MIT** (`LICENSE`). Original experimental data and figures: **CC BY 4.0** (`DATA_LICENSE.md`). Third-party terms and attribution are listed in `THIRD_PARTY.md`. Use `CITATION.cff` to cite this work; it is an unpublished manuscript and no journal DOI is assigned.

Computing hardware was personally funded and privately owned by the author. Contact: 21620242202487@stu.xmu.edu.cn.

## Reanalysis verification and recording consistency

The trace analyzer was executed from a fresh checkout of the public repository with Python 3.12, NumPy 2.4.4 and Matplotlib 3.11.1. It reconstructed all 36 neural-run summaries and eight moving-versus-static group contrasts from CSV traces. The original-network contrasts are +81.1667 and -64.3333 Hz; the three null networks reproduce the reported large reductions. This check used an existing analysis environment; it was not an independent clean installation or a fresh neural/body simulation. Installing the pinned analysis requirements in a suitable Python environment avoids mixing the system Matplotlib binary with NumPy 2.

`data/mechanism/recapture_comparison.csv` compares the nine original-network synchronized recordings with their behavioral-batch counterparts. Eight yaw values agree at stored precision. Positive motion, seed 1730, gives 34.047248 degrees in the behavioral batch and 31.135036 degrees in the synchronized recapture. The corresponding three-seed summaries are +36.98 ± 3.01 and +36.01 ± 4.52 degrees. Their configurations and recorded initial states match; the source of the numerical difference has not been isolated. The two cohorts are retained separately and are not pooled as independent replicates.

## Representative neural replay

The accompanying `publication_reproduction.zip` supplies the derived CSR, T4/T5 map and nine cached retinal movies required for neural replay. It is a separate reviewer reproduction archive, not a pretrained-weight mirror. After extracting it and obtaining the selected FlyVis checkpoint, run its `code/scripts/replay_common_input.py --variant original --case-id speed_positive60_seed1729 --out /path/to/new/output` in a user systemd unit with `MemoryMax=5G`, `MemorySwapMax=0` and `CPUQuota=300%`. Pass `--flyvis-env` and `--model-root` for your environment paths. The archive README supplies the complete command. Without these derived inputs, the public checkout supports output reanalysis but not fresh neural replay.

The representative 13-s replay used about 114 s of computation (128 s total service runtime) and a 1.5-GiB memory peak on the research workstation. It matched all recorded DNp15 rates, raw locomotor drive and network spike counts. The replay configuration is the actual frozen high-contrast experiment configuration, rather than the project's general-purpose worker configuration.
