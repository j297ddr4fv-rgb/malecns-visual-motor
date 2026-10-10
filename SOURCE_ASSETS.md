# Provenance of the source model and local reproduction bundle

The public release contains source code, experiment outputs and figures. The descriptions below identify assets used by the source local reproduction bundle. The public release excludes CSR arrays, retinal replay inputs, the T4/T5 map and pretrained/body models; see THIRD_PARTY.md. Workstation-specific paths are shown as placeholders and are not required installation paths.


Paths and versions below are the values recorded on the workstation used to prepare this local bundle. The heavy FlyVis checkpoint and FlyGym body assets remain external.

## Project code

- Project: `malecns-gpu`; source checkout: `<workstation>/Projects/malecns-gpu`; `VERSION`: `0.1.1-dev`.
- The checkout has no configured Git remote. Its local tags include `v0.1.0`. The base source commit for the publication replay and analyzer is `90f25b50ddffbd0b1b9e0d95c8b8f889f54e82e8`; the bundle replay entry point adapts that source to bundled inputs and explicit model/CSR paths. The bundle contains only the replay's transitive source files under `code/`.
- Source components are copied from project `scripts/`, `config/`, and the local working copies in `Projects/scratch/malecns-bc-mechanism/` and `Projects/scratch/malecns-visual-illusions/`. Those scratch files have no separate verified license.
- No public repository URL or public release identifier for `malecns-gpu` is recorded.

## MaleCNS connectome

- Dataset: neuPrint `male-cns:v1.0`; publisher page: <https://male-cns.janelia.org/download/>; terms: CC BY 4.0.
- The local raw source tables are under `<external-assets>/datasets/raw/malecns/v1.0/`; the exact GCS object URLs, generations, sizes, and publisher MD5/CRC32C values are in that directory's `download-manifest.tsv`. The direct object prefix is `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`; no raw table is copied into this bundle.

  | Source table | Rows × columns | Bytes | GCS generation | Local SHA-256 |
  |---|---:|---:|---:|---|
  | `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 211,577 × 36 | 14,483,314 | 1780494878811468 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
  | `body-neurotransmitters-male-cns-v1.0.feather` | 1,835,518 × 10 | 43,282,834 | 1780894899156750 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
  | `body-stats-male-cns-v1.0-minconf-0.5.feather` | 88,384,522 × 11 | 778,062,826 | 1780494888472305 | `ca5dc83a26382ae70c8d8f42fc09ce2dbc1af7c03f3a001a1936b5e142540647` |
  | `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 151,856,684 × 3 | 1,051,241,946 | 1780494887545976 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |
- Included graph data: `data/csr-flyb/`, 176,422 neurons, 6,287,749 directed connections, 90,296,905 synapses, and 9,619 retina entries. The copied `meta.json` retains source metadata from the original CSR build, including its build-time source path `<workstation>/Projects/fly-brain-minecraft/src/main/resources/connectome/malecns-v1.0.flyb.gz`; that path is provenance text, not a runtime dependency.
- Data transformation: the upstream `fly-brain-minecraft` provenance records a threshold of at least five synapses, removal of 40 autapses, body-ID sorting/dense re-indexing, unsigned 16-bit synapse counts, separate neurotransmitter signs, and photoreceptor column assignment from the strongest hex-bearing postsynaptic target. The historical project conversion `scripts/flyb_to_csr.py` parses that FLYB v1 file and writes the NumPy CSR arrays, annotation tables, signs, and retina table without changing edge ordering; that converter and the FLYB source file are not included, because this bundle ships one derived CSR copy. The local FLYB source had SHA-256 `e33df182bed7a6f3ea279daf4790a82b05706d3d41e819a6a80c0473e8c559f3`. Its source repository README records `https://github.com/blendi-remade/fly-brain-minecraft`; the local source copy has no Git metadata.
- Neurotransmitter sign mapping in that source provenance: acetylcholine `+1`; glutamate, GABA, and histamine `-1`; dopamine, octopamine, serotonin, and unclear/missing `+1`. Predicted transmitter labels are used only when the consensus label is not in the retained set and predicted confidence is at least 0.5; otherwise cell-type prediction is the fallback.
- Included CSR file SHA-256 values:

  | File | SHA-256 |
  |---|---|
  | `annotations.npz` | `00c566182480894db54436061598f4378032531287aa7e7c50d312a43e841595` |
  | `ids.npy` | `0be708c85482f0d2d42235c37235cf2f758b2314ebe16962ca807174ce507c96` |
  | `meta.json` | `2c970067dbbd6bae7ee4fdf3702a658c5f464901754aa19fda2fe00c63c1b332` |
  | `offsets.npy` | `c7d9ded6bf125e8a072333604d3b36d978fa5221a05e3890fba8c249a44ebbcc` |
  | `retina.npy` | `fde520751890ad2d4050492b4fc4a80f02b57991897c120b2acfabd03dfa7fa4` |
  | `sign.npy` | `406c13433074eed2fe19fe7ad2dc8bd9f9397df8e035190299b0e9d875993024` |
  | `tables.json` | `d0360d11b018be618d5fdd107b6bd7b89302940532487005e8b56374f0d8f911` |
  | `target.npy` | `8360ef72d8c80f9b2be597aa694b275fca885c688a69318d2e3936303b08342f` |
  | `weight.npy` | `8b92f6f5526ab45c6b3cf97e7354f6be43835121795b6cacd7d9f243dc6b20ec` |

## FlyVis optic-lobe model

- Runtime package: FlyVis `1.2.0`; Python `3.12.3`; installed package terms: MIT according to PyPI release metadata. PyPI provenance records package commit `92b3845cc426dd309a1a0e1b3890156c42e14021`.
- Environment: `<external-assets>/environments/projects/flyvis-opticlobe`; executable: `<external-assets>/environments/projects/flyvis-opticlobe/bin/python`.
- Model root: `<external-assets>/models/flyvis`; selected view: `flow/0000/000`; checkpoint: `results/flow/0000/000/best_chkpt`; local checkpoint SHA-256 `d8a57a022aa18ca338599be713af9db29b89cde4b00098ffe42b78f0d583e46a`.
- Official downloader archive `results_pretrained_models.zip` has expected SHA-256 `71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1`; downloader Google Drive folder ID `15_8gPaVVJV6wGAspwkrdN8r-2NxMY4Kj`. The archive and checkpoint are not included because no separate asset-level license was found.
- Included mapping `code/artifacts/flyvis-eye/t4t5_dirs.npz`: 13,582 target rows; fields `idx`, `dirs`, `w`, `type`; SHA-256 `66bedaf757e967fdd89bcd00bd3be7e49f477df54fe025212c180d12e6f749ed`. The worker cross-checks target indices/subtypes against the MaleCNS CSR annotations. This mapping is distinct from the FlyVis checkpoint and has no separate asset license record.

## FlyGym / NeuroMechFly body

- FlyGym repository release v2.1.0 declares Apache-2.0; its default packaged NeuroMechFly v2 model is the adult-female mesh/configuration. Larger FlyBody/FlyMimic assets are fetched separately and are outside this bundle.
- Recorded runtime: FlyGym `2.1.0`, Python `3.12.3`, MuJoCo `3.9.0`, environment `<external-assets>/environments/projects/flygym-reference`; upstream commit `ca65a510c2afe6ac61c51df4f274c8d190c2f95f`; Apache-2.0.
- The project reference runtime uses FlyGym v2's default adult-female NeuroMechFly model; it is not a MaleCNS-specific male body reconstruction. Body packages/assets are not in this bundle and are not used by the fixed-input replay.
- Included scene asset `models/optomotor-panel/stripe_panel.gltf` is 7,466 bytes, SHA-256 `ad214cf3fe9aeb9c8508b3ec648ee4d67b1026604e564867b8d86e0481649b9b`. Its scene metadata identifies a procedurally authored CC0-1.0 stripe panel.

## Replayed input data

- `data/common-inputs/` contains nine compressed NPZ archives from the intact-body reference runs, one for each static, +60 deg/s, and -60 deg/s stimulus at seeds 1729, 1730, and 1731. Each stores 2,600 binocular pixel frames, 2,600 thorax quaternions, and the FlyVis target index order. Their archive checksums and schedule/spec checksums are in `conditions.json`.
- Pixels and quaternions are replayed directly. `data/experiment/` includes the completed 36-run protocol/state, paired/group CSVs, analysis, figures, and only per-run trace, summary, and graph-intervention records; worker subdirectories and logs are omitted. The package does not include original body trajectories, a body model, or the source 302-run output set.
