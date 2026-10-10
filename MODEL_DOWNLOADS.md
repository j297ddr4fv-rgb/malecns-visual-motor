# Download the upstream models

This repository supplies experiment code and outputs. Obtain models from their original providers; weights and third-party body assets are not mirrored here.

## FlyVis fitted optic-lobe model

- Official repository: https://github.com/TuragaLab/flyvis
- Documentation: https://turagalab.github.io/flyvis/
- Project runtime version: `flyvis==1.2.0`.
- Official pretrained archive folder: https://drive.google.com/drive/folders/15_8gPaVVJV6wGAspwkrdN8r-2NxMY4Kj

In the selected compatible FlyVis environment, run:

```sh
export FLYVIS_ROOT_DIR=/path/to/flyvis-data
flyvis download-pretrained
```

The model selected for this work is `flow/0000/000`, with checkpoint at `results/flow/0000/000/best_chkpt` relative to the FlyVis data root. The downloader may also retrieve its UMAP archive.

| File | SHA-256 recorded for this study |
|---|---|
| `results_pretrained_models.zip` | `71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1` |
| Selected `best_chkpt` | `d8a57a022aa18ca338599be713af9db29b89cde4b00098ffe42b78f0d583e46a` |

Package code licensing and checkpoint redistribution rights are distinct. Follow upstream terms when obtaining the checkpoint.

## MaleCNS connectome

- Official v1.0 dataset download page: https://male-cns.janelia.org/download/
- Original flat-table object prefix: https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
- Project graph size after transformation: 176,422 neurons and 6,287,749 directed connections.

The downloadable original tables are not the project's derived CSR arrays. SOURCE_ASSETS.md documents the threshold, autapse removal, index order, sign conventions and graph-file checksums. `scripts/flyb_to_csr.py` converts the intermediate FLYB format; obtaining the raw tables alone does not recreate every transformation automatically. The project-specific T4/T5 target map also remains a separate required asset for fresh simulation.

## Six-legged body model

- Pinned upstream release: https://github.com/NeLy-EPFL/flygym/tree/v2.1.0
- Recorded runtime: FlyGym 2.1.0, MuJoCo 3.9.0.
- Body model: the default NeuroMechFly v2 adult-female body supplied with FlyGym, coupled here to the male CNS graph.

Use the upstream installation instructions to obtain the runtime and packaged body assets. The package/version reference files in this repository record the research environment and are not portable installation lockfiles.
