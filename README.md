# 3D Nucleus Classifier

A 3D CNN pipeline for classifying the state of cell nuclei in time-series fluorescence microscopy. Given a nucleus at time `t`, the model uses three consecutive volumes (`t-1`, `t`, `t+1`) and the segmentation mask at `t` to predict one of four states: **stable**, **mitotic**, **death**, or **new_daughter**.

The flagship model — **ncnn4** — is a 4-channel 3D CNN that consumes raw intensity volumes alongside the binary nucleus mask. Comparison baselines (single-frame CNN, 3-channel ncnn3, RNN) are kept in-tree for reference but are no longer maintained.

---

## Table of Contents

1. [Why this exists](#why-this-exists)
2. [Repository layout](#repository-layout)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Data preparation](#data-preparation)
6. [Models](#models)
7. [Training](#training)
8. [Prediction & evaluation](#prediction--evaluation)
9. [Video generation](#video-generation)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## Why this exists

Lineage-tracked 3D microscopy of early embryos produces thousands of nuclei per timepoint. Hand-labeling cellular events — divisions, deaths, freshly appeared daughters — is the bottleneck for downstream lineage analysis. This project trains a classifier that consumes the local 3D context around a nucleus and labels its state automatically, so downstream lineage pipelines can be run end-to-end.

Inputs and outputs:

- **Input**: 4-channel 3D tensor of shape `(4, D, H, W)` per nucleus — `[raw_{t-1}, raw_t, raw_{t+1}, mask_t]`, default crop `64×64×64`.
- **Output**: Softmax over `{stable, mitotic, death, new_daughter}`.

---

## Repository layout

```
.
├── preperation/        Data extraction & cleaning (LineageGraph.json → cropped nuclei)
├── model/
│   ├── ncnn4/          Flagship 4-channel CNN (recommended)
│   ├── ncnn3/          3-channel baseline (t-1, t, t+1, no mask)
│   ├── cnn/            Single-frame baseline
│   ├── rnn/            Recurrent baseline
│   └── notebooks/      Exploration & analysis notebooks
├── video_generation/   Render classifier predictions over time-series volumes
├── raw-data/           (Local) Input microscopy stacks + LineageGraph.json
├── requirements.txt
└── pyproject.toml
```

> Note: the `preperation/` directory name is preserved for backward compatibility with existing scripts and SLURM jobs.

---

## Installation

Tested on Python 3.10, PyTorch 2.7 with CUDA.

```bash
git clone https://github.com/dineshchhantyal/3d-nucleus-classifier.git
cd 3d-nucleus-classifier

python3.10 -m venv ~/venvs/jupyter-gpu
source ~/venvs/jupyter-gpu/bin/activate

pip install -r requirements.txt
```

Pinned dependency versions live in `requirements.txt` and `pyproject.toml`. For HPC use, `model/ncnn4/slurm_train.sh` and the various `preperation/sbatch_*.sh` scripts assume the venv at `~/venvs/jupyter-gpu`.

---

## Quick start

After installation:

```bash
# 1. Extract & crop nuclei from a registered microscopy dataset
python preperation/process_dataset.py \
    --dataset 230212_stack6 \
    --timeframe 1 \
    --output_dir data/nuclei_state_dataset

# 2. Train ncnn4
python model/ncnn4/train.py

# 3. Predict on a held-out sample folder
python model/ncnn4/predict.py \
    --model_path model/ncnn4/training_outputs/best_model.pth \
    --folder_path <sample_folder>
```

Each step is documented in detail below and in the subfolder READMEs (`preperation/README.md`, `model/README.md`, `video_generation/README.md`).

---

## Data preparation

The pipeline expects registered 3D stacks plus a lineage graph:

```
raw-data/<dataset>/
├── LineageGraph.json          # cell tracking & event labels
├── registered_images/         # raw fluorescence stacks per frame
└── registered_label_images/   # segmented nucleus labels per frame
```

`preperation/process_dataset.py` walks the lineage graph and emits a per-nucleus folder for every classified event, with the three timeframes and a binary mask:

```
data/nuclei_state_dataset/
├── stable/
├── mitotic/
├── death/
└── new_daughter/
    └── {dataset}_frame_{f}_nucleus_{id}_count_{n}/
        ├── t-1/raw_cropped.tif
        ├── t/
        │   ├── raw_cropped.tif
        │   ├── label_cropped.tif
        │   └── binary_label_cropped.tif
        └── t+1/raw_cropped.tif
```

Common invocations:

```bash
# Built-in dataset name (resolved from preperation/process_dataset.py)
python preperation/process_dataset.py \
    --dataset 220321_stack11 --timeframe 1 \
    --output_dir data/nuclei_state_dataset \
    --cube_max_size 64 --verbose

# Custom dataset directory and lineage file
python preperation/process_dataset.py \
    --dataset_dir raw-data/230212_stack6 \
    --lineage_file raw-data/230212_stack6/LineageGraph.json \
    --timeframe 1 \
    --output_dir data/nuclei_state_dataset \
    --max_samples 1000
```

See [`preperation/README.md`](preperation/README.md) for the full argument reference, metadata schema, and SLURM templates.

---

## Models

| Model | Input channels | Mask aware | Status | Location |
|-------|----------------|------------|--------|----------|
| **ncnn4** | `raw_{t-1}, raw_t, raw_{t+1}, mask_t` (4) | yes | **recommended** | `model/ncnn4/` |
| ncnn3 | `raw_{t-1}, raw_t, raw_{t+1}` (3) | no | baseline | `model/ncnn3/` |
| cnn   | `raw_t` (1) | no | baseline | `model/cnn/` |
| rnn   | sequence of raw volumes | no | experimental | `model/rnn/` |

Augmentations on the raw channels (defined in `model/ncnn4/cnn_model.py`): random flips, small rotations, intensity/contrast/brightness jitter, additive Gaussian noise. All channels are normalized to `[0, 1]` before forward pass.

---

## Training

Defaults live in `model/ncnn4/config.py`. Override at the command line:

```bash
python model/ncnn4/train.py \
    --output_dir model/ncnn4/training_outputs/run01 \
    --num_epochs 100 \
    --dataset_dir data/nuclei_state_dataset
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir`  | `HPARAMS["output_dir"]` or `training_outputs` | Where checkpoints, plots, reports go |
| `--num_epochs`  | `HPARAMS["num_epochs"]` or `100`              | Training epochs |
| `--dataset_dir` | `DATA_ROOT_DIR`                                | Root of the prepared dataset |

On SLURM:

```bash
cd model/ncnn4
sbatch slurm_train.sh
```

Outputs written to `model/ncnn4/training_outputs/<run>/`:

- `best_model.pth` — best-F1 checkpoint
- `training_metrics.png` — loss / accuracy curves
- `final_confusion_matrix.png` — validation confusion matrix
- `final_classification_report.txt` — per-class precision / recall / F1

Full reference: [`model/README.md`](model/README.md).

---

## Prediction & evaluation

**Batch prediction on pre-cropped sample folders:**

```bash
python model/ncnn4/predict.py \
    --model_path model/ncnn4/training_outputs/best_model.pth \
    --folder_path <sample_folder1> <sample_folder2> ...
```

Each `<sample_folder>` must contain `t-1/raw_cropped.tif`, `t/raw_cropped.tif`, `t/binary_label_cropped.tif`, and `t+1/raw_cropped.tif`.

**Full-volume prediction** (classify every nucleus in a timepoint):

```bash
python model/ncnn4/predict.py \
    --model_path model/ncnn4/training_outputs/best_model.pth \
    --volumes <t-1.tif> <t.tif> <t+1.tif> <mask.tif> \
    --full_timestamp [--nuclei_ids 1,2,3]
```

**Programmatic inference** (numpy arrays / `.npz` / `.tif`) — see `model/ncnn4/inference_example.ipynb`.

**Benchmark all checkpoints** in `training_outputs/`:

```bash
bash model/ncnn4/benchmark_all_models.sh <sample_folder1> <sample_folder2> ...
```

---

## Video generation

`video_generation/` renders classifier predictions over a time-series, producing MP4 / GIF visualizations with per-nucleus overlays. Sliding-window processing keeps memory bounded to three frames at a time. See [`video_generation/README.md`](video_generation/README.md) for configuration presets, projection options, and SLURM templates.

> Status: actively iterated on. APIs may shift.

---

## Troubleshooting

- **Class mismatch errors during training** — update `num_classes`, `classes_names`, and `class_weights` in `model/ncnn4/config.py` to match your prepared dataset.
- **Missing data / config errors** — verify `DATA_ROOT_DIR` in `config.py` resolves to a directory with the four class subfolders.
- **GPU OOM in video generation** — lower `config.batch_size` or set `config.use_gpu = False`.
- **HPC paths** — the SLURM scripts hard-code paths under `/mnt/home/dchhantyal/...`. Edit them to match your cluster.

---

## References

1. *Robust 3D Nuclear Instance Segmentation of the Early Mouse Embryo* — BlastoSPIM. <https://blastospim.flatironinstitute.org/html/>
