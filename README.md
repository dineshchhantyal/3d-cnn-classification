# 3D Nucleus State Classification in Time-Series Microscopy

## Project Summary

This project is a solution for extracting, cleaning, and classifying the states of cell nuclei in 3D time-series microscopy images. The main model, **4ncnn**, uses both spatial and temporal information, plus segmentation masks, to achieve high accuracy. Other models (CNN-only, 3ncnn, RNN) are included for comparison, but may not have the latest improvements.

---

## Folder Guide

-   `preperation/` — Scripts for extracting and cleaning data, and generating metadata
-   `data/` — Prepared datasets, organized by cell state and event
-   `model/4ncnn/` — The latest 4-channel CNN model: code, training, evaluation, benchmarking
-   `model/rnn/` — RNN/ConvLSTM models for sequence modeling
-   `video_generation/` — Video and visualization tools (**work in progress**)
-   `analysis_plots/` — Figures, workflow diagrams, and results

---

## Setup Environment

1. **Clone the repository:**
    ```bash
    git clone https://github.com/dineshchhantyal/3d-cnn-classification.git
    cd 3d-cnn-classification
    ```
2. **Create a virtual environment:**
    ```bash
    python -m venv ~/venvs/jupyter-gpu
    source ~/venvs/jupyter-gpu/bin/activate
    ```
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## How the Data is Prepared

1. **Extraction:**

    - For each nucleus event, crops are taken from three timepoints: t-1, t, and t+1, using the segmentation mask at t to localize the nucleus.
    - Data extraction scripts are in `preperation/` and can be run to generate metadata and crop volumes for each nucleus.
    - Each nucleus is saved in its own folder, containing raw crops for each timepoint and a binary mask for t.
    - The input for the 4ncnn model is a 4-channel 3D tensor: [t-1], [t], [t+1], [mask at t].
    - Augmentation includes random flips, small rotations, changes in intensity/contrast/brightness, and Gaussian noise (applied to raw channels). See `model/[model name]/cnn_model.py` for details.
    - All data is normalized to the [0, 1] range before training or inference.
    - For custom extraction or cleaning, modify scripts in `preperation/` and re-run them as needed.

More details on the extraction process can be found in the `preperation/README.md`.

---

## How to Train the 4ncnn Model

1. Complete setup as described in the "Setup Environment" section.

2. **Set training parameters:**
    - Edit `model/ncnn4/config.py` as needed
3. **Start training:**
    ```bash
    python model/ncnn4/train.py
    ```
    - Training logs and results will be saved in `model/ncnn4/training_outputs/`
4. **Benchmark all models:**
    ```bash
    bash model/ncnn4/benchmark_all_models.sh
    ```

More details on training and benchmarking can be found in the `model/README.md`.

---

## How to Evaluate and Visualize Results

1. **Metrics:**

    - After training, logs and checkpoints are saved in `training_outputs/`.
    - For prediction, use `predict.py` with the appropriate arguments to generate results and analysis. Example:
        ```bash
        python model/ncnn4/predict.py --model_path model/ncnn4/training_outputs/best_model.pth --folder_path <sample_folder1> <sample_folder2> ...
        ```
        or for full volume prediction:
        ```bash
        python model/ncnn4/predict.py --model_path model/ncnn4/training_outputs/best_model.pth --volumes <t-1.tif> <t.tif> <t+1.tif> <mask.tif> --full_timestamp
        ```
    - Prediction results and analysis are saved in the directory specified by `--output_dir` (default: `./analysis_output`).
    - For visualization and evaluation of training results, use the Jupyter notebooks in `model/notebooks/` (e.g., `visualization.ipynb` for overlays and projections, `check_random_label_data.ipynb` for label inspection).
    - Notebooks may require additional setup (e.g., installing `matplotlib`, `seaborn`).
    - For publication-quality figures or custom analysis, modify or create new notebooks in `model/notebooks/`.
    - For more details and troubleshooting, see the full documentation in `model/README.md`.

---

## Notes

-   **4ncnn** is the latest and recommended model. Older models may not include the latest improvements.
-   The `video_generation/` folder is a work in progress and may not be fully functional yet.

---

## References

1. Robust 3D Nuclear Instance Segmentation of the Early Mouse Embryo
   . [https://blastospim.flatironinstitute.org/html/](https://blastospim.flatironinstitute.org/html/)
