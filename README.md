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

## How the Data is Prepared

1. **Extraction:**
    - For each nucleus event, crops are taken from three timepoints: t-1, t, and t+1
    - The segmentation mask at t is used to localize the nucleus
2. **Saving:**
    - Each nucleus gets its own folder with raw crops, and a binary mask for t
3. **Input for 4ncnn:**
    - Each sample is a 4-channel 3D tensor: [t-1], [t], [t+1], [mask at t]
4. **Augmentation:**
    - Random flips, small rotations, changes in intensity/contrast/brightness, and Gaussian noise (applied to raw channels) (more at `model/[model name]/cnn_model.py`)
5. **Normalization:**
    - All data is scaled to the [0, 1] range

---

## How to Train the 4ncnn Model

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2. **Set training parameters:**
    - Edit `model/ncnn4/config.py` as needed
3. **Start training:**
    ```bash
    python model/ncnn4/cnn_model.py
    ```
    - Training logs and results will be saved in `model/ncnn4/training_outputs/`
4. **Benchmark all models:**
    ```bash
    bash model/ncnn4/benchmark_all_models.sh
    ```

---

## How to Evaluate and Visualize Results

1. **Metrics:**
    - Accuracy, F1-score, confusion matrix, and classification report are generated automatically
2. **Visualization:**
    - Use the Jupyter notebooks in `model/notebooks/` to see max projections, overlays, and more (might require additional setup)
    - Example:
    ```bash
    jupyter notebook model/notebooks/visualization.ipynb
    ```

---

## Notes

-   **4ncnn** is the latest and recommended model. Older models may not include the latest improvements.
-   The `video_generation/` folder is a work in progress and may not be fully functional yet.

---

## References

1. Robust 3D Nuclear Instance Segmentation of the Early Mouse Embryo
   . [https://blastospim.flatironinstitute.org/html/](https://blastospim.flatironinstitute.org/html/)
