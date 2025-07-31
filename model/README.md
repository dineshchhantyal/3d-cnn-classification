# ncnn4 Model Training and Prediction Documentation

> **Note:** All commands and instructions assume your working directory is `model/ncnn4`.

## Prerequisites

-   Ensure you are in the `model/ncnn4` directory:
    ```bash
    cd model/ncnn4
    ```

## Configuration

-   All training and prediction parameters are set in `config.py`.
-   Edit `config.py` to adjust hyperparameters, data paths, and model options.

## Model Training

1. **Prepare your dataset**

    - Place your training and validation data in the folders specified in `config.py` (see `DATASET_DIR`, etc.).

2. **Configure training parameters**

    - Edit `config.py` to set hyperparameters (epochs, batch size, learning rate, etc.).

3. **Start training**

    ```bash
    python train.py
    ```

    - Training logs and checkpoints will be saved in the `training_outputs/` directory.

    **Run training on HPC (SLURM):**

    ```bash
    sbatch slurm_train.sh
    ```

    - This will submit the training job to the cluster using the configuration in `slurm_train.sh`.
    - Always activate your Python environment before running training or prediction jobs.

    ```bash
    source ~/venvs/jupyter-gpu/bin/activate
    ```

    The default output directory for training results is `training_outputs/`. You can change this in `config.py`.

### Training Script Arguments

You can customize training by passing arguments to `train.py`:

| Argument      | Type | Default Value                                | Description                                                       |
| ------------- | ---- | -------------------------------------------- | ----------------------------------------------------------------- |
| --output_dir  | str  | HPARAMS["output_dir"] or "training_outputs"  | Directory to save all training outputs and artifacts.             |
| --num_epochs  | int  | HPARAMS["num_epochs"] or 100                 | Number of training epochs.                                        |
| --dataset_dir | str  | DATA_ROOT_DIR (from HPARAMS or default path) | Base directory for the dataset (if not using predefined datasets) |

#### Example usage:

```bash
python train.py --output_dir ./results --num_epochs 50 --dataset_dir /path/to/dataset
```

All arguments are optional and have sensible defaults from your config. You can override any default by specifying the argument on the command line.

### Training Workflow

1. **Environment Check**: The script checks if your data directory and config are set up correctly before starting.
2. **Data Loading**: Loads and prints the class distribution for your dataset. Supports per-class sample limits.
3. **Augmentation**: Applies conservative 3D augmentations for robust training.
4. **Training Loop**: Trains the model, validates, and saves the best checkpoint based on F1-score. Early stopping is supported.
5. **Artifacts**: Saves training metrics, confusion matrix, and a detailed classification report in the output directory.

### Output Files

-   `best_model.pth`: Best model checkpoint (highest validation F1-score)
-   `training_metrics.png`: Training/validation loss and accuracy plots
-   `final_confusion_matrix.png`: Confusion matrix for validation set
-   `final_classification_report.txt`: Detailed classification report and environment info

### Troubleshooting

-   If you see errors about missing data or config, check the paths in `config.py` and your command-line arguments.
-   For class mismatch errors, update `num_classes`, `classes_names`, and `class_weights` in `config.py`.

---

## Model Prediction

1. **Prepare input data**

    - Format your input data as required (see comments in `predict.py`).

2. **Run prediction**

    - For batch prediction on pre-cropped sample folders:
        ```bash
        python predict.py --model_path training_outputs/best_model.pth --folder_path <sample_folder1> <sample_folder2> ...
        ```
    - For prediction on full timestamp .tif volumes:
        ```bash
        python predict.py --model_path training_outputs/best_model.pth --volumes <t-1.tif> <t.tif> <t+1.tif> <mask.tif> --full_timestamp [--nuclei_ids 1,2,3]
        ```
    - For single sample prediction from .tif files:
        ```bash
        python predict.py --model_path training_outputs/best_model.pth --volumes <t-1.tif> <t.tif> <t+1.tif> <mask.tif>
        ```
    - Optional arguments:
        - `--save_analysis` : Save preprocessing and model analysis visualizations
        - `--output_dir <dir>` : Directory to save analysis outputs

    **Run prediction benchmarks on HPC (SLURM):**

    ```bash
    bash benchmark_all_models.sh <sample_folder1> <sample_folder2> ...
    ```

    - This will run predictions for all models in `training_outputs/` on the provided sample folders, using SLURM and the configuration in `benchmark_all_models.sh`.

## Output

-   Training outputs (logs, checkpoints) are saved in `training_outputs/`.
-   Prediction results and analysis are saved in the directory specified by `--output_dir` (default: `./analysis_output`).

## Troubleshooting & Customization

-   For troubleshooting, refer to logs in `training_outputs/` and analysis outputs.
-   For custom configurations, modify `config.py` and re-run the scripts.

---

---

## Example Commands (Copiable)

### Environment Activation

```bash
source ~/venvs/jupyter-gpu/bin/activate
```

### Training

```bash
sbatch slurm_train.sh
```

### Prediction

# Predict on a single sample folder

```bash
srun python predict.py --model_path /mnt/home/dchhantyal/3d-cnn-classification/model/ncnn4/training_outputs/no-aug/best_model.pth --folder_path /mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_065_nucleus_018_count_17
```

# Predict on 4 cropped volumes

```bash
srun python ncnn4/predict.py --model_path /mnt/home/dchhantyal/3d-cnn-classification/model/ncnn4/training_outputs/no-aug/best_model.pth --volumes /mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_027_nucleus_014_count_16/t-1/raw_cropped.tif /mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_027_nucleus_014_count_16/t/raw_cropped.tif /mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_027_nucleus_014_count_16/t+1/raw_cropped.tif /mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_027_nucleus_014_count_16/t/binary_label_cropped.tif -v
```

# Predict on full timestamp .tif volumes

```bash
srun python predict.py --model_path training_outputs/best_model.pth --volumes <t-1.tif> <t.tif> <t+1.tif> <mask.tif> --full_timestamp [--nuclei_ids 1,2,3]
```

Example 1: All nuclei in the volume will be predicted and results saved in `./analysis_output`.

```bash
srun python predict.py --model_path /mnt/home/dchhantyal/3d-cnn-classification/model/ncnn4/training_outputs/no-aug/best_model.pth --volumes /mnt/home/dchhantyal/3d-cnn-classification/raw-data/230212_stack6/registered_images/nuclei_reg8_29.tif /mnt/home/dchhantyal/3d-cnn-classification/raw-data/230212_stack6/registered_images/nuclei_reg8_30.tif /mnt/home/dchhantyal/3d-cnn-classification/raw-data/230212_stack6/registered_images/nuclei_reg8_31.tif /mnt/home/dchhantyal/3d-cnn-classification/raw-data/230212_stack6/registered_label_images/label_reg8_30.tif --full_timestamp
```

Example 2: Predict specific nuclei by IDs (e.g., 1, 2, 3).

```bash
srun python predict.py --model_path /mnt/home/dchhantyal/3d-cnn-classification/model/ncnn4/training_outputs/no-aug/best_model.pth --volumes /mnt/home/dchhantyal/3d-cnn-classification/raw-data/230212_stack6/registered_images/nuclei_reg8_29.tif /mnt/home/dchhantyal/3d-cnn-classification/raw-data/230212_stack6/registered_images/nuclei_reg8_30.tif /mnt/home/dchhantyal/3d-cnn-classification/raw-data/230212_stack6/registered_images/nuclei_reg8_31.tif /mnt/home/dchhantyal/3d-cnn-classification/raw-data/230212_stack6/registered_label_images/label_reg8_30.tif --full_timestamp --nuclei_ids "16,17"
```

### Benchmark All Models

```bash
bash benchmark_all_models.sh <sample_folder1> <sample_folder2> ...
```

---

**This documentation only covers ncnn4. For previous versions, there might be few changes required.**
