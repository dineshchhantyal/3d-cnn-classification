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
    sbatch sbatch.sh
    ```

    - This will submit the training job to the cluster using the configuration in `sbatch.sh`.
    - Always activate your Python environment before running training or prediction jobs.

    ```bash
    source ~/venvs/jupyter-gpu/bin/activate
    ```

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
sbatch sbatch.sh
```

### Prediction

# Predict on a single sample folder

```bash
srun python predict.py --model_path /mnt/home/dchhantyal/3d-cnn-classification/model/ncnn4/training_outputs/no-aug/best_model.pth --folder_path /mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_065_nucleus_018_count_17
```

# Predict on 4 cropped volumes

```bash
srun python predict.py --model_path /mnt/home/dchhantyal/3d-cnn-classification/model/ncnn4/training_outputs/no-aug/best_model.pth --volumes /mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_027_nucleus_014_count_16/t-1/raw_cropped.tif /mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_027_nucleus_014_count_16/t/raw_cropped.tif /mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_027_nucleus_014_count_16/t+1/raw_cropped.tif /mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_027_nucleus_014_count_16/t/binary_label_cropped.tif
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
