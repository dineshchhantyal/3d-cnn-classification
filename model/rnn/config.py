# config.py
# Shared configuration for 4-channel CNN model

import torch

# --- Shared Hyperparameters ---
HPARAMS = {
    # --- MODIFIED: Input dimensions reduced to 32x32x32 ---
    "input_depth": 32,
    "input_height": 32,
    "input_width": 32,
    "num_classes": 3,
    "classes_names": ["mitotic", "new_daughter", "stable"],
    # --- Training Parameters ---
    "learning_rate": 1e-4,
    "batch_size": 16,
    "num_epochs": 300,
    "num_input_channels": 3,  # [t-1, t, t+1, segmentation_mask]
    # --- Regularization & Early Stopping ---
    "dropout_rate": 0.5,
    "weight_decay": 1e-5,  # L2 regularization for the Adam optimizer
    "early_stopping_patience": 25,  # Epochs to wait for improvement before stopping
    # --- Data Handling ---
    "max_samples_per_class": {
        "mitotic": 329,
        "new_daughter": 329,
        "stable": 329,
    },
    "class_weights": [1.0, 1.0, 1.0],  # [mitotic, new_daughter, stable]
}

# --- Shared Constants ---
CLASS_NAMES = HPARAMS["classes_names"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If using class weights, pass them to your loss function as a tensor
# Ensure they are float32 for the loss function
CLASS_WEIGHTS_TENSOR = torch.tensor(HPARAMS["class_weights"], dtype=torch.float32).to(
    DEVICE
)

# --- File Structure Constants ---
TIME_POINTS = ["t-1", "t", "t+1"]
RAW_FILE_NAME = "raw_cropped.tif"
LABEL_FILE_NAME = "binary_label_cropped.tif"
