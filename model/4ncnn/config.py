# config.py
# Shared configuration for 4-channel CNN model

import torch

# --- Shared Hyperparameters ---
HPARAMS = {
    "input_depth": 64,
    "input_height": 64,
    "input_width": 64,
    "num_classes": 3,
    "classes_names": ["mitotic", "new_daughter", "stable"],
    # A more standard starting learning rate. Consider using a scheduler.
    "learning_rate": 1e-4,
    # Adjust based on your GPU VRAM. May need to be lowered.
    "batch_size": 16,
    "num_epochs": 300,
    "num_input_channels": 4,  # [t-1, t, t+1, segmentation_mask]
    # Class-specific sample limits (matches your dataset)
    "max_samples_per_class": {
        "mitotic": 213,
        "new_daughter": 433,
        "stable": 20919,
        "death": 24,
    },
    # Class weights calculated to balance the training loss
    "class_weights": [25.34, 12.47, 0.26],  # [mitotic, new_daughter, stable, death]
}

# --- Shared Constants ---
CLASS_NAMES = HPARAMS["classes_names"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# If using class weights, pass them to your loss function as a tensor
CLASS_WEIGHTS_TENSOR = torch.tensor(HPARAMS["class_weights"], device=DEVICE)

# --- File Structure Constants ---
TIME_POINTS = ["t-1", "t", "t+1"]
RAW_FILE_NAME = "raw_cropped.tif"
LABEL_FILE_NAME = "binary_label_cropped.tif"
