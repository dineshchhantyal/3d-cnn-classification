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
    "learning_rate": 1e-5,
    "batch_size": 16,
    "num_epochs": 300,
    "num_input_channels": 4,  # [t-1, t, t+1, segmentation_mask]
    
    # Class-specific sample limits to reflect biological distribution
    "max_samples_per_class": {
        "mitotic": 216,
        "new_daughter": 216, 
        "stable": 216  # 3x more stable samples (more common in nature)
    },
    
    # Class weights to balance training loss (inverse of sample ratios)
    "class_weights": [1.0, 1.0, 1.0],  # [mitotic, new_daughter, stable]
}

# --- Shared Constants ---
CLASS_NAMES = HPARAMS["classes_names"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- File Structure Constants ---
TIME_POINTS = ["t-1", "t", "t+1"]
RAW_FILE_NAME = "raw_cropped.tif"
LABEL_FILE_NAME = "label_cropped.tif"
