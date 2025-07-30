# ==============================================================================
#           Configuration for 4-Channel CNN Cell Classification Model
# ==============================================================================
# Welcome! This file contains all the settings for training the deep learning
# model. You can change these values to experiment with different model
# behaviors and training setups.
#
# Each setting is explained with a comment. Read the comments carefully before
# making changes.
# ==============================================================================

import torch

# --- Core Hyperparameters ---
# Hyperparameters are the "dials" you can turn to tune the model's performance.
HPARAMS = {
    # --------------------------------------------------------------------------
    # (1) Input Data Shape
    # Description: These settings define the size of the 3D image chunks
    #              (also called "cubes" or "voxels") that the model will see.
    #              The model processes data in the format (depth, height, width).
    #
    # Example: A 32x32x32 cube of pixels from a 3D microscope image.
    # --------------------------------------------------------------------------
    "input_depth": 32,  # The number of layers in the 3D cube (Z-axis).
    "input_height": 32,  # The height of the cube in pixels (Y-axis).
    "input_width": 32,  # The width of the cube in pixels (X-axis).
    "num_input_channels": 4,  # The number of data channels for each cube.
    # Default is 4: [Image at time t-1, Image at time t,
    #               Image at time t+1, Segmentation Mask]
    # --------------------------------------------------------------------------
    # (2) Classification Settings
    # Description: Define what the model is trying to classify. You need to
    #              specify the different categories (classes) and their names.
    #
    # IMPORTANT: The number of items in `num_classes`, `classes_names`, and
    #            `class_weights` must be the same.
    # --------------------------------------------------------------------------
    "num_classes": 3,  # The total number of categories the model predicts.
    # e.g., 4 for (mitotic, new_daughter, stable, death).
    "classes_names": [  # The names for each category. The order matters!
        "mitotic",  # Class 0
        "new_daughter",  # Class 1
        "stable",  # Class 2
    ],
    "class_weights": [  # Use to balance uneven datasets.
        1.0,
        1.0,
        1.0,
        # A higher value gives a class more importance during
    ],  # training. E.g., if 'death' is rare, you might set
    # its weight higher, like 5.0.
    # --------------------------------------------------------------------------
    # (3) Training & Optimization
    # Description: These settings control how the model learns from the data.
    # --------------------------------------------------------------------------
    "data_root_dir": "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3",  # The main folder where the training data is stored.
    # Change this if your data is in a different location.
    "learning_rate": 1e-5,  # How big of a step the model takes during learning.
    # Smaller values (e.g., 1e-6) are safer but slower.
    # Larger values (e.g., 1e-4) can learn faster but
    # might become unstable.
    "batch_size": 16,  # The number of data samples the model processes at
    # once. A larger batch size requires more memory (VRAM).
    # Powers of 2 (e.g., 8, 16, 32) are common.
    "num_epochs": 300,  # The total number of times the model will cycle
    # through the entire training dataset.
    # --------------------------------------------------------------------------
    # (4) Regularization & Early Stopping
    # Description: Techniques to prevent the model from "memorizing" the data
    #              (overfitting) and to stop training when it's no longer
    #              improving.
    # --------------------------------------------------------------------------
    "dropout_rate": 0.1,  # The probability of randomly ignoring some neurons
    # during training to make the model more robust.
    # A value between 0.1 and 0.5 is typical.
    "weight_decay": 1e-5,  # A penalty to keep the model's internal parameters
    # small, which helps prevent overfitting.
    "early_stopping_patience": 35,  # Number of epochs to wait for the model's
    # performance to improve before stopping the
    # training process automatically.
    # --------------------------------------------------------------------------
    # (5) Data Handling & Output
    # Description: Settings for managing the dataset and where to save results.
    # --------------------------------------------------------------------------
    "max_samples_per_class": {  # Limit the number of samples for each class.
        "mitotic": None,  # Set to `None` to use all available samples.
        "new_daughter": None,  # Or set a number, e.g., 1000, to limit it.
        "stable": None,
        "death": None,
    },
    "output_dir": "training_outputs",  # The folder where training results, logs,
    # and saved models will be stored.
}


# ==============================================================================
# --- Shared Constants (Usually Not Changed) ---
# These values are derived from the hyperparameters above. It's best not to
# edit these directly.
# ==============================================================================

# Automatically set the device to use the GPU (cuda) if available, otherwise CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A convenient list of the class names.
CLASS_NAMES = HPARAMS["classes_names"]

# Converts the class weights list into a special format (Tensor) for PyTorch.
# This tensor is sent to the specified device (GPU or CPU).
CLASS_WEIGHTS_TENSOR = torch.tensor(HPARAMS["class_weights"], dtype=torch.float32).to(
    DEVICE
)


# ==============================================================================
# --- File Structure Constants ---
# Description: Defines the expected names for files within your dataset folders.
#              Change these if your files have different names.
# ==============================================================================

# The names of the different time points used as input channels.
TIME_POINTS = ["t-1", "t", "t+1"]

# The filename for the raw image data.
RAW_FILE_NAME = "raw_cropped.tif"

# The filename for the corresponding segmentation mask.
LABEL_FILE_NAME = "binary_label_cropped.tif"

# --- End of Configuration ---
print(f"Configuration loaded. Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
