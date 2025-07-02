import numpy as np
import torch
import tifffile
from pathlib import Path
import os

PROJECT_ROOT = Path("/mnt/home/dchhantyal/3d-cnn-classification")
import sys

sys.path.append(str(PROJECT_ROOT))
from model import ConvRNN, Config, resize_volume

config = Config()
def predict_single_sample(model, file_paths, device):
    """
    Runs a prediction on a single sample of three .tif files.

    Args:
        model (nn.Module): The trained PyTorch model.
        file_paths (list): A list of 3 file paths for ["previous", "current", "next"].
        device (str): The device to run the prediction on ('cuda' or 'cpu').

    Returns:
        tuple: The predicted class name and the confidence score.
    """
    # 1. Set model to evaluation mode
    model.eval()

    # 2. Load and preprocess the data
    volumes = []
    for f_path in file_paths:
        vol = tifffile.imread(f_path)
        vol_resized = resize_volume(
            vol, config.input_depth, config.input_height, config.input_width
        )
        volumes.append(vol_resized)

    sequence = np.stack(volumes, axis=0).astype(np.float32)

    # Normalize
    min_val, max_val = sequence.min(), sequence.max()
    if max_val > min_val:
        sequence = (sequence - min_val) / (max_val - min_val)

    # Add channel dimension
    sequence = np.expand_dims(sequence, axis=1)  # Shape: (time, channels, D, H, W)

    # Convert to tensor and add batch dimension
    sequence_tensor = torch.from_numpy(sequence).unsqueeze(
        0
    )  # Shape: (batch, time, C, D, H, W)
    sequence_tensor = sequence_tensor.to(device)

    # 3. Perform inference
    with torch.no_grad():
        output_logits = model(sequence_tensor)

        # Convert logits to probabilities
        probabilities = torch.softmax(output_logits, dim=1)

        # Get the top prediction
        confidence, predicted_idx = torch.max(probabilities, 1)

    # 4. Map index to class name
    class_names = ["mitotic", "new_daughter", "stable", "death"]
    predicted_class = class_names[predicted_idx.item()]

    return predicted_class, confidence.item()
