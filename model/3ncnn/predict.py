# predict.py

import torch
import numpy as np
import os
import tifffile
from scipy.ndimage import zoom
import argparse

# --- Import Model and Config ---
# Import the model and hyperparameters from your central model definition file.
from cnn_model import Simple3DCNN, HPARAMS

# --- Prediction-Specific Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["mitotic", "new_daughter", "stable", "death"]


# --- Data Preprocessing ---
def transform_and_pad(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resizes, pads, and normalizes a single 3D volume."""
    original_shape = volume.shape
    ratios = np.array(target_shape) / np.array(original_shape)
    resize_factor = np.min(ratios)

    resized_shape = (np.array(original_shape) * resize_factor).astype(int)
    start_indices = (np.array(target_shape) - resized_shape) // 2

    resized = zoom(volume, resize_factor, order=1, mode='constant', cval=0.0)

    padded = np.zeros(target_shape, dtype=np.float32)
    slices = tuple(slice(start, start + size) for start, size in zip(start_indices, resized.shape))
    padded[slices] = resized

    min_val, max_val = padded.min(), padded.max()
    if max_val > min_val:
        padded = (padded - min_val) / (max_val - min_val)

    return padded

def preprocess_input(folder_path: str = None, volume_paths: list = None) -> torch.Tensor:
    """Loads and preprocesses input data from either a folder or direct file paths."""
    target_shape = (HPARAMS["input_depth"], HPARAMS["input_height"], HPARAMS["input_width"])

    if folder_path:
        paths = [
            os.path.join(folder_path, "t-1", "raw_cropped.tif"),
            os.path.join(folder_path, "t", "raw_cropped.tif"),
            os.path.join(folder_path, "t+1", "raw_cropped.tif"),
        ]
    elif volume_paths and len(volume_paths) == 3:
        paths = volume_paths
    else:
        raise ValueError("You must provide either --folder_path or --volumes.")

    all_volumes = []
    for path in paths:
        if os.path.exists(path):
            volume = tifffile.imread(path).astype(np.float32)
            processed_vol = transform_and_pad(volume, target_shape)
            all_volumes.append(processed_vol)
        else:
            print(f"⚠️ Warning: File not found at {path}. Using a blank volume.")
            all_volumes.append(np.zeros(target_shape, dtype=np.float32))

    volume_stack = np.stack(all_volumes, axis=0)
    return torch.from_numpy(volume_stack).float().unsqueeze(0)


# --- Prediction Logic ---
def run_prediction(model_path: str, input_tensor: torch.Tensor):
    """Loads a trained model and runs prediction on the input tensor."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    # Initialize model from the imported class
    model = Simple3DCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print(f"🚀 Model loaded from {model_path} and running on {DEVICE}.")

    with torch.no_grad():
        input_tensor = input_tensor.to(DEVICE)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    return predicted_idx.item(), predicted_class, confidence.item()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D CNN prediction on nucleus state data.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pth file).")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--folder_path", type=str, help="Path to the sample folder with t-1, t, t+1 subdirs.")
    input_group.add_argument("--volumes", nargs=3, help="Three space-separated paths to the t-1, t, and t+1 .tif files.")

    args = parser.parse_args()

    try:
        print("Preprocessing input data...")
        input_tensor = preprocess_input(folder_path=args.folder_path, volume_paths=args.volumes)
        print(f"Input tensor created with shape: {input_tensor.shape}")

        pred_index, pred_class, pred_confidence = run_prediction(args.model_path, input_tensor)

        print("\n--- Prediction Result ---")
        print(f"Predicted Class Index: {pred_index}")
        print(f"Predicted Class Name:  {pred_class.upper()}")
        print(f"Confidence:            {pred_confidence:.2%}")
        print("-------------------------\n")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")