# predict.py
# Updated to work with 4-channel CNN model (3 temporal + 1 binary segmentation mask)
# Added batch processing for multiple samples with single model loading

import torch
import numpy as np
import os
import tifffile
from scipy.ndimage import zoom
import argparse

# --- Import Model and Config ---
# Import the model and hyperparameters al model definition file.
from cnn_model import Simple3DCNN, HPARAMS

# --- Prediction-Specific Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["mitotic", "new_daughter", "stable"]


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
        label_path = os.path.join(folder_path, "t", "label_cropped.tif")
    elif volume_paths and len(volume_paths) >= 3:
        paths = volume_paths[:3]  # Take first 3 as temporal volumes
        label_path = volume_paths[3] if len(volume_paths) >= 4 else None  # 4th as label
    else:
        raise ValueError("You must provide either --folder_path or --volumes (3-4 paths).")

    # --- Step 1: Load t volume to determine transformation params ---
    t_path = paths[1]  # "t" volume
    if os.path.exists(t_path):
        t_volume = tifffile.imread(t_path).astype(np.float32)
    else:
        print(f"⚠️ Warning: t volume not found at {t_path}. Using blank volume.")
        t_volume = np.zeros((32, 32, 32), dtype=np.float32)

    # Calculate resize factor and padding based on the t volume
    original_shape = t_volume.shape
    ratios = np.array(target_shape) / np.array(original_shape)
    resize_factor = np.min(ratios)

    resized_shape = (np.array(original_shape) * resize_factor).astype(int)
    start_indices = (np.array(target_shape) - resized_shape) // 2

    # --- Step 2: Extract nucleus ID from folder name (if using folder_path) ---
    nucleus_id = None
    if folder_path:
        folder_name = os.path.basename(folder_path)
        parts = folder_name.split('_')
        for i, part in enumerate(parts):
            if part == 'nucleus' and i + 1 < len(parts):
                try:
                    nucleus_id = int(parts[i + 1])
                    break
                except ValueError:
                    continue

    # --- Step 3: Define transformation function ---
    def transform_and_pad_consistent(volume, is_label=False, target_nucleus_id=None):
        # Resize using the factor from the t volume
        resized = zoom(volume, resize_factor, order=0 if is_label else 1, mode="constant", cval=0.0)
        padded = np.zeros(target_shape, dtype=np.float32)

        # Create slices to embed the resized volume centrally
        rz_shape = resized.shape
        slices = tuple(
            slice(start, start + size)
            for start, size in zip(start_indices, rz_shape)
        )
        rz_slices = tuple(slice(0, s.stop - s.start) for s in slices)

        padded[slices] = resized[rz_slices]

        # Handle different volume types
        if not is_label:
            # Normalize raw volumes
            min_val, max_val = padded.min(), padded.max()
            if max_val > min_val:
                padded = (padded - min_val) / (max_val - min_val)
        else:
            # For labels, create binary mask for specific nucleus
            if target_nucleus_id is not None:
                padded = (padded == target_nucleus_id).astype(np.float32)
            else:
                padded = (padded > 0).astype(np.float32)
    
        return padded

    # --- Step 4: Process temporal volumes ---
    all_volumes = []
    time_points = ["t-1", "t", "t+1"]

    for i, (tp, path) in enumerate(zip(time_points, paths)):
        if os.path.exists(path):
            if i == 1:  # "t" volume - already loaded
                vol_to_process = t_volume
            else:
                vol_to_process = tifffile.imread(path).astype(np.float32)
            processed_vol = transform_and_pad_consistent(vol_to_process, is_label=False)
            all_volumes.append(processed_vol)
        else:
            print(f"⚠️ Warning: File not found at {path}. Using a blank volume.")
            all_volumes.append(np.zeros(target_shape, dtype=np.float32))

    # --- Step 5: Process binary segmentation mask for current timestamp (t) ---
    if label_path and os.path.exists(label_path):
        label_volume = tifffile.imread(label_path).astype(np.float32)
        # Create binary mask for the specific nucleus
        processed_label = transform_and_pad_consistent(label_volume, is_label=True, target_nucleus_id=nucleus_id)
    else:
        # Create empty mask if label file is missing
        processed_label = np.zeros(target_shape, dtype=np.float32)
        if label_path:
            print(f"⚠️ Warning: Label file not found at {label_path}. Using blank mask.")

    # Add the binary segmentation mask as the 4th channel
    all_volumes.append(processed_label)

    # --- Step 6: Stack volumes ---
    volume_stack = np.stack(all_volumes, axis=0)  # [t-1, t, t+1, binary_mask]
    return torch.from_numpy(volume_stack).float().unsqueeze(0)


# --- Prediction Logic ---
def load_model(model_path: str):
    """Loads and returns the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    model = Simple3DCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    print(f"🚀 Model loaded from {model_path} and running on {DEVICE}.")
    return model

def run_single_prediction(model, input_tensor: torch.Tensor):
    """Runs prediction on a single input tensor using pre-loaded model."""
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
    input_group.add_argument("--folder_path", nargs='+', help="One or more paths to sample folders with t-1, t, t+1 subdirs.")
    input_group.add_argument("--volumes", nargs='+', help="3-4 space-separated paths: t-1, t, t+1 .tif files, and optionally label file.")

    args = parser.parse_args()

    try:
        # Validate input arguments
        if args.volumes:
            if len(args.volumes) < 3:
                raise ValueError("--volumes requires at least 3 paths (t-1, t, t+1)")
            elif len(args.volumes) > 4:
                raise ValueError("--volumes accepts maximum 4 paths (t-1, t, t+1, label)")
            
            # Process single sample with volume paths
            folder_paths = None
            volume_paths = args.volumes
        else:
            # Process one or more folder paths
            folder_paths = args.folder_path
            volume_paths = None

        # Load model once
        print("Loading model...")
        model = load_model(args.model_path)
        
        if folder_paths:
            # Batch processing for multiple folders
            total_samples = len(folder_paths)
            print(f"\n🚀 Processing {total_samples} sample(s)...")
            print("=" * 50)
            
            results = []
            for i, folder_path in enumerate(folder_paths):
                sample_name = os.path.basename(folder_path)
                print(f"\n📁 Sample {i+1}/{total_samples}: {sample_name}")
                
                try:
                    # Preprocess single sample
                    input_tensor = preprocess_input(folder_path=folder_path)
                    print(f"   Input tensor shape: {input_tensor.shape}")
                    
                    # Run prediction
                    pred_index, pred_class, pred_confidence = run_single_prediction(model, input_tensor)
                    
                    # Store result
                    result = {
                        'sample': sample_name,
                        'index': pred_index,
                        'class': pred_class,
                        'confidence': pred_confidence
                    }
                    results.append(result)
                    
                    print(f"   Predicted: {pred_class.upper()} ({pred_confidence:.2%})")
                    
                except Exception as e:
                    print(f"   ❌ Error processing {sample_name}: {e}")
                    results.append({
                        'sample': sample_name,
                        'error': str(e)
                    })
            
            # Summary
            print(f"\n{'='*50}")
            print("🎉 BATCH PROCESSING COMPLETE")
            print(f"{'='*50}")
            for i, result in enumerate(results):
                if 'error' not in result:
                    print(f"{i+1}. {result['sample'][:40]:40} → {result['class'].upper():12} ({result['confidence']:.1%})")
                else:
                    print(f"{i+1}. {result['sample'][:40]:40} → ERROR: {result['error']}")
        
        else:
            # Single sample processing with volume paths
            print("Preprocessing input data...")
            input_tensor = preprocess_input(volume_paths=volume_paths)
            print(f"Input tensor created with shape: {input_tensor.shape}")

            pred_index, pred_class, pred_confidence = run_single_prediction(model, input_tensor)

            print("\n--- Prediction Result ---")
            print(f"Predicted Class Index: {pred_index}")
            print(f"Predicted Class Name:  {pred_class.upper()}")
            print(f"Confidence:            {pred_confidence:.2%}")
            print("-------------------------\n")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")