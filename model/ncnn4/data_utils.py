# data_utils.py
# Shared data preprocessing utilities for training and prediction

import numpy as np
import os
import tifffile
import torch
from datetime import datetime
from config import HPARAMS, TIME_POINTS, RAW_FILE_NAME, LABEL_FILE_NAME


def extract_nucleus_id_from_folder(folder_path: str) -> int:
    """Extract nucleus ID from folder name pattern like '..._nucleus_077_...'"""
    folder_name = os.path.basename(folder_path)
    parts = folder_name.split("_")

    for i, part in enumerate(parts):
        if part == "nucleus" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return None


def transform_and_pad_volume(
    volume: np.ndarray,
    target_shape: tuple,
    is_label: bool = False,
    target_nucleus_id: int = None,
    v_min: float = None,
    v_max: float = None,
) -> np.ndarray:
    """
    Center-crop or pad a 3D volume to the target shape without resizing.
    Args:
        volume: Input 3D volume
        target_shape: Target shape (D, H, W)
        is_label: Whether this is a label volume (binary mask)
        target_nucleus_id: Nucleus ID for binary mask creation
    """
    # Center crop or pad without resizing
    processed = center_crop_or_pad(volume, target_shape)

    if not is_label:
        # Normalize raw volumes to [0, 1]
        if v_min is None or v_max is None:
            v_min, v_max = processed.min(), processed.max()
        if v_max <= v_min:
            raise ValueError(
                f"Invalid volume range: min={v_min}, max={v_max}. Cannot normalize."
            )
        processed = (processed - v_min) / (v_max - v_min)
    else:
        # Check if already binary (only 0 and 1)
        unique_vals = np.unique(processed)
        if set(unique_vals).issubset({0, 1}):
            processed = processed.astype(np.float32)
        elif target_nucleus_id is not None:
            processed = (processed == target_nucleus_id).astype(np.float32)
        else:
            processed = (processed > 0).astype(np.float32)
            print(
                f"⚠️ Warning: No nucleus ID provided for binary mask. Using all non-zero values: {unique_vals}, looking for {target_nucleus_id}"
            )

    return processed


def load_temporal_volumes(folder_path: str) -> tuple:
    """
    Load temporal volumes from a sample folder.

    Returns:
        tuple: (temporal_paths, label_path)
    """
    temporal_paths = []
    for tp in TIME_POINTS:
        path = os.path.join(folder_path, tp, RAW_FILE_NAME)
        temporal_paths.append(path)

    label_path = os.path.join(folder_path, "t", LABEL_FILE_NAME)

    return temporal_paths, label_path


def preprocess_sample(
    folder_path: str = None,
    volume_paths: list = None,
    for_training: bool = False,
    save_analysis: bool = False,
    analysis_output_dir: str = None,
) -> torch.Tensor:
    """
    Unified preprocessing function for both training and prediction.

    Args:
        folder_path: Path to sample folder (standard format)
        volume_paths: Direct paths to volume files [t-1, t, t+1, label]
        for_training: Whether this is for training (affects error handling)
        save_analysis: Whether to save preprocessing analysis visualizations
        analysis_output_dir: Directory to save analysis outputs

    Returns:
        torch.Tensor: Preprocessed 4-channel volume [t-1, t, t+1, binary_mask]
    """
    target_shape = (
        HPARAMS["input_depth"],
        HPARAMS["input_height"],
        HPARAMS["input_width"],
    )

    # Determine file paths
    if folder_path:
        temporal_paths, label_path = load_temporal_volumes(folder_path)
        nucleus_id = extract_nucleus_id_from_folder(folder_path)
    elif volume_paths and len(volume_paths) >= 3:
        temporal_paths = volume_paths[:3]  # [t-1, t, t+1]
        label_path = volume_paths[3] if len(volume_paths) >= 4 else None
        nucleus_id = None
        print("Using provided volume paths for preprocessing.")
    else:
        raise ValueError("Must provide either folder_path or volume_paths (3-4 paths)")
    # --- Step 1: Load 't' volume to determine transformation parameters ---
    t_path = temporal_paths[1]  # "t" volume
    if os.path.exists(t_path):
        t_volume = tifffile.imread(t_path).astype(np.float32)
    else:
        if for_training:
            raise FileNotFoundError(f"Required 't' volume not found: {t_path}")
        print(f"⚠️ Warning: t volume not found at {t_path}. Using blank volume.")
        t_volume = np.zeros((32, 32, 32), dtype=np.float32)

    # Calculate transformation parameters based on 't' volume

    # --- Step 2: Process all temporal volumes ---
    all_volumes = []
    v_min, v_max = t_volume.min(), t_volume.max()

    for i, (tp, path) in enumerate(zip(TIME_POINTS, temporal_paths)):
        if os.path.exists(path):
            if i == 1:  # "t" volume - already loaded
                vol_to_process = t_volume
            else:
                vol_to_process = tifffile.imread(path).astype(np.float32)
                v_min, v_max = min(vol_to_process.min(), v_min), max(
                    vol_to_process.max(), v_max
                )

            all_volumes.append(vol_to_process)
        else:
            if for_training:
                raise FileNotFoundError(f"Required temporal volume not found: {path}")
            print(f"⚠️ Warning: File not found at {path}. Using blank volume.")
            all_volumes.append(np.zeros(target_shape, dtype=np.float32))

    # Normalize and transform all temporal volumes
    for i, vol in enumerate(all_volumes):
        processed_vol = transform_and_pad_volume(
            vol,
            target_shape,
            is_label=False,
            target_nucleus_id=nucleus_id,
            v_min=v_min,
            v_max=v_max,
        )
        all_volumes[i] = processed_vol

    # --- Step 3: Process binary segmentation mask ---
    if label_path and os.path.exists(label_path):
        label_volume = tifffile.imread(label_path).astype(np.float32)
        processed_label = transform_and_pad_volume(
            label_volume,
            target_shape,
            is_label=True,
            target_nucleus_id=nucleus_id,
        )
    else:
        if for_training and label_path:
            raise FileNotFoundError(f"Required label file not found: {label_path}")
        if label_path and not for_training:
            print(f"⚠️ Warning: Label file not found at {label_path}. Using blank mask.")
        processed_label = np.zeros(target_shape, dtype=np.float32)

    # Add binary mask as 4th channel
    all_volumes.append(processed_label)

    # --- Step 4: Stack and return ---
    volume_stack = np.stack(all_volumes, axis=0)  # [t-1, t, t+1, binary_mask]

    # --- Step 5: Add a subtle focus on channels 0, 1, 2, based on the binary mask ---
    if False and processed_label is not None and processed_label.sum() > 0:
        # This factor determines how much to dim the background (e.g., 0.5 means 50% brightness).
        background_factor = 0.5

        # Ensure the mask is boolean for indexing
        # This assumes processed_label has values of 0 and 1.
        mask_boolean = processed_label.astype(bool)

        # Apply the highlighting effect to the first three channels
        for i in range(3):
            channel = volume_stack[i]

            # Create a copy to avoid modifying the original array in place while indexing
            highlighted_channel = channel.copy()

            # Select the background (where the mask is False) and dim it
            highlighted_channel[~mask_boolean] *= background_factor

            # Update the volume stack with the highlighted channel
            volume_stack[i] = highlighted_channel

        # --- Step 6: Save Analysis (if requested) ---
    if save_analysis and analysis_output_dir:
        try:
            from visualization_utils import (
                create_output_structure,
                save_volume_statistics,
                save_volume_slices,
                save_segmentation_overlay,
                save_preprocessing_comparison,
            )

            # Determine sample name
            sample_name = os.path.basename(folder_path) if folder_path else "sample"
            if not sample_name or sample_name == ".":
                sample_name = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create output structure
            dirs = create_output_structure(analysis_output_dir, sample_name)
            print(f"💾 Saving preprocessing analysis to: {dirs['sample']}")

            # Save original volumes for comparison
            original_volumes = {}
            for i, (tp, path) in enumerate(zip(TIME_POINTS, temporal_paths)):
                if os.path.exists(path):
                    original_volumes[tp] = tifffile.imread(path).astype(np.float32)
                else:
                    original_volumes[tp] = None
            original_volumes["mask"] = (
                label_volume if label_path and os.path.exists(label_path) else None
            )
            # Save volume statistics
            volumes_dict = {
                "t-1_original": original_volumes.get("t-1"),
                "t_original": original_volumes.get("t"),
                "t+1_original": original_volumes.get("t+1"),
                "t-1_processed": all_volumes[0],
                "t_processed": all_volumes[1],
                "t+1_processed": all_volumes[2],
                "segmentation_mask": all_volumes[3],
            }

            save_volume_statistics(
                volumes_dict,
                os.path.join(dirs["preprocessing"], "volume_statistics.json"),
            )

            # Save volume slice visualizations
            for i, tp in enumerate(TIME_POINTS):
                if all_volumes[i] is not None:
                    save_volume_slices(
                        all_volumes[i],
                        f"Processed {tp} Volume",
                        os.path.join(
                            dirs["preprocessing"],
                            f'{tp.replace("-", "minus")}_slices.png',
                        ),
                    )

            # Save segmentation mask slices
            save_volume_slices(
                all_volumes[3],
                "Segmentation Mask",
                os.path.join(dirs["preprocessing"], "segmentation_mask.png"),
                cmap="RdYlBu_r",
                is_binary=True,
            )

            # Save segmentation overlay
            save_segmentation_overlay(
                all_volumes[1],  # Use 't' volume
                all_volumes[3],  # Segmentation mask
                os.path.join(dirs["preprocessing"], "segmentation_overlay.png"),
            )

            # Save preprocessing comparison
            save_preprocessing_comparison(
                original_volumes,
                all_volumes,  # Only temporal volumes
                os.path.join(dirs["preprocessing"], "preprocessing_comparison.png"),
            )

            print(f"✅ Preprocessing analysis saved successfully!")

        except Exception as e:
            print(f"⚠️ Warning: Could not save preprocessing analysis: {e}")

    return torch.from_numpy(volume_stack).float().unsqueeze(0)


def center_crop_or_pad(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Center-crop or pad a 3D volume to the target shape without resizing.
    Args:
        volume: Input 3D numpy array.
        target_shape: Desired output shape (D, H, W).
    Returns:
        Cropped or padded 3D numpy array.
    """
    result = np.zeros(target_shape, dtype=volume.dtype)
    in_shape = np.array(volume.shape)
    out_shape = np.array(target_shape)
    # Calculate cropping and padding
    crop_start = np.maximum((in_shape - out_shape) // 2, 0)
    crop_end = crop_start + np.minimum(in_shape, out_shape)
    pad_start = np.maximum((out_shape - in_shape) // 2, 0)
    pad_end = pad_start + np.minimum(in_shape, out_shape)
    # Place the cropped region into the padded result
    result[
        pad_start[0] : pad_end[0], pad_start[1] : pad_end[1], pad_start[2] : pad_end[2]
    ] = volume[
        crop_start[0] : crop_end[0],
        crop_start[1] : crop_end[1],
        crop_start[2] : crop_end[2],
    ]
    return result
