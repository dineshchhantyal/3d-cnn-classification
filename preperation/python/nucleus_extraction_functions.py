"""
Nucleus Extraction Functions

This module contains the core functions for extracting and cleaning nucleus volumes
from 3D microscopy data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tifffile
from skimage import measure, morphology, filters
from skimage.segmentation import clear_border
import warnings

warnings.filterwarnings("ignore")


def clean_nucleus_volume(
    raw_volume, label_volume, target_nucleus_id, remove_small_objects=True, min_size=50
):
    """
    Clean the nucleus volume by removing foreign objects and noise

    Args:
        raw_volume: Raw image volume (3D array)
        label_volume: Label volume (3D array)
        target_nucleus_id: ID of the target nucleus
        remove_small_objects: Whether to remove small connected components
        min_size: Minimum size for small object removal

    Returns:
        dict: Contains cleaned and original versions of images and labels
    """
    # Create target nucleus mask
    target_mask = (label_volume == target_nucleus_id).astype(np.uint8)

    # Clean the target mask
    cleaned_mask = target_mask.copy()

    if remove_small_objects:
        # Remove small connected components in 3D
        cleaned_mask = morphology.remove_small_objects(
            cleaned_mask.astype(bool), min_size=min_size
        ).astype(np.uint8)

    # Fill holes in 2D slices
    for z in range(cleaned_mask.shape[0]):
        if np.any(cleaned_mask[z]):
            # Fill holes in this slice
            filled = morphology.binary_closing(cleaned_mask[z], morphology.disk(2))
            cleaned_mask[z] = filled.astype(np.uint8)

    # Create cleaned volumes
    cleaned_raw = raw_volume.copy()
    cleaned_raw[cleaned_mask == 0] = 0  # Set background to 0

    # Create cleaned label volume
    cleaned_label = np.zeros_like(label_volume)
    cleaned_label[cleaned_mask == 1] = target_nucleus_id

    # Calculate statistics
    original_volume = np.sum(target_mask)
    cleaned_volume = np.sum(cleaned_mask)
    volume_change = cleaned_volume - original_volume

    return {
        "raw_original": raw_volume,
        "raw_cleaned": cleaned_raw,
        "label_original": label_volume,
        "label_cleaned": cleaned_label,
        "mask_original": target_mask,
        "mask_cleaned": cleaned_mask,
        "stats": {
            "original_volume": original_volume,
            "cleaned_volume": cleaned_volume,
            "volume_change": volume_change,
            "volume_change_percent": (
                (volume_change / original_volume * 100) if original_volume > 0 else 0
            ),
        },
    }


def find_nucleus_bounding_box(label_volume, nucleus_id, padding_factor=1.2):
    """
    Find 3D bounding box around a nucleus with optional padding

    Args:
        label_volume: 3D label array
        nucleus_id: Target nucleus ID
        padding_factor: Factor to expand bounding box (1.0 = no expansion)

    Returns:
        tuple: (z_min, z_max, y_min, y_max, x_min, x_max)
    """
    # Find all positions where nucleus exists
    positions = np.where(label_volume == nucleus_id)

    if len(positions[0]) == 0:
        return None

    # Get bounding box coordinates
    z_min, z_max = positions[0].min(), positions[0].max()
    y_min, y_max = positions[1].min(), positions[1].max()
    x_min, x_max = positions[2].min(), positions[2].max()

    # Apply padding
    if padding_factor > 1.0:
        z_center = (z_min + z_max) // 2
        y_center = (y_min + y_max) // 2
        x_center = (x_min + x_max) // 2

        z_size = int((z_max - z_min + 1) * padding_factor)
        y_size = int((y_max - y_min + 1) * padding_factor)
        x_size = int((x_max - x_min + 1) * padding_factor)

        z_min = max(0, z_center - z_size // 2)
        z_max = min(label_volume.shape[0] - 1, z_center + z_size // 2)
        y_min = max(0, y_center - y_size // 2)
        y_max = min(label_volume.shape[1] - 1, y_center + y_size // 2)
        x_min = max(0, x_center - x_size // 2)
        x_max = min(label_volume.shape[2] - 1, x_center + x_size // 2)

    return (z_min, z_max, y_min, y_max, x_min, x_max)


def extract_nucleus_time_series(data_path, nucleus_id, event_frame, config):
    """
    Extract 3-frame time series (previous, current, next) for a nucleus
    Returns both cleaned and original versions

    Args:
        data_path: Path to dataset
        nucleus_id: Target nucleus ID
        event_frame: Frame where the event occurs
        config: NucleusExtractorConfig object

    Returns:
        dict: Complete extraction results
    """
    p = Path(data_path)
    img_dir = p / "registered_images"
    lbl_dir = p / "registered_label_images"

    # Define frame range (previous, current, next)
    frames = [
        event_frame - config.time_window,
        event_frame,
        event_frame + config.time_window,
    ]
    frame_labels = ["previous", "current", "next"]

    print(f"🔍 Extracting nucleus {nucleus_id} from frames {frames}")

    # Find the nucleus in the event frame first to get reference position
    event_lbl_file = list(lbl_dir.glob(f"label_reg8_{event_frame}.tif"))
    if not event_lbl_file:
        print(f"❌ Event frame label file not found: {event_frame}")
        return None

    event_lbl = tifffile.imread(event_lbl_file[0])
    bbox = find_nucleus_bounding_box(event_lbl, nucleus_id, config.padding_factor)

    if bbox is None:
        print(f"❌ Nucleus {nucleus_id} not found in event frame {event_frame}")
        return None

    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    print(
        f"📦 Bounding box: Z[{z_min}:{z_max}], Y[{y_min}:{y_max}], X[{x_min}:{x_max}]"
    )

    # Storage for results
    results = {
        "nucleus_id": nucleus_id,
        "event_frame": event_frame,
        "frames": frames,
        "bounding_box": bbox,
        "time_series": {},
        "extraction_success": True,
        "config": config,
    }

    # Extract each frame
    for i, (frame, label) in enumerate(zip(frames, frame_labels)):
        print(f"  📸 Processing {label} frame {frame}...")

        # Find files
        img_files = list(img_dir.glob(f"nuclei_reg8_{frame}.tif"))
        lbl_files = list(lbl_dir.glob(f"label_reg8_{frame}.tif"))

        if not img_files or not lbl_files:
            print(f"    ❌ Files not found for frame {frame}")
            results["extraction_success"] = False
            continue

        # Load full volumes
        img_full = tifffile.imread(img_files[0])
        lbl_full = tifffile.imread(lbl_files[0])

        # Extract region of interest
        img_roi = img_full[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
        lbl_roi = lbl_full[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]

        # Check if nucleus exists in this frame
        nucleus_present = nucleus_id in np.unique(lbl_roi)

        # Clean the volume
        if nucleus_present:
            cleaned_data = clean_nucleus_volume(img_roi, lbl_roi, nucleus_id)
            print(
                f"    ✅ Nucleus found | Volume: {cleaned_data['stats']['original_volume']} → {cleaned_data['stats']['cleaned_volume']} pixels"
            )
        else:
            print(f"    ⚠️  Nucleus not found in {label} frame")
            # Create empty cleaned data
            cleaned_data = {
                "raw_original": img_roi,
                "raw_cleaned": np.zeros_like(img_roi),
                "label_original": lbl_roi,
                "label_cleaned": np.zeros_like(lbl_roi),
                "mask_original": np.zeros_like(lbl_roi, dtype=np.uint8),
                "mask_cleaned": np.zeros_like(lbl_roi, dtype=np.uint8),
                "stats": {
                    "original_volume": 0,
                    "cleaned_volume": 0,
                    "volume_change": 0,
                    "volume_change_percent": 0,
                },
            }

        # Store results for this frame
        results["time_series"][label] = {
            "frame_number": frame,
            "nucleus_present": nucleus_present,
            "files": {"image": img_files[0].name, "label": lbl_files[0].name},
            "data": cleaned_data,
        }

    # Calculate summary statistics
    total_frames = len(
        [f for f in results["time_series"].values() if f["nucleus_present"]]
    )
    results["summary"] = {
        "frames_with_nucleus": total_frames,
        "nucleus_persistence": total_frames / len(frames),
        "roi_shape": img_roi.shape,
        "extraction_successful": results["extraction_success"],
    }

    print(
        f"✅ Extraction complete | Nucleus found in {total_frames}/{len(frames)} frames"
    )
    return results


def batch_extract_nuclei(
    data_path, metadata, config, max_samples=None, event_types=None
):
    """
    Batch extract nuclei from classification data

    Args:
        data_path: Path to dataset
        metadata: Metadata dictionary from read_death_and_mitotic_class
        config: NucleusExtractorConfig object
        max_samples: Maximum number of samples to extract (None for all)
        event_types: List of event types to extract ['mitotic', 'death', 'both', 'normal']

    Returns:
        list: List of extraction results
    """
    if "classes" not in metadata:
        print("❌ No classification data found")
        return []

    df = metadata["classes"]

    # Filter by event types if specified
    if event_types:
        mask = pd.Series([False] * len(df))

        if "mitotic" in event_types:
            mask |= (df["mitotic"] == 1) & (df["death"] == 0)
        if "death" in event_types:
            mask |= (df["mitotic"] == 0) & (df["death"] == 1)
        if "both" in event_types:
            mask |= (df["mitotic"] == 1) & (df["death"] == 1)
        if "normal" in event_types:
            mask |= (df["mitotic"] == 0) & (df["death"] == 0)

        df_filtered = df[mask]
        print(f"🔍 Filtered to {len(df_filtered)} events of types: {event_types}")
    else:
        df_filtered = df

    # Limit samples if specified
    if max_samples and len(df_filtered) > max_samples:
        df_filtered = df_filtered.head(max_samples)
        print(f"📊 Limited to {max_samples} samples")

    print(f"🚀 Starting batch extraction of {len(df_filtered)} nuclei...")

    results = []
    successful_extractions = 0

    for idx, row in df_filtered.iterrows():
        nucleus_id = int(row["nucleus_id"])
        event_frame = int(row["frame"])
        is_mitotic = int(row["mitotic"])
        is_death = int(row["death"])

        # Determine event type
        if is_mitotic and is_death:
            event_type = "mitotic_death"
        elif is_mitotic:
            event_type = "mitotic"
        elif is_death:
            event_type = "death"
        else:
            event_type = "normal"

        print(
            f"\n[{idx+1}/{len(df_filtered)}] Processing nucleus {nucleus_id} (frame {event_frame}) - {event_type}"
        )

        try:
            result = extract_nucleus_time_series(
                data_path, nucleus_id, event_frame, config
            )

            if result and result["extraction_success"]:
                # Add classification info
                result["event_type"] = event_type
                result["is_mitotic"] = is_mitotic
                result["is_death"] = is_death
                results.append(result)
                successful_extractions += 1
                print(f"  ✅ Success ({successful_extractions}/{len(df_filtered)})")
            else:
                print(f"  ❌ Failed extraction")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    print(f"\n🎯 Batch extraction complete:")
    print(
        f"  • Successful: {successful_extractions}/{len(df_filtered)} ({successful_extractions/len(df_filtered)*100:.1f}%)"
    )
    print(f"  • Failed: {len(df_filtered) - successful_extractions}")

    return results
