from datetime import datetime
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from volume_utils import generate_frame_label
import tifffile
import json
import os


def create_main_metadata(result, dataset_name, classification):
    """Create main nucleus metadata following V2 pipeline specification"""
    node_info = result.get("node_info", {})
    summary = result.get("summary", {})
    time_series = result.get("time_series", {})

    # Determine available and missing frames
    # Generate expected frame labels based on the actual frame range
    event_frame = result["event_frame"]
    frames = result.get("frames", [])

    # Generate expected frame labels using utility function
    expected_frames = []
    for frame_num in frames:
        expected_frames.append(generate_frame_label(frame_num, event_frame))

    available_frames = list(time_series.keys())
    missing_frames = [f for f in expected_frames if f not in available_frames]

    # Convert NumPy types to Python native types for JSON serialization
    nucleus_id = result["nucleus_id"]
    if hasattr(nucleus_id, "item"):  # NumPy scalar
        nucleus_id = nucleus_id.item()

    event_frame = result["event_frame"]
    if hasattr(event_frame, "item"):  # NumPy scalar
        event_frame = event_frame.item()

    return {
        "nucleus_summary": {
            "dataset_name": dataset_name,
            "nucleus_id": str(nucleus_id),
            "event_frame": int(event_frame),
            "total_nuclei_in_frame": len(
                time_series.get("t", {})
                .get("data", {})
                .get("unique_labels_in_region", [nucleus_id])
            ),  # This is count in cropped region only
            "classification": classification,
            "extraction_date": datetime.now().isoformat() + "Z",
            "available_frames": available_frames,
            "missing_frames": missing_frames,
        },
        "lineage_info": {
            "parent_cells": (
                [node_info.get("parent")] if node_info.get("parent") else []
            ),
            "daughter_cells": node_info.get("children", []),
            "division_events": len(node_info.get("children", [])),
            "death_frame": result["event_frame"] if classification == "death" else None,
            "cell_fate": classification,
        },
        "extraction_config": {
            "crop_padding": 2.0,  # Default padding from README
            "time_window": len(result.get("frames", [])) // 2,
            "min_object_size": 20,  # Default from README
            "hole_filling_enabled": True,
        },
    }


def create_frame_metadata(result, frame_info, dataset_name, classification):
    """Create frame-specific metadata following V2 pipeline specification"""
    frame_data = frame_info["data"]
    node_info = result.get("node_info", {})

    # Calculate centroid from bounding box
    bbox = frame_data.get("bbox", result.get("bounding_box"))
    centroid_z = float((bbox[0] + bbox[1]) / 2) if bbox else 0.0
    centroid_y = float((bbox[2] + bbox[3]) / 2) if bbox else 0.0
    centroid_x = float((bbox[4] + bbox[5]) / 2) if bbox else 0.0

    # Convert NumPy types to Python native types for JSON serialization
    nucleus_id = result["nucleus_id"]
    if hasattr(nucleus_id, "item"):  # NumPy scalar
        nucleus_id = nucleus_id.item()

    frame_number = frame_info["frame_number"]
    if hasattr(frame_number, "item"):  # NumPy scalar
        frame_number = frame_number.item()

    # Convert bounding box values to native Python types
    bbox_dict = {}
    if bbox:
        bbox_dict = {
            "min_z": int(bbox[0]),
            "max_z": int(bbox[1]),
            "min_y": int(bbox[2]),
            "max_y": int(bbox[3]),
            "min_x": int(bbox[4]),
            "max_x": int(bbox[5]),
        }

    return {
        "extraction_info": {
            "dataset_name": dataset_name,
            "frame_number": int(frame_number),
            "nucleus_id": str(nucleus_id),
            "timestamp": frame_info.get("frame_label", "t"),
            "total_nuclei_in_frame": len(
                frame_data.get("unique_labels_in_region", [nucleus_id])
            ),  # This is count in cropped region only
            "extraction_date": datetime.now().isoformat() + "Z",
        },
        "nucleus_properties": {
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "centroid_z": centroid_z,
            "bounding_box": bbox_dict,
            "volume_pixels": int(frame_data.get("volume_size", 0)),
        },
        "extraction_parameters": {
            "crop_padding": 2.0,
            "min_object_size": 20,
            "hole_filling_enabled": True,
            "image_dimensions": {
                "width": (
                    frame_data.get("raw_original", []).shape[2]
                    if frame_data.get("raw_original") is not None
                    else 0
                ),
                "height": (
                    frame_data.get("raw_original", []).shape[1]
                    if frame_data.get("raw_original") is not None
                    else 0
                ),
                "depth": (
                    frame_data.get("raw_original", []).shape[0]
                    if frame_data.get("raw_original") is not None
                    else 0
                ),
            },
        },
        "classification": {
            "category": classification,
            "lineage_analysis": {
                "parent_cells": (
                    [node_info.get("parent")] if node_info.get("parent") else []
                ),
                "daughter_cells": node_info.get("children", []),
                "division_events": len(node_info.get("children", [])),
                "death_frame": (
                    result["event_frame"] if classification == "death" else None
                ),
            },
        },
    }


def print_classification_distribution(
    classification_counts, max_samples=None, output_dir=None, save=False
):
    """
    Print and optionally save the classification distribution.

    Args:
        classification_counts (dict): Dictionary with classification counts
        max_samples (int, optional): Maximum samples per classification to display
        output_dir (str, optional): Directory to save the distribution Plot
        save (bool, optional): Whether to save the distribution chart or not
    """
    print(f"\n📊 CLASSIFICATION DISTRIBUTION:")
    for classification, count in classification_counts.items():
        if max_samples and count > max_samples:
            print(f"   • {classification.upper()}: {max_samples} (limited)")
        else:
            print(f"   • {classification.upper()}: {count}")

    if save and output_dir:
        # Use today's date for subfolder
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_dir = Path(output_dir) / date_str
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "classification_distribution.png"

        plt.figure(figsize=(10, 6))
        plt.bar(classification_counts.keys(), classification_counts.values())
        plt.xlabel("Classification")
        plt.ylabel("Count")
        plt.title("Nucleus Classification Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"   Distribution chart saved to {output_file}")


def save_single_nucleus_immediate(
    result,
    base_output_dir,
    dataset_name,
    classification,
    total_nuclei_in_entire_frame=None,
):
    """
    Save a single extracted nucleus immediately to proper folder structure.

    Args:
        result: Extraction result with time_series data
        base_output_dir: Base directory for saving
        dataset_name: Dataset name for file naming
        classification: Classification type for folder structure

    Returns:
        str: Path to the saved nucleus directory
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)

    nucleus_id = result["nucleus_id"]
    event_frame = result["event_frame"]
    time_series = result["time_series"]

    # Create classification directory
    class_dir = os.path.join(base_output_dir, classification)
    os.makedirs(class_dir, exist_ok=True)

    # Count total nuclei in cropped region (for metadata)
    event_frame_data = time_series.get("t", {}).get("data", {})
    total_nuclei_in_cropped_region = len(
        event_frame_data.get("unique_labels_in_region", [nucleus_id])
    )

    # Use entire frame count for folder naming, fallback to cropped region count
    folder_nucleus_count = (
        total_nuclei_in_entire_frame
        if total_nuclei_in_entire_frame is not None
        else total_nuclei_in_cropped_region
    )

    # Create nucleus directory with ENTIRE FRAME count
    nucleus_dir_name = f"{dataset_name}_frame_{event_frame:03d}_nucleus_{nucleus_id:03d}_count_{folder_nucleus_count}"
    nucleus_dir_path = os.path.join(class_dir, nucleus_dir_name)
    os.makedirs(nucleus_dir_path, exist_ok=True)

    # Save main nucleus metadata
    main_metadata = create_main_metadata(result, dataset_name, classification)
    main_metadata_path = os.path.join(nucleus_dir_path, "metadata.json")
    with open(main_metadata_path, "w") as f:
        json.dump(main_metadata, f, indent=2)

    # Process each time frame
    for frame_label, frame_info in time_series.items():
        frame_data = frame_info["data"]
        is_event_frame = frame_info["is_event_frame"]

        # Create timestamp subdirectory
        timestamp_dir = os.path.join(nucleus_dir_path, frame_label)
        os.makedirs(timestamp_dir, exist_ok=True)

        # Save raw cropped image
        if "raw_original" in frame_data and frame_data["raw_original"] is not None:
            raw_path = os.path.join(timestamp_dir, "raw_cropped.tif")
            tifffile.imwrite(raw_path, frame_data["raw_original"])

        # Save label cropped image
        if "label_original" in frame_data and frame_data["label_original"] is not None:
            label_path = os.path.join(timestamp_dir, "label_cropped.tif")
            tifffile.imwrite(label_path, frame_data["label_original"])

        # For event frame, save additional processed data
        if is_event_frame:
            # Save binary mask if available
            if "target_mask" in frame_data and frame_data["target_mask"] is not None:
                binary_path = os.path.join(timestamp_dir, "binary_label_cropped.tif")
                tifffile.imwrite(binary_path, frame_data["target_mask"])

            # Save raw image cropped using label (only target nucleus visible)
            if "raw_cropped" in frame_data and frame_data["raw_cropped"] is not None:
                raw_nucleus_path = os.path.join(timestamp_dir, "raw_image_cropped.tif")
                tifffile.imwrite(raw_nucleus_path, frame_data["raw_cropped"])

        # Save frame-specific metadata
        frame_metadata = create_frame_metadata(
            result, frame_info, dataset_name, classification
        )
        frame_metadata_path = os.path.join(timestamp_dir, "metadata.json")
        with open(frame_metadata_path, "w") as f:
            json.dump(frame_metadata, f, indent=2)

    return nucleus_dir_path
