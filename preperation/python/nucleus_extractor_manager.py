#!/usr/bin/env python3
"""
Nucleus Extractor Manager
A comprehensive management script for extracting and processing nucleus data from 3D time series.

This script provides functions to:
- Extract 3-frame time series (previous, current, next) for nuclei
- Clean volumes by removing noise and foreign objects
- Save both cleaned and original versions
- Batch process multiple nuclei
- Manage different datasets

Author: Generated for 3D CNN Classification Project
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
from skimage import measure, morphology, filters
from skimage.segmentation import clear_border
import warnings

warnings.filterwarnings("ignore")

# Import the classification reader
from read_death_and_mitotic_class import read_death_and_mitotic_class


class NucleusExtractorConfig:
    """Configuration class for nucleus extraction parameters"""

    def __init__(self):
        self.crop_padding = 1.2  # Factor to expand the bounding box (20% padding)
        self.time_window = (
            1  # Number of frames before and after (previous, current, next)
        )
        self.min_object_size = 50  # Minimum size for object cleaning
        self.enable_hole_filling = True  # Fill holes in masks

    def __repr__(self):
        return f"NucleusExtractorConfig(crop_padding={self.crop_padding}, time_window={self.time_window}, min_object_size={self.min_object_size})"


class NucleusExtractorManager:
    """Main class for managing nucleus extraction operations"""

    def __init__(self, data_path, config=None):
        """
        Initialize the nucleus extractor manager

        Args:
            data_path: Path to the dataset
            config: NucleusExtractorConfig object (optional)
        """
        self.data_path = Path(data_path)
        self.config = config or NucleusExtractorConfig()
        self.metadata = None
        self.img_dir = self.data_path / "registered_images"
        self.lbl_dir = self.data_path / "registered_label_images"

        # Load metadata
        self.load_metadata()

    def load_metadata(self):
        """Load classification metadata"""
        print(f"📂 Loading metadata from {self.data_path}")
        self.metadata = read_death_and_mitotic_class(str(self.data_path))

        if "classes" in self.metadata:
            df = self.metadata["classes"]
            print(f"✅ Loaded {len(df)} classifications")
            return True
        else:
            print("❌ Failed to load metadata")
            return False

    def clean_nucleus_volume(self, raw_volume, label_volume, target_nucleus_id):
        """
        Clean the nucleus volume by removing foreign objects and noise

        Args:
            raw_volume: Raw image volume (3D array)
            label_volume: Label volume (3D array)
            target_nucleus_id: ID of the target nucleus

        Returns:
            dict: Contains cleaned and original versions of images and labels
        """
        # Create target nucleus mask
        target_mask = (label_volume == target_nucleus_id).astype(np.uint8)

        # Clean the target mask
        cleaned_mask = target_mask.copy()

        if self.config.min_object_size > 0:
            # Remove small connected components in 3D
            cleaned_mask = morphology.remove_small_objects(
                cleaned_mask.astype(bool), min_size=self.config.min_object_size
            ).astype(np.uint8)

        # Fill holes in 2D slices if enabled
        if self.config.enable_hole_filling:
            for z in range(cleaned_mask.shape[0]):
                if np.any(cleaned_mask[z]):
                    # Fill holes in this slice
                    filled = morphology.binary_closing(
                        cleaned_mask[z], morphology.disk(2)
                    )
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
                    (volume_change / original_volume * 100)
                    if original_volume > 0
                    else 0
                ),
            },
        }

    def find_nucleus_bounding_box(self, label_volume, nucleus_id):
        """
        Find 3D bounding box around a nucleus with optional padding

        Args:
            label_volume: 3D label array
            nucleus_id: Target nucleus ID

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
        if self.config.crop_padding > 1.0:
            z_center = (z_min + z_max) // 2
            y_center = (y_min + y_max) // 2
            x_center = (x_min + x_max) // 2

            z_size = int((z_max - z_min + 1) * self.config.crop_padding)
            y_size = int((y_max - y_min + 1) * self.config.crop_padding)
            x_size = int((x_max - x_min + 1) * self.config.crop_padding)

            z_min = max(0, z_center - z_size // 2)
            z_max = min(label_volume.shape[0] - 1, z_center + z_size // 2)
            y_min = max(0, y_center - y_size // 2)
            y_max = min(label_volume.shape[1] - 1, y_center + y_size // 2)
            x_min = max(0, x_center - x_size // 2)
            x_max = min(label_volume.shape[2] - 1, x_center + x_size // 2)

        return (z_min, z_max, y_min, y_max, x_min, x_max)

    def extract_nucleus_time_series(self, nucleus_id, event_frame):
        """
        Extract 3-frame time series (previous, current, next) for a nucleus
        Returns both cleaned and original versions

        Args:
            nucleus_id: Target nucleus ID
            event_frame: Frame where the event occurs

        Returns:
            dict: Complete extraction results
        """
        # Define frame range (previous, current, next)
        frames = [
            event_frame - self.config.time_window,
            event_frame,
            event_frame + self.config.time_window,
        ]
        frame_labels = ["previous", "current", "next"]

        print(f"🔍 Extracting nucleus {nucleus_id} from frames {frames}")

        # Find the nucleus in the event frame first to get reference position
        event_lbl_file = list(self.lbl_dir.glob(f"label_reg8_{event_frame}.tif"))
        if not event_lbl_file:
            print(f"❌ Event frame label file not found: {event_frame}")
            return None

        event_lbl = tifffile.imread(event_lbl_file[0])
        bbox = self.find_nucleus_bounding_box(event_lbl, nucleus_id)

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
            "config": self.config,
        }

        # Extract each frame
        for i, (frame, label) in enumerate(zip(frames, frame_labels)):
            print(f"  📸 Processing {label} frame {frame}...")

            # Find files
            img_files = list(self.img_dir.glob(f"nuclei_reg8_{frame}.tif"))
            lbl_files = list(self.lbl_dir.glob(f"label_reg8_{frame}.tif"))

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
                cleaned_data = self.clean_nucleus_volume(img_roi, lbl_roi, nucleus_id)
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

    def batch_extract_nuclei(self, max_samples=None, event_types=None):
        """
        Batch extract nuclei from classification data

        Args:
            max_samples: Maximum number of samples to extract (None for all)
            event_types: List of event types to extract ['mitotic', 'death', 'both', 'normal']

        Returns:
            list: List of extraction results
        """
        if not self.metadata or "classes" not in self.metadata:
            print("❌ No classification data found")
            return []

        df = self.metadata["classes"]

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
                result = self.extract_nucleus_time_series(nucleus_id, event_frame)

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

    def save_extraction_results(self, results, output_path, dataset_name):
        """
        Save extraction results to files

        Args:
            results: List of extraction results
            output_path: Base output directory
            dataset_name: Name of the dataset
        """
        output_dir = Path(output_path) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving {len(results)} extraction results to {output_dir}")

        # Save each nucleus extraction
        for i, result in enumerate(results):
            nucleus_id = result["nucleus_id"]
            event_frame = result["event_frame"]
            event_type = result["event_type"]

            # Get the number of unique nuclei in the event frame for folder naming
            # Load the label image for the event frame to count unique nuclei
            event_lbl_file = list(self.lbl_dir.glob(f"label_reg8_{event_frame}.tif"))
            if event_lbl_file:
                event_lbl = tifffile.imread(event_lbl_file[0])
                # Count unique nucleus IDs (excluding background, typically 0)
                unique_nuclei = np.unique(event_lbl)
                unique_nuclei = unique_nuclei[
                    unique_nuclei > 0
                ]  # Remove background (0)
                number_of_nuclei = len(unique_nuclei)
            else:
                # Fallback if label file not found
                number_of_nuclei = 0
                print(
                    f"    ⚠️  Warning: Could not load label file for frame {event_frame}"
                )

            # Create directory for this nucleus with nuclei count
            nucleus_dir = (
                output_dir
                / f"nucleus_{nucleus_id}_frame_{event_frame}_{event_type}_{number_of_nuclei}"
            )
            nucleus_dir.mkdir(exist_ok=True)

            # Save each frame's data
            for frame_label, frame_data in result["time_series"].items():
                frame_dir = nucleus_dir / frame_label
                frame_dir.mkdir(exist_ok=True)

                data = frame_data["data"]

                # Save volumes as TIFF files (6 images total per frame)
                tifffile.imwrite(frame_dir / "raw_original.tif", data["raw_original"])
                tifffile.imwrite(frame_dir / "raw_cleaned.tif", data["raw_cleaned"])
                tifffile.imwrite(
                    frame_dir / "label_original.tif", data["label_original"]
                )
                tifffile.imwrite(frame_dir / "label_cleaned.tif", data["label_cleaned"])
                tifffile.imwrite(frame_dir / "mask_original.tif", data["mask_original"])
                tifffile.imwrite(frame_dir / "mask_cleaned.tif", data["mask_cleaned"])

                # Save metadata (convert numpy types for JSON serialization)
                def convert_numpy_types(obj):
                    """Convert numpy types to JSON serializable types"""
                    if isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return obj

                metadata = {
                    "nucleus_id": int(nucleus_id),
                    "frame_number": int(frame_data["frame_number"]),
                    "nucleus_present": bool(frame_data["nucleus_present"]),
                    "files": frame_data["files"],
                    "stats": convert_numpy_types(data["stats"]),
                    "number_of_nuclei_in_frame": int(number_of_nuclei),
                }

                with open(frame_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

            # Save overall nucleus metadata (convert numpy types)
            nucleus_metadata = {
                "nucleus_id": int(nucleus_id),
                "event_frame": int(event_frame),
                "event_type": event_type,
                "number_of_nuclei_in_event_frame": int(number_of_nuclei),
                "is_mitotic": bool(result["is_mitotic"]),
                "is_death": bool(result["is_death"]),
                "frames": result["frames"],
                "bounding_box": convert_numpy_types(result["bounding_box"]),
                "summary": convert_numpy_types(result["summary"]),
                "config": {
                    "crop_padding": result["config"].crop_padding,
                    "time_window": result["config"].time_window,
                    "min_object_size": result["config"].min_object_size,
                    "enable_hole_filling": result["config"].enable_hole_filling,
                },
            }

            with open(nucleus_dir / "nucleus_metadata.json", "w") as f:
                json.dump(nucleus_metadata, f, indent=2)

        print(f"✅ Results saved to {output_dir}")
        return output_dir


def main():
    """Example usage of the NucleusExtractorManager"""

    # Available datasets
    datasets = {
        "230212_stack6": "/mnt/ceph/users/lbrown/MouseData/Rebecca/230212_stack6/",
        "220321_stack11": "/mnt/ceph/users/lbrown/MouseData/Rebecca/220321_stack11/",
        "221016_FUCCI_Nanog_stack_3": "/mnt/ceph/users/lbrown/Labels3DMouse/Abhishek/RebeccaData/221016_FUCCI_Nanog_stack_3/",
    }

    # Configuration
    config = NucleusExtractorConfig()
    config.crop_padding = 1.3  # 30% padding
    config.time_window = 1  # Previous, current, next

    # Initialize manager
    data_path = datasets["230212_stack6"]
    manager = NucleusExtractorManager(data_path, config)

    print("🎯 Nucleus Extractor Manager initialized")
    print(f"📂 Dataset: {data_path}")
    print(f"⚙️ Config: {config}")

    # Example: Extract a small batch
    if manager.metadata:
        results = manager.batch_extract_nuclei(max_samples=5, event_types=["mitotic"])

        if results:
            output_dir = manager.save_extraction_results(
                results,
                "/mnt/home/dchhantyal/3d-cnn-classification/extracted_nuclei",
                "230212_stack6_test",
            )
            print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
