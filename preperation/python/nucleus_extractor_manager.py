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
- Use new data directory structure: data/nuclei_state_dataset/stable/

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
    """
    Configuration class for nucleus extraction parameters

    Attributes:
        crop_padding (float): Factor to expand the bounding box (e.g., 2.0 = 200% padding)
        time_window (int): Number of frames before and after for default 3-frame extraction
        frame_offsets (list, optional): Custom frame offsets relative to event frame
            - If None, uses default 3-frame window: [event_frame - time_window, event_frame, event_frame + time_window]
            - If provided, extracts frames at [event_frame + offset for offset in frame_offsets]
            - Examples:
              * [-1, 0, 1] for 3-frame window (equivalent to default)
              * [-2, -1, 0, 1, 2] for 5-frame window
              * [0] for single frame (event frame only)
              * [-5, -3, -1, 0, 1, 3, 5] for sparse sampling
        min_object_size (int): Minimum size for object cleaning (pixels)
        enable_hole_filling (bool): Whether to fill holes in masks
    """

    def __init__(self):
        self.crop_padding = 2  # Factor to expand the bounding box (200% padding)
        self.time_window = (
            1  # Number of frames before and after (previous, current, next)
        )
        self.frame_offsets = None  # Custom frame offsets relative to event frame (e.g., [-2, -1, 0, 1, 2])
        self.min_object_size = 20  # Minimum size for object cleaning
        self.enable_hole_filling = True  # Fill holes in masks

    def __repr__(self):
        return f"NucleusExtractorConfig(crop_padding={self.crop_padding}, time_window={self.time_window}, frame_offsets={self.frame_offsets}, min_object_size={self.min_object_size})"


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
        Extract time series for a nucleus with flexible frame selection
        Returns both cleaned and original versions

        Args:
            nucleus_id: Target nucleus ID
            event_frame: Frame where the event occurs

        Returns:
            dict: Complete extraction results
        """
        # Define frame range - use custom offsets if provided, otherwise use default 3-frame window
        if self.config.frame_offsets is not None:
            # Custom frame offsets (e.g., [-2, -1, 0, 1, 2] for 5-frame series)
            frames = [event_frame + offset for offset in self.config.frame_offsets]
            # Generate labels based on offsets
            frame_labels = []
            for offset in self.config.frame_offsets:
                if offset < 0:
                    frame_labels.append(f"t{offset}")  # e.g., "t-2", "t-1"
                elif offset == 0:
                    frame_labels.append("current")
                else:
                    frame_labels.append(f"t+{offset}")  # e.g., "t+1", "t+2"
        else:
            # Default 3-frame window (backward compatibility)
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

    def save_extraction_results(self, results, dataset_name, output_base_path=None):
        """
        Save extraction results to files using new data structure

        Args:
            results: List of extraction results
            dataset_name: Name of the dataset (e.g., "230212_stack6")
            output_base_path: Optional base path override (defaults to data/nuclei_state_dataset)
        """
        import re

        # Helper function to convert numpy types
        def convert_numpy_types(obj):
            """Convert numpy types to JSON serializable types"""
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.complexfloating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, "item"):  # For any remaining numpy scalars
                return obj.item()
            else:
                return obj

        # Set up the new data structure
        if output_base_path is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"

            # Create symlink to ceph storage if it doesn't exist
            ceph_target = Path("/mnt/home/dchhantyal/ceph/nuclei_state_dataset")
            symlink_path = data_dir / "nuclei_state_dataset"

            if not symlink_path.exists():
                data_dir.mkdir(exist_ok=True)
                ceph_target.mkdir(parents=True, exist_ok=True)
                symlink_path.symlink_to(ceph_target)

            output_base_path = symlink_path

        # Extract stack number from dataset name (e.g., "230212_stack6" -> "stack6")
        stack_match = re.search(r"stack(\d+)", dataset_name)
        if stack_match:
            stack_number = f"stack{stack_match.group(1)}"
        else:
            # Fallback if no stack number found
            stack_number = dataset_name.replace("_", "")

        print(
            f"💾 Saving {len(results)} extraction results with stack identifier: {stack_number}"
        )

        # Save each nucleus extraction
        for i, result in enumerate(results):
            nucleus_id = result["nucleus_id"]
            event_frame = result["event_frame"]
            original_event_type = result["event_type"]

            # Determine event type directory
            if original_event_type == "normal":
                event_type = "stable"  # Replace 'normal' with 'stable'
            elif original_event_type == "mitotic_death":
                event_type = "mitotic"  # Put combined events in mitotic folder
            else:
                event_type = original_event_type

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

            # Create event type directory
            event_dir = output_base_path / event_type
            event_dir.mkdir(parents=True, exist_ok=True)

            # Create directory with labeled naming convention: {stack_number}_nucleus{nucleus_id}_frame{frame_id}_count{number_of_nuclei}
            nucleus_dir = (
                event_dir
                / f"{stack_number}_nucleus{nucleus_id}_frame{event_frame}_count{number_of_nuclei}"
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

                # Save metadata
                metadata = {
                    "nucleus_id": int(nucleus_id),
                    "frame_number": int(frame_data["frame_number"]),
                    "stack_number": stack_number,
                    "nucleus_present": bool(frame_data["nucleus_present"]),
                    "files": frame_data["files"],
                    "stats": convert_numpy_types(data["stats"]),
                    "number_of_nuclei_in_frame": int(number_of_nuclei),
                }

                with open(frame_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

            # Save overall nucleus metadata (convert numpy types)
            nucleus_metadata = {
                "nucleus_id": nucleus_id,
                "event_frame": event_frame,
                "stack_number": stack_number,
                "event_type": event_type,
                "original_event_type": original_event_type,  # Keep track of original
                "number_of_nuclei_in_event_frame": number_of_nuclei,
                "dataset_name": dataset_name,
                "storage_path": str(nucleus_dir.relative_to(output_base_path)),
                "is_mitotic": result["is_mitotic"],
                "is_death": result["is_death"],
                "frames": convert_numpy_types(result["frames"]),
                "bounding_box": convert_numpy_types(result["bounding_box"]),
                "summary": convert_numpy_types(result["summary"]),
                "config": {
                    "crop_padding": result["config"].crop_padding,
                    "time_window": result["config"].time_window,
                    "min_object_size": result["config"].min_object_size,
                    "enable_hole_filling": result["config"].enable_hole_filling,
                },
            }
            # Apply convert_numpy_types to the entire metadata structure
            nucleus_metadata = convert_numpy_types(nucleus_metadata)

            with open(nucleus_dir / "nucleus_metadata.json", "w") as f:
                json.dump(nucleus_metadata, f, indent=2)

            print(f"  ✅ Saved nucleus {nucleus_id} → {event_type}/{nucleus_dir.name}")

        print(f"✅ Results saved organized by event type in: {output_base_path}")
        print(f"📍 Full path: {output_base_path}")
        return output_base_path

    def plot_nucleus_3d_shape(self, result, frame_type="current", output_dir=None):
        """
        Plot 3D visualization of the nucleus shape
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("❌ 3D plotting requires matplotlib with 3D support")
            return None

        if frame_type not in result["time_series"]:
            print(f"❌ Frame type '{frame_type}' not found")
            return None

        data = result["time_series"][frame_type]["data"]
        mask = data["mask_cleaned"]
        nucleus_id = result["nucleus_id"]
        frame_number = result["time_series"][frame_type]["frame_number"]

        # Get 3D coordinates of nucleus voxels
        z, y, x = np.where(mask > 0)

        if len(z) == 0:
            print(f"❌ No nucleus found in {frame_type} frame")
            return None

        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot nucleus voxels
        ax.scatter(x, y, z, c=z, cmap="viridis", alpha=0.6, s=20)

        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_zlabel("Z (slices)")
        ax.set_title(
            f"3D Shape of Nucleus {nucleus_id} - {frame_type.capitalize()} Frame {frame_number}\n"
            f"Total voxels: {len(z)} | Bounding box: {mask.shape}"
        )

        # Add bounding box wireframe
        bbox = result["bounding_box"]
        z_min, z_max, y_min, y_max, x_min, x_max = bbox

        # Draw bounding box edges
        from itertools import product, combinations

        corners = list(product([x_min, x_max], [y_min, y_max], [z_min, z_max]))
        for s, e in combinations(corners, 2):
            if (
                sum(abs(a - b) for a, b in zip(s, e)) == (x_max - x_min)
                or sum(abs(a - b) for a, b in zip(s, e)) == (y_max - y_min)
                or sum(abs(a - b) for a, b in zip(s, e)) == (z_max - z_min)
            ):
                ax.plot3D(*zip(s, e), color="red", alpha=0.6, linewidth=2)

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            filename = f"nucleus_{nucleus_id}_frame_{frame_number}_3D_shape.png"
            plt.savefig(Path(output_dir) / filename, dpi=150, bbox_inches="tight")
            print(f"✅ 3D shape plot saved: {filename}")

        return fig
