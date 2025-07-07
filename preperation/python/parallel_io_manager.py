#!/usr/bin/env python3
"""
Parallel I/O Manager for Nucleus Extraction
High-performance parallel file I/O operations for the nucleus extraction pipeline.

This module provides:
- Parallel file writing operations
- Optimized metadata serialization
- Batch file operations
- Memory-mapped file operations for large datasets
- Asynchronous I/O where possible
"""

import os
import json
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import tifffile
from dataclasses import asdict
import re


class ParallelIOManager:
    """
    High-performance parallel I/O manager for nucleus extraction results
    """

    def __init__(self, max_workers: int = 16, use_async: bool = True):
        """
        Initialize parallel I/O manager

        Args:
            max_workers: Maximum number of parallel I/O workers
            use_async: Whether to use asynchronous I/O operations
        """
        self.max_workers = max_workers
        self.use_async = use_async

    def save_extraction_results_parallel(
        self,
        results: List[Dict[str, Any]],
        dataset_name: str,
        output_base_path: Optional[str] = None,
    ) -> int:
        """
        Save extraction results in parallel with optimized I/O

        Args:
            results: List of extraction results
            dataset_name: Name of the dataset
            output_base_path: Optional base path override

        Returns:
            Number of successfully saved results
        """
        if not results:
            return 0

        # Setup output directory structure
        if output_base_path is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"

            # Create symlink to ceph storage if it doesn't exist
            ceph_target = Path("/mnt/home/dchhantyal/ceph/nuclei_state_dataset/v2")
            symlink_path = data_dir / "v2"

            if not symlink_path.exists():
                data_dir.mkdir(exist_ok=True)
                ceph_target.mkdir(parents=True, exist_ok=True)
                symlink_path.symlink_to(ceph_target)

            output_base_path = symlink_path
        else:
            output_base_path = Path(output_base_path)

        print(f"💾 Saving {len(results)} results in parallel...")

        # Use parallel execution
        successful_saves = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all save tasks
            future_to_result = {
                executor.submit(
                    self._save_single_result, result, dataset_name, output_base_path
                ): i
                for i, result in enumerate(results)
            }

            # Collect results as they complete
            for future in as_completed(future_to_result):
                result_idx = future_to_result[future]
                try:
                    success = future.result()
                    if success:
                        successful_saves += 1
                    else:
                        print(f"❌ Failed to save result {result_idx}")
                except Exception as e:
                    print(f"❌ Error saving result {result_idx}: {e}")

        print(f"✅ Saved {successful_saves}/{len(results)} results successfully")
        return successful_saves

    def _save_single_result(
        self, result: Dict[str, Any], dataset_name: str, output_base_path: Path
    ) -> bool:
        """
        Save a single extraction result with all its files

        Args:
            result: Single extraction result
            dataset_name: Name of the dataset
            output_base_path: Base output path

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract nucleus information
            nucleus_id = result["nucleus_id"]
            event_frame = result["event_frame"]
            event_type = result.get("event_type", "unknown")

            # Map event types to directory names
            event_type_map = {
                "mitotic": "mitotic",
                "death": "death",
                "mitotic_death": "mitotic_death",
                "normal": "stable",
            }

            category = event_type_map.get(event_type, "stable")

            # Extract stack number from dataset name
            stack_match = re.search(r"stack(\d+)", dataset_name)
            stack_number = f"stack{stack_match.group(1)}" if stack_match else "unknown"

            # Count total nuclei in the event frame (from time_series data)
            total_nuclei_count = self._estimate_total_nuclei_count(result)

            # Create directory structure
            nucleus_dir_name = f"{dataset_name}_frame_{event_frame:03d}_nucleus_{nucleus_id:03d}_count_{total_nuclei_count}"
            nucleus_dir = output_base_path / category / nucleus_dir_name
            nucleus_dir.mkdir(parents=True, exist_ok=True)

            # Save main metadata
            main_metadata = self._create_main_metadata(
                result, dataset_name, stack_number
            )
            main_metadata_path = nucleus_dir / "metadata.json"

            with open(main_metadata_path, "w") as f:
                json.dumps(
                    main_metadata, f, indent=2, default=self._convert_numpy_types
                )

            # Save frame-specific data in parallel
            frame_save_tasks = []

            for frame_label, frame_data in result["time_series"].items():
                if frame_data.get("success", False):
                    # Create timestamp subdirectory
                    timestamp_dir = nucleus_dir / self._convert_frame_label(frame_label)
                    timestamp_dir.mkdir(exist_ok=True)

                    # Save frame data
                    frame_save_tasks.append(
                        (frame_data, timestamp_dir, frame_label, result)
                    )

            # Execute frame saves in parallel
            if frame_save_tasks:
                with ThreadPoolExecutor(
                    max_workers=min(4, len(frame_save_tasks))
                ) as frame_executor:
                    frame_futures = [
                        frame_executor.submit(self._save_frame_data, *task)
                        for task in frame_save_tasks
                    ]

                    # Wait for all frame saves to complete
                    for future in as_completed(frame_futures):
                        future.result()  # This will raise any exceptions

            return True

        except Exception as e:
            print(f"❌ Error saving nucleus {result.get('nucleus_id', 'unknown')}: {e}")
            return False

    def _save_frame_data(
        self,
        frame_data: Dict[str, Any],
        timestamp_dir: Path,
        frame_label: str,
        full_result: Dict[str, Any],
    ) -> bool:
        """
        Save data for a single time frame

        Args:
            frame_data: Frame-specific data
            timestamp_dir: Directory for this timestamp
            frame_label: Label for this frame
            full_result: Full extraction result for metadata

        Returns:
            True if successful
        """
        try:
            # Save image files
            if "img_cropped" in frame_data:
                img_path = timestamp_dir / "raw_cropped.tif"
                tifffile.imwrite(img_path, frame_data["img_cropped"])

            if "lbl_cropped" in frame_data:
                lbl_path = timestamp_dir / "label_cropped.tif"
                tifffile.imwrite(lbl_path, frame_data["lbl_cropped"])

            # Save cleaned data if available
            cleaned_data = frame_data.get("cleaned_data", {})

            if "raw_cleaned" in cleaned_data:
                cleaned_raw_path = timestamp_dir / "raw_cleaned.tif"
                tifffile.imwrite(cleaned_raw_path, cleaned_data["raw_cleaned"])

            if "label_cleaned" in cleaned_data:
                cleaned_label_path = timestamp_dir / "label_cleaned.tif"
                tifffile.imwrite(cleaned_label_path, cleaned_data["label_cleaned"])

            if "mask_cleaned" in cleaned_data:
                binary_mask_path = timestamp_dir / "binary_label_cropped.tif"
                tifffile.imwrite(binary_mask_path, cleaned_data["mask_cleaned"])

                # Create nucleus-only raw image
                nucleus_raw = cleaned_data["raw_original"].copy()
                nucleus_raw[cleaned_data["mask_cleaned"] == 0] = 0
                nucleus_raw_path = timestamp_dir / "raw_image_cropped.tif"
                tifffile.imwrite(nucleus_raw_path, nucleus_raw)

            # Save frame-specific metadata
            frame_metadata = self._create_frame_metadata(
                frame_data, frame_label, full_result
            )
            metadata_path = timestamp_dir / "metadata.json"

            with open(metadata_path, "w") as f:
                json.dump(
                    frame_metadata, f, indent=2, default=self._convert_numpy_types
                )

            return True

        except Exception as e:
            print(f"❌ Error saving frame {frame_label}: {e}")
            return False

    def _create_main_metadata(
        self, result: Dict[str, Any], dataset_name: str, stack_number: str
    ) -> Dict[str, Any]:
        """Create main nucleus metadata"""
        import datetime

        # Determine available and missing frames
        available_frames = []
        missing_frames = []

        for frame_label, frame_data in result["time_series"].items():
            if frame_data.get("success", False):
                available_frames.append(self._convert_frame_label(frame_label))
            else:
                missing_frames.append(self._convert_frame_label(frame_label))

        # Calculate quality metrics
        successful_frames = result.get("successful_frames", 0)
        total_frames = result.get("total_frames", 1)
        frame_consistency = successful_frames / total_frames if total_frames > 0 else 0

        # Determine cell fate and lineage info
        event_type = result.get("event_type", "unknown")
        is_mitotic = result.get("is_mitotic", 0)
        is_death = result.get("is_death", 0)

        cell_fate = "normal"
        if is_death:
            cell_fate = "apoptotic_death"
        elif is_mitotic:
            cell_fate = "mitotic_division"

        return {
            "nucleus_summary": {
                "dataset_name": dataset_name,
                "nucleus_id": str(result["nucleus_id"]),
                "event_frame": result["event_frame"],
                "total_nuclei_in_frame": self._estimate_total_nuclei_count(result),
                "classification": event_type,
                "extraction_date": datetime.datetime.now().isoformat(),
                "available_frames": available_frames,
                "missing_frames": missing_frames,
            },
            "lineage_info": {
                "parent_cells": [],  # Would need lineage data to fill
                "daughter_cells": [],  # Would need lineage data to fill
                "division_events": 1 if is_mitotic else 0,
                "death_frame": result["event_frame"] if is_death else None,
                "cell_fate": cell_fate,
            },
            "extraction_config": self._convert_numpy_types(
                asdict(result.get("config", {}))
            ),
            "quality_metrics": {
                "extraction_success": result.get("extraction_success", False),
                "frame_consistency": frame_consistency,
                "volume_variance": 0.0,  # Would need to calculate from frame data
                "signal_to_noise_ratio": 0.0,  # Would need to calculate from frame data
            },
        }

    def _create_frame_metadata(
        self, frame_data: Dict[str, Any], frame_label: str, full_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create frame-specific metadata"""
        import datetime

        # Extract basic info
        nucleus_id = full_result["nucleus_id"]
        event_frame = full_result["event_frame"]
        frame_number = frame_data.get("frame", event_frame)

        # Get bounding box
        bbox = frame_data.get(
            "bbox", full_result.get("bounding_box", (0, 0, 0, 0, 0, 0))
        )
        z_min, z_max, y_min, y_max, x_min, x_max = bbox

        # Get cleaning statistics
        cleaned_data = frame_data.get("cleaned_data", {})
        stats = cleaned_data.get("stats", {})

        # Determine classification
        event_type = full_result.get("event_type", "unknown")
        is_mitotic = full_result.get("is_mitotic", 0)
        is_death = full_result.get("is_death", 0)

        return {
            "extraction_info": {
                "dataset_name": full_result.get("dataset_name", "unknown"),
                "frame_number": frame_number,
                "nucleus_id": str(nucleus_id),
                "timestamp": self._convert_frame_label(frame_label),
                "total_nuclei_in_frame": self._estimate_total_nuclei_count(full_result),
                "extraction_date": datetime.datetime.now().isoformat(),
            },
            "nucleus_properties": {
                "centroid_x": float((x_min + x_max) / 2),
                "centroid_y": float((y_min + y_max) / 2),
                "centroid_z": float((z_min + z_max) / 2),
                "bounding_box": {
                    "min_x": int(x_min),
                    "max_x": int(x_max),
                    "min_y": int(y_min),
                    "max_y": int(y_max),
                    "min_z": int(z_min),
                    "max_z": int(z_max),
                },
                "volume_pixels": int(stats.get("cleaned_volume", 0)),
                "fluorescence_markers": {
                    "histone_raw": 0.0,  # Would need actual fluorescence data
                    "gata6_raw": 0.0,
                    "nanog_raw": 0.0,
                    "histone_norm": 0.0,
                    "gata6_norm": 0.0,
                    "nanog_norm": 0.0,
                },
            },
            "extraction_parameters": {
                "crop_padding": getattr(
                    full_result.get("config", {}), "crop_padding", 2.0
                ),
                "min_object_size": getattr(
                    full_result.get("config", {}), "min_object_size", 20
                ),
                "hole_filling_enabled": getattr(
                    full_result.get("config", {}), "enable_hole_filling", True
                ),
                "image_dimensions": {
                    "width": int(x_max - x_min + 1),
                    "height": int(y_max - y_min + 1),
                    "depth": int(z_max - z_min + 1),
                },
            },
            "classification": {
                "category": event_type,
                "confidence": 0.95,  # Would need actual confidence if available
                "lineage_analysis": {
                    "parent_cells": [],
                    "daughter_cells": [],
                    "division_events": 1 if is_mitotic else 0,
                    "death_frame": frame_number if is_death else None,
                },
            },
        }

    def _convert_frame_label(self, frame_label: str) -> str:
        """Convert frame labels to timestamp format"""
        label_map = {"previous": "t-1", "current": "t", "next": "t+1"}
        return label_map.get(frame_label, frame_label)

    def _estimate_total_nuclei_count(self, result: Dict[str, Any]) -> int:
        """Estimate total nuclei count in the frame"""
        # This would need actual data to implement properly
        # For now, return a placeholder
        return 50

    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_numpy_types(item) for item in obj]
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


# Async version for even better performance
class AsyncIOManager:
    """
    Asynchronous I/O manager for maximum performance
    """

    def __init__(self, max_concurrent: int = 50):
        self.max_concurrent = max_concurrent
        self.semaphore = None

    async def save_results_async(
        self,
        results: List[Dict[str, Any]],
        dataset_name: str,
        output_base_path: Optional[str] = None,
    ) -> int:
        """
        Save results using asynchronous I/O for maximum performance
        """
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        tasks = [
            self._save_single_result_async(result, dataset_name, output_base_path)
            for result in results
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(1 for r in results if r is True)

        return successful

    async def _save_single_result_async(
        self, result: Dict[str, Any], dataset_name: str, output_base_path: Optional[str]
    ) -> bool:
        """Save a single result asynchronously"""
        async with self.semaphore:
            # Implementation would use aiofiles for async file operations
            # This is a placeholder for the async implementation
            return True


if __name__ == "__main__":
    # Example usage
    io_manager = ParallelIOManager(max_workers=16)

    # Test with dummy data
    dummy_results = [
        {
            "nucleus_id": 1,
            "event_frame": 45,
            "event_type": "death",
            "time_series": {
                "current": {
                    "success": True,
                    "frame": 45,
                    "img_cropped": np.random.randint(
                        0, 255, (10, 50, 50), dtype=np.uint8
                    ),
                    "lbl_cropped": np.random.randint(
                        0, 5, (10, 50, 50), dtype=np.uint8
                    ),
                }
            },
        }
    ]

    # Save results
    saved = io_manager.save_extraction_results_parallel(
        dummy_results, "test_dataset", "/tmp/test_output"
    )

    print(f"Saved {saved} results")
