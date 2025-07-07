#!/usr/bin/env python3
"""
Parallel Nucleus Extractor Manager
High-performance parallel implementation of the nucleus extraction pipeline.

This module provides parallelized versions of nucleus extraction operations:
- Parallel batch processing of multiple nuclei
- Parallel time frame processing for each nucleus
- Parallel I/O operations for file reading/writing
- Parallel image processing operations
- Memory-efficient processing with configurable worker pools

Key Features:
- Multi-level parallelization (batch, frame, I/O, processing)
- Progress tracking and monitoring
- Memory management and resource optimization
- Error handling and recovery
- Scalable to large datasets
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager
import threading
from queue import Queue

import numpy as np
import pandas as pd
import tifffile
from skimage import measure, morphology, filters
from skimage.segmentation import clear_border

warnings.filterwarnings("ignore")

# Import the classification reader
from read_death_and_mitotic_class import read_death_and_mitotic_class
from parallel_io_manager import ParallelIOManager


@dataclass
class ParallelConfig:
    """Configuration for parallel processing parameters"""

    # Core extraction parameters (inherited from original config)
    crop_padding: float = 2.0
    time_window: int = 1
    frame_offsets: Optional[List[int]] = None
    min_object_size: int = 20
    enable_hole_filling: bool = True

    # Parallel processing parameters
    max_workers_batch: int = 4  # Number of nuclei to process in parallel
    max_workers_frames: int = 8  # Number of frames to process in parallel per nucleus
    max_workers_io: int = 16  # Number of I/O operations in parallel
    use_process_pool: bool = True  # Use processes vs threads for CPU-bound work
    use_thread_pool_io: bool = True  # Use threads for I/O operations

    # Memory management
    max_memory_gb: float = 16.0  # Maximum memory usage in GB
    chunk_size: int = 50  # Number of nuclei to process in each batch chunk
    preload_images: bool = False  # Whether to preload images into memory

    # Progress and monitoring
    progress_update_interval: int = 5  # Seconds between progress updates
    enable_detailed_logging: bool = True
    save_intermediate_results: bool = True  # Save results as they complete


class ProgressTracker:
    """Thread-safe progress tracking for parallel operations"""

    def __init__(self, total_items: int):
        self.total_items = total_items
        self.completed_items = 0
        self.successful_items = 0
        self.failed_items = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def update(self, success: bool = True):
        """Update progress counters"""
        with self.lock:
            self.completed_items += 1
            if success:
                self.successful_items += 1
            else:
                self.failed_items += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        with self.lock:
            elapsed_time = time.time() - self.start_time
            completion_rate = (
                self.completed_items / self.total_items if self.total_items > 0 else 0
            )

            return {
                "total": self.total_items,
                "completed": self.completed_items,
                "successful": self.successful_items,
                "failed": self.failed_items,
                "completion_rate": completion_rate,
                "elapsed_time": elapsed_time,
                "estimated_total_time": (
                    elapsed_time / completion_rate
                    if completion_rate > 0
                    else float("inf")
                ),
                "items_per_second": (
                    self.completed_items / elapsed_time if elapsed_time > 0 else 0
                ),
            }


class ParallelNucleusExtractor:
    """
    High-performance parallel nucleus extraction manager
    """

    def __init__(self, data_path: str, config: ParallelConfig = None):
        """
        Initialize the parallel nucleus extractor

        Args:
            data_path: Path to dataset directory containing raw-data
            config: Parallel configuration parameters
        """
        self.data_path = Path(data_path)
        self.config = config or ParallelConfig()

        # Initialize paths
        self.img_dir = None
        self.lbl_dir = None
        self.metadata = None

        # Thread-safe result storage
        self.manager = Manager()
        self.results_queue = Queue()

        # Initialize parallel I/O manager
        self.io_manager = ParallelIOManager(max_workers=self.config.max_workers_io)

        print(f"🚀 Initialized Parallel Nucleus Extractor")
        print(f"   • Batch workers: {self.config.max_workers_batch}")
        print(f"   • Frame workers: {self.config.max_workers_frames}")
        print(f"   • I/O workers: {self.config.max_workers_io}")
        print(f"   • Memory limit: {self.config.max_memory_gb} GB")

    def load_dataset(self, dataset_name: str) -> bool:
        """Load dataset and classification data"""
        dataset_path = self.data_path / "raw-data" / dataset_name

        if not dataset_path.exists():
            print(f"❌ Dataset not found: {dataset_path}")
            return False

        self.img_dir = dataset_path / "registered_images"
        self.lbl_dir = dataset_path / "registered_label_images"

        if not self.img_dir.exists() or not self.lbl_dir.exists():
            print(f"❌ Image directories not found in {dataset_path}")
            return False

        # Load classification data
        print(f"📊 Loading classification data for {dataset_name}...")
        try:
            self.metadata = read_death_and_mitotic_class(str(dataset_path))
            print(f"✅ Loaded {len(self.metadata['classes'])} classified nuclei")
            return True
        except Exception as e:
            print(f"❌ Error loading classification data: {e}")
            return False

    def _load_image_parallel(self, file_path: Path) -> Optional[np.ndarray]:
        """Load a single image file in a thread-safe manner"""
        try:
            if file_path.exists():
                return tifffile.imread(file_path)
            return None
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None

    def _process_frame_data(self, frame_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single frame's data (cropping, cleaning, etc.)
        This can be parallelized per frame
        """
        frame_num = frame_info["frame"]
        label = frame_info["label"]
        nucleus_id = frame_info["nucleus_id"]
        bbox = frame_info["bbox"]

        z_min, z_max, y_min, y_max, x_min, x_max = bbox

        # Find and load image files in parallel
        img_files = list(self.img_dir.glob(f"nuclei_reg8_{frame_num}.tif"))
        lbl_files = list(self.lbl_dir.glob(f"label_reg8_{frame_num}.tif"))

        if not img_files or not lbl_files:
            return {
                "frame": frame_num,
                "label": label,
                "success": False,
                "error": f"Files not found for frame {frame_num}",
            }

        # Parallel I/O loading
        with ThreadPoolExecutor(max_workers=2) as io_executor:
            img_future = io_executor.submit(self._load_image_parallel, img_files[0])
            lbl_future = io_executor.submit(self._load_image_parallel, lbl_files[0])

            img_volume = img_future.result()
            lbl_volume = lbl_future.result()

        if img_volume is None or lbl_volume is None:
            return {
                "frame": frame_num,
                "label": label,
                "success": False,
                "error": f"Failed to load image data for frame {frame_num}",
            }

        # Crop volumes
        img_cropped = img_volume[
            z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1
        ]
        lbl_cropped = lbl_volume[
            z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1
        ]

        # Clean volumes (this can be CPU intensive, good for process pool)
        cleaned_data = self._clean_nucleus_volume_parallel(
            img_cropped, lbl_cropped, nucleus_id
        )

        return {
            "frame": frame_num,
            "label": label,
            "success": True,
            "img_cropped": img_cropped,
            "lbl_cropped": lbl_cropped,
            "cleaned_data": cleaned_data,
            "bbox": bbox,
        }

    def _clean_nucleus_volume_parallel(
        self, raw_volume: np.ndarray, label_volume: np.ndarray, target_nucleus_id: int
    ) -> Dict[str, Any]:
        """
        Parallel version of nucleus volume cleaning
        """
        # Create target nucleus mask
        target_mask = (label_volume == target_nucleus_id).astype(np.uint8)
        cleaned_mask = target_mask.copy()

        # Parallel processing of different cleaning operations
        def remove_small_objects_task():
            if self.config.min_object_size > 0:
                return morphology.remove_small_objects(
                    cleaned_mask.astype(bool), min_size=self.config.min_object_size
                ).astype(np.uint8)
            return cleaned_mask

        def fill_holes_task(mask):
            if not self.config.enable_hole_filling:
                return mask

            result = mask.copy()

            # Parallel processing of Z slices
            def process_slice(z):
                if np.any(mask[z]):
                    return morphology.binary_closing(
                        mask[z], morphology.disk(2)
                    ).astype(np.uint8)
                return mask[z]

            # Process slices in parallel
            with ThreadPoolExecutor(max_workers=min(4, mask.shape[0])) as executor:
                slice_futures = {
                    z: executor.submit(process_slice, z) for z in range(mask.shape[0])
                }

                for z, future in slice_futures.items():
                    result[z] = future.result()

            return result

        # Execute cleaning operations
        if self.config.use_process_pool and self.config.min_object_size > 0:
            # For CPU-intensive operations, consider using process pool
            cleaned_mask = remove_small_objects_task()

        if self.config.enable_hole_filling:
            cleaned_mask = fill_holes_task(cleaned_mask)

        # Create cleaned volumes
        cleaned_raw = raw_volume.copy()
        cleaned_raw[cleaned_mask == 0] = 0

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
                "original_volume": int(original_volume),
                "cleaned_volume": int(cleaned_volume),
                "volume_change": int(volume_change),
                "volume_change_percent": (
                    float(volume_change / original_volume * 100)
                    if original_volume > 0
                    else 0.0
                ),
            },
        }

    def extract_nucleus_time_series_parallel(
        self, nucleus_id: int, event_frame: int
    ) -> Optional[Dict[str, Any]]:
        """
        Parallel extraction of nucleus time series
        Processes all time frames in parallel
        """
        # Determine frame range
        if self.config.frame_offsets is not None:
            frames = [event_frame + offset for offset in self.config.frame_offsets]
            frame_labels = []
            for offset in self.config.frame_offsets:
                if offset < 0:
                    frame_labels.append(f"t{offset}")
                elif offset == 0:
                    frame_labels.append("current")
                else:
                    frame_labels.append(f"t+{offset}")
        else:
            frames = [
                event_frame - self.config.time_window,
                event_frame,
                event_frame + self.config.time_window,
            ]
            frame_labels = ["previous", "current", "next"]

        # Find nucleus bounding box from event frame
        event_lbl_file = list(self.lbl_dir.glob(f"label_reg8_{event_frame}.tif"))
        if not event_lbl_file:
            return None

        event_lbl = tifffile.imread(event_lbl_file[0])
        bbox = self._find_nucleus_bounding_box(event_lbl, nucleus_id)

        if bbox is None:
            return None

        # Prepare frame processing tasks
        frame_tasks = []
        for frame, label in zip(frames, frame_labels):
            frame_tasks.append(
                {"frame": frame, "label": label, "nucleus_id": nucleus_id, "bbox": bbox}
            )

        # Process all frames in parallel
        frame_results = {}
        executor_class = (
            ProcessPoolExecutor if self.config.use_process_pool else ThreadPoolExecutor
        )

        with executor_class(max_workers=self.config.max_workers_frames) as executor:
            # Submit all frame processing tasks
            future_to_frame = {
                executor.submit(self._process_frame_data, task): task["label"]
                for task in frame_tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_frame):
                frame_label = future_to_frame[future]
                try:
                    result = future.result()
                    frame_results[frame_label] = result
                except Exception as e:
                    print(f"❌ Error processing frame {frame_label}: {e}")
                    frame_results[frame_label] = {"success": False, "error": str(e)}

        # Compile final results
        successful_frames = sum(
            1 for r in frame_results.values() if r.get("success", False)
        )

        return {
            "nucleus_id": nucleus_id,
            "event_frame": event_frame,
            "frames": frames,
            "bounding_box": bbox,
            "time_series": frame_results,
            "extraction_success": successful_frames > 0,
            "successful_frames": successful_frames,
            "total_frames": len(frames),
            "config": asdict(self.config),
        }

    def _find_nucleus_bounding_box(
        self, label_volume: np.ndarray, nucleus_id: int
    ) -> Optional[Tuple[int, ...]]:
        """Find 3D bounding box around a nucleus with optional padding"""
        positions = np.where(label_volume == nucleus_id)

        if len(positions[0]) == 0:
            return None

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

    def batch_extract_nuclei_parallel(
        self,
        max_samples: Optional[int] = None,
        event_types: Optional[List[str]] = None,
        dataset_name: str = None,
        output_base_path: Optional[str] = None,
    ) -> int:
        """
        High-performance parallel batch extraction of nuclei

        Args:
            max_samples: Maximum number of samples to extract
            event_types: List of event types to extract
            dataset_name: Name of the dataset for saving
            output_base_path: Optional base path override

        Returns:
            Number of successful extractions
        """
        if not self.metadata or "classes" not in self.metadata:
            print("❌ No classification data found")
            return 0

        if dataset_name is None:
            raise ValueError("dataset_name must be provided for saving results.")

        df = self.metadata["classes"]

        # Filter by event types
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
        else:
            df_filtered = df

        # Limit samples
        if max_samples and len(df_filtered) > max_samples:
            df_filtered = df_filtered.head(max_samples)

        total_nuclei = len(df_filtered)
        print(f"🚀 Starting parallel batch extraction of {total_nuclei} nuclei...")
        print(f"   • Batch workers: {self.config.max_workers_batch}")
        print(f"   • Frame workers per nucleus: {self.config.max_workers_frames}")

        # Initialize progress tracking
        progress = ProgressTracker(total_nuclei)

        # Progress monitoring thread
        def monitor_progress():
            while progress.completed_items < progress.total_items:
                time.sleep(self.config.progress_update_interval)
                stats = progress.get_stats()
                print(
                    f"📊 Progress: {stats['completed']}/{stats['total']} "
                    f"({stats['completion_rate']*100:.1f}%) - "
                    f"{stats['items_per_second']:.2f} nuclei/sec - "
                    f"ETA: {(stats['estimated_total_time'] - stats['elapsed_time'])/60:.1f} min"
                )

        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()

        # Process nuclei in parallel batches
        successful_extractions = 0

        # Split data into chunks for better memory management
        chunk_size = self.config.chunk_size
        chunks = [
            df_filtered[i : i + chunk_size]
            for i in range(0, len(df_filtered), chunk_size)
        ]

        for chunk_idx, chunk in enumerate(chunks):
            print(
                f"\n🔄 Processing chunk {chunk_idx + 1}/{len(chunks)} "
                f"({len(chunk)} nuclei)"
            )

            # Process chunk in parallel
            executor_class = (
                ProcessPoolExecutor
                if self.config.use_process_pool
                else ThreadPoolExecutor
            )

            with executor_class(max_workers=self.config.max_workers_batch) as executor:
                # Submit extraction tasks
                future_to_nucleus = {}

                for idx, row in chunk.iterrows():
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

                    future = executor.submit(
                        self.extract_nucleus_time_series_parallel,
                        nucleus_id,
                        event_frame,
                    )

                    future_to_nucleus[future] = {
                        "nucleus_id": nucleus_id,
                        "event_frame": event_frame,
                        "event_type": event_type,
                        "is_mitotic": is_mitotic,
                        "is_death": is_death,
                        "row_idx": idx,
                    }

                # Collect results as they complete
                for future in as_completed(future_to_nucleus):
                    nucleus_info = future_to_nucleus[future]

                    try:
                        result = future.result()

                        if result and result["extraction_success"]:
                            # Add classification info
                            result["event_type"] = nucleus_info["event_type"]
                            result["is_mitotic"] = nucleus_info["is_mitotic"]
                            result["is_death"] = nucleus_info["is_death"]

                            # Save immediately if configured
                            if self.config.save_intermediate_results:
                                self.io_manager.save_extraction_results_parallel(
                                    [result], dataset_name, output_base_path
                                )
                            else:
                                self.results_queue.put(result)

                            successful_extractions += 1
                            progress.update(success=True)

                            if self.config.enable_detailed_logging:
                                print(
                                    f"  ✅ Nucleus {nucleus_info['nucleus_id']} "
                                    f"(frame {nucleus_info['event_frame']}) - "
                                    f"{nucleus_info['event_type']}"
                                )
                        else:
                            progress.update(success=False)
                            if self.config.enable_detailed_logging:
                                print(
                                    f"  ❌ Failed: Nucleus {nucleus_info['nucleus_id']} "
                                    f"(frame {nucleus_info['event_frame']})"
                                )

                    except Exception as e:
                        progress.update(success=False)
                        if self.config.enable_detailed_logging:
                            print(
                                f"  ❌ Error: Nucleus {nucleus_info['nucleus_id']}: {e}"
                            )

        # Final statistics
        final_stats = progress.get_stats()
        print(f"\n🎯 Parallel batch extraction complete:")
        print(f"  • Total time: {final_stats['elapsed_time']/60:.2f} minutes")
        print(
            f"  • Successful: {final_stats['successful']}/{final_stats['total']} "
            f"({final_stats['successful']/final_stats['total']*100:.1f}%)"
        )
        print(f"  • Failed: {final_stats['failed']}")
        print(f"  • Average rate: {final_stats['items_per_second']:.2f} nuclei/sec")

        return successful_extractions


def create_optimized_config(
    dataset_size: int = 1000,
    available_cores: int = None,
    available_memory_gb: float = 16.0,
) -> ParallelConfig:
    """
    Create an optimized configuration based on system resources and dataset size

    Args:
        dataset_size: Number of nuclei to process
        available_cores: Number of CPU cores available (auto-detect if None)
        available_memory_gb: Available memory in GB

    Returns:
        Optimized ParallelConfig
    """
    if available_cores is None:
        available_cores = cpu_count()

    # Calculate optimal worker distribution
    # Reserve some cores for system and I/O
    usable_cores = max(1, available_cores - 2)

    # For large datasets, use more batch workers
    if dataset_size > 1000:
        batch_workers = min(8, usable_cores // 2)
    elif dataset_size > 100:
        batch_workers = min(4, usable_cores // 3)
    else:
        batch_workers = min(2, usable_cores // 4)

    # Frame workers per nucleus
    frame_workers = min(8, usable_cores // max(1, batch_workers))

    # I/O workers (can be higher since they're mostly waiting)
    io_workers = min(16, usable_cores * 2)

    # Memory-based chunk size
    chunk_size = max(10, min(100, int(available_memory_gb * 5)))

    config = ParallelConfig(
        max_workers_batch=batch_workers,
        max_workers_frames=frame_workers,
        max_workers_io=io_workers,
        max_memory_gb=available_memory_gb,
        chunk_size=chunk_size,
        use_process_pool=True,  # Better for CPU-bound tasks
        use_thread_pool_io=True,  # Better for I/O-bound tasks
    )

    print(f"🔧 Optimized configuration for {dataset_size} nuclei:")
    print(f"   • Available cores: {available_cores}")
    print(f"   • Batch workers: {batch_workers}")
    print(f"   • Frame workers: {frame_workers}")
    print(f"   • I/O workers: {io_workers}")
    print(f"   • Chunk size: {chunk_size}")
    print(f"   • Memory limit: {available_memory_gb} GB")

    return config


# Example usage and main execution
if __name__ == "__main__":
    # Example: High-performance parallel extraction
    data_path = "/mnt/home/dchhantyal/3d-cnn-classification"
    dataset_name = "230212_stack6"

    # Create optimized configuration
    config = create_optimized_config(
        dataset_size=500,  # Expected number of nuclei
        available_memory_gb=32.0,  # Available system memory
    )

    # Initialize parallel extractor
    extractor = ParallelNucleusExtractor(data_path, config)

    # Load dataset
    if extractor.load_dataset(dataset_name):
        # Parallel batch extraction
        successful = extractor.batch_extract_nuclei_parallel(
            max_samples=100,  # Start with smaller batch for testing
            event_types=["death", "mitotic"],
            dataset_name=dataset_name,
        )

        print(f"🎉 Completed parallel extraction: {successful} nuclei processed")
    else:
        print("❌ Failed to load dataset")
