"""
Data Pipeline for Video Generation
Handles sliding window processing of time-series nucleus data
"""

import os
import json
import numpy as np
import tifffile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from dataclasses import dataclass
import torch
import logging

from .config import VideoConfig
from .model_interface import ModelInferenceEngine

# Setup logger
logger = logging.getLogger("video_generation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)


@dataclass
class FrameData:
    """Container for frame-level data."""

    timestamp: int
    raw_volume: np.ndarray
    label_volume: np.ndarray
    nuclei_ids: List[int]


class SlidingWindowProcessor:
    """
    Efficiently processes entire datasets using a sliding window approach.
    Handles memory management and parallel processing.
    """

    def __init__(self, config: VideoConfig):
        """
        Initialize the sliding window processor.

        Args:
            config: Video generation configuration
        """
        self.config = config
        self.raw_path = Path(config.raw_data_path)
        self.label_path = Path(config.label_data_path)
        self.cache_dir = Path(config.cache_dir)

        logger.info(f"🔍 SlidingWindowProcessor init:")
        logger.info(f"   config.raw_data_path: {config.raw_data_path}")
        logger.info(f"   config.label_data_path: {config.label_data_path}")
        logger.info(f"   self.raw_path: {self.raw_path}")
        logger.info(f"   self.label_path: {self.label_path}")

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Frame cache - keeps 3 frames in memory
        self.frame_cache = {}

        # Discover dataset structure
        self.frame_count, self.frame_range = self._discover_frame_count()
        # Use actual frame range instead of assuming 0-based indexing
        if self.frame_range:
            start_frame, end_frame = self.frame_range
            self.frame_indices = list(
                range(start_frame + 1, end_frame)
            )  # Skip boundaries for sliding window
        else:
            self.frame_indices = list(range(1, self.frame_count - 1))  # Fallback

        logger.info(f"📊 Dataset discovered:")
        logger.info(f"   Raw data: {self.raw_path}")
        logger.info(f"   Label data: {self.label_path}")
        logger.info(f"   Total frames: {self.frame_count}")
        if self.frame_range:
            logger.info(
                f"   Frame range: {self.frame_range[0]} to {self.frame_range[1]}"
            )
        logger.info(f"   Processable frames: {len(self.frame_indices)}")
        if self.frame_indices:
            logger.info(
                f"   Processing range: {self.frame_indices[0]} to {self.frame_indices[-1]}"
            )

    def _discover_frame_count(self) -> tuple:
        """Discover the number of frames and frame range in the dataset."""
        # Try to find frame files with different naming patterns
        raw_files = []

        # Pattern 1: nuclei_reg8_XXX.tif (your current data)
        pattern1_files = list(self.raw_path.glob("nuclei_reg8_*.tif"))
        if pattern1_files:
            raw_files = pattern1_files

        # Pattern 2: frame_XXX.tif
        if not raw_files:
            raw_files = list(self.raw_path.glob("frame_*.tif"))

        # Pattern 3: XXX.tif
        if not raw_files:
            raw_files = list(self.raw_path.glob("*.tif"))

        if raw_files:
            # Extract frame numbers from filenames
            frame_numbers = []
            for file_path in raw_files:
                filename = file_path.stem
                try:
                    # Try different patterns
                    if "nuclei_reg8_" in filename:
                        frame_num = int(filename.split("nuclei_reg8_")[1])
                    elif "frame_" in filename:
                        frame_num = int(filename.split("frame_")[1])
                    else:
                        # Try to extract number from filename
                        frame_num = int("".join(filter(str.isdigit, filename)))
                    frame_numbers.append(frame_num)
                except:
                    continue

            if frame_numbers:
                frame_numbers.sort()
                frame_range = (min(frame_numbers), max(frame_numbers))
                frame_count = max(frame_numbers) + 1
                print(
                    f"🔍 Discovered frames: {len(frame_numbers)} files, range {frame_range[0]}-{frame_range[1]}"
                )
                return frame_count, frame_range

        # If no direct files, look for subdirectories with frame numbers
        subdirs = [d for d in self.raw_path.iterdir() if d.is_dir()]
        if subdirs:
            # Try to extract frame numbers from directory names
            frame_numbers = []
            for subdir in subdirs:
                try:
                    # Assuming format like "frame_001", "frame_002", etc.
                    frame_num = int(subdir.name.split("_")[-1])
                    frame_numbers.append(frame_num)
                except:
                    continue

            if frame_numbers:
                frame_numbers.sort()
                frame_range = (min(frame_numbers), max(frame_numbers))
                frame_count = max(frame_numbers) + 1
                return frame_count, frame_range

        # Default fallback
        print("⚠️ Could not auto-detect frame count, using default of 100")
        return 100, None

    def _load_frame(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw and label volumes for a specific frame.

        Args:
            frame_idx: Frame index to load

        Returns:
            Tuple of (raw_volume, label_volume)
        """
        # DEBUG: Print the base paths being used
        logger.info(f"🔍 _load_frame: Looking for frame {frame_idx}")
        logger.debug(f"   Raw path base: {self.raw_path}")
        logger.debug(f"   Label path base: {self.label_path}")

        # Pattern 1: nuclei_reg8_XXX.tif and label_reg8_XXX.tif (your current data)
        raw_file = self.raw_path / f"nuclei_reg8_{frame_idx}.tif"
        label_file = self.label_path / f"label_reg8_{frame_idx}.tif"

        logger.debug(f"   Trying pattern 1: {raw_file}")

        # Check if the primary pattern exists
        if raw_file.exists() and label_file.exists():
            try:
                raw_volume = tifffile.imread(str(raw_file)).astype(np.float32)
                label_volume = tifffile.imread(str(label_file)).astype(np.int32)
                return raw_volume, label_volume
            except Exception as e:
                logger.warning(
                    f"⚠️ Error loading primary pattern for frame {frame_idx}: {e}"
                )

        # Pattern 2: frame_XXX.tif format
        if not (raw_file.exists() and label_file.exists()):
            raw_file = self.raw_path / f"frame_{frame_idx:03d}.tif"
            label_file = self.label_path / f"frame_{frame_idx:03d}.tif"
            logger.debug(f"   Trying pattern 2: {raw_file}")

            if raw_file.exists() and label_file.exists():
                try:
                    raw_volume = tifffile.imread(str(raw_file)).astype(np.float32)
                    label_volume = tifffile.imread(str(label_file)).astype(np.int32)
                    return raw_volume, label_volume
                except Exception as e:
                    logger.warning(
                        f"⚠️ Error loading pattern 2 for frame {frame_idx}: {e}"
                    )

        # Pattern 3: XXX.tif format (last resort)
        if not (raw_file.exists() and label_file.exists()):
            raw_file = self.raw_path / f"{frame_idx:03d}.tif"
            label_file = self.label_path / f"{frame_idx:03d}.tif"
            logger.debug(f"   Trying pattern 3: {raw_file}")

            if raw_file.exists() and label_file.exists():
                try:
                    raw_volume = tifffile.imread(str(raw_file)).astype(np.float32)
                    label_volume = tifffile.imread(str(label_file)).astype(np.int32)
                    return raw_volume, label_volume
                except Exception as e:
                    logger.warning(
                        f"⚠️ Error loading pattern 3 for frame {frame_idx}: {e}"
                    )

        # If we get here, the frame doesn't exist with any pattern
        logger.error(f"❌ Frame {frame_idx} not found with any naming pattern")
        if hasattr(self, "frame_range") and self.frame_range:
            min_frame, max_frame = self.frame_range
            logger.info(f"   Available frame range: {min_frame} to {max_frame}")
            if frame_idx < min_frame or frame_idx > max_frame:
                logger.warning(f"   Frame {frame_idx} is outside available range!")

        # Return empty volumes as fallback
        logger.warning(f"   Returning empty volumes for frame {frame_idx}")
        return np.zeros((64, 64, 64), dtype=np.float32), np.zeros(
            (64, 64, 64), dtype=np.int32
        )

    def ensure_frames_loaded(self, timestamp: int):
        """
        Ensure frames [t-1, t, t+1] are loaded in memory.

        Args:
            timestamp: Current timestamp being processed
        """
        required_frames = [timestamp - 1, timestamp, timestamp + 1]

        for frame_idx in required_frames:
            cache_key = f"frame_{frame_idx}"
            if cache_key not in self.frame_cache:
                raw_vol, label_vol = self._load_frame(frame_idx)
                self.frame_cache[cache_key] = {
                    "raw": raw_vol,
                    "label": label_vol,
                    "timestamp": frame_idx,
                }

    def extract_nuclei_ids(self, timestamp: int) -> List[int]:
        """
        Extract all nucleus IDs present in label volume at given timestamp.

        Args:
            timestamp: Frame timestamp

        Returns:
            List of nucleus IDs
        """
        cache_key = f"frame_{timestamp}"
        if cache_key not in self.frame_cache:
            self.ensure_frames_loaded(timestamp)

        label_volume = self.frame_cache[cache_key]["label"]
        unique_ids = np.unique(label_volume)

        # Remove background (ID 0)
        nuclei_ids = [int(nid) for nid in unique_ids if nid > 0]
        return nuclei_ids

    def extract_nucleus_sequence(
        self, nucleus_id: int, timestamp: int
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract 3-frame sequence for a specific nucleus.

        Args:
            nucleus_id: ID of the nucleus
            timestamp: Central timestamp

        Returns:
            Dictionary with keys 't-1', 't', 't+1' containing cropped volumes,
            or None if nucleus not found in central frame
        """
        # Check if nucleus exists in central frame
        central_label = self.frame_cache[f"frame_{timestamp}"]["label"]
        if nucleus_id not in central_label:
            return None

        # Get bounding box from central frame
        bbox = self._get_nucleus_bbox(nucleus_id, central_label)
        if bbox is None:
            return None

        # Extract volumes for all timepoints
        sequence = {}
        timepoints = ["t-1", "t", "t+1"]
        frame_indices = [timestamp - 1, timestamp, timestamp + 1]

        for tp, frame_idx in zip(timepoints, frame_indices):
            cache_key = f"frame_{frame_idx}"
            if cache_key in self.frame_cache:
                raw_vol = self.frame_cache[cache_key]["raw"]
                label_vol = self.frame_cache[cache_key]["label"]

                # Crop using bounding box
                cropped_raw = self._crop_volume(raw_vol, bbox)
                sequence[tp] = cropped_raw

                # Add label for 4ncnn models
                if self.config.model_type == "4ncnn" and tp == "t":
                    cropped_label = self._crop_volume(label_vol, bbox)
                    # Create binary mask for this nucleus
                    binary_mask = (cropped_label == nucleus_id).astype(np.float32)
                    sequence["label"] = binary_mask

        return sequence if len(sequence) >= 3 else None

    def _get_nucleus_bbox(
        self, nucleus_id: int, label_volume: np.ndarray
    ) -> Optional[Tuple[slice, ...]]:
        """Get fixed-size crop centered at nucleus centroid."""
        mask = label_volume == nucleus_id
        if not np.any(mask):
            return None

        coords = np.where(mask)
        centroid = [int(np.round(np.mean(c))) for c in coords]

        # Get desired output shape from config (assume tuple: (D, H, W))
        if hasattr(self.config, "input_shape"):
            crop_shape = self.config.input_shape
        elif hasattr(self.config, "crop_shape"):
            crop_shape = self.config.crop_shape
        else:
            crop_shape = (64, 64, 64)  # fallback default

        slices = []
        for i, center in enumerate(centroid):
            shape_max = label_volume.shape[i]
            half = crop_shape[i] // 2
            slice_min = max(0, center - half)
            slice_max = min(shape_max, center + half)
            # If crop_shape is odd, ensure correct size
            if (slice_max - slice_min) < crop_shape[i]:
                if slice_min == 0:
                    slice_max = min(shape_max, slice_min + crop_shape[i])
                else:
                    slice_min = max(0, slice_max - crop_shape[i])
            slices.append(slice(slice_min, slice_max))

        return tuple(slices)

    def _crop_volume(self, volume: np.ndarray, bbox: Tuple[slice, ...]) -> np.ndarray:
        """Crop volume using bounding box."""
        return volume[bbox]

    def parallel_extract_nuclei(
        self, nuclei_list: List[int], timestamp: int
    ) -> Dict[int, Optional[Dict[str, np.ndarray]]]:
        """
        Extract nucleus sequences in parallel.

        Args:
            nuclei_list: List of nucleus IDs to extract
            timestamp: Central timestamp

        Returns:
            Dictionary mapping nucleus ID to sequence data
        """
        if not self.config.use_parallel_processing:
            # Sequential processing
            results = {}
            for nucleus_id in nuclei_list:
                results[nucleus_id] = self.extract_nucleus_sequence(
                    nucleus_id, timestamp
                )
            return results

        # Parallel processing using threads (since we're I/O bound)
        results = {}
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_to_nucleus = {
                executor.submit(
                    self.extract_nucleus_sequence, nucleus_id, timestamp
                ): nucleus_id
                for nucleus_id in nuclei_list
            }

            for future in future_to_nucleus:
                nucleus_id = future_to_nucleus[future]
                try:
                    results[nucleus_id] = future.result()
                except Exception as e:
                    logger.warning(f"⚠️ Error extracting nucleus {nucleus_id}: {e}")
                    results[nucleus_id] = None

        return results

    def cleanup_old_frames(self, current_timestamp: int):
        """Remove old frames from cache to manage memory."""
        frames_to_remove = []
        for cache_key in self.frame_cache:
            frame_idx = int(cache_key.split("_")[1])
            if frame_idx < current_timestamp - 1:
                frames_to_remove.append(cache_key)

        for key in frames_to_remove:
            del self.frame_cache[key]

    def format_results(
        self, timestamp: int, predictions: Dict[int, Tuple[int, str, float]]
    ) -> Dict[str, Any]:
        """
        Format prediction results for a timestamp.

        Args:
            timestamp: Frame timestamp
            predictions: Dictionary mapping nucleus_id to (class_idx, class_name, confidence)

        Returns:
            Formatted results dictionary
        """
        formatted_predictions = {}
        for nucleus_id, (class_idx, class_name, confidence) in predictions.items():
            formatted_predictions[str(nucleus_id)] = {
                "class_index": class_idx,
                "class_name": class_name,
                "confidence": float(confidence),
                "nucleus_id": nucleus_id,
            }

        return {
            "timestamp": timestamp,
            "frame_index": timestamp,
            "nuclei_count": len(predictions),
            "predictions": formatted_predictions,
        }

    def process_full_dataset(self, model_engine: ModelInferenceEngine) -> str:
        """
        Process the entire dataset using sliding window approach.

        Args:
            model_engine: Loaded model inference engine

        Returns:
            Path to predictions file
        """
        predictions_file = self.cache_dir / "predictions.jsonl"

        print(f"🚀 Starting dataset processing...")
        print(f"   Processing {len(self.frame_indices)} frames")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Output file: {predictions_file}")

        start_time = time.time()

        with open(predictions_file, "w") as f:
            for i, timestamp in enumerate(self.frame_indices):
                frame_start_time = time.time()

                # Load required frames [t-1, t, t+1]
                self.ensure_frames_loaded(timestamp)

                # Get all nuclei present in frame t
                nuclei_list = self.extract_nuclei_ids(timestamp)

                if not nuclei_list:
                    print(f"⚠️ No nuclei found in frame {timestamp}")
                    continue

                # Parallel extraction of nucleus sequences
                nucleus_data = self.parallel_extract_nuclei(nuclei_list, timestamp)

                # Filter out failed extractions and preprocess
                valid_nuclei = []
                processed_tensors = []

                for nucleus_id, sequence in nucleus_data.items():
                    if sequence is not None:
                        try:
                            processed_tensor = model_engine.preprocess_nucleus(sequence)
                            valid_nuclei.append(nucleus_id)
                            processed_tensors.append(processed_tensor)
                        except Exception as e:
                            logger.warning(
                                f"⚠️ Failed to preprocess nucleus {nucleus_id}: {e}"
                            )

                # Batch prediction on GPU
                if processed_tensors:
                    batch_predictions = model_engine.batch_predict_gpu(
                        processed_tensors
                    )

                    # Convert to results format
                    predictions = {}
                    for j, nucleus_id in enumerate(valid_nuclei):
                        probs = batch_predictions[j]
                        confidence, class_idx = torch.max(probs, 0)
                        class_name = model_engine.class_names[class_idx.item()]

                        predictions[nucleus_id] = (
                            class_idx.item(),
                            class_name,
                            confidence.item(),
                        )
                else:
                    predictions = {}

                # Format and save results
                frame_result = self.format_results(timestamp, predictions)
                f.write(json.dumps(frame_result) + "\n")
                f.flush()  # Ensure data is written immediately

                # Memory cleanup
                self.cleanup_old_frames(timestamp)

                # Progress reporting
                frame_time = time.time() - frame_start_time
                total_time = time.time() - start_time
                avg_time = total_time / (i + 1)
                remaining_frames = len(self.frame_indices) - (i + 1)
                eta = remaining_frames * avg_time

                logger.info(
                    f"📊 Frame {timestamp:3d}/{self.frame_indices[-1]} | "
                    f"Nuclei: {len(valid_nuclei):2d} | "
                    f"Time: {frame_time:.1f}s | "
                    f"ETA: {eta/60:.1f}min"
                )

        total_time = time.time() - start_time
        logger.info(f"✅ Dataset processing complete!")
        logger.info(f"   Total time: {total_time/60:.1f} minutes")
        logger.info(
            f"   Average time per frame: {total_time/len(self.frame_indices):.1f}s"
        )
        logger.info(f"   Predictions saved to: {predictions_file}")

        return str(predictions_file)


if __name__ == "__main__":
    # Test the data pipeline
    from config import VideoConfig

    logger.info("Testing SlidingWindowProcessor...")

    # Create test configuration
    config = VideoConfig()
    config.raw_data_path = "/dummy/raw"
    config.label_data_path = "/dummy/labels"
    config.cache_dir = "./test_cache"

    try:
        processor = SlidingWindowProcessor(config)
        logger.info("✅ SlidingWindowProcessor created successfully")
        logger.info(f"   Frame count: {processor.frame_count}")
        logger.info(f"   Processable frames: {len(processor.frame_indices)}")
    except Exception as e:
        logger.error(f"❌ Failed to create processor: {e}")

    logger.info("Data pipeline testing complete!")
