"""
Nucleus Extraction Module with Sliding Window Processing

This module provides efficient nucleus extraction from 3D time series data using
a sliding window approach that processes all nuclei at each timestamp together.

Important Note: Nucleus IDs are randomly assigned in each frame and cannot be
compared across different frames. This module only uses nucleus IDs for the
event frame to calculate bounding boxes, then applies the same spatial region
to all frames in the time series.

"""

from collections import defaultdict
from pathlib import Path
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial
from sliding_window_volume_manager import SlidingWindowVolumeManager
from volume_utils import (
    find_nucleus_bounding_box,
    check_if_frame_exists,
    get_fixed_size_bbox,
    generate_frame_label,
    safe_bounds,
)
from lineage_tree import classify_node
from metadata_utils import (
    print_classification_distribution,
    save_single_nucleus_immediate,
)

STABLE_WINDOW_LIMIT = 8


def process_single_nucleus_threadsafe(
    candidate,
    timestamp,
    timeframe,
    volume_list,
    output_dir,
    dataset_name="230212_stack6",
    total_nuclei_in_entire_frame=None,
    fixed_size=None,
):
    """
    Thread-safe function to process a single nucleus extraction and save.

    This function is designed to be called concurrently for multiple nuclei
    within the same timestamp. It handles all extraction and saving for one nucleus.

    Args:
        candidate: Dictionary containing 'node' and 'classification'
        timestamp: Current timestamp being processed
        timeframe: Number of frames before/after the event frame
        volume_list: List of (frame_num, reg_volume, lbl_volume) tuples
        output_dir: Base output directory for saving
        dataset_name: Dataset name for file naming

    Returns:
        dict: Processing result with success status and metadata
    """
    thread_id = threading.get_ident()
    node = candidate["node"]
    classification = candidate["classification"]

    # Convert nucleus_id to integer to match numpy array data types
    try:
        nucleus_id = int(node.label)
    except (ValueError, TypeError):
        return {
            "success": False,
            "error": f"Invalid nucleus ID format: {node.label}",
            "nucleus_id": node.label,
            "thread_id": thread_id,
            "classification": classification,
        }

    print(
        f"      [Thread {thread_id}] Processing nucleus {nucleus_id} ({classification.upper()})"
    )

    # Extract time series for this nucleus
    node_info = {
        "node_id": node.node_id,
        "timestamp": timestamp,
        "parent": node.parent.node_id if node.parent else None,
        "children": (list(node.id_to_child.keys()) if node.id_to_child else []),
    }

    try:
        result = extract_nucleus_time_series(
            nucleus_id,
            timestamp,
            timeframe,
            volume_list,
            classification=classification,
            node_info=node_info,
            output_dir=output_dir,
            dataset_name=dataset_name,
            total_nuclei_in_entire_frame=total_nuclei_in_entire_frame,
            fixed_size=fixed_size,
        )

        if result and result.get("extraction_success"):
            return {
                "success": True,
                "nucleus_id": nucleus_id,
                "event_frame": timestamp,
                "saved": result.get("saved", False),
                "save_path": result.get("save_path"),
                "volume_size": result.get("volume_size", 0),
                "thread_id": thread_id,
                "classification": classification,
            }
        else:
            return {
                "success": False,
                "error": "Extraction failed",
                "nucleus_id": nucleus_id,
                "thread_id": thread_id,
                "classification": classification,
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Exception during processing: {str(e)}",
            "nucleus_id": nucleus_id,
            "thread_id": thread_id,
            "classification": classification,
        }


def get_extraction_plan_with_sample_limits(
    extraction_plan, valid_timestamps, max_samples
):
    """
    Apply sample limits to the extraction plan.

    Args:
        extraction_plan (dict): The initial extraction plan.
        valid_timestamps (list): List of valid timestamps for extraction.
        max_samples (int, optional): Maximum samples per classification.

    Returns:
        dict: Updated extraction plan with sample limits applied.
    """
    if max_samples is None:
        return extraction_plan

    class_samples = defaultdict(int)
    filtered_plan = {}

    for timestamp in valid_timestamps:
        filtered_candidates = []
        for candidate in extraction_plan[timestamp]:
            classification = candidate["classification"]
            if class_samples[classification] < max_samples:
                filtered_candidates.append(candidate)
                class_samples[classification] += 1

        if filtered_candidates:  # Only include timestamps with valid candidates
            filtered_plan[timestamp] = filtered_candidates

    print(f"\n🎯 LIMITED TO {max_samples} SAMPLES PER CLASS:")
    for classification, count in class_samples.items():
        print(f"   • {classification.upper()}: {count} samples")

    return filtered_plan


def nucleus_extractor(
    forest,
    timeframe=1,
    base_dir="data",
    output_dir="extracted_nuclei",
    max_samples=None,
    fixed_size=None,
):
    """
    Extract nuclei by processing all nuclei at each timestamp together.

    Algorithm:
    1. Load the initial sliding window (e.g., frames [0,1,2,3,4] for timestamp 2, timeframe=2).
    2. Process ALL nuclei at the current timestamp using the same loaded volumes.
    3. Slide the window by one frame and repeat for the next timestamp.

    This minimizes volume loading and maximizes efficiency.

    Args:
        forest: Forest object containing the lineage tree.
        timeframe: Timeframe for extraction (default is 1).
        base_dir: Base directory of the dataset.
        output_dir: Directory to save extracted nuclei.
        max_samples: Maximum number of samples to extract per classification (None for all). Note: this is per classification, not total. Only the first max_samples will be extracted per classification. TODO: Random sampling.
        fixed_size: Optional fixed size for bounding box (if None, uses dynamic bounding box) e.g [32, 32, 32], center of the nucleus will be at the center of the bounding box.

    Returns:
        dict: Extraction results organized by classification.
    """
    print("🚀 Starting nucleus extraction with timestamp-based processing...")
    # print configuration
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Timeframe: ±{timeframe} frames")
    print(
        f"Max samples per classification: {max_samples if max_samples else 'unlimited'}"
    )
    print(f"Fixed size for bounding box: {fixed_size if fixed_size else 'dynamic'}")
    print("=" * 60)

    forest.find_tracks_and_lineages()
    # Group nodes by timestamp
    nodes_by_timestamp = defaultdict(list)
    for node in forest.id_to_node.values():
        nodes_by_timestamp[node.timestamp_ordinal].append(node)

    sorted_timestamps = sorted(nodes_by_timestamp.keys())
    final_frame = max(sorted_timestamps)

    # Check what frames are actually available in the dataset
    print(f"📁 Checking available frames in dataset...")
    available_frames = []
    for frame in sorted_timestamps:
        if check_if_frame_exists(base_dir, frame):
            available_frames.append(frame)

    if not available_frames:
        print(f"❌ No volumes found in dataset!")
        return {}

    first_available_frame = min(available_frames)
    last_available_frame = max(available_frames)

    # Filter timestamps to ensure we have enough frames for time series
    # For timeframe=2, we need [t-2, t-1, t, t+1, t+2] so t must be >= first_frame + timeframe
    valid_timestamps = [
        t
        for t in sorted_timestamps
        if (
            t >= first_available_frame + timeframe
            and t <= last_available_frame - timeframe
            and t in available_frames
        )
    ]

    print(f"📅 EXTRACTION PLAN:")
    print(f"   Total timestamps: {len(sorted_timestamps)}")
    print(
        f"   Available frames: {len(available_frames)} (range: {first_available_frame}-{last_available_frame})"
    )
    print(f"   Valid timestamps for extraction: {len(valid_timestamps)}")
    print(
        f"   First valid timestamp: {valid_timestamps[0] if valid_timestamps else 'None'}"
    )
    print(f"   First five timestamps: {valid_timestamps[:5]}")
    print(f"   Timeframe: ±{timeframe} frames")
    print(f"   Final frame: {final_frame}")

    # Collect extraction candidates with sample limits
    extraction_plan = {}
    classification_counts = defaultdict(int)

    for timestamp in valid_timestamps[76:]:  # Skip first 76 timestamps for testing
        nodes = nodes_by_timestamp[timestamp]
        timestamp_candidates = []

        for node in nodes:
            classification = classify_node(
                node, final_frame, forest, STABLE_WINDOW_LIMIT
            )
            classification_counts[classification] += 1
            timestamp_candidates.append(
                {"node": node, "classification": classification}
            )

        extraction_plan[timestamp] = timestamp_candidates

    # show classification distribution
    print_classification_distribution(
        classification_counts, max_samples, output_dir, save=True
    )

    # Apply sample limits if specified

    if max_samples:
        extraction_plan = get_extraction_plan_with_sample_limits(
            extraction_plan, valid_timestamps, max_samples
        )

    # Initialize sliding window volume manager
    volume_manager = SlidingWindowVolumeManager(base_dir, timeframe)

    # Process timestamps sequentially
    results = defaultdict(list)
    successful_extractions = 0
    total_candidates = sum(len(candidates) for candidates in extraction_plan.values())

    print(
        f"\n🔄 STARTING EXTRACTIONS ({total_candidates} total nuclei across {len(extraction_plan)} timestamps)..."
    )

    for timestamp_idx, timestamp in enumerate(sorted(extraction_plan.keys())):
        candidates = extraction_plan[timestamp]

        print(
            f"\n🕒 PROCESSING TIMESTAMP {timestamp} ({timestamp_idx + 1}/{len(extraction_plan)})"
        )
        print(f"   Nuclei to extract: {len(candidates)}")

        # Load/slide window to this timestamp
        volume_manager.slide_to_frame(timestamp)
        volume_list = volume_manager.get_volumes_for_extraction()

        # Find the center frame (current timestamp) volumes for bbox calculations
        center_volumes = None
        for frame_num, reg_vol, lbl_vol in volume_list:
            if frame_num == timestamp:
                center_volumes = {"registered_image": reg_vol, "label_image": lbl_vol}
                break

        if center_volumes is None or center_volumes["label_image"] is None:
            print(f"   ❌ Center frame volumes not available for timestamp {timestamp}")
            continue

        # Get all unique nucleus IDs in this timestamp's label volume
        center_label_volume = center_volumes["label_image"]
        all_nucleus_ids = np.unique(center_label_volume)
        all_nucleus_ids = all_nucleus_ids[all_nucleus_ids > 0]  # Remove background

        print(
            f"   📊 Found {len(all_nucleus_ids)} total nuclei in label volume: {all_nucleus_ids[:10].tolist()}{'...' if len(all_nucleus_ids) > 10 else ''}"
        )

        # Use parallel/concurrent processing for the candidates
        # Filter candidates to only include nuclei that exist in this timestamp
        valid_candidates = []
        for candidate in candidates:
            try:
                nucleus_id = int(candidate["node"].label)
                if nucleus_id in all_nucleus_ids:
                    valid_candidates.append(candidate)
                else:
                    print(
                        f"      ❌ Nucleus {nucleus_id} not found in timestamp {timestamp}"
                    )
            except (ValueError, TypeError):
                print(f"      ❌ Invalid nucleus ID format: {candidate['node'].label}")
                continue

        if not valid_candidates:
            print(f"   ❌ No valid candidates found for timestamp {timestamp}")
            continue

        print(
            f"   🎯 Processing {len(valid_candidates)} valid nuclei with concurrent extraction..."
        )

        # Use ThreadPoolExecutor for concurrent processing of nuclei within this timestamp
        # We use a conservative number of threads to avoid overwhelming the system
        max_workers = min(len(valid_candidates), os.cpu_count() or 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a partial function with fixed arguments for all nuclei in this timestamp
            process_func = partial(
                process_single_nucleus_threadsafe,
                timestamp=timestamp,
                timeframe=timeframe,
                volume_list=volume_list,
                output_dir=output_dir,
                dataset_name=Path(base_dir).name,
                total_nuclei_in_entire_frame=len(all_nucleus_ids),
                fixed_size=fixed_size,  # Use dynamic bounding box by default
            )

            # Submit all tasks
            future_to_candidate = {
                executor.submit(process_func, candidate): candidate
                for candidate in valid_candidates
            }

            # Collect results as they complete
            timestamp_successes = 0
            for future in as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    result = future.result()

                    if result["success"]:
                        timestamp_successes += 1
                        successful_extractions += 1

                        # Keep minimal tracking for summary
                        results[result["classification"]].append(
                            {
                                "nucleus_id": result["nucleus_id"],
                                "event_frame": timestamp,
                                "saved": result.get("saved", False),
                                "save_path": result.get("save_path"),
                                "volume_size": result.get("volume_size", 0),
                            }
                        )

                        print(
                            f"      ✅ [Thread {result['thread_id']}] Nucleus {result['nucleus_id']} extraction successful"
                        )
                    else:
                        print(
                            f"      ❌ [Thread {result['thread_id']}] Nucleus {result['nucleus_id']} failed: {result.get('error', 'Unknown error')}"
                        )

                except Exception as e:
                    nucleus_id = candidate["node"].label
                    print(
                        f"      ❌ Exception processing nucleus {nucleus_id}: {str(e)}"
                    )

        print(
            f"   📊 Timestamp {timestamp} completed: {timestamp_successes}/{len(valid_candidates)} nuclei extracted successfully"
        )

    print(f"\n🎯 EXTRACTION COMPLETE:")
    print(f"   Successful: {successful_extractions}/{total_candidates}")
    print(f"   Success rate: {successful_extractions/total_candidates*100:.1f}%")

    for classification, class_results in results.items():
        saved_count = sum(1 for r in class_results if r.get("saved", False))
        failed_count = len(class_results) - saved_count
        print(f"   • {classification.upper()}: {saved_count} saved successfully")
        if failed_count > 0:
            print(f"     └─ {failed_count} failures")

    return dict(results)


def extract_nucleus_time_series(
    nucleus_id,
    event_frame,
    timeframe,
    volume_list,
    classification=None,
    node_info=None,
    output_dir=None,
    dataset_name="230212_stack6",
    total_nuclei_in_entire_frame=None,
    fixed_size=None,
):
    """
    Extract time series for a nucleus and save immediately to disk.

    Since nucleus IDs are randomly assigned in each frame and cannot be compared
    across frames, this function:
    1. Uses the target nucleus_id only for the event frame to calculate bounding box
    2. Applies the same bounding box to all frames in the time series
    3. For event frame: provides detailed processing (cropped, dual-color, etc.)
    4. For other frames: only provides raw original data within the bounding box
    5. Saves each nucleus immediately to proper folder structure

    Args:
        nucleus_id: Target nucleus ID (only valid for event frame)
        event_frame: Frame where the event occurs (center frame)
        timeframe: Number of frames before and after
        volume_list: List of (frame_num, reg_volume, lbl_volume) tuples
        classification: Classification type (for folder structure)
        node_info: Node information (for metadata)
        output_dir: Output directory for saving (if None, saves to memory only)
        dataset_name: Dataset name for file naming
        total_nuclei_in_entire_frame: Total nuclei count in the entire frame (for metadata)
        fixed_size: Optional fixed size for bounding box (if None, uses dynamic bounding box else uses fixed_size) e.g [32, 32, 32], center of the nucelus will be at the center of the bounding box

    Returns:
        dict: Minimal extraction results (save status and path)
    """
    # Generate dynamic frame range
    frames = list(range(event_frame - timeframe, event_frame + timeframe + 1))

    print(
        f"      🔍 Extracting nucleus {nucleus_id} from {len(frames)} frames: {frames}"
    )

    # Find event frame volume for bounding box calculation
    event_volume_data = None
    for frame_num, reg_vol, lbl_vol in volume_list:
        if frame_num == event_frame:
            event_volume_data = {"registered_image": reg_vol, "label_image": lbl_vol}
            break

    if event_volume_data is None or event_volume_data["label_image"] is None:
        print(f"      ❌ Event frame data not found: {event_frame}")
        return None

    # Calculate bounding box from the event frame only
    bbox = find_nucleus_bounding_box(event_volume_data["label_image"], nucleus_id)
    if bbox is None:
        print(f"      ❌ Nucleus {nucleus_id} not found in event frame {event_frame}")
        return {
            "nucleus_id": nucleus_id,
            "event_frame": event_frame,
            "frames": frames,
            "extraction_success": False,
            "error": f"Nucleus {nucleus_id} not found in event frame {event_frame}",
        }

    # If fixed_size is provided, override bbox
    if fixed_size is not None:
        # Compute centroid from bbox
        zc = (bbox[0] + bbox[1]) // 2
        yc = (bbox[2] + bbox[3]) // 2
        xc = (bbox[4] + bbox[5]) // 2
        centroid = (zc, yc, xc)
        bbox = get_fixed_size_bbox(
            centroid, fixed_size, event_volume_data["label_image"].shape
        )

    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    print(
        f"      📦 Bounding box: Z[{z_min}:{z_max}], Y[{y_min}:{y_max}], X[{x_min}:{x_max}]"
    )

    # Storage for results
    results = {
        "nucleus_id": nucleus_id,
        "event_frame": event_frame,
        "frames": frames,
        "bounding_box": bbox,
        "time_series": {},
        "extraction_success": True,
        "extraction_method": "sliding_window",
    }

    # Extract each frame using the SAME bounding box from event frame
    for frame_num, reg_volume, lbl_volume in volume_list:
        # Generate dynamic label using utility function
        frame_label = generate_frame_label(frame_num, event_frame)

        print(f"         📸 Processing {frame_label} (frame {frame_num})")

        if reg_volume is None or lbl_volume is None:
            print(f"         ❌ Frame {frame_num} not available")
            results["extraction_success"] = False
            continue

        # Extract SAME region of interest using the bounding box from event frame
        try:
            # Use safe_bounds utility function
            bounds = safe_bounds(reg_volume.shape, bbox)
            z_start, z_end = bounds["z"]
            y_start, y_end = bounds["y"]
            x_start, x_end = bounds["x"]

            img_roi = reg_volume[z_start:z_end, y_start:y_end, x_start:x_end]
            lbl_roi = lbl_volume[z_start:z_end, y_start:y_end, x_start:x_end]

            if img_roi.size == 0 or lbl_roi.size == 0:
                print(f"         ❌ Empty ROI extracted for frame {frame_num}")
                results["extraction_success"] = False
                continue

        except Exception as e:
            print(f"         ❌ Error extracting ROI for frame {frame_num}: {e}")
            results["extraction_success"] = False
            continue

        # Analyze what's in this region
        unique_labels = np.unique(lbl_roi)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background (0)

        # For current/event frame, provide detailed processing
        if frame_num == event_frame:
            # Current frame - we know the target nucleus ID exists here
            target_nucleus_in_frame = nucleus_id

            if target_nucleus_in_frame in unique_labels:
                # Create mask for the target nucleus
                target_mask = (lbl_roi == target_nucleus_in_frame).astype(np.uint8)

                # Create cropped versions with only target nucleus
                raw_cropped = img_roi.copy()
                raw_cropped[target_mask == 0] = 0

                label_cropped = np.zeros_like(lbl_roi)
                label_cropped[target_mask == 1] = target_nucleus_in_frame

                # Create dual-color label (target=1, others=2)
                label_dual_color = np.zeros_like(lbl_roi)
                label_dual_color[target_mask == 1] = 1  # Target nucleus
                label_dual_color[(lbl_roi > 0) & (target_mask == 0)] = 2  # Other nuclei

                volume_size = np.sum(target_mask)
                print(
                    f"         ✅ Event frame - Target nucleus {target_nucleus_in_frame} | Volume: {volume_size} pixels"
                )

                # Store comprehensive data for event frame
                frame_data = {
                    "raw_original": img_roi,
                    "label_original": lbl_roi,
                    "raw_cropped": raw_cropped,
                    "label_cropped": label_cropped,
                    "label_dual_color": label_dual_color,
                    "target_mask": target_mask,
                    "volume_size": volume_size,
                    "bbox": bbox,
                    "unique_labels_in_region": unique_labels.tolist(),
                }
            else:
                print(
                    f"         ❌ Target nucleus {target_nucleus_in_frame} not found in event frame"
                )
                # Create empty processed data
                frame_data = {
                    "raw_original": img_roi,
                    "label_original": lbl_roi,
                    "raw_cropped": None,
                    "label_cropped": None,
                    "label_dual_color": None,
                    "target_mask": None,
                    "volume_size": 0,
                    "bbox": bbox,
                    "unique_labels_in_region": unique_labels.tolist(),
                }
        else:
            # Other frames - just extract same spatial region, ignore nucleus IDs completely
            # We don't care what nucleus IDs are in this region - just capture whatever is spatially there
            print(
                f"         📸 Frame {frame_num} - Extracting spatial region (ignoring nucleus IDs)"
            )

            # Create a simple dual-color label for this frame (any nuclei=1, background=0)
            simple_dual_color = np.zeros_like(lbl_roi)
            simple_dual_color[lbl_roi > 0] = (
                1  # Any nucleus present = 1, background = 0
            )

            frame_data = {
                "raw_original": img_roi,
                "label_original": lbl_roi,
                "label_dual_color": simple_dual_color,  # Simple binary mask
                "unique_labels_in_region": unique_labels.tolist(),
                "bbox": bbox,
                "spatial_extraction": True,  # Flag to indicate this is spatial-only extraction
            }

        # Store results for this frame
        results["time_series"][frame_label] = {
            "frame_number": frame_num,
            "is_event_frame": frame_num == event_frame,
            "data": frame_data,
        }

    # Calculate summary statistics
    event_frame_data = results["time_series"].get("t", {}).get("data", {})
    has_event_frame_data = "volume_size" in event_frame_data

    # Extraction is successful only if:
    # 1. We have event frame data AND
    # 2. extraction_success flag is still True (no missing frames or errors)
    final_success = has_event_frame_data and results["extraction_success"]
    results["extraction_success"] = final_success

    # Count how many frames we actually extracted
    frames_extracted = len(results["time_series"])
    expected_frames = len(frames)

    results["summary"] = {
        "total_frames": expected_frames,
        "frames_extracted": frames_extracted,
        "all_frames_present": frames_extracted == expected_frames,
        "event_frame_processed": has_event_frame_data,
        "event_frame_volume_size": (
            event_frame_data.get("volume_size", 0) if has_event_frame_data else 0
        ),
        "roi_shape": img_roi.shape if "img_roi" in locals() else None,
        "extraction_successful": final_success,
        "method": "sliding_window",
    }

    if final_success:
        print(
            f"      ✅ Extraction complete | Event frame volume: {event_frame_data.get('volume_size', 0)} pixels | Frames extracted: {frames_extracted}/{expected_frames}"
        )
    elif has_event_frame_data:
        print(
            f"      ⚠️ Partial success | Event frame found but missing {expected_frames - frames_extracted} frames"
        )
    else:
        print(f"      ❌ Extraction failed | No target nucleus found in event frame")

    # Save immediately to disk if output_dir and classification are provided
    if output_dir and classification and final_success:
        try:
            # Add required metadata for saving
            results["classification"] = classification
            results["node_info"] = node_info or {}

            # Save to proper folder structure
            nucleus_dir_path = save_single_nucleus_immediate(
                results,
                output_dir,
                dataset_name,
                classification,
                total_nuclei_in_entire_frame,
            )

            print(f"      💾 Saved to: {nucleus_dir_path}")

            # Return minimal result for memory efficiency
            return {
                "nucleus_id": nucleus_id,
                "event_frame": event_frame,
                "extraction_success": True,
                "saved": True,
                "save_path": nucleus_dir_path,
                "volume_size": event_frame_data.get("volume_size", 0),
            }

        except Exception as e:
            print(f"      ❌ Save failed: {e}")
            return {
                "nucleus_id": nucleus_id,
                "event_frame": event_frame,
                "extraction_success": True,
                "saved": False,
                "error": str(e),
            }

    # Return full results if not saving immediately
    return results
