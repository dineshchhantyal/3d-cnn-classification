# predict.py
# Description: This script performs inference using a trained 4D CNN model on nucleus state data.
# It supports multiple prediction modes:
#   1. Batch processing of pre-cropped sample folders.
#   2. Prediction on a single pre-cropped sample from .tif files.
#   3. Prediction on specified nuclei from full-timestamp .tif volumes.

import torch
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from tifffile import imread
import traceback
import numpy as np

# --- Import Shared Modules ---
# These modules are assumed to be in the same directory or in the Python path.
from config import HPARAMS, CLASS_NAMES, DEVICE
from utils.data_utils import preprocess_sample, transform_and_pad_volume
from utils.model_utils import load_model, run_inference
from utils.prediction_utils import (
    get_true_label_from_path,
    get_volumes_by_nuclei_ids_from_full_volumes,
)


from utils.prediction_reports_utils import (
    print_batch_summary,
    generate_analysis_reports,
    generate_benchmark_summary,
)


def get_volumes_by_nuclei_ids(
    volume_paths: List[str], nuclei_ids: Optional[List[int]] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Get the cropped volumes for specific nuclei IDs from full timestamp volumes.

    Args:
        volume_paths (List[str]): List of paths to .tif files for t-1, t, t+1 and mask.
        If `nuclei_ids` is provided, only those IDs will be processed.
        If `nuclei_ids` is None, all nuclei found in the segmentation will be predicted.
        nuclei_ids (Optional[List[int]]): List of nuclei IDs to predict. If None, predicts all.

    Returns:
        Dict[int, Dict[str, Any]]: Dictionary where keys are nuclei IDs and values are dicts
                                    with keys 't-1', 't', 't+1', mask and their corresponding volumes.
    """
    # This function is expected to return a dictionary where keys are nuclei IDs
    # and values are dictionaries of {'t-1': vol, 't': vol, 't+1': vol, 'mask': vol (if available)}

    if not volume_paths or len(volume_paths) < 4:
        raise ValueError("At least 4 volume paths (t-1, t, t+1, mask) are required.")

    volume_previous = imread(volume_paths[0])
    volume_current = imread(volume_paths[1])
    volume_next = imread(volume_paths[2])
    volume_mask = imread(volume_paths[3])

    print(
        f"Loaded volumes: {volume_previous.shape}, {volume_current.shape}, {volume_next.shape}, {volume_mask.shape}"
    )

    if not nuclei_ids:
        # If no nuclei IDs provided, use all unique IDs from the mask
        print("No nuclei IDs provided, using all the nuclei IDs from the mask")
        nuclei_ids = np.unique(volume_mask.flatten()).tolist()

    print("Nuclei IDs to process:", nuclei_ids)

    all_nuclei_volumes = get_volumes_by_nuclei_ids_from_full_volumes(
        [volume_previous, volume_current, volume_next, volume_mask], nuclei_ids
    )

    return all_nuclei_volumes


def parse_arguments() -> argparse.Namespace:
    """
    Parses and validates command-line arguments for the prediction script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run 3D CNN prediction on nucleus state data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.pth file).",
    )

    # --- Input Data Arguments ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--folder_path",
        nargs="+",
        help="One or more paths to pre-cropped sample folders.",
    )
    input_group.add_argument(
        "--volumes",
        nargs="+",
        help="3-4 paths to .tif files: t-1, t, t+1, and optional segmentation label file.",
    )

    # --- Full Timestamp Processing Arguments ---
    parser.add_argument(
        "--full_timestamp",
        action="store_true",
        default=False,
        help="Treat --volumes as full timestamp images, not pre-cropped. Requires --volumes.",
    )
    parser.add_argument(
        "--nuclei_ids",
        type=str,
        default=None,
        help="Comma-separated list of nuclei IDs to predict from full timestamp volumes. Predicts all if not set.",
    )

    # --- Analysis & Output Arguments ---
    parser.add_argument(
        "--save_analysis",
        action="store_true",
        help="Save preprocessing and model analysis visualizations.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis_output",
        help="Directory to save analysis outputs.",
    )

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.full_timestamp and not args.volumes:
        parser.error("--full_timestamp requires --volumes to be set.")
    if args.volumes and not (3 <= len(args.volumes) <= 4):
        parser.error(
            "--volumes requires 3 or 4 paths (t-1, t, t+1, and optional label)."
        )

    return args


def get_sample_name(
    folder_path: Optional[str] = None,
    sample_name_override: Optional[str] = None,
) -> str:
    """
    Get a standardized sample name based on folder path or an override.

    Args:
        folder_path (Optional[str]): Path to the sample folder.
        sample_name_override (Optional[str]): A name to use for the sample, overriding default naming.

    Returns:
        str: A standardized sample name.
    """
    if sample_name_override:
        return sample_name_override
    elif folder_path:
        return os.path.basename(folder_path)
    else:
        return f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def process_single_sample_by_output_folder(
    model: torch.nn.Module,
    args: argparse.Namespace,
    folder_path: Optional[str] = None,
    volume_paths: Optional[List[str]] = None,
    sample_name_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Preprocesses and runs inference on a single sample.

    Args:
        model: The loaded PyTorch model.
        args: Command-line arguments.
        folder_path: Path to the sample folder.
        volume_paths: List of paths to volume files.
        sample_name_override: A name to use for the sample, overriding default naming.

    Returns:
        A dictionary containing the prediction results for the sample.
    """
    sample_name = get_sample_name(folder_path, sample_name_override)

    true_class = get_true_label_from_path(folder_path) if folder_path else None
    analysis_dir = (
        os.path.join(args.output_dir, sample_name) if args.save_analysis else None
    )

    input_tensor = preprocess_sample(
        folder_path=folder_path,
        volume_paths=volume_paths,
        for_training=False,
        save_analysis=args.save_analysis,
        analysis_output_dir=analysis_dir,
    )
    print(f"   Input tensor shape: {input_tensor.shape}")

    pred_index, pred_class, pred_confidence = run_inference(
        model,
        input_tensor[:, : HPARAMS["num_input_channels"]],
        save_analysis=args.save_analysis,
        analysis_output_dir=analysis_dir,
        sample_name=sample_name,
    )

    is_correct = (true_class == pred_class) if true_class is not None else None

    return {
        "sample": sample_name,
        "true_class": true_class,
        "predicted_class": pred_class,
        "index": pred_index,
        "confidence": pred_confidence,
        "correct": is_correct,
    }


def process_single_sample_by_np_volumes(
    volumes: Dict[str, Any],
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Processes a single sample using numpy arrays for the volume data. First it will find the min and max volume intensity and call transform_and_pad_volume to get the final vector. (Note: Label volumne will not be be normalized.)

    Args:
        volumes (Dict[str, Any]): Dictionary containing volume data.
        args (argparse.Namespace): Command-line arguments.

    Returns:
        Dict[str, Any]: Dictionary containing the prediction results.
    """

    sample_name = get_sample_name()

    analysis_dir = (
        os.path.join(args.output_dir, sample_name) if args.save_analysis else None
    )

    v_min, v_max = float("inf"), float("-inf")

    for key, vol in volumes.items():
        if key != "mask":
            v_min = min(v_min, vol.min())
            v_max = max(v_max, vol.max())
    print(f"   Volume intensity range: {v_min} to {v_max}")

    # Normalize and pad volumes
    for key, vol in volumes.items():
        volumes[key] = transform_and_pad_volume(
            vol,
            target_shape=[
                HPARAMS["input_depth"],
                HPARAMS["input_height"],
                HPARAMS["input_width"],
            ],
            v_min=v_min,
            v_max=v_max,
            is_label=(key == "mask"),
        )
        print(f"   {key} volume shape after transform: {volumes[key].shape}")

    # Convert volume data to tensors

    input_tensors = [
        torch.tensor(np.ascontiguousarray(value), dtype=torch.float32).unsqueeze(0)
        for key, value in volumes.items()
    ]
    input_tensor = torch.stack(input_tensors, dim=1)

    # Run inference
    pred_index, pred_class, pred_confidence = run_inference(
        model,
        input_tensor[:, : HPARAMS["num_input_channels"]],
        save_analysis=args.save_analysis,
        analysis_output_dir=analysis_dir,
        sample_name=sample_name,
    )

    return {
        "sample": sample_name,
        "true_class": None,
        "predicted_class": pred_class,
        "index": pred_index,
        "confidence": pred_confidence,
        "correct": None,
    }


def handle_batch_folder_prediction(
    model: torch.nn.Module, args: argparse.Namespace
) -> List[Dict[str, Any]]:
    """Handles prediction for a batch of sample folders."""
    print(f"\n🚀 Processing {len(args.folder_path)} sample(s) from folders...")
    print("=" * 50)
    results = []
    for i, folder in enumerate(args.folder_path):
        sample_start_time = datetime.now()
        sample_name = os.path.basename(folder)
        print(f"\n📁 Sample {i+1}/{len(args.folder_path)}: {sample_name}")
        try:
            result = process_single_sample_by_output_folder(
                model, args, folder_path=folder
            )
            result["processing_time"] = (
                datetime.now() - sample_start_time
            ).total_seconds()
            results.append(result)
            # Display immediate feedback
            true_label = str(result.get("true_class", "UNKNOWN") or "UNKNOWN").upper()
            pred_class_value = result.get("predicted_class", "UNKNOWN")
            pred_label = str(
                pred_class_value if pred_class_value is not None else "UNKNOWN"
            ).upper()
            confidence = result.get("confidence", 0)
            icon = (
                "✅"
                if result.get("correct")
                else "❌" if result.get("correct") is False else "❓"
            )
            print(
                f"   True: {true_label} → Predicted: {pred_label} ({confidence:.2%}) {icon}"
            )
        except Exception as e:
            print(f"   ❌ Error processing {sample_name}: {e}")
            results.append({"sample": sample_name, "error": str(e)})
    return results


def handle_full_timestamp_prediction(
    model: torch.nn.Module, args: argparse.Namespace
) -> List[Dict[str, Any]]:
    """Handles prediction on nuclei extracted from full timestamp volumes."""
    print("\n🔬 Processing nuclei from full timestamp volumes...")
    if args.nuclei_ids:
        try:
            nuclei_ids_list = [
                int(x.strip()) for x in args.nuclei_ids.split(",") if x.strip()
            ]
            if not nuclei_ids_list:
                raise ValueError("No valid nuclei IDs provided.")
            print(f"Targeting nuclei IDs: {nuclei_ids_list}")
        except ValueError as e:
            raise ValueError(
                f"Invalid nuclei_ids format: {args.nuclei_ids}. Expected comma-separated integers."
            ) from e
    else:
        nuclei_ids_list = None
        print("Targeting all nuclei found in the segmentation.")
    print("=" * 50)

    # This function is expected to return a dictionary where keys are nuclei IDs
    # and values are dictionaries of {'t-1': vol, 't': vol, 't+1': vol}
    nuclei_volumes = get_volumes_by_nuclei_ids(
        volume_paths=args.volumes, nuclei_ids=nuclei_ids_list
    )

    if not nuclei_volumes:
        print("Could not extract any nuclei volumes. Please check inputs.")
        return []
    print(f"Found {len(nuclei_volumes)} nuclei to process.")
    print(nuclei_volumes.keys())
    results = []
    total_nuclei = len(nuclei_volumes)
    for i, (nucleus_id, vol_dict) in enumerate(nuclei_volumes.items()):
        sample_start_time = datetime.now()
        sample_name = f"nucleus_{nucleus_id}"
        print(f"\n🔬 Predicting for Nucleus {i+1}/{total_nuclei}: {sample_name}")
        try:
            result = process_single_sample_by_np_volumes(
                volumes=vol_dict,
                model=model,
                args=args,
            )
            result["processing_time"] = (
                datetime.now() - sample_start_time
            ).total_seconds()
            results.append(result)
            # Display immediate feedback
            pred_label = str(result.get("predicted_class", "UNKNOWN")).upper()
            confidence = result.get("confidence", 0)
            print(f"   Predicted: {pred_label} ({confidence:.2%})")
        except Exception as e:
            print(f"   ❌ Error processing {sample_name}: {e}")
            results.append({"sample": sample_name, "error": str(e)})
    return results


def handle_single_volume_prediction(
    model: torch.nn.Module, args: argparse.Namespace
) -> List[Dict[str, Any]]:
    """Handles prediction for a single pre-cropped volume."""
    print("\n🔬 Processing single pre-cropped sample from volume paths...")
    result = process_single_sample_by_output_folder(
        model, args, volume_paths=args.volumes
    )
    print("\n--- Prediction Result ---")
    print(f"Predicted Class Index: {result['index']}")
    pred_class_name = result["predicted_class"]
    print(
        f"Predicted Class Name:  {(str(pred_class_name).upper() if pred_class_name is not None else 'UNKNOWN')}"
    )
    print(f"Confidence:            {result['confidence']:.2%}")
    print("-------------------------\n")
    return [result]


def main():
    """Main function to orchestrate the prediction process."""
    try:
        args = parse_arguments()
        start_time = datetime.now()

        print("Loading model...")
        model = load_model(args.model_path)
        model.eval()
        print(
            f"Model loaded in {(datetime.now() - start_time).total_seconds():.2f} seconds"
        )

        if args.save_analysis:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"📁 Analysis outputs will be saved to: {args.output_dir}")

        results = []
        if args.full_timestamp:
            results = handle_full_timestamp_prediction(model, args)
        elif args.folder_path:
            results = handle_batch_folder_prediction(model, args)
        elif args.volumes:
            results = handle_single_volume_prediction(model, args)

        if results:
            print_batch_summary(results)
            generate_analysis_reports(results, args)
            generate_benchmark_summary(results, start_time, args)
        else:
            print("\nNo predictions were made.")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred in the main process: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
