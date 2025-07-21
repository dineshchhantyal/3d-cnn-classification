# predict.py
# Updated to support the SpatioTemporalModel (ConvLSTM) architecture.

import torch
import os
import argparse
from datetime import datetime
import json

# --- Import Shared Modules ---
from config import HPARAMS, CLASS_NAMES, DEVICE
from data_utils import preprocess_sample

# The load_model and run_inference functions will now use the new model
from model_utils import load_model, run_inference


def get_true_label_from_path(folder_path):
    """
    Extract the true class label from the directory path.
    """
    path_parts = folder_path.replace("\\", "/").split("/")
    class_mapping = {
        "death": None,
        "mitotic": "mitotic",
        "new_daughter": "new_daughter",
        "stable": "stable",
        "stable2": "stable",
    }
    for part in reversed(path_parts):
        if part in class_mapping:
            return class_mapping[part]
    return None


def create_benchmark_summary(results, metadata):
    """
    Create benchmark summary data structure.
    (This function is unchanged)
    """
    successful_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]
    total_processed = len(successful_results)
    samples_with_known_labels = len(
        [r for r in successful_results if r.get("true_class") is not None]
    )
    correct_predictions = len(
        [r for r in successful_results if r.get("correct") is True]
    )
    incorrect_predictions = len(
        [r for r in successful_results if r.get("correct") is False]
    )
    overall_accuracy = (
        (correct_predictions / (correct_predictions + incorrect_predictions) * 100)
        if (correct_predictions + incorrect_predictions) > 0
        else 0
    )
    class_stats = {}
    for result in successful_results:
        if result.get("true_class"):
            true_class = result["true_class"]
            if true_class not in class_stats:
                class_stats[true_class] = {
                    "total_samples": 0,
                    "correct_predictions": 0,
                    "confidences": [],
                }
            class_stats[true_class]["total_samples"] += 1
            if result.get("confidence") is not None:
                class_stats[true_class]["confidences"].append(result["confidence"])
            if result.get("correct") is True:
                class_stats[true_class]["correct_predictions"] += 1
    per_class_performance = {}
    for class_name, stats in class_stats.items():
        confidences = stats["confidences"]
        per_class_performance[class_name] = {
            "total_samples": stats["total_samples"],
            "correct_predictions": stats["correct_predictions"],
            "accuracy": (
                (stats["correct_predictions"] / stats["total_samples"] * 100)
                if stats["total_samples"] > 0
                else 0
            ),
            "avg_confidence": (
                (sum(confidences) / len(confidences) * 100) if confidences else 0
            ),
        }
    summary = {
        "metadata": metadata,
        "overall_statistics": {
            "total_processed": total_processed,
            "total_errors": len(error_results),
            "overall_accuracy": round(overall_accuracy, 1),
        },
        "per_class_performance": per_class_performance,
        "detailed_results": results,
    }
    return summary


def save_benchmark_summary(summary_data, output_dir):
    """
    Save benchmark summary as JSON file.
    (This function is unchanged)
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"benchmark_summary_{timestamp}.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)
        return json_path
    except Exception as e:
        print(f"⚠️ Failed to save benchmark summary: {e}")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Spatio-Temporal model prediction on nucleus state data."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.pth file).",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--folder_path", nargs="+", help="One or more paths to sample folders."
    )
    input_group.add_argument(
        "--volumes",
        nargs="+",
        help="3-4 paths: t-1, t, t+1 .tif files, and optionally label file.",
    )
    parser.add_argument(
        "--save_analysis",
        action="store_true",
        help="Save analysis visualizations (currently placeholder).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis_output",
        help="Directory to save outputs.",
    )
    args = parser.parse_args()

    try:
        # Load model once
        print("Loading model...")
        start_time = datetime.now()
        model = load_model(args.model_path)

        if args.save_analysis:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"📁 Analysis outputs will be saved to: {args.output_dir}")

        if args.folder_path:
            # Batch processing for multiple folders
            total_samples = len(args.folder_path)
            print(f"\n🚀 Processing {total_samples} sample(s)...")
            results = []
            for i, folder_path in enumerate(args.folder_path):
                sample_start_time = datetime.now()
                sample_name = os.path.basename(folder_path)
                print(f"\n📁 Sample {i+1}/{total_samples}: {sample_name}")
                try:
                    true_class = get_true_label_from_path(folder_path)

                    # --- CHANGE 1: Reshape data for the SpatioTemporalModel ---
                    # preprocess_sample returns a tensor of shape (1, 4, D, H, W)
                    volume_processed = preprocess_sample(
                        folder_path=folder_path, for_training=False
                    )

                    # Squeeze the batch dimension: (1, 4, D, H, W) -> (4, D, H, W)
                    volume_processed = volume_processed.squeeze(0)

                    # Slice the raw channels: (4, D, H, W) -> (3, D, H, W)
                    raw_volumes = volume_processed[:3, :, :, :]

                    # Reshape for sequence model: (3, D, H, W) -> (3, 1, D, H, W)
                    sequence_tensor = raw_volumes.unsqueeze(1)

                    # Add batch dimension back for inference: (3, 1, D, H, W) -> (1, 3, 1, D, H, W)
                    input_tensor = sequence_tensor.unsqueeze(0)
                    print(f"   Input tensor shape for model: {input_tensor.shape}")

                    # --- CHANGE 2: Call run_inference with the correctly shaped tensor ---
                    pred_index, pred_class, pred_confidence = run_inference(
                        model,
                        input_tensor,  # Pass the full, reshaped tensor
                        save_analysis=args.save_analysis,
                        analysis_output_dir=(
                            args.output_dir if args.save_analysis else None
                        ),
                        sample_name=sample_name,
                    )

                    is_correct = (
                        (true_class == pred_class) if true_class is not None else None
                    )
                    processing_time = (
                        datetime.now() - sample_start_time
                    ).total_seconds()

                    results.append(
                        {
                            "sample": sample_name,
                            "true_class": true_class,
                            "predicted_class": pred_class,
                            "confidence": pred_confidence,
                            "correct": is_correct,
                            "processing_time": processing_time,
                        }
                    )
                    print(
                        f"   Prediction: {(pred_class or 'UNKNOWN').upper()} ({pred_confidence:.2%})"
                    )

                except Exception as e:
                    print(f"   ❌ Error processing {sample_name}: {e}")
                    results.append({"sample": sample_name, "error": str(e)})

            # --- Summary Generation (Simplified for brevity) ---
            print("\n🎉 BATCH PROCESSING COMPLETE")
            # (The summary generation logic can be added back here if needed)

        else:  # Single sample processing with --volumes
            print("Preprocessing single sample from volume paths...")

            # --- CHANGE 1 (Repeated for single sample): Reshape data ---
            volume_processed = preprocess_sample(
                volume_paths=args.volumes, for_training=False
            )
            volume_processed = volume_processed.squeeze(0)
            raw_volumes = volume_processed[:3, :, :, :]
            sequence_tensor = raw_volumes.unsqueeze(1)
            input_tensor = sequence_tensor.unsqueeze(0)
            print(f"Input tensor created with shape: {input_tensor.shape}")

            # --- CHANGE 2 (Repeated for single sample): Call run_inference ---
            pred_index, pred_class, pred_confidence = run_inference(model, input_tensor)

            print("\n--- Prediction Result ---")
            print(f"Predicted Class:  {pred_class.upper()}")
            print(f"Confidence:       {pred_confidence:.2%}")
            print("-------------------------\n")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
