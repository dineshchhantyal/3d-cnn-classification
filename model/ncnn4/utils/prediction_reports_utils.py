import os
import json
from datetime import datetime
from typing import List, Dict, Any
import argparse


def print_batch_summary(results: List[Dict[str, Any]]):
    """Prints a detailed summary table and accuracy statistics for a batch run."""
    if not results or len(results) <= 1:
        return  # No summary needed for single runs or empty results

    print(f"\n{'='*50}")
    print("🎉 BATCH PROCESSING COMPLETE")
    print(f"{'='*50}")
    # (The rest of the function is identical to the previous version)
    for i, result in enumerate(results):
        if "error" in result:
            print(f"{i+1:2d}. {result['sample'][:35]:35} | ERROR: {result['error']}")
            continue
        true_class = result.get("true_class", "UNKNOWN").upper()
        pred_class = result.get("predicted_class", "UNKNOWN").upper()
        confidence = result.get("confidence", 0)
        correctness = result.get("correct")
        icon = "✅" if correctness is True else "❌" if correctness is False else "❓"
        print(
            f"{i+1:2d}. {result['sample'][:35]:35} | True: {true_class:12} → Pred: {pred_class:12} ({confidence:.1%}) {icon}"
        )
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        print("\nNo valid samples were processed.")
        return
    correct_predictions = sum(1 for r in valid_results if r.get("correct") is True)
    incorrect_predictions = sum(1 for r in valid_results if r.get("correct") is False)
    total_classified = correct_predictions + incorrect_predictions
    print(f"\n{'='*50}\n📊 ACCURACY SUMMARY\n{'='*50}")
    print(f"Total samples processed successfully: {len(valid_results)}")
    if total_classified > 0:
        accuracy = (correct_predictions / total_classified) * 100
        print(f"Correct predictions:     {correct_predictions}")
        print(f"Incorrect predictions:   {incorrect_predictions}")
        print(f"Overall accuracy:        {accuracy:.1f}%")
    class_stats = {}
    for r in valid_results:
        true_class = r.get("true_class")
        if true_class:
            class_stats.setdefault(true_class, {"correct": 0, "total": 0})
            class_stats[true_class]["total"] += 1
            if r.get("correct"):
                class_stats[true_class]["correct"] += 1
    if class_stats:
        print(f"\n📈 PER-CLASS ACCURACY:")
        for class_name in sorted(class_stats.keys()):
            stats = class_stats[class_name]
            class_acc = (
                (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            )
            print(
                f"  {class_name.upper():12}: {stats['correct']:2d}/{stats['total']:2d} ({class_acc:.1f}%)"
            )


def generate_analysis_reports(results: List[Dict[str, Any]], args: argparse.Namespace):
    """Generates individual HTML summary reports if requested."""
    if not args.save_analysis or not results:
        return
    print(f"\n📋 Generating analysis reports...")
    try:
        from utils.visualization_utils import generate_summary_report

        for result in results:
            if "error" in result:
                continue
            sample_name = result["sample"]
            sample_dir = os.path.join(args.output_dir, sample_name)
            stats_path = os.path.join(
                sample_dir, "preprocessing", "volume_statistics.json"
            )
            if os.path.exists(sample_dir) and os.path.exists(stats_path):
                report_path = generate_summary_report(
                    {"sample": sample_dir},
                    sample_name,
                    result["predicted_class"],
                    result["confidence"],
                    stats_path,
                )
                print(f"   📄 Report for {sample_name}: {report_path}")
        print(f"✅ All analysis reports generated successfully!")
        print(f"🌐 Open the HTML reports in your browser for detailed analysis.")
    except Exception as e:
        print(f"⚠️ Warning: Could not generate individual summary reports: {e}")


def generate_benchmark_summary(
    results: List[Dict[str, Any]], start_time: datetime, args: argparse.Namespace
):
    """Generates a comprehensive JSON and HTML benchmark summary for a batch run."""
    if not results or len(results) <= 1:
        return
    print(f"\n📊 Generating benchmark summary...")
    try:
        metadata = {
            "timestamp": start_time.isoformat(),
            "model_path": args.model_path,
            "model_timestamp": (
                os.path.basename(os.path.dirname(args.model_path))
                if "training_outputs" in args.model_path
                else "unknown"
            ),
            "total_samples": len(results),
            "total_processing_time": (datetime.now() - start_time).total_seconds(),
            "output_dir": args.output_dir,
        }
        summary_data = create_benchmark_summary(results, metadata)
        json_path, html_path = save_benchmark_summary(summary_data, args.output_dir)
        if json_path and html_path:
            print(f"   📄 JSON summary: {os.path.basename(json_path)}")
            print(f"   🌐 HTML report: {os.path.basename(html_path)}")
            print(f"✅ Benchmark summary generated successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not generate benchmark summary: {e}")


def create_benchmark_summary(results, metadata):
    """
    Create benchmark summary data structure.

    Args:
        results (list): List of prediction results
        metadata (dict): Run metadata (model_path, timing, etc.)

    Returns:
        dict: Complete summary data structure
    """
    # Filter successful results
    successful_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]

    # Calculate overall statistics
    total_processed = len(successful_results)
    samples_with_known_labels = len(
        [r for r in successful_results if r.get("true_class") is not None]
    )
    samples_with_unknown_labels = total_processed - samples_with_known_labels
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

    # Calculate per-class statistics
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
            confidence = result.get("confidence")
            if confidence is not None:
                class_stats[true_class]["confidences"].append(confidence)
            if result.get("correct") is True:
                class_stats[true_class]["correct_predictions"] += 1

    # Calculate class-level metrics
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
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "confidence_range": (
                [min(confidences), max(confidences)] if confidences else [0, 0]
            ),
        }

    # Confidence analysis
    all_confidences = [
        r.get("confidence", 0)
        for r in successful_results
        if r.get("confidence") is not None
    ]
    overall_avg_confidence = (
        sum(all_confidences) / len(all_confidences) if all_confidences else 0
    )
    high_confidence_samples = len([c for c in all_confidences if c > 0.8])
    medium_confidence_samples = len([c for c in all_confidences if 0.6 <= c <= 0.8])
    low_confidence_samples = len([c for c in all_confidences if c < 0.6])

    # Create summary structure
    summary = {
        "metadata": metadata,
        "overall_statistics": {
            "total_processed": total_processed,
            "total_errors": len(error_results),
            "samples_with_known_labels": samples_with_known_labels,
            "samples_with_unknown_labels": samples_with_unknown_labels,
            "overall_accuracy": round(overall_accuracy, 1),
            "correct_predictions": correct_predictions,
            "incorrect_predictions": incorrect_predictions,
        },
        "per_class_performance": per_class_performance,
        "confidence_analysis": {
            "overall_avg_confidence": round(overall_avg_confidence, 3),
            "high_confidence_samples": high_confidence_samples,
            "medium_confidence_samples": medium_confidence_samples,
            "low_confidence_samples": low_confidence_samples,
        },
        "detailed_results": [
            {
                "sample_name": r["sample"],
                "true_class": r.get("true_class"),
                "predicted_class": r.get("predicted_class"),
                "confidence": r.get("confidence"),
                "correct": r.get("correct"),
                "processing_time": r.get("processing_time"),
                "error": r.get("error"),
            }
            for r in results
        ],
    }

    return summary


def save_benchmark_summary(summary_data, output_dir):
    """
    Save benchmark summary as JSON and HTML files.

    Args:
        summary_data (dict): Summary data from create_benchmark_summary
        output_dir (str): Directory to save files

    Returns:
        tuple: (json_path, html_path) or (None, None) if failed
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON summary
        json_filename = f"benchmark_summary_{timestamp}.json"
        json_path = os.path.join(output_dir, json_filename)

        with open(json_path, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

        return json_path, None

    except Exception as e:
        print(f"⚠️ Failed to save benchmark summary: {e}")
        return None, None
