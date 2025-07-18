# predict.py
# Refactored to use shared modules - eliminates code duplication
# Batch processing for multiple samples with single model loading

import torch
import os
import argparse
from datetime import datetime
import json

# --- Import Shared Modules ---
from config import HPARAMS, CLASS_NAMES, DEVICE
from data_utils import preprocess_sample
from model_utils import load_model, run_inference


def get_true_label_from_path(folder_path):
    """
    Extract the true class label from the directory path.

    Args:
        folder_path (str): Path to the sample folder

    Returns:
        str: True class label, or None if cannot be determined
    """
    # Split path and look for class directory names
    path_parts = folder_path.replace("\\", "/").split("/")

    # Mapping from directory names to model class names
    class_mapping = {
        "death": None,  # Death not in 3-class model
        "mitotic": "mitotic",
        "new_daughter": "new_daughter",
        "stable": "stable",
        "stable2": "stable",  # stable2 maps to stable
    }

    # Look for class directory in path (typically second-to-last or third-to-last)
    for part in reversed(path_parts):
        if part in class_mapping:
            return class_mapping[part]

    # If no known class found, return None
    return None


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

        # Create simple HTML report
        html_content = create_html_report(summary_data)
        html_filename = f"benchmark_report_{timestamp}.html"
        html_path = os.path.join(output_dir, html_filename)

        with open(html_path, "w") as f:
            f.write(html_content)

        return json_path, html_path

    except Exception as e:
        print(f"⚠️ Failed to save benchmark summary: {e}")
        return None, None


def create_html_report(summary_data):
    """
    Create HTML report from summary data.

    Args:
        summary_data (dict): Summary data structure

    Returns:
        str: HTML content
    """
    metadata = summary_data["metadata"]
    overall = summary_data["overall_statistics"]
    per_class = summary_data["per_class_performance"]
    confidence = summary_data["confidence_analysis"]
    results = summary_data["detailed_results"]

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Analysis Report - {metadata.get('model_timestamp', 'Unknown')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .correct {{ color: #27ae60; }}
        .incorrect {{ color: #e74c3c; }}
        .unknown {{ color: #f39c12; }}
        .summary-section {{ margin: 30px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Benchmark Analysis Report</h1>
        
        <div class="summary-section">
            <h2>📊 Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{overall['overall_accuracy']:.1f}%</div>
                    <div class="metric-label">Overall Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{confidence['overall_avg_confidence']:.1f}%</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{overall['total_processed']}</div>
                    <div class="metric-label">Samples Processed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metadata.get('total_processing_time', 0):.1f}s</div>
                    <div class="metric-label">Total Time</div>
                </div>
            </div>
        </div>
        
        <div class="summary-section">
            <h2>🎯 Per-Class Performance</h2>
            <table>
                <thead>
                    <tr><th>Class</th><th>Samples</th><th>Correct</th><th>Accuracy</th><th>Avg Confidence</th></tr>
                </thead>
                <tbody>"""

    for class_name, stats in per_class.items():
        html += f"""
                    <tr>
                        <td>{class_name.upper()}</td>
                        <td>{stats['total_samples']}</td>
                        <td>{stats['correct_predictions']}</td>
                        <td>{stats['accuracy']:.1f}%</td>
                        <td>{stats['avg_confidence']:.1f}%</td>
                    </tr>"""

    html += f"""
                </tbody>
            </table>
        </div>
        
        <div class="summary-section">
            <h2>📋 Detailed Results</h2>
            <table>
                <thead>
                    <tr><th>Sample</th><th>True</th><th>Predicted</th><th>Confidence</th><th>Status</th><th>Time (s)</th></tr>
                </thead>
                <tbody>"""

    for result in results:
        if result.get("error"):
            status_class = "incorrect"
            status_text = "ERROR"
            true_class = "N/A"
            pred_class = "N/A"
            confidence = "N/A"
        else:
            true_class = (result.get("true_class") or "UNKNOWN").upper()
            pred_class = (result.get("predicted_class") or "UNKNOWN").upper()
            confidence = f"{result.get('confidence', 0):.1%}"

            if result.get("correct") is True:
                status_class = "correct"
                status_text = "✅"
            elif result.get("correct") is False:
                status_class = "incorrect"
                status_text = "❌"
            else:
                status_class = "unknown"
                status_text = "❓"

        processing_time = result.get("processing_time", 0)

        html += f"""
                    <tr>
                        <td>{result['sample_name'][:40]}</td>
                        <td>{true_class}</td>
                        <td>{pred_class}</td>
                        <td>{confidence}</td>
                        <td class="{status_class}">{status_text}</td>
                        <td>{processing_time:.1f}</td>
                    </tr>"""

    html += f"""
                </tbody>
            </table>
        </div>
        
        <div class="summary-section">
            <h2>ℹ️ Run Information</h2>
            <p><strong>Model:</strong> {metadata.get('model_path', 'Unknown')}</p>
            <p><strong>Timestamp:</strong> {metadata.get('timestamp', 'Unknown')}</p>
            <p><strong>Analysis Level:</strong> {metadata.get('analysis_level', 'Unknown')}</p>
        </div>
    </div>
</body>
</html>"""

    return html


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run 3D CNN prediction on nucleus state data."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.pth file).",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--folder_path",
        nargs="+",
        help="One or more paths to sample folders with t-1, t, t+1 subdirs.",
    )
    input_group.add_argument(
        "--volumes",
        nargs="+",
        help="3-4 space-separated paths: t-1, t, t+1 .tif files, and optionally label file.",
    )

    # Analysis arguments
    parser.add_argument(
        "--save_analysis",
        action="store_true",
        help="Save preprocessing and model analysis visualizations.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis_output",
        help="Directory to save analysis outputs (default: ./analysis_output).",
    )
    parser.add_argument(
        "--analysis_level",
        choices=["basic", "detailed", "full"],
        default="detailed",
        help="Level of analysis detail to save (default: detailed).",
    )

    args = parser.parse_args()

    try:
        # Validate input arguments
        if args.volumes:
            if len(args.volumes) < 3:
                raise ValueError("--volumes requires at least 3 paths (t-1, t, t+1)")
            elif len(args.volumes) > 4:
                raise ValueError(
                    "--volumes accepts maximum 4 paths (t-1, t, t+1, label)"
                )

            # Process single sample with volume paths
            folder_paths = None
            volume_paths = args.volumes
        else:
            # Process one or more folder paths
            folder_paths = args.folder_path
            volume_paths = None

        # Load model once
        print("Loading model...")
        start_time = datetime.now()
        model = load_model(args.model_path)

        # Setup analysis output directory if needed
        if args.save_analysis:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"📁 Analysis outputs will be saved to: {args.output_dir}")

        if folder_paths:
            # Batch processing for multiple folders
            total_samples = len(folder_paths)
            print(f"\n🚀 Processing {total_samples} sample(s)...")
            print("=" * 50)

            results = []
            for i, folder_path in enumerate(folder_paths):
                sample_start_time = datetime.now()
                sample_name = os.path.basename(folder_path)
                print(f"\n📁 Sample {i+1}/{total_samples}: {sample_name}")

                try:
                    # Extract true label from directory path
                    true_class = get_true_label_from_path(folder_path)

                    # Preprocess single sample using shared function
                    input_tensor = preprocess_sample(
                        folder_path=folder_path,
                        for_training=False,
                        save_analysis=args.save_analysis,
                        analysis_output_dir=(
                            args.output_dir if args.save_analysis else None
                        ),
                    )
                    print(f"   Input tensor shape: {input_tensor.shape}")

                    # Run prediction using shared function
                    pred_index, pred_class, pred_confidence = run_inference(
                        model,
                        input_tensor[:, : HPARAMS["num_input_channels"]],
                        save_analysis=args.save_analysis,
                        analysis_output_dir=(
                            args.output_dir if args.save_analysis else None
                        ),
                        sample_name=sample_name,
                    )

                    # Determine if prediction is correct
                    is_correct = (
                        (true_class == pred_class) if true_class is not None else None
                    )
                    correctness_icon = (
                        "✅"
                        if is_correct is True
                        else "❌" if is_correct is False else "❓"
                    )

                    # Calculate processing time
                    processing_time = (
                        datetime.now() - sample_start_time
                    ).total_seconds()

                    # Store result
                    result = {
                        "sample": sample_name,
                        "true_class": true_class,
                        "predicted_class": pred_class,
                        "index": pred_index,
                        "confidence": pred_confidence,
                        "correct": is_correct,
                        "processing_time": processing_time,
                    }
                    results.append(result)

                    # Display result with true vs predicted
                    if true_class is not None:
                        print(
                            f"   True: {true_class.upper()} → Predicted: {(pred_class or 'UNKNOWN').upper()} ({pred_confidence:.2%}) {correctness_icon}"
                        )
                    else:
                        print(
                            f"   True: UNKNOWN → Predicted: {(pred_class or 'UNKNOWN').upper()} ({pred_confidence:.2%}) ❓"
                        )

                except Exception as e:
                    processing_time = (
                        datetime.now() - sample_start_time
                    ).total_seconds()
                    print(f"   ❌ Error processing {sample_name}: {e}")
                    results.append(
                        {
                            "sample": sample_name,
                            "error": str(e),
                            "processing_time": processing_time,
                        }
                    )

            # Summary with optional analysis report generation
            print(f"\n{'='*50}")
            print("🎉 BATCH PROCESSING COMPLETE")
            print(f"{'='*50}")

            # Calculate accuracy statistics
            total_samples = len([r for r in results if "error" not in r])
            correct_predictions = len(
                [r for r in results if "error" not in r and r.get("correct") is True]
            )
            incorrect_predictions = len(
                [r for r in results if "error" not in r and r.get("correct") is False]
            )
            unknown_labels = len(
                [r for r in results if "error" not in r and r.get("correct") is None]
            )

            # Display detailed results
            for i, result in enumerate(results):
                if "error" not in result:
                    true_class = result.get("true_class") or "UNKNOWN"
                    pred_class = result.get("predicted_class") or "UNKNOWN"
                    confidence = result.get("confidence", 0)
                    correctness = result.get("correct")

                    if correctness is True:
                        icon = "✅"
                    elif correctness is False:
                        icon = "❌"
                    else:
                        icon = "❓"

                    print(
                        f"{i+1:2d}. {result['sample'][:35]:35} | True: {true_class.upper():12} → Pred: {pred_class.upper():12} ({confidence:.1%}) {icon}"
                    )
                else:
                    print(
                        f"{i+1:2d}. {result['sample'][:35]:35} | ERROR: {result['error']}"
                    )

            # Display accuracy summary
            if total_samples > 0:
                print(f"\n{'='*50}")
                print("📊 ACCURACY SUMMARY")
                print(f"{'='*50}")
                print(f"Total samples processed: {total_samples}")
                if correct_predictions + incorrect_predictions > 0:
                    accuracy = (
                        correct_predictions
                        / (correct_predictions + incorrect_predictions)
                        * 100
                    )
                    print(f"Correct predictions:     {correct_predictions}")
                    print(f"Incorrect predictions:   {incorrect_predictions}")
                    print(f"Overall accuracy:        {accuracy:.1f}%")
                if unknown_labels > 0:
                    print(f"Unknown true labels:     {unknown_labels}")

                # Per-class accuracy breakdown
                class_stats = {}
                for result in results:
                    if "error" not in result and result.get("true_class"):
                        true_class = result["true_class"]
                        if true_class and true_class not in class_stats:
                            class_stats[true_class] = {"correct": 0, "total": 0}
                        if true_class:
                            class_stats[true_class]["total"] += 1
                            if result.get("correct") is True:
                                class_stats[true_class]["correct"] += 1

                if class_stats:
                    print(f"\n📈 PER-CLASS ACCURACY:")
                    for class_name in sorted(class_stats.keys()):
                        if class_name:  # Additional safety check
                            stats = class_stats[class_name]
                            class_accuracy = (
                                stats["correct"] / stats["total"] * 100
                                if stats["total"] > 0
                                else 0
                            )
                            print(
                                f"  {class_name.upper():12}: {stats['correct']:2d}/{stats['total']:2d} ({class_accuracy:.1f}%)"
                            )

            # Generate summary reports if analysis was saved
            if args.save_analysis:
                print(f"\n📋 Generating summary reports...")
                try:
                    from visualization_utils import generate_summary_report

                    for result in results:
                        if "error" not in result:
                            sample_name = result["sample"]
                            dirs = {
                                "sample": os.path.join(args.output_dir, sample_name)
                            }

                            if os.path.exists(dirs["sample"]):
                                report_path = generate_summary_report(
                                    dirs,
                                    sample_name,
                                    result["predicted_class"],
                                    result["confidence"],
                                    os.path.join(
                                        dirs["sample"],
                                        "preprocessing",
                                        "volume_statistics.json",
                                    ),
                                )
                                print(f"   📄 Report for {sample_name}: {report_path}")

                    print(f"✅ All analysis reports generated successfully!")
                    print(
                        f"🌐 Open the HTML reports in your browser for detailed analysis."
                    )

                except Exception as e:
                    print(f"⚠️ Warning: Could not generate summary reports: {e}")

            # Generate comprehensive benchmark summary
            if len(results) > 1:  # Only for batch processing
                try:
                    print(f"\n📊 Generating benchmark summary...")

                    # Create metadata
                    end_time = datetime.now()
                    total_processing_time = (end_time - start_time).total_seconds()

                    metadata = {
                        "timestamp": start_time.isoformat(),
                        "model_path": args.model_path,
                        "model_timestamp": (
                            os.path.basename(os.path.dirname(args.model_path))
                            if "training_outputs" in args.model_path
                            else "unknown"
                        ),
                        "total_samples": len(results),
                        "total_processing_time": total_processing_time,
                        "analysis_level": args.analysis_level,
                        "output_dir": args.output_dir,
                    }

                    # Generate summary
                    summary_data = create_benchmark_summary(results, metadata)
                    json_path, html_path = save_benchmark_summary(
                        summary_data, args.output_dir
                    )

                    if json_path and html_path:
                        print(f"   📄 JSON summary: {os.path.basename(json_path)}")
                        print(f"   🌐 HTML report: {os.path.basename(html_path)}")
                        print(f"✅ Benchmark summary generated successfully!")

                except Exception as e:
                    print(f"⚠️ Warning: Could not generate benchmark summary: {e}")

        else:
            # Single sample processing with volume paths
            print("Preprocessing input data...")

            # Determine sample name for single sample
            if args.volumes:
                sample_name = (
                    f"single_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

            input_tensor = preprocess_sample(
                volume_paths=volume_paths,
                for_training=False,
                save_analysis=args.save_analysis,
                analysis_output_dir=args.output_dir if args.save_analysis else None,
            )
            print(f"Input tensor created with shape: {input_tensor.shape}")

            pred_index, pred_class, pred_confidence = run_inference(
                model,
                input_tensor,
                save_analysis=args.save_analysis,
                analysis_output_dir=args.output_dir if args.save_analysis else None,
                sample_name=sample_name if args.save_analysis else None,
            )

            print("\n--- Prediction Result ---")
            print(f"Predicted Class Index: {pred_index}")
            print(f"Predicted Class Name:  {pred_class.upper()}")
            print(f"Confidence:            {pred_confidence:.2%}")
            print("-------------------------\n")

            # Generate summary report for single sample
            if args.save_analysis:
                try:
                    from visualization_utils import generate_summary_report

                    dirs = {"sample": os.path.join(args.output_dir, sample_name)}
                    if os.path.exists(dirs["sample"]):
                        report_path = generate_summary_report(
                            dirs,
                            sample_name,
                            pred_class,
                            pred_confidence,
                            os.path.join(
                                dirs["sample"],
                                "preprocessing",
                                "volume_statistics.json",
                            ),
                        )
                        print(f"📄 Analysis report generated: {report_path}")
                        print(
                            f"🌐 Open the HTML report in your browser for detailed analysis."
                        )

                except Exception as e:
                    print(f"⚠️ Warning: Could not generate summary report: {e}")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
