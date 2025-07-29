# utils.visualization_utils.py
# Utilities for saving preprocessing analysis and model interpretability

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
import os
import json
import torch
import torch.nn.functional as F
from datetime import datetime
import tifffile
from config import HPARAMS, CLASS_NAMES


def create_output_structure(base_output_dir: str, sample_name: str) -> dict:
    """Create organized output directory structure for a sample."""
    sample_dir = os.path.join(base_output_dir, sample_name)

    dirs = {
        "sample": sample_dir,
        "preprocessing": os.path.join(sample_dir, "preprocessing"),
        "model_analysis": os.path.join(sample_dir, "model_analysis"),
        "raw_data": os.path.join(sample_dir, "raw_data"),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def save_volume_statistics(volumes_dict: dict, output_path: str):
    """Save detailed statistics about volumes to JSON."""
    stats = {}

    for name, volume in volumes_dict.items():
        if volume is not None:
            stats[name] = {
                "shape": list(volume.shape),
                "dtype": str(volume.dtype),
                "min": float(np.min(volume)),
                "max": float(np.max(volume)),
                "mean": float(np.mean(volume)),
                "std": float(np.std(volume)),
                "unique_values": int(len(np.unique(volume))),
                "non_zero_voxels": int(np.count_nonzero(volume)),
                "total_voxels": int(volume.size),
            }
        else:
            stats[name] = {"status": "not_available"}

    # Add metadata
    stats["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "target_shape": [
            HPARAMS["input_depth"],
            HPARAMS["input_height"],
            HPARAMS["input_width"],
        ],
        "preprocessing_params": {
            "input_depth": HPARAMS["input_depth"],
            "input_height": HPARAMS["input_height"],
            "input_width": HPARAMS["input_width"],
        },
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)


def save_volume_slices(
    volume: np.ndarray,
    title: str,
    output_path: str,
    cmap: str = "gray",
    is_binary: bool = False,
):
    """Save representative slices from a 3D volume."""
    if volume is None or volume.size == 0:
        return

    # Select slices from different depths
    depth = volume.shape[0]
    slice_indices = [depth // 4, depth // 2, 3 * depth // 4]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title} - Representative Slices", fontsize=14, fontweight="bold")

    for i, slice_idx in enumerate(slice_indices):
        slice_2d = volume[slice_idx]

        if is_binary:
            # For binary masks, use custom colormap
            im = axes[i].imshow(slice_2d, cmap="RdYlBu_r", vmin=0, vmax=1)
        else:
            im = axes[i].imshow(slice_2d, cmap=cmap)

        axes[i].set_title(f"Slice {slice_idx}/{depth}")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_segmentation_overlay(
    raw_volume: np.ndarray, mask_volume: np.ndarray, output_path: str
):
    """Save overlay of segmentation mask on raw volume with 2x4 grid:
    Col 1: Max projection (raw, overlay)
    Col 2-4: Slices at 1/4, 1/2, 3/4 depth (raw, overlay)
    """
    if raw_volume is None or mask_volume is None:
        return

    depth = raw_volume.shape[0]
    slice_indices = [depth // 4, depth // 2, 3 * depth // 4]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Segmentation Mask Overlay Analysis", fontsize=16, fontweight="bold")

    # --- Column 1: Max projection ---
    raw_max = np.max(raw_volume, axis=0)
    mask_max = np.max(mask_volume, axis=0)

    # Row 1, Col 1: Raw max projection
    axes[0, 0].imshow(raw_max, cmap="gray")
    axes[0, 0].set_title("Raw Max Projection")
    axes[0, 0].axis("off")

    # Row 2, Col 1: Overlay max projection
    axes[1, 0].imshow(raw_max, cmap="gray")
    axes[1, 0].imshow(mask_max, cmap="Reds", alpha=0.5)
    axes[1, 0].set_title("Overlay Max Projection")
    axes[1, 0].axis("off")

    # --- Columns 2-4: Slices ---
    for i, slice_idx in enumerate(slice_indices):
        # Row 1: Raw slice
        axes[0, i + 1].imshow(raw_volume[slice_idx], cmap="gray")
        axes[0, i + 1].set_title(f"Raw Slice {slice_idx}")
        axes[0, i + 1].axis("off")

        # Row 2: Overlay slice
        axes[1, i + 1].imshow(raw_volume[slice_idx], cmap="gray", alpha=0.7)
        mask_rgba = np.zeros((*mask_volume[slice_idx].shape, 4))
        mask_rgba[..., 0] = mask_volume[slice_idx]  # Red channel
        mask_rgba[..., 3] = mask_volume[slice_idx] * 0.5  # Alpha channel
        axes[1, i + 1].imshow(mask_rgba)
        axes[1, i + 1].set_title(f"Overlay Slice {slice_idx}")
        axes[1, i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_preprocessing_comparison(
    original_volumes: dict, processed_volumes: dict, output_path: str
):
    """
    Save before/after comparison of preprocessing steps using max intensity projections.

    Args:
        original_volumes (dict): Dictionary with keys 't-1', 't', 't+1', 'mask' mapping to 3D volumes.
        processed_volumes (dict or list): List or dict of processed volumes in order [t-1, t, t+1, mask].
        output_path (str): File path to save the comparison image.
    """
    time_points = ["t-1", "t", "t+1", "mask"]

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(
        "Preprocessing Comparison: Original vs Processed (Max Projection)",
        fontsize=16,
        fontweight="bold",
    )

    for i, tp in enumerate(time_points):
        # --- Row 1: Original ---
        if tp in original_volumes and original_volumes[tp] is not None:
            orig_vol = original_volumes[tp]
            orig_slice = np.max(orig_vol, axis=0)
            im1 = axes[0, i].imshow(orig_slice, cmap="gray")
            axes[0, i].set_title(f"Original {tp}\n{orig_vol.shape}")
            axes[0, i].axis("off")
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        else:
            axes[0, i].text(
                0.5,
                0.5,
                f"{tp}\nNot Available",
                ha="center",
                va="center",
                transform=axes[0, i].transAxes,
            )
            axes[0, i].axis("off")

        # --- Row 2: Processed ---
        if isinstance(processed_volumes, dict):
            proc_vol = processed_volumes.get(tp, None)
        else:
            proc_vol = processed_volumes[i] if i < len(processed_volumes) else None

        if proc_vol is not None:
            proc_slice = np.max(proc_vol, axis=0)
            im2 = axes[1, i].imshow(proc_slice, cmap="gray")
            axes[1, i].set_title(f"Processed {tp}\n{proc_vol.shape}")
            axes[1, i].axis("off")
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
        else:
            axes[1, i].text(
                0.5,
                0.5,
                f"{tp}\nNot Available",
                ha="center",
                va="center",
                transform=axes[1, i].transAxes,
            )
            axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_model_activations(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    output_path: str,
    device: str = "cuda",
):
    """Extract and save activation maps from intermediate layers."""
    model.eval()
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()

        return hook

    # Register hooks for convolutional layers
    hooks = []
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv3d, torch.nn.MaxPool3d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
            layer_names.append(name)

    # Forward pass
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        _ = model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Save activation visualizations
    if activations:
        n_layers = min(len(activations), 4)  # Show max 4 layers
        fig, axes = plt.subplots(2, n_layers, figsize=(20, 8))
        if n_layers == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle(
            "Model Activation Maps (Middle Slices)", fontsize=14, fontweight="bold"
        )

        for i, (layer_name, activation) in enumerate(
            list(activations.items())[:n_layers]
        ):
            # Take middle slice and first channel
            if len(activation.shape) == 5:  # [batch, channels, depth, height, width]
                middle_slice = activation[0, 0, activation.shape[2] // 2].numpy()
                mean_activation = (
                    activation[0].mean(dim=0)[activation.shape[2] // 2].numpy()
                )
            else:
                continue

            # Plot first channel
            im1 = axes[0, i].imshow(middle_slice, cmap="viridis")
            axes[0, i].set_title(f"{layer_name}\nFirst Channel")
            axes[0, i].axis("off")
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

            # Plot mean across channels
            im2 = axes[1, i].imshow(mean_activation, cmap="viridis")
            axes[1, i].set_title(f"{layer_name}\nMean Activation")
            axes[1, i].axis("off")
            plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def save_prediction_analysis(
    predictions: torch.Tensor, pred_class: str, confidence: float, output_path: str
):
    """Save prediction confidence and class probability analysis."""
    probabilities = F.softmax(predictions, dim=1)[0].cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Class probability bar chart
    colors = ["#ff7f7f", "#7fbfff", "#7fff7f"]  # Light red, blue, green
    bars = ax1.bar(CLASS_NAMES, probabilities, color=colors)
    ax1.set_title("Class Probabilities", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(0, 1)

    # Highlight predicted class
    pred_idx = CLASS_NAMES.index(pred_class)
    bars[pred_idx].set_color("#ff4444")  # Bright red for prediction
    bars[pred_idx].set_edgecolor("black")
    bars[pred_idx].set_linewidth(2)

    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{prob:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Confidence gauge
    angles = np.linspace(0, np.pi, 100)
    confidence_angle = confidence * np.pi

    # Create semicircle gauge
    x_gauge = np.cos(angles)
    y_gauge = np.sin(angles)
    ax2.plot(x_gauge, y_gauge, "k-", linewidth=3)

    # Color segments based on confidence levels
    confidence_colors = ["red", "orange", "yellow", "lightgreen", "green"]
    for i, color in enumerate(confidence_colors):
        start_angle = i * np.pi / 5
        end_angle = (i + 1) * np.pi / 5
        angles_segment = np.linspace(start_angle, end_angle, 20)
        x_seg = np.cos(angles_segment)
        y_seg = np.sin(angles_segment)
        ax2.fill_between(x_seg, 0, y_seg, color=color, alpha=0.3)

    # Confidence needle
    needle_x = np.cos(confidence_angle)
    needle_y = np.sin(confidence_angle)
    ax2.arrow(
        0,
        0,
        needle_x * 0.8,
        needle_y * 0.8,
        head_width=0.05,
        head_length=0.05,
        fc="red",
        ec="red",
        linewidth=3,
    )

    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title(
        f"Prediction Confidence: {confidence:.1%}", fontsize=14, fontweight="bold"
    )

    # Add confidence labels
    for i, label in enumerate(["Very Low", "Low", "Medium", "High", "Very High"]):
        angle = (i + 0.5) * np.pi / 5
        x_label = np.cos(angle) * 1.1
        y_label = np.sin(angle) * 1.1
        ax2.text(
            x_label,
            y_label,
            label,
            ha="center",
            va="center",
            fontsize=8,
            rotation=np.degrees(angle) - 90 if angle > np.pi / 2 else np.degrees(angle),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_summary_report(
    dirs: dict, sample_name: str, pred_class: str, confidence: float, stats_file: str
):
    """Generate an HTML summary report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analysis Report - {sample_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; 
                         padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 20px; }}
            .section {{ margin: 30px 0; padding: 20px; background-color: #f9f9f9; border-radius: 8px; }}
            .prediction {{ background-color: #e8f5e8; padding: 20px; border-radius: 8px; 
                          border-left: 5px solid #4CAF50; margin: 20px 0; }}
            .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                          gap: 20px; margin: 20px 0; }}
            .image-container {{ text-align: center; }}
            .image-container img {{ max-width: 100%; height: auto; border-radius: 5px; 
                                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .stats {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 10px; }}
            .timestamp {{ color: #7f8c8d; font-style: italic; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔬 3D CNN Nucleus State Analysis Report</h1>
                <h2>Sample: {sample_name}</h2>
                <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="prediction">
                <h2>🎯 Prediction Results</h2>
                <p><strong>Predicted Class:</strong> <span style="color: #e74c3c; font-size: 1.2em;">{pred_class.upper()}</span></p>
                <p><strong>Confidence:</strong> <span style="color: #27ae60; font-size: 1.2em;">{confidence:.1%}</span></p>
            </div>
            
            <div class="section">
                <h2>📊 Preprocessing Analysis</h2>
                <div class="image-grid">
                    <div class="image-container">
                        <h3>Volume Slices Comparison</h3>
                        <img src="preprocessing/preprocessing_comparison.png" alt="Preprocessing Comparison">
                    </div>
                    <div class="image-container">
                        <h3>Segmentation Overlay</h3>
                        <img src="preprocessing/segmentation_overlay.png" alt="Segmentation Overlay">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🧠 Model Analysis</h2>
                <div class="image-grid">
                    <div class="image-container">
                        <h3>Prediction Analysis</h3>
                        <img src="model_analysis/prediction_analysis.png" alt="Prediction Analysis">
                    </div>
                    <div class="image-container">
                        <h3>Neural Network Activations</h3>
                        <img src="model_analysis/activation_maps.png" alt="Activation Maps">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Volume Statistics</h2>
                <div class="stats">
                    <p>Detailed volume statistics and preprocessing parameters are available in: 
                    <a href="preprocessing/volume_statistics.json">volume_statistics.json</a></p>
                    <p><strong>Target Shape:</strong> {HPARAMS['input_depth']} × {HPARAMS['input_height']} × {HPARAMS['input_width']}</p>
                    <p><strong>Input Channels:</strong> {HPARAMS['num_input_channels']} (t-1, t, t+1, segmentation mask)</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    report_path = os.path.join(dirs["sample"], "analysis_report.html")
    with open(report_path, "w") as f:
        f.write(html_content)

    return report_path
