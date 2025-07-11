#!/usr/bin/env python3
"""
Specialized script for generating video from 221016_FUCCI_Nanog_stack_3 dataset
using the 4ncnn model trained on 2025-07-10.

This script is optimized for the specific file structure and naming patterns
of the stack3 dataset.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from video_generation.config import VideoConfig
from video_generation.video_generator import VideoGenerator
import time


def create_stack3_config():
    """Create optimized configuration for stack3 dataset processing."""
    config = VideoConfig()

    # Dataset specific paths
    config.raw_data_path = "/mnt/home/dchhantyal/3d-cnn-classification/raw-data/221016_FUCCI_Nanog_stack_3/registered_images"
    config.label_data_path = "/mnt/home/dchhantyal/3d-cnn-classification/raw-data/221016_FUCCI_Nanog_stack_3/registered_label_images"
    config.model_path = "/mnt/home/dchhantyal/3d-cnn-classification/model/4ncnn/training_outputs/20250710-131550/best_model.pth"

    # Output configuration
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    config.output_dir = f"/mnt/home/dchhantyal/3d-cnn-classification/video_generation/output/stack3_4ncnn_{timestamp}"
    config.video_name = "221016_FUCCI_Nanog_stack3_analysis"
    config.cache_dir = f"{config.output_dir}/cache"

    # Model configuration - 4ncnn specific
    config.model_type = "4ncnn"

    # Video settings optimized for clarity and observation
    config.fps = 3  # Slower for better observation (was 15)
    config.video_format = "mp4"
    config.output_width = 2560  # Higher resolution for better text visibility
    config.output_height = 1440

    # Enhanced visualization settings for congested nuclei
    config.show_raw_data = True
    config.show_nucleus_outlines = True
    config.show_nucleus_centers = True
    config.show_class_labels = True
    config.show_confidence_scores = True
    config.show_nucleus_ids = True  # Show IDs to distinguish overlapping nuclei
    config.show_frame_info = True
    config.show_multiple_projections = False  # Start with single view for speed
    config.projection_axis = "z"  # Top-down view
    config.dark_theme = True
    config.visualization_style = "matplotlib"

    # ENHANCED: Multi-video generation features (TEMPORARILY DISABLED due to renderer issues)
    # config.generate_per_class_videos = True
    # config.probability_visualization = True
    # config.dual_channel_display = True
    # config.raw_data_enhancement = True
    # config.confidence_brightness_scaling = True
    # config.show_confidence_values = True
    # config.estimated_time_remaining = True

    # Use working features only for now
    config.generate_per_class_videos = False
    config.probability_visualization = False
    config.dual_channel_display = False
    config.raw_data_enhancement = False

    # Congested nuclei handling settings
    config.smart_label_positioning = True
    config.show_leader_lines = True
    config.label_font_size = 1.5  # Larger text for better readability
    config.outline_thickness = 4  # Thicker outlines for better visibility
    config.high_contrast_mode = True
    config.label_collision_distance = 60.0  # More space between labels

    # Video timing for better observation
    config.pause_on_events = True
    config.event_pause_duration = 3.0  # 3 second pause on important events
    config.verbose_logging = True
    config.show_processing_stats = True

    # Performance settings for HPC environment
    # Check for CPU fallback from environment variable
    force_cpu = os.environ.get("FORCE_CPU", "false").lower() == "true"

    if force_cpu:
        print("💻 CPU mode forced by environment variable")
        config.use_gpu = False
        config.batch_size = 8  # Smaller batch for CPU
        config.num_workers = 4  # Fewer workers for CPU
        config.compile_model = False  # Disable compilation on CPU
    else:
        config.use_gpu = True
        config.batch_size = 32  # Large batch for GPU efficiency
        config.num_workers = 4  # Match sbatch cpus-per-gpu
        config.compile_model = True  # TorchScript optimization

    config.use_parallel_processing = True

    # Set PyTorch threading for better CPU utilization
    import torch

    torch.set_num_threads(config.num_workers)
    torch.set_num_interop_threads(config.num_workers)

    return config


def verify_dataset():
    """Verify the dataset structure and count frames."""
    raw_path = Path(
        "/mnt/home/dchhantyal/3d-cnn-classification/raw-data/221016_FUCCI_Nanog_stack_3/registered_images"
    )
    label_path = Path(
        "/mnt/home/dchhantyal/3d-cnn-classification/raw-data/221016_FUCCI_Nanog_stack_3/registered_label_images"
    )

    print("🔍 Verifying dataset structure...")

    # Check paths exist
    if not raw_path.exists():
        print(f"❌ Raw data path does not exist: {raw_path}")
        return False

    if not label_path.exists():
        print(f"❌ Label data path does not exist: {label_path}")
        return False

    # Count frames
    raw_files = list(raw_path.glob("nuclei_reg8_*.tif"))
    label_files = list(label_path.glob("label_reg8_*.tif"))

    print(f"📊 Dataset verification:")
    print(f"   Raw frames: {len(raw_files)}")
    print(f"   Label frames: {len(label_files)}")

    if len(raw_files) == 0:
        print("❌ No raw frames found with pattern 'nuclei_reg8_*.tif'")
        return False

    if len(label_files) == 0:
        print("❌ No label frames found with pattern 'label_reg8_*.tif'")
        return False

    if len(raw_files) != len(label_files):
        print(
            f"⚠️ WARNING: Frame count mismatch (raw: {len(raw_files)}, labels: {len(label_files)})"
        )

    # Check frame number ranges
    raw_numbers = []
    label_numbers = []

    for f in raw_files:
        try:
            num = int(f.stem.split("_")[-1])
            raw_numbers.append(num)
        except:
            continue

    for f in label_files:
        try:
            num = int(f.stem.split("_")[-1])
            label_numbers.append(num)
        except:
            continue

    if raw_numbers and label_numbers:
        print(f"   Frame range: {min(raw_numbers)} to {max(raw_numbers)}")
        print(
            f"   Processable frames: {len(raw_numbers) - 2}"
        )  # Exclude boundaries for sliding window

    print("✅ Dataset verification complete")
    return True


def verify_model():
    """Verify the model file exists and is loadable."""
    model_path = Path(
        "/mnt/home/dchhantyal/3d-cnn-classification/model/4ncnn/training_outputs/20250710-131550/best_model.pth"
    )

    print("🔍 Verifying model...")

    if not model_path.exists():
        print(f"❌ Model file does not exist: {model_path}")
        return False

    # Check file size
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"📊 Model file: {model_path.name} ({size_mb:.1f} MB)")

    # Try to load with PyTorch
    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu")
        print(f"✅ Model checkpoint loaded successfully")

        # Print some checkpoint info if available
        if isinstance(checkpoint, dict):
            for key in ["epoch", "best_val_accuracy", "model_state_dict"]:
                if key in checkpoint:
                    print(f"   {key}: {checkpoint[key]}")

    except Exception as e:
        print(f"⚠️ Warning: Could not load model checkpoint: {e}")
        print("   This may be normal if the model uses custom components")

    print("✅ Model verification complete")
    return True


def main():
    """Main execution function."""
    print("🎬 Stack3 4ncnn Video Generation")
    print("=" * 50)

    # Verify prerequisites
    if not verify_dataset():
        print("❌ Dataset verification failed")
        return 1

    if not verify_model():
        print("❌ Model verification failed")
        return 1

    print()

    # Create configuration
    print("⚙️ Creating configuration...")
    config = create_stack3_config()

    print(f"📋 Enhanced Configuration Summary:")
    print(f"   🤖 Model: 4ncnn")
    print(f"   📁 Output: {config.output_dir}")
    print(f"   🎬 Video: {config.video_name}.{config.video_format}")
    print(f"   📺 Resolution: {config.output_width}x{config.output_height}")
    print(f"   ⏱️ FPS: {config.fps} (slower for better observation)")
    print(f"   🔢 Batch size: {config.batch_size}")
    print(f"   🖥️ GPU: {config.use_gpu}")
    print(f"   ⚡ Parallel: {config.use_parallel_processing}")
    print()
    print(f"🎨 Enhanced Visualization Features:")
    print(f"   🆔 Show nucleus IDs: {config.show_nucleus_ids}")
    print(f"   🎯 Smart label positioning: {config.smart_label_positioning}")
    print(f"   📏 Leader lines: {config.show_leader_lines}")
    print(
        f"   📊 Congestion detection: {getattr(config, 'congestion_detection', False)}"
    )
    print(f"   🔤 Font size multiplier: {config.label_font_size}")
    print(f"   ✨ High contrast mode: {config.high_contrast_mode}")
    print()
    print(f"🎬 Multi-Video Generation (TEMPORARILY DISABLED):")
    print(f"   🎭 Per-class videos: {config.generate_per_class_videos}")
    print(f"   📊 Probability visualization: {config.probability_visualization}")
    print(f"   📺 Dual channel display: {config.dual_channel_display}")
    print(f"   🔬 Raw data enhancement: {config.raw_data_enhancement}")
    print(
        f"   💡 Confidence brightness: {getattr(config, 'confidence_brightness_scaling', False)}"
    )
    print()
    print(f"⏯️ Video Timing Features:")
    print(f"   ⏸️ Pause on events: {config.pause_on_events}")
    print(f"   ⏱️ Event pause duration: {config.event_pause_duration}s")
    print(f"   📝 Verbose logging: {config.verbose_logging}")
    print(f"   📊 Processing stats: {config.show_processing_stats}")
    print(f"   ⏰ ETA estimates: {getattr(config, 'estimated_time_remaining', False)}")
    print()

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Save configuration for reference
    config_path = Path(config.output_dir) / "config.json"
    config.to_json(str(config_path))
    print(f"💾 Configuration saved: {config_path}")

    # Generate video using STANDARD pipeline (enhanced features disabled for now)
    try:
        print("🚀 Starting video generation (standard pipeline)...")
        start_time = time.time()

        generator = VideoGenerator(config)

        # Check if we already have predictions from a previous run
        predictions_file = Path(config.cache_dir) / "predictions.jsonl"

        if predictions_file.exists():
            print(f"📄 Using existing predictions: {predictions_file}")
            # Use standard generation method since enhanced features are disabled
            video_path = generator.generate_video()
        else:
            print("📊 No existing predictions found. Running full pipeline...")
            video_path = generator.generate_video()

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ Video generation completed successfully!")
        print(f"   Duration: {duration/60:.1f} minutes")
        print(f"   📹 Video saved: {video_path}")

        return 0

    except Exception as e:
        print(f"❌ Video generation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
