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
    raw_dataset_path = Path(
        "/mnt/home/dchhantyal/3d-cnn-classification/raw-data/230101_Gata6Nanog_stack_19/stack_19_channel_1_obj_left/"
    )

    # Dataset specific paths
    config.raw_data_path = str(raw_dataset_path / "registered_images")
    config.label_data_path = str(raw_dataset_path / "registered_label_images")
    config.model_path = "/mnt/home/dchhantyal/3d-cnn-classification/model/ncnn4/training_outputs/latest-4ncnn/best_model.pth"

    # Output configuration
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    config.output_dir = f"/mnt/home/dchhantyal/3d-cnn-classification/video_generation/output/stack11_ncnn4_{timestamp}"
    config.video_name = "stack11_video"
    config.cache_dir = f"{config.output_dir}/cache"

    # Model configuration - ncnn4 specific
    config.model_type = "ncnn4"

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

    # Scientific visualization mode for publication quality
    config.scientific_mode = True  # Enable clean raw data visualization
    config.boundary_thickness = 1  # Thinner boundaries for subtle outlines
    config.boundary_opacity = 0.7  # Semi-transparent boundaries
    config.show_raw_background = True  # Show actual microscopy data

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


def main():
    """Main execution function."""
    print("🎬 Stack3 4ncnn Video Generation")
    print("=" * 50)
    # Create configuration
    print("⚙️ Creating configuration...")
    config = create_stack3_config()

    # DEBUG: Print the actual paths being used
    print("🔍 DEBUG: Configuration paths:")
    print(f"   Raw data path: {config.raw_data_path}")
    print(f"   Label data path: {config.label_data_path}")
    print(f"   Model path: {config.model_path}")
    print(f"   Output dir: {config.output_dir}")
    print("=" * 60)

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
