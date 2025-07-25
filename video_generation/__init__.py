"""
3D CNN Video Generation System

A comprehensive pipeline for generating videos from 3D CNN nucleus classification predictions.
Supports both 3ncnn and 4ncnn models with GPU acceleration and parallel processing.

Main Components:
- VideoConfig: Configuration management
- ModelInferenceEngine: Model loading and inference
- SlidingWindowProcessor: Efficient data processing
- FrameRenderer: Visualization and rendering
- VideoGenerator: Complete pipeline orchestration

Quick Usage:
    from video_generation import VideoGenerator

    video_path = VideoGenerator.quick_video(
        raw_data_path="/path/to/raw",
        label_data_path="/path/to/labels",
        model_path="/path/to/model",
        output_dir="./output"
    )

Custom Usage:
    from video_generation import VideoConfig, VideoGenerator

    config = VideoConfig()
    config.raw_data_path = "/path/to/raw"
    config.model_path = "/path/to/model"
    config.use_gpu = True
    config.show_multiple_projections = True

    generator = VideoGenerator(config)
    video_path = generator.generate_video()
"""

# Core components
from .config import VideoConfig, ConfigPresets
from .model_interface import ModelInferenceEngine
from .data_pipeline import SlidingWindowProcessor, FrameData
from .renderer import FrameRenderer, NucleusVisualization
from .video_generator import VideoGenerator

# Convenience imports
__all__ = [
    # Main classes
    "VideoGenerator",
    "VideoConfig",
    "ConfigPresets",
    "ModelInferenceEngine",
    "SlidingWindowProcessor",
    "FrameRenderer",
    # Data structures
    "FrameData",
    "NucleusVisualization",
    # Version info
    "__version__",
]

# Package-level configuration
import warnings
import torch


def _check_dependencies():
    """Check if required dependencies are available."""
    try:
        import cv2
        import matplotlib
        import tifffile
        import numpy as np
        from PIL import Image
    except ImportError as e:
        warnings.warn(f"Missing required dependency: {e}")
        return False
    return True


def _check_gpu_availability():
    """Check GPU availability and provide helpful info."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU acceleration available: {gpu_count} device(s)")
        print(f"   Primary GPU: {gpu_name}")
        return True
    else:
        print("⚠️ GPU not available, using CPU mode")
        return False


# Initialize package
_check_dependencies()
_gpu_available = _check_gpu_availability()


# Default configuration hints
def get_recommended_config():
    """Get recommended configuration based on system capabilities."""
    if _gpu_available:
        return ConfigPresets.create_high_quality()
    else:
        config = ConfigPresets.create_quick_preview()
        config.use_gpu = False
        config.batch_size = 4  # Smaller batches for CPU
        return config


# Helper function for first-time users
def quick_start_guide():
    """Print quick start guide."""
    print(
        """
    🎬 3D CNN Video Generation - Quick Start Guide
    
    1. Basic Usage:
       from video_generation import VideoGenerator
       
       video_path = VideoGenerator.quick_video(
           raw_data_path="/path/to/raw/data",
           label_data_path="/path/to/labels",
           model_path="/path/to/model", 
           output_dir="./output"
       )
    
    2. Run Examples:
       python examples.py quick          # Quick video demo
       python examples.py test_model     # Test model loading
       python examples.py presets        # View configuration options
    
    3. Command Line:
       python video_generator.py --raw-data /path/to/raw \\
                                 --label-data /path/to/labels \\
                                 --model /path/to/model \\
                                 --output ./output
    
    4. Need Help?
       - Check README.md for detailed documentation
       - Run examples.py for working demonstrations
       - Use ConfigPresets for common configurations
    """
    )


# Version check and compatibility warnings
import sys

if sys.version_info < (3, 8):
    warnings.warn("Python 3.8+ recommended for best performance")

# Optional: Auto-configure based on environment
try:
    # Try to detect if we're in a notebook environment
    get_ipython()
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False

if IN_NOTEBOOK:
    # Notebook-specific optimizations
    import matplotlib

    matplotlib.use("inline")
    print("📓 Jupyter notebook environment detected")
    print("   Use %matplotlib inline for best visualization")
