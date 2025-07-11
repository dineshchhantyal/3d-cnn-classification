"""
Configuration System for Video Generation
Handles all visualization options, model settings, and video parameters
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict


@dataclass
class VideoConfig:
    """
    Complete configuration for video generation pipeline.
    All visualization options are toggleable as requested.
    """

    # ===== MODEL SETTINGS =====
    model_type: str = "3ncnn"  # '3ncnn' or '4ncnn'
    model_path: Optional[str] = None  # Custom model path (auto-detect if None)
    batch_size: int = 32  # GPU batch size
    device: str = "auto"  # 'auto', 'cuda', 'cpu', 'cuda:0'

    # ===== DATASET SETTINGS =====
    raw_data_path: str = ""  # Path to raw volume data
    label_data_path: str = ""  # Path to label volume data
    cache_dir: str = "./cache"  # Directory for temporary files
    output_dir: str = "./output"  # Directory for final videos

    # ===== VISUALIZATION OPTIONS (All Toggleable) =====
    # Basic display
    show_classifications: bool = True  # Color-code nuclei by prediction
    show_confidence_scores: bool = False  # Show confidence as text labels
    show_confidence_intensity: bool = False  # Show confidence as color intensity
    show_confidence_bars: bool = False  # Show confidence as bar overlays
    show_bounding_boxes: bool = False  # Draw bounding boxes around nuclei
    show_nucleus_ids: bool = True  # Display nucleus ID numbers
    show_3d_volumes: bool = False  # Show 3D volume renderings
    show_axes: bool = False  # Show plot axes

    # Congested nuclei handling
    smart_label_positioning: bool = (
        True  # Use intelligent label placement to avoid overlaps
    )
    show_leader_lines: bool = (
        True  # Draw lines connecting labels to nuclei when repositioned
    )
    congestion_detection: bool = True  # Detect and highlight congested regions
    auto_zoom_congested: bool = (
        False  # Automatically create zoom insets for congested areas
    )
    label_collision_distance: float = (
        50.0  # Minimum distance between labels to avoid collisions
    )

    # Text and visual enhancement
    label_font_size: float = 1.0  # Font size multiplier for labels
    outline_thickness: int = 3  # Thickness of nucleus outlines
    label_background_alpha: float = 0.8  # Transparency of label backgrounds
    high_contrast_mode: bool = True  # Enhanced contrast for better visibility

    # Frame display options
    show_raw_data: bool = True  # Show raw microscopy data
    show_nucleus_outlines: bool = True  # Show nucleus outlines
    show_nucleus_centers: bool = False  # Show nucleus center points
    show_class_labels: bool = True  # Show classification labels
    show_frame_info: bool = True  # Show frame information overlay
    show_multiple_projections: bool = False  # Show multiple projection views

    # Visual style options
    dark_theme: bool = True  # Use dark background theme
    visualization_style: str = "matplotlib"  # Visualization backend
    projection_axis: str = "z"  # Default projection axis ('x', 'y', 'z')
    output_width: int = 1920  # Output frame width
    output_height: int = 1080  # Output frame height

    # Raw data visualization
    raw_data_enhancement: bool = True  # Enhance raw microscopy data visibility
    raw_data_contrast: float = 2.0  # Contrast multiplier for raw data
    raw_data_brightness: float = 1.2  # Brightness multiplier for raw data
    show_raw_data_colormap: str = (
        "gray"  # Colormap for raw data ('gray', 'viridis', 'plasma')
    )

    # Label visualization
    label_transparency: float = 0.6  # Transparency of label overlays
    label_outline_only: bool = False  # Show only outlines instead of filled regions
    label_colormap: str = "distinct"  # 'distinct', 'viridis', 'tab10'

    # Single nucleus study options
    highlight_nucleus_id: Optional[int] = None  # Highlight specific nucleus
    dim_other_nuclei: bool = False  # Dim non-highlighted nuclei
    highlight_opacity: float = 0.3  # Opacity for dimmed nuclei

    # ===== STANDARD COLORS (As Requested) =====
    classification_colors: Dict[str, str] = None

    # ===== IMAGE PROCESSING =====
    projection_type: str = "mip"  # 'mip', 'mean', 'slice'
    slice_index: Optional[int] = None  # Specific slice for 'slice' mode
    contrast_enhancement: bool = True  # Enhance image contrast
    overlay_opacity: float = 0.7  # Opacity for overlays

    # ===== VIDEO SETTINGS =====
    video_format: str = "mp4"  # 'mp4' or 'gif'
    fps: int = 2  # Frames per second (15min real-life per frame)
    resolution: tuple = (1920, 1080)  # Output video resolution (width, height)
    quality: str = "high"  # 'low', 'medium', 'high'

    # Multi-video generation options
    generate_per_class_videos: bool = (
        True  # Generate separate video for each classification
    )
    generate_combined_video: bool = True  # Generate combined video with all classes
    generate_raw_label_comparison: bool = (
        True  # Generate side-by-side raw vs label videos
    )

    # Enhanced visualization modes
    probability_visualization: bool = (
        True  # Show probability as transparency/brightness
    )
    dual_channel_display: bool = True  # Show raw and label side by side
    overlay_mode: bool = False  # Overlay predictions on raw data (alternative to dual)

    # Probability display settings
    min_confidence_threshold: float = 0.5  # Minimum confidence to display nucleus
    probability_square_size: int = 20  # Size of probability indicator squares
    confidence_brightness_scaling: bool = True  # Scale brightness by confidence
    show_uncertainty_indicators: bool = True  # Show visual uncertainty indicators

    # Video timing and pacing
    pause_on_events: bool = True  # Pause video on important events
    event_pause_duration: float = 2.0  # Seconds to pause on mitotic/death events
    intro_duration: float = 3.0  # Duration of intro frame with dataset info
    outro_duration: float = 2.0  # Duration of outro frame with summary

    # Progress and feedback
    verbose_logging: bool = True  # Detailed progress logging
    show_processing_stats: bool = True  # Show memory/GPU usage during processing
    estimated_time_remaining: bool = True  # Show ETA during processing

    # ===== PERFORMANCE SETTINGS =====
    use_parallel_processing: bool = True  # Enable CPU parallelization
    num_workers: int = 4  # Number of CPU workers
    memory_limit_gb: float = 8.0  # Memory limit for processing

    def __post_init__(self):
        """Initialize default values that depend on other settings."""
        if self.classification_colors is None:
            self.classification_colors = self._get_standard_colors()

    def _get_standard_colors(self) -> Dict[str, str]:
        """Standard color scheme for classifications."""
        return {
            "stable": "#1f77b4",  # Blue
            "mitotic": "#ff7f0e",  # Orange
            "new_daughter": "#2ca02c",  # Green
            "death": "#d62728",  # Red
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VideoConfig":
        """Create config from dictionary with proper default handling."""
        # Create a default instance first to get all default values
        default_config = cls()
        default_dict = asdict(default_config)

        # Update defaults with provided values
        default_dict.update(config_dict)

        return cls(**default_dict)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "VideoConfig":
        """Load configuration from JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, json_path: Union[str, Path]):
        """Save configuration to JSON file."""
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_to_file(self, filepath: Union[str, Path]):
        """Save configuration to file (JSON format)."""
        self.to_json(filepath)

    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

    def validate(self):
        """Validate configuration parameters."""
        errors = []

        # Model validation
        if self.model_type not in ["3ncnn", "4ncnn"]:
            errors.append(f"Invalid model_type: {self.model_type}")

        # Path validation
        if not self.raw_data_path:
            errors.append("raw_data_path is required")
        if not self.label_data_path:
            errors.append("label_data_path is required")

        # Video format validation
        if self.video_format not in ["mp4", "gif"]:
            errors.append(f"Invalid video_format: {self.video_format}")

        # Projection type validation
        if self.projection_type not in ["mip", "mean", "slice"]:
            errors.append(f"Invalid projection_type: {self.projection_type}")

        # Numeric validations
        if self.fps <= 0:
            errors.append("fps must be positive")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if not (0.0 <= self.overlay_opacity <= 1.0):
            errors.append("overlay_opacity must be between 0 and 1")
        if not (0.0 <= self.highlight_opacity <= 1.0):
            errors.append("highlight_opacity must be between 0 and 1")

        if errors:
            raise ValueError(
                "Configuration validation failed:\n"
                + "\n".join(f"- {e}" for e in errors)
            )

        return True


class ConfigPresets:
    """Predefined configuration presets for common use cases."""

    @staticmethod
    def basic_video() -> VideoConfig:
        """Basic video with just classifications."""
        return VideoConfig(
            show_classifications=True,
            show_confidence_scores=False,
            show_bounding_boxes=False,
            video_format="mp4",
            fps=2,
        )

    @staticmethod
    def detailed_analysis() -> VideoConfig:
        """Detailed analysis with all visualization options."""
        return VideoConfig(
            show_classifications=True,
            show_confidence_scores=True,
            show_confidence_intensity=True,
            show_bounding_boxes=True,
            show_nucleus_ids=True,
            video_format="mp4",
            fps=1,
            resolution=(1280, 1280),
        )

    @staticmethod
    def single_nucleus_study(nucleus_id: int) -> VideoConfig:
        """Study a specific nucleus with highlighting."""
        return VideoConfig(
            show_classifications=True,
            show_confidence_scores=True,
            highlight_nucleus_id=nucleus_id,
            dim_other_nuclei=True,
            highlight_opacity=0.2,
            video_format="mp4",
            fps=1,
        )

    @staticmethod
    def high_quality_presentation() -> VideoConfig:
        """High-quality video for presentations."""
        return VideoConfig(
            show_classifications=True,
            show_confidence_scores=False,
            show_bounding_boxes=True,
            contrast_enhancement=True,
            video_format="mp4",
            fps=3,
            resolution=(1920, 1920),
            quality="high",
        )

    @staticmethod
    def quick_preview() -> VideoConfig:
        """Quick, low-quality preview for testing."""
        return VideoConfig(
            show_classifications=True,
            video_format="gif",
            fps=4,
            resolution=(512, 512),
            quality="low",
            batch_size=64,
        )


def create_config_from_args(args) -> VideoConfig:
    """Create configuration from command line arguments."""
    config = VideoConfig()

    # Update config with provided arguments
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    return config


def load_or_create_config(
    config_path: Optional[str] = None, **overrides
) -> VideoConfig:
    """
    Load configuration from file or create default, with optional overrides.

    Args:
        config_path: Path to JSON config file (optional)
        **overrides: Additional parameters to override

    Returns:
        VideoConfig instance
    """
    if config_path and Path(config_path).exists():
        config = VideoConfig.from_json(config_path)
        print(f"📄 Loaded configuration from {config_path}")
    else:
        config = VideoConfig()
        print("🔧 Using default configuration")

    # Apply any overrides
    if overrides:
        config.update(**overrides)
        print(f"🔄 Applied {len(overrides)} configuration overrides")

    # Validate final config
    config.validate()

    return config


if __name__ == "__main__":
    # Test configuration system
    print("Testing VideoConfig...")

    # Test default config
    config = VideoConfig()
    print("✅ Default config created")
    print(f"Default colors: {config.classification_colors}")

    # Test validation
    try:
        config.validate()
        print("✅ Default config validation passed")
    except ValueError as e:
        print(f"❌ Validation failed: {e}")

    # Test presets
    presets = [
        ("Basic Video", ConfigPresets.basic_video()),
        ("Detailed Analysis", ConfigPresets.detailed_analysis()),
        ("Single Nucleus Study", ConfigPresets.single_nucleus_study(123)),
        ("High Quality", ConfigPresets.high_quality_presentation()),
        ("Quick Preview", ConfigPresets.quick_preview()),
    ]

    for name, preset in presets:
        try:
            preset.raw_data_path = "/dummy/path"  # Add required paths for validation
            preset.label_data_path = "/dummy/path"
            preset.validate()
            print(f"✅ {name} preset validated")
        except ValueError as e:
            print(f"❌ {name} preset validation failed: {e}")

    # Test JSON serialization
    try:
        test_config = ConfigPresets.basic_video()
        test_config.raw_data_path = "/test/raw"
        test_config.label_data_path = "/test/labels"

        # Save and load
        test_config.to_json("test_config.json")
        loaded_config = VideoConfig.from_json("test_config.json")

        print("✅ JSON serialization/deserialization works")

        # Cleanup
        Path("test_config.json").unlink()

    except Exception as e:
        print(f"❌ JSON test failed: {e}")

    print("Configuration system testing complete!")
