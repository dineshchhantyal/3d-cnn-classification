"""
Visualization and Rendering System
Creates beautiful frames for video generation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont
import time
import tifffile
from dataclasses import dataclass

from .config import VideoConfig


@dataclass
class NucleusVisualization:
    """Container for nucleus visualization data."""

    nucleus_id: int
    position: Tuple[int, int, int]  # (z, y, x) coordinates
    class_name: str
    confidence: float
    color: Tuple[int, int, int]  # RGB color
    radius: int
    volume_data: Optional[np.ndarray] = None


class FrameRenderer:
    """
    High-quality frame rendering for video generation.
    Supports multiple visualization modes and styling options.
    """

    def __init__(self, config: VideoConfig):
        """
        Initialize the frame renderer.

        Args:
            config: Video generation configuration
        """
        self.config = config

        # Set up visualization parameters
        self.setup_visual_parameters()

        # Create output directory
        self.frames_dir = Path(config.output_dir) / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎨 FrameRenderer initialized:")
        print(
            f"   Output size: {getattr(config, 'output_width', 1920)}x{getattr(config, 'output_height', 1080)}"
        )
        print(f"   Frames directory: {self.frames_dir}")
        print(f"   Style: {getattr(config, 'visualization_style', 'matplotlib')}")

    def setup_visual_parameters(self):
        """Set up colors, fonts, and visual styling."""
        # Define class colors (adjust based on your classes)
        self.class_colors = {
            "stable": (0, 255, 0),  # Green
            "mitotic": (255, 165, 0),  # Orange
            "death": (255, 0, 0),  # Red
            "division": (255, 255, 0),  # Yellow
            "unknown": (128, 128, 128),  # Gray
        }

        # Enhanced color scheme for new rendering modes
        self.colors = {
            "background": (25, 25, 25),  # Dark background
            "nucleus": (100, 255, 100),  # Bright green for default
            "text": (255, 255, 255),  # White text
            "label_bg": (50, 50, 50),  # Dark gray for label backgrounds
            "congested_bg": (80, 40, 40),  # Darker red for congested areas
            "legend_bg": (40, 40, 40),  # Background for legends
            "divider": (128, 128, 128),  # Gray divider line
            # Class-specific colors for better distinction
            "class_stable": (100, 255, 100),  # Green - healthy/stable
            "class_mitotic": (255, 255, 100),  # Yellow - active division
            "class_new_daughter": (100, 200, 255),  # Light blue - newly formed
            "class_death": (255, 100, 100),  # Red - cell death
        }

        # Set up fonts for PIL-based rendering
        try:
            # Try to load a nice font
            self.font = ImageFont.truetype("arial.ttf", 16)
            self.small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            # Fallback to default font
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

        # Matplotlib style settings
        plt.style.use(
            "dark_background" if getattr(self.config, "dark_theme", True) else "default"
        )
        self.font_size = max(8, getattr(self.config, "output_width", 1920) // 100)

        # Figure setup for consistent sizing
        self.fig_dpi = 100
        self.fig_width = getattr(self.config, "output_width", 1920) / self.fig_dpi
        self.fig_height = getattr(self.config, "output_height", 1080) / self.fig_dpi

    def load_frame_data(self, timestamp: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load raw and label volumes for a specific frame.

        Args:
            timestamp: Frame timestamp

        Returns:
            Tuple of (raw_volume, label_volume)
        """
        # Construct file paths based on your data structure
        raw_path = Path(self.config.raw_data_path) / f"nuclei_reg8_{timestamp}.tif"
        label_path = Path(self.config.label_data_path) / f"label_reg8_{timestamp}.tif"

        # Pattern 2: frame_XXX.tif
        if not raw_path.exists():
            raw_path = Path(self.config.raw_data_path) / f"frame_{timestamp:03d}.tif"
        if not label_path.exists():
            label_path = (
                Path(self.config.label_data_path) / f"frame_{timestamp:03d}.tif"
            )

        # Pattern 3: XXX.tif
        if not raw_path.exists():
            raw_path = Path(self.config.raw_data_path) / f"{timestamp:03d}.tif"
        if not label_path.exists():
            label_path = Path(self.config.label_data_path) / f"{timestamp:03d}.tif"

        try:
            raw_volume = tifffile.imread(str(raw_path)).astype(np.float32)
            label_volume = tifffile.imread(str(label_path)).astype(np.int32)
            return raw_volume, label_volume
        except Exception as e:
            print(f"⚠️ Error loading frame {timestamp}: {e}")
            # Return dummy data
            shape = (64, 64, 64)
            return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.int32)

    def extract_nucleus_visualizations(
        self, timestamp: int, predictions: Dict[str, Dict]
    ) -> List[NucleusVisualization]:
        """
        Extract visualization data for all nuclei in a frame.

        Args:
            timestamp: Frame timestamp
            predictions: Prediction results for this frame

        Returns:
            List of nucleus visualization objects
        """
        # Load frame data
        raw_volume, label_volume = self.load_frame_data(timestamp)

        visualizations = []

        for nucleus_id_str, pred_data in predictions.items():
            nucleus_id = int(nucleus_id_str)

            # Find nucleus in label volume
            mask = label_volume == nucleus_id
            if not np.any(mask):
                continue

            # Calculate center position
            coords = np.where(mask)
            center_z = int(np.mean(coords[0]))
            center_y = int(np.mean(coords[1]))
            center_x = int(np.mean(coords[2]))

            # Get class information
            class_name = pred_data["class_name"]
            confidence = pred_data["confidence"]

            # Determine color and radius based on confidence
            base_color = self.class_colors.get(class_name, self.class_colors["unknown"])
            confidence_factor = max(0.3, confidence)  # Minimum 30% opacity
            color = tuple(int(c * confidence_factor) for c in base_color)
            radius = max(3, int(10 * confidence))

            # Extract volume data if needed
            volume_data = None
            if getattr(self.config, "show_3d_volumes", False):
                bbox = self._get_nucleus_bbox(mask)
                volume_data = raw_volume[bbox]

            viz = NucleusVisualization(
                nucleus_id=nucleus_id,
                position=(center_z, center_y, center_x),
                class_name=class_name,
                confidence=confidence,
                color=color,
                radius=radius,
                volume_data=volume_data,
            )

            visualizations.append(viz)

        return visualizations

    def _get_nucleus_bbox(self, mask: np.ndarray) -> Tuple[slice, ...]:
        """Get bounding box for a nucleus mask."""
        coords = np.where(mask)
        min_coords = [int(np.min(c)) for c in coords]
        max_coords = [int(np.max(c)) for c in coords]

        # Add small padding
        padding = 2
        slices = []
        for i, (min_c, max_c) in enumerate(zip(min_coords, max_coords)):
            shape_max = mask.shape[i]
            slice_min = max(0, min_c - padding)
            slice_max = min(shape_max, max_c + padding + 1)
            slices.append(slice(slice_min, slice_max))

        return tuple(slices)

    def create_2d_projection(
        self,
        raw_volume: np.ndarray,
        label_volume: np.ndarray,
        nuclei_viz: List[NucleusVisualization],
        projection_axis: str = "z",
    ) -> np.ndarray:
        """
        Create 2D projection of the 3D volume.

        Args:
            raw_volume: Raw 3D volume
            label_volume: Label 3D volume
            nuclei_viz: List of nucleus visualizations
            projection_axis: Axis to project along ('x', 'y', 'z')

        Returns:
            2D projection image as RGB array
        """
        # Create projection
        axis_map = {"x": 2, "y": 1, "z": 0}
        axis_idx = axis_map[projection_axis]

        # Project raw volume (max intensity projection)
        if getattr(self.config, "show_raw_data", True):
            raw_proj = np.max(raw_volume, axis=axis_idx)
            # Normalize to 0-255
            raw_proj = (
                (raw_proj - raw_proj.min())
                / (raw_proj.max() - raw_proj.min() + 1e-8)
                * 255
            ).astype(np.uint8)
        else:
            raw_proj = np.zeros(
                (
                    raw_volume.shape[1:3]
                    if axis_idx == 0
                    else (
                        (raw_volume.shape[0], raw_volume.shape[2])
                        if axis_idx == 1
                        else raw_volume.shape[:2]
                    )
                ),
                dtype=np.uint8,
            )

        # Convert to RGB
        if len(raw_proj.shape) == 2:
            rgb_image = np.stack([raw_proj, raw_proj, raw_proj], axis=-1)
        else:
            rgb_image = raw_proj

        # Overlay nucleus annotations
        if getattr(self.config, "show_nucleus_outlines", True) or getattr(
            self.config, "show_class_labels", True
        ):
            rgb_image = self._overlay_nucleus_annotations(
                rgb_image, label_volume, nuclei_viz, axis_idx
            )

        # Add congestion detection and indicators
        if getattr(self.config, "congestion_detection", False):
            congested_regions = self._detect_congested_regions(nuclei_viz)
            if congested_regions:
                rgb_image = self._add_congestion_indicators(
                    rgb_image, congested_regions
                )

        return rgb_image

    def _overlay_nucleus_annotations(
        self,
        rgb_image: np.ndarray,
        label_volume: np.ndarray,
        nuclei_viz: List[NucleusVisualization],
        axis_idx: int,
    ) -> np.ndarray:
        """Overlay nucleus outlines and labels on RGB image."""
        result_image = rgb_image.copy()

        for viz in nuclei_viz:
            nucleus_id = viz.nucleus_id

            # Get nucleus mask for this projection
            mask = label_volume == nucleus_id
            if axis_idx == 0:  # Z projection
                mask_2d = np.any(mask, axis=0)
                center_y, center_x = viz.position[1], viz.position[2]
            elif axis_idx == 1:  # Y projection
                mask_2d = np.any(mask, axis=1)
                center_y, center_x = viz.position[0], viz.position[2]
            else:  # X projection
                mask_2d = np.any(mask, axis=2)
                center_y, center_x = viz.position[0], viz.position[1]

            if not np.any(mask_2d):
                continue

            # Draw nucleus outline with enhanced thickness
            if getattr(self.config, "show_nucleus_outlines", True):
                contours, _ = cv2.findContours(
                    mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                outline_thickness = getattr(self.config, "outline_thickness", 2)
                cv2.drawContours(
                    result_image, contours, -1, viz.color, outline_thickness
                )

            # Draw center point with enhanced visibility
            if getattr(self.config, "show_nucleus_centers", False):
                center_radius = max(3, getattr(self.config, "outline_thickness", 2))
                cv2.circle(
                    result_image, (center_x, center_y), center_radius, viz.color, -1
                )
                # Add white border for better visibility
                if getattr(self.config, "high_contrast_mode", False):
                    cv2.circle(
                        result_image,
                        (center_x, center_y),
                        center_radius + 1,
                        (255, 255, 255),
                        1,
                    )

            # Add class label
            if getattr(self.config, "show_class_labels", True):
                label_text = f"{viz.class_name}"
                if getattr(self.config, "show_confidence_scores", False):
                    label_text += f" ({viz.confidence:.2f})"

                # Position label near nucleus center
                label_x = max(10, center_x - 30)
                label_y = max(20, center_y - 10)

                # Draw text with background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5 * getattr(self.config, "label_font_size", 1.0)
                thickness = max(
                    1, int(getattr(self.config, "outline_thickness", 2) / 2)
                )

                # Enhanced label text with nucleus ID
                label_parts = []
                if getattr(self.config, "show_nucleus_ids", False):
                    label_parts.append(f"#{viz.nucleus_id}")
                label_parts.append(viz.class_name)
                if getattr(self.config, "show_confidence_scores", False):
                    label_parts.append(f"({viz.confidence:.2f})")

                label_text = " ".join(label_parts)

                # Smart label positioning to avoid overlaps
                if getattr(self.config, "smart_label_positioning", False):
                    label_x, label_y, needs_leader = self._find_optimal_label_position(
                        center_x,
                        center_y,
                        label_text,
                        font,
                        font_scale,
                        thickness,
                        rgb_image.shape,
                        nuclei_viz,
                    )

                    # Draw leader line if label was moved far from nucleus
                    if needs_leader and getattr(self.config, "show_leader_lines", True):
                        cv2.line(
                            rgb_image,
                            (center_x, center_y),
                            (label_x + 10, label_y - 5),
                            viz.color,
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    # Original positioning
                    label_x = max(10, center_x - 30)
                    label_y = max(20, center_y - 10)

                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, font, font_scale, thickness
                )

                # Enhanced background with better visibility
                bg_alpha = getattr(self.config, "label_background_alpha", 0.8)
                padding = 4

                # Create background rectangle
                bg_color = (
                    (0, 0, 0)
                    if getattr(self.config, "dark_theme", True)
                    else (255, 255, 255)
                )
                if getattr(self.config, "high_contrast_mode", False):
                    # Add white border around text for better contrast
                    cv2.rectangle(
                        rgb_image,
                        (label_x - padding - 1, label_y - text_height - padding - 1),
                        (label_x + text_width + padding + 1, label_y + padding + 1),
                        (255, 255, 255),
                        -1,
                    )

                cv2.rectangle(
                    rgb_image,
                    (label_x - padding, label_y - text_height - padding),
                    (label_x + text_width + padding, label_y + padding),
                    bg_color,
                    -1,
                )

                # Text with enhanced visibility
                cv2.putText(
                    rgb_image,
                    label_text,
                    (label_x, label_y),
                    font,
                    font_scale,
                    viz.color,
                    thickness,
                    cv2.LINE_AA,
                )

        return result_image

    def create_matplotlib_frame(
        self, timestamp: int, predictions: Dict[str, Dict]
    ) -> str:
        """
        Create a high-quality frame using matplotlib.

        Args:
            timestamp: Frame timestamp
            predictions: Prediction results for this frame

        Returns:
            Path to saved frame image
        """
        # Load frame data
        raw_volume, label_volume = self.load_frame_data(timestamp)

        # Extract nucleus visualizations
        nuclei_viz = self.extract_nucleus_visualizations(timestamp, predictions)

        # Create figure
        fig = plt.figure(figsize=(self.fig_width, self.fig_height), dpi=self.fig_dpi)
        fig.patch.set_facecolor(
            "black" if getattr(self.config, "dark_theme", True) else "white"
        )

        # Determine layout based on enabled features
        if getattr(self.config, "show_multiple_projections", False):
            # Show multiple projection views
            self._create_multi_projection_layout(
                fig, raw_volume, label_volume, nuclei_viz, timestamp
            )
        else:
            # Single projection view
            self._create_single_projection_layout(
                fig, raw_volume, label_volume, nuclei_viz, timestamp
            )

        # Save frame
        frame_path = self.frames_dir / f"frame_{timestamp:06d}.png"
        plt.savefig(
            frame_path,
            dpi=self.fig_dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.close(fig)

        return str(frame_path)

    def _create_single_projection_layout(
        self,
        fig,
        raw_volume: np.ndarray,
        label_volume: np.ndarray,
        nuclei_viz: List[NucleusVisualization],
        timestamp: int,
    ):
        """Create single projection view layout."""
        ax = fig.add_subplot(111)

        # Create 2D projection
        projection = self.create_2d_projection(
            raw_volume,
            label_volume,
            nuclei_viz,
            getattr(self.config, "projection_axis", "z"),
        )

        # Display projection
        ax.imshow(projection, origin="lower", aspect="equal")

        # Add frame information
        if getattr(self.config, "show_frame_info", True):
            self._add_frame_info(ax, timestamp, nuclei_viz)

        ax.set_title(
            f"Frame {timestamp} - {getattr(self.config, 'projection_axis', 'z').upper()} Projection",
            fontsize=self.font_size + 2,
        )
        ax.axis("off" if not getattr(self.config, "show_axes", False) else "on")

    def _create_multi_projection_layout(
        self,
        fig,
        raw_volume: np.ndarray,
        label_volume: np.ndarray,
        nuclei_viz: List[NucleusVisualization],
        timestamp: int,
    ):
        """Create multi-projection view layout."""
        projections = ["z", "y", "x"]
        titles = ["Z (Top-Down)", "Y (Front)", "X (Side)"]

        for i, (proj_axis, title) in enumerate(zip(projections, titles)):
            ax = fig.add_subplot(1, 3, i + 1)

            # Create projection
            projection = self.create_2d_projection(
                raw_volume, label_volume, nuclei_viz, proj_axis
            )

            # Display
            ax.imshow(projection, origin="lower", aspect="equal")
            ax.set_title(f"{title}\nFrame {timestamp}", fontsize=self.font_size)
            ax.axis("off")

        # Add global frame info
        if getattr(self.config, "show_frame_info", True):
            fig.suptitle(
                f"Multi-View Analysis - Frame {timestamp}",
                fontsize=self.font_size + 4,
                y=0.95,
            )

    def _add_frame_info(
        self, ax, timestamp: int, nuclei_viz: List[NucleusVisualization]
    ):
        """Add frame information overlay."""
        # Count nuclei by class
        class_counts = {}
        for viz in nuclei_viz:
            class_counts[viz.class_name] = class_counts.get(viz.class_name, 0) + 1

        # Create info text
        info_lines = [f"Frame: {timestamp}"]
        info_lines.append(f"Total Nuclei: {len(nuclei_viz)}")

        for class_name, count in sorted(class_counts.items()):
            color = self.class_colors.get(class_name, self.class_colors["unknown"])
            info_lines.append(f"{class_name.title()}: {count}")

        # Position text box
        info_text = "\n".join(info_lines)
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=self.font_size,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
            color="white" if getattr(self.config, "dark_theme", True) else "black",
        )

    def render_frame(self, timestamp: int, predictions_data: Dict) -> str:
        """
        Main method to render a complete frame.

        Args:
            timestamp: Frame timestamp
            predictions_data: Complete prediction data for this frame

        Returns:
            Path to rendered frame
        """
        try:
            predictions = predictions_data.get("predictions", {})

            if not predictions:
                print(f"⚠️ No predictions found for frame {timestamp}")
                # Create empty frame
                return self._create_empty_frame(timestamp)

            # Render frame
            if (
                getattr(self.config, "visualization_style", "matplotlib")
                == "matplotlib"
            ):
                frame_path = self.create_matplotlib_frame(timestamp, predictions)
            else:
                # Default to matplotlib for now
                frame_path = self.create_matplotlib_frame(timestamp, predictions)

            return frame_path

        except Exception as e:
            print(f"❌ Error rendering frame {timestamp}: {e}")
            return self._create_empty_frame(timestamp)

    def _create_empty_frame(self, timestamp: int) -> str:
        """Create an empty frame for error cases."""
        fig = plt.figure(figsize=(self.fig_width, self.fig_height), dpi=self.fig_dpi)
        fig.patch.set_facecolor(
            "black" if getattr(self.config, "dark_theme", True) else "white"
        )

        ax = fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            f"No Data\nFrame {timestamp}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=self.font_size + 4,
            color="white" if getattr(self.config, "dark_theme", True) else "black",
        )
        ax.axis("off")

        frame_path = self.frames_dir / f"frame_{timestamp:06d}.png"
        plt.savefig(
            frame_path,
            dpi=self.fig_dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.close(fig)

        return str(frame_path)

    def _find_optimal_label_position(
        self,
        center_x: int,
        center_y: int,
        label_text: str,
        font,
        font_scale: float,
        thickness: int,
        image_shape: tuple,
        all_nuclei: List,
    ) -> tuple:
        """
        Find optimal position for label to avoid overlaps with other labels.

        Returns:
            (label_x, label_y, needs_leader_line)
        """
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        collision_distance = getattr(self.config, "label_collision_distance", 50.0)

        # Try positions in order of preference (closer to nucleus first)
        candidate_positions = [
            # Close positions first
            (center_x - 30, center_y - 10, False),  # Original position
            (center_x + 10, center_y - 10, False),  # Right of nucleus
            (center_x - 30, center_y + 20, False),  # Below nucleus
            (center_x + 10, center_y + 20, False),  # Bottom-right
            # Farther positions if needed
            (center_x - 60, center_y - 10, True),  # Far left
            (center_x + 40, center_y - 10, True),  # Far right
            (center_x - 30, center_y - 40, True),  # Far above
            (center_x - 30, center_y + 40, True),  # Far below
            # Corner positions as last resort
            (center_x - 80, center_y - 40, True),  # Top-left
            (center_x + 60, center_y - 40, True),  # Top-right
            (center_x - 80, center_y + 40, True),  # Bottom-left
            (center_x + 60, center_y + 40, True),  # Bottom-right
        ]

        image_height, image_width = image_shape[:2]

        for candidate_x, candidate_y, needs_leader in candidate_positions:
            # Check if position is within image bounds
            if (
                candidate_x < 10
                or candidate_y < 20
                or candidate_x + text_width > image_width - 10
                or candidate_y > image_height - 10
            ):
                continue

            # Check for collisions with other nucleus positions
            collision_found = False
            for other_viz in all_nuclei:
                if other_viz.nucleus_id == getattr(self, "_current_nucleus_id", None):
                    continue  # Don't check collision with self

                # Get other nucleus center (simplified - using position directly)
                other_x, other_y = (
                    other_viz.position[2],
                    other_viz.position[1],
                )  # X, Y from position

                # Check distance between label positions
                distance = (
                    (candidate_x - other_x) ** 2 + (candidate_y - other_y) ** 2
                ) ** 0.5
                if distance < collision_distance:
                    collision_found = True
                    break

            if not collision_found:
                return candidate_x, candidate_y, needs_leader

        # If no good position found, use original with offset
        return max(10, center_x - 30), max(20, center_y - 10), False

    def _detect_congested_regions(self, nuclei_viz: List) -> List[tuple]:
        """
        Detect regions with high nucleus density (congestion).

        Returns:
            List of (x, y, radius) tuples for congested regions
        """
        if not getattr(self.config, "congestion_detection", False):
            return []

        congested_regions = []
        min_distance_threshold = getattr(self.config, "label_collision_distance", 50.0)

        # Group nuclei by proximity
        nucleus_groups = []
        for viz in nuclei_viz:
            center_x, center_y = viz.position[2], viz.position[1]  # X, Y coordinates

            # Find if this nucleus belongs to an existing group
            group_found = False
            for group in nucleus_groups:
                for other_x, other_y, _ in group:
                    distance = (
                        (center_x - other_x) ** 2 + (center_y - other_y) ** 2
                    ) ** 0.5
                    if (
                        distance < min_distance_threshold * 1.5
                    ):  # Slightly larger threshold for grouping
                        group.append((center_x, center_y, viz.nucleus_id))
                        group_found = True
                        break
                if group_found:
                    break

            if not group_found:
                nucleus_groups.append([(center_x, center_y, viz.nucleus_id)])

        # Identify congested groups (3 or more nuclei in close proximity)
        for group in nucleus_groups:
            if len(group) >= 3:
                # Calculate center and radius of congested region
                x_coords = [pos[0] for pos in group]
                y_coords = [pos[1] for pos in group]

                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)

                # Calculate radius as max distance from center plus some padding
                max_distance = max(
                    ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    for x, y, _ in group
                )
                radius = max_distance + 30  # Add padding

                congested_regions.append((int(center_x), int(center_y), int(radius)))

        return congested_regions

    def _add_congestion_indicators(
        self, rgb_image: np.ndarray, congested_regions: List[tuple]
    ) -> np.ndarray:
        """
        Add visual indicators for congested regions.
        """
        if not congested_regions:
            return rgb_image

        result_image = rgb_image.copy()

        for center_x, center_y, radius in congested_regions:
            # Draw subtle circle around congested region
            cv2.circle(result_image, (center_x, center_y), radius, (255, 255, 0), 2)

            # Add congestion warning text
            warning_text = "CONGESTED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1

            (text_width, text_height), _ = cv2.getTextSize(
                warning_text, font, font_scale, thickness
            )

            # Position warning text above the congested region
            warning_x = center_x - text_width // 2
            warning_y = max(center_y - radius - 10, text_height + 5)

            # Background for warning text
            cv2.rectangle(
                result_image,
                (warning_x - 2, warning_y - text_height - 2),
                (warning_x + text_width + 2, warning_y + 2),
                (0, 0, 0),
                -1,
            )

            # Warning text
            cv2.putText(
                result_image,
                warning_text,
                (warning_x, warning_y),
                font,
                font_scale,
                (255, 255, 0),
                thickness,
            )

        return result_image

    def render_enhanced_frame(
        self,
        timestamp: int,
        frame_data: Dict,
        mode: str = "combined",
        class_filter: str = None,
    ) -> str:
        """
        Render a frame in enhanced mode - FALLBACK to working matplotlib renderer.

        Args:
            timestamp: Frame timestamp
            frame_data: Frame prediction data
            mode: Rendering mode ('combined', 'raw_only', 'dual_channel', 'single_class')
            class_filter: Class to show when mode='single_class'

        Returns:
            Path to rendered frame
        """
        # For now, use the working matplotlib renderer for all modes
        # This ensures we get visible nuclei instead of black frames

        if mode == "single_class" and class_filter:
            # Filter frame_data to only include the specified class
            filtered_frame_data = frame_data.copy()
            filtered_predictions = {}
            for nucleus_id, nucleus_data in frame_data.get("predictions", {}).items():
                if nucleus_data.get("class_name") == class_filter:
                    filtered_predictions[nucleus_id] = nucleus_data
            filtered_frame_data["predictions"] = filtered_predictions
            frame_data = filtered_frame_data

        # Use the working matplotlib renderer (render_frame method)
        frame_path = self.render_frame(timestamp, frame_data)

        # Rename the file to include the mode for organization
        original_path = Path(frame_path)
        mode_suffix = f"_{class_filter}" if class_filter else ""
        new_filename = f"frame_{timestamp:06d}_{mode}{mode_suffix}.png"
        new_path = original_path.parent / new_filename

        # Move/rename the file
        if original_path.exists():
            original_path.rename(new_path)
            return str(new_path)
        else:
            return frame_path

    def _render_raw_only_frame(
        self, frame_data: Dict, width: int, height: int
    ) -> Image.Image:
        """Render frame showing only raw microscopy data."""
        # Create base image
        img = Image.new("RGB", (width, height), self.colors["background"])

        # Load and enhance raw data if available
        raw_data_enhanced = getattr(self.config, "raw_data_enhancement", True)
        if raw_data_enhanced:
            # Apply brightness/contrast enhancement to make nuclei more visible
            # This would require access to actual raw image data
            pass

        # Add minimal overlay (just timestamp)
        draw = ImageDraw.Draw(img)
        timestamp_text = f"Frame {frame_data['timestamp']}"
        draw.text((20, 20), timestamp_text, fill=self.colors["text"], font=self.font)

        return img

    def _render_dual_channel_frame(
        self, frame_data: Dict, width: int, height: int
    ) -> Image.Image:
        """Render frame with raw data on left, labels on right."""
        # Create dual-width canvas
        img = Image.new("RGB", (width, height), self.colors["background"])

        # Split into left (raw) and right (labels) halves
        half_width = width // 2

        # Left side: Raw data
        raw_img = self._render_raw_only_frame(frame_data, half_width, height)
        img.paste(raw_img, (0, 0))

        # Right side: Labels
        label_img = self._render_label_side(frame_data, half_width, height)
        img.paste(label_img, (half_width, 0))

        # Add center divider line
        draw = ImageDraw.Draw(img)
        draw.line(
            [(half_width, 0), (half_width, height)],
            fill=self.colors["divider"],
            width=2,
        )

        # Add headers
        draw.text((20, 20), "Raw Microscopy", fill=self.colors["text"], font=self.font)
        draw.text(
            (half_width + 20, 20),
            "CNN Predictions",
            fill=self.colors["text"],
            font=self.font,
        )

        return img

    def _render_single_class_frame(
        self, frame_data: Dict, class_filter: str, width: int, height: int
    ) -> Image.Image:
        """Render frame showing only nuclei of specified class."""
        img = Image.new("RGB", (width, height), self.colors["background"])
        draw = ImageDraw.Draw(img)

        # Filter predictions to only show specified class
        filtered_predictions = {}
        for nucleus_id, nucleus_data in frame_data.get("predictions", {}).items():
            if nucleus_data.get("class_name") == class_filter:
                filtered_predictions[nucleus_id] = nucleus_data

        # Create filtered frame data
        filtered_frame_data = frame_data.copy()
        filtered_frame_data["predictions"] = filtered_predictions

        # Render nuclei of this class only
        self._render_nucleus_predictions(
            draw, filtered_frame_data, show_all_labels=True
        )

        # Add class-specific header
        class_count = len(filtered_predictions)
        header_text = f"Class: {class_filter.title()} | Count: {class_count} | Frame: {frame_data['timestamp']}"

        # Use class color for header
        class_color = self.colors.get(f"class_{class_filter}", self.colors["text"])
        draw.text((20, 20), header_text, fill=class_color, font=self.font)

        # Add class legend
        self._add_single_class_legend(draw, class_filter, width, height)

        return img

    def _render_combined_enhanced_frame(
        self, frame_data: Dict, width: int, height: int
    ) -> Image.Image:
        """Render enhanced combined frame with probability visualization."""
        img = Image.new("RGB", (width, height), self.colors["background"])
        draw = ImageDraw.Draw(img)

        # Apply probability visualization if enabled
        probability_viz = getattr(self.config, "probability_visualization", False)

        if probability_viz:
            self._render_probability_nuclei(draw, frame_data)
        else:
            self._render_nucleus_predictions(draw, frame_data)

        # Enhanced header with more information
        self._add_enhanced_header(draw, frame_data, width)

        # Enhanced legend with confidence information
        self._add_enhanced_legend(draw, frame_data, width, height)

        return img

    def _render_probability_nuclei(self, draw: ImageDraw.Draw, frame_data: Dict):
        """Render nuclei with probability-based transparency and size."""
        for nucleus_id, nucleus_data in frame_data.get("predictions", {}).items():
            try:
                # Get nucleus properties
                x = int(nucleus_data.get("x", 0))
                y = int(nucleus_data.get("y", 0))
                confidence = float(nucleus_data.get("confidence", 0.0))
                class_name = nucleus_data.get("class_name", "unknown")

                # Scale position to output dimensions
                display_x, display_y = self._scale_position(x, y)

                # Get base color for this class
                base_color = self.colors.get(
                    f"class_{class_name}", self.colors["nucleus"]
                )

                # Adjust transparency based on confidence
                confidence_brightness = getattr(
                    self.config, "confidence_brightness_scaling", True
                )
                if confidence_brightness:
                    # Higher confidence = brighter/more opaque
                    alpha = int(255 * max(0.3, confidence))  # Minimum 30% opacity
                    # Convert RGB to RGBA
                    if isinstance(base_color, tuple) and len(base_color) == 3:
                        nucleus_color = base_color + (alpha,)
                    else:
                        nucleus_color = base_color
                else:
                    nucleus_color = base_color

                # Adjust size based on confidence
                base_size = getattr(self.config, "nucleus_size", 8)
                confidence_size = int(
                    base_size * (0.5 + 0.5 * confidence)
                )  # 50-100% of base size

                # Draw probability visualization
                if (
                    getattr(self.config, "probability_visualization_style", "square")
                    == "square"
                ):
                    # Draw semi-transparent squares
                    bbox = [
                        display_x - confidence_size,
                        display_y - confidence_size,
                        display_x + confidence_size,
                        display_y + confidence_size,
                    ]
                    draw.rectangle(bbox, fill=nucleus_color, outline=base_color)
                else:
                    # Draw circles with varying sizes
                    bbox = [
                        display_x - confidence_size,
                        display_y - confidence_size,
                        display_x + confidence_size,
                        display_y + confidence_size,
                    ]
                    draw.ellipse(bbox, fill=nucleus_color, outline=base_color)

                # Add confidence text if enabled
                if getattr(self.config, "show_confidence_values", False):
                    conf_text = f"{confidence:.2f}"
                    text_y = display_y + confidence_size + 2
                    draw.text(
                        (display_x - 10, text_y),
                        conf_text,
                        fill=self.colors["text"],
                        font=self.small_font,
                    )

            except (ValueError, KeyError) as e:
                if getattr(self.config, "verbose_logging", False):
                    print(f"⚠️ Error rendering probability nucleus {nucleus_id}: {e}")
                continue

    def _render_label_side(
        self, frame_data: Dict, width: int, height: int
    ) -> Image.Image:
        """Render just the label/prediction side for dual-channel display."""
        img = Image.new("RGB", (width, height), self.colors["background"])
        draw = ImageDraw.Draw(img)

        # Render predictions on this side
        self._render_nucleus_predictions(draw, frame_data)

        # Add mini legend
        self._add_compact_legend(draw, frame_data, width, height)

        return img

    def _add_enhanced_header(self, draw: ImageDraw.Draw, frame_data: Dict, width: int):
        """Add enhanced header with comprehensive frame information."""
        timestamp = frame_data["timestamp"]
        nuclei_count = len(frame_data.get("predictions", {}))

        # Count nuclei by class
        class_counts = {}
        total_confidence = 0
        for nucleus_data in frame_data.get("predictions", {}).values():
            class_name = nucleus_data.get("class_name", "unknown")
            confidence = float(nucleus_data.get("confidence", 0.0))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += confidence

        avg_confidence = total_confidence / nuclei_count if nuclei_count > 0 else 0

        # Main header
        header_text = f"Frame {timestamp} | {nuclei_count} nuclei | Avg confidence: {avg_confidence:.2f}"
        draw.text((20, 20), header_text, fill=self.colors["text"], font=self.font)

        # Class breakdown (if space allows)
        if len(class_counts) <= 4:  # Only show if not too crowded
            class_text = " | ".join(
                [f"{cls}: {count}" for cls, count in sorted(class_counts.items())]
            )
            draw.text(
                (20, 50), class_text, fill=self.colors["text"], font=self.small_font
            )

    def _add_enhanced_legend(
        self, draw: ImageDraw.Draw, frame_data: Dict, width: int, height: int
    ):
        """Add enhanced legend with confidence information."""
        legend_x = width - 250
        legend_y = height - 120

        # Background for legend
        draw.rectangle(
            [legend_x - 10, legend_y - 10, width - 10, height - 10],
            fill=self.colors["legend_bg"],
            outline=self.colors["text"],
        )

        # Title
        draw.text(
            (legend_x, legend_y),
            "Classification Legend",
            fill=self.colors["text"],
            font=self.font,
        )

        # Class entries with confidence ranges
        y_offset = 25
        for class_name in ["stable", "mitotic", "new_daughter", "death"]:
            class_color = self.colors.get(f"class_{class_name}", self.colors["nucleus"])

            # Count and average confidence for this class
            class_nuclei = [
                n
                for n in frame_data.get("predictions", {}).values()
                if n.get("class_name") == class_name
            ]
            class_count = len(class_nuclei)

            if class_count > 0:
                avg_conf = (
                    sum(float(n.get("confidence", 0)) for n in class_nuclei)
                    / class_count
                )
                legend_text = f"{class_name}: {class_count} ({avg_conf:.2f})"
            else:
                legend_text = f"{class_name}: 0"

            # Color square
            square_size = 12
            draw.rectangle(
                [
                    legend_x,
                    legend_y + y_offset,
                    legend_x + square_size,
                    legend_y + y_offset + square_size,
                ],
                fill=class_color,
                outline=self.colors["text"],
            )

            # Text
            draw.text(
                (legend_x + 20, legend_y + y_offset),
                legend_text,
                fill=self.colors["text"],
                font=self.small_font,
            )

            y_offset += 20

    def _add_single_class_legend(
        self, draw: ImageDraw.Draw, class_filter: str, width: int, height: int
    ):
        """Add legend specific to single class view."""
        legend_x = width - 200
        legend_y = height - 80

        # Background
        draw.rectangle(
            [legend_x - 10, legend_y - 10, width - 10, height - 10],
            fill=self.colors["legend_bg"],
            outline=self.colors["text"],
        )

        # Class color and description
        class_color = self.colors.get(f"class_{class_filter}", self.colors["nucleus"])
        draw.rectangle(
            [legend_x, legend_y, legend_x + 15, legend_y + 15],
            fill=class_color,
            outline=self.colors["text"],
        )

        draw.text(
            (legend_x + 25, legend_y),
            f"{class_filter.title()} nuclei",
            fill=self.colors["text"],
            font=self.font,
        )

        # Add description
        descriptions = {
            "stable": "Non-dividing cells",
            "mitotic": "Actively dividing",
            "new_daughter": "Recently divided",
            "death": "Undergoing apoptosis",
        }

        description = descriptions.get(class_filter, "Unknown class")
        draw.text(
            (legend_x, legend_y + 25),
            description,
            fill=self.colors["text"],
            font=self.small_font,
        )

    def _add_compact_legend(
        self, draw: ImageDraw.Draw, frame_data: Dict, width: int, height: int
    ):
        """Add compact legend for dual-channel label side."""
        legend_x = width - 150
        legend_y = height - 100

        draw.text(
            (legend_x, legend_y),
            "Classes:",
            fill=self.colors["text"],
            font=self.small_font,
        )

        y_offset = 15
        for class_name in ["stable", "mitotic", "new_daughter", "death"]:
            class_color = self.colors.get(f"class_{class_name}", self.colors["nucleus"])

            # Small color square
            draw.rectangle(
                [legend_x, legend_y + y_offset, legend_x + 8, legend_y + y_offset + 8],
                fill=class_color,
            )

            # Abbreviated text
            abbrev = class_name[:3].upper()
            draw.text(
                (legend_x + 12, legend_y + y_offset - 2),
                abbrev,
                fill=self.colors["text"],
                font=self.small_font,
            )

            y_offset += 12

    def _scale_position(self, x: int, y: int) -> Tuple[int, int]:
        """
        Scale nucleus position from original coordinates to display coordinates.

        Args:
            x, y: Original coordinates

        Returns:
            Scaled display coordinates
        """
        # Get image dimensions
        display_width = getattr(self.config, "output_width", 1920)
        display_height = getattr(self.config, "output_height", 1080)

        # For now, assume 1:1 mapping - this may need adjustment based on your data
        # You might need to scale based on the original image dimensions
        original_width = getattr(self.config, "original_width", display_width)
        original_height = getattr(self.config, "original_height", display_height)

        scaled_x = int((x / original_width) * display_width)
        scaled_y = int((y / original_height) * display_height)

        return scaled_x, scaled_y

    def _render_nucleus_predictions(
        self, draw: ImageDraw.Draw, frame_data: Dict, show_all_labels: bool = False
    ):
        """
        Render nucleus predictions using PIL drawing.

        Args:
            draw: PIL ImageDraw object
            frame_data: Frame prediction data
            show_all_labels: Whether to show all labels or use smart positioning
        """
        for nucleus_id, nucleus_data in frame_data.get("predictions", {}).items():
            try:
                # Get nucleus properties
                x = int(nucleus_data.get("x", 0))
                y = int(nucleus_data.get("y", 0))
                confidence = float(nucleus_data.get("confidence", 0.0))
                class_name = nucleus_data.get("class_name", "unknown")

                # Scale position to output dimensions
                display_x, display_y = self._scale_position(x, y)

                # Get color for this class
                nucleus_color = self.colors.get(
                    f"class_{class_name}", self.colors["nucleus"]
                )

                # Draw nucleus indicator
                nucleus_size = getattr(self.config, "nucleus_size", 8)
                bbox = [
                    display_x - nucleus_size,
                    display_y - nucleus_size,
                    display_x + nucleus_size,
                    display_y + nucleus_size,
                ]

                # Draw circle or square based on config
                visualization_style = getattr(self.config, "nucleus_shape", "circle")
                if visualization_style == "square":
                    draw.rectangle(
                        bbox, fill=nucleus_color, outline=self.colors["text"]
                    )
                else:
                    draw.ellipse(bbox, fill=nucleus_color, outline=self.colors["text"])

                # Add labels if enabled
                if getattr(self.config, "show_nucleus_ids", True) or show_all_labels:
                    label_text = f"{nucleus_id}"
                    if getattr(self.config, "show_class_labels", True):
                        label_text += f" ({class_name})"
                    if getattr(self.config, "show_confidence_scores", True):
                        label_text += f" {confidence:.2f}"

                    # Position label
                    label_x = display_x + nucleus_size + 5
                    label_y = display_y - nucleus_size

                    # Smart label positioning if enabled
                    if (
                        getattr(self.config, "smart_label_positioning", True)
                        and not show_all_labels
                    ):
                        label_x, label_y = self._get_smart_label_position(
                            display_x, display_y, label_text, frame_data
                        )

                    # Draw label background for better readability
                    text_bbox = draw.textbbox(
                        (label_x, label_y), label_text, font=self.font
                    )
                    draw.rectangle(
                        text_bbox, fill=self.colors["label_bg"], outline=nucleus_color
                    )

                    # Draw label text
                    draw.text(
                        (label_x, label_y),
                        label_text,
                        fill=self.colors["text"],
                        font=self.font,
                    )

            except (ValueError, KeyError) as e:
                if getattr(self.config, "verbose_logging", False):
                    print(f"⚠️ Error rendering nucleus {nucleus_id}: {e}")
                continue

    def _get_smart_label_position(
        self, x: int, y: int, text: str, frame_data: Dict
    ) -> Tuple[int, int]:
        """
        Calculate smart label position to avoid overlaps.

        Args:
            x, y: Nucleus center position
            text: Label text
            frame_data: Frame data for collision detection

        Returns:
            Optimal label position
        """
        # Default position
        label_x = x + 15
        label_y = y - 10

        # Check for collision distance
        collision_distance = getattr(self.config, "label_collision_distance", 50.0)

        # Try different positions if there are overlaps
        positions = [
            (x + 15, y - 10),  # Right-top
            (x - 50, y - 10),  # Left-top
            (x + 15, y + 20),  # Right-bottom
            (x - 50, y + 20),  # Left-bottom
            (x, y - 25),  # Top-center
            (x, y + 30),  # Bottom-center
        ]

        for pos_x, pos_y in positions:
            # Check if this position is clear
            is_clear = True
            for other_id, other_data in frame_data.get("predictions", {}).items():
                other_x, other_y = self._scale_position(
                    int(other_data.get("x", 0)), int(other_data.get("y", 0))
                )

                distance = ((pos_x - other_x) ** 2 + (pos_y - other_y) ** 2) ** 0.5
                if distance < collision_distance:
                    is_clear = False
                    break

            if is_clear:
                label_x, label_y = pos_x, pos_y
                break

        return label_x, label_y

    @property
    def output_dir(self) -> Path:
        """Get output directory for frames."""
        return self.frames_dir


if __name__ == "__main__":
    # Test the renderer
    from config import VideoConfig

    print("Testing FrameRenderer...")

    # Create test configuration
    config = VideoConfig()
    config.output_dir = "./test_output"
    config.raw_data_path = "/dummy/raw"
    config.label_data_path = "/dummy/labels"

    try:
        renderer = FrameRenderer(config)
        print("✅ FrameRenderer created successfully")
        print(f"   Frames directory: {renderer.frames_dir}")
        print(f"   Figure size: {renderer.fig_width}x{renderer.fig_height}")
    except Exception as e:
        print(f"❌ Failed to create renderer: {e}")

    print("Renderer testing complete!")
