"""
Video Generator - Main Pipeline
Orchestrates the complete video generation process
"""

import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
from PIL import Image
import torch

from .config import VideoConfig, ConfigPresets
from .model_interface import ModelInferenceEngine
from .data_pipeline import SlidingWindowProcessor
from .renderer import FrameRenderer


class VideoGenerator:
    """
    Complete video generation pipeline.
    Orchestrates model inference, frame rendering, and video assembly.
    """

    def __init__(self, config: VideoConfig):
        """
        Initialize the video generator.

        Args:
            config: Video generation configuration
        """
        self.config = config

        # Initialize components
        print("🚀 Initializing Video Generator...")

        # Model inference engine
        self.model_engine = ModelInferenceEngine(config)
        print("✅ Model engine loaded")

        # Data pipeline
        self.data_processor = SlidingWindowProcessor(config)
        print("✅ Data processor initialized")

        # Frame renderer
        self.frame_renderer = FrameRenderer(config)
        print("✅ Frame renderer initialized")

        # Output paths
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.video_path = self.output_dir / f"{config.video_name}.{config.video_format}"
        self.frames_dir = self.frame_renderer.frames_dir

        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎬 Video will be saved as: {self.video_path}")

    def generate_video(self) -> str:
        """
        Generate enhanced videos with multiple formats and views.

        Returns:
            Path to the main generated video file
        """
        print("🎬 Starting ENHANCED video generation pipeline...")
        start_time = time.time()

        try:
            # Step 1: Process dataset and generate predictions
            print("\n📊 Step 1: Processing dataset with model predictions...")
            predictions_file = self.data_processor.process_full_dataset(
                self.model_engine
            )

            # Step 2: Enhanced frame rendering with multiple modes
            print("\n🎨 Step 2: Enhanced frame rendering...")
            all_frame_sets = self._render_enhanced_frames(
                predictions_file, mode="single_class", class_filter="mitotic"
            )

            # Step 3: Generate multiple videos
            print("\n🎞️ Step 3: Generating multiple video formats...")
            video_paths = self._generate_multiple_videos(all_frame_sets)

            # Step 4: Generate summary
            print("\n📋 Step 4: Generating comprehensive summary...")
            self._generate_enhanced_summary(predictions_file, all_frame_sets)

            total_time = time.time() - start_time
            print(f"\n✅ Enhanced video generation complete!")
            print(f"   Total time: {total_time/60:.1f} minutes")
            print(f"   Videos generated: {len(video_paths)}")
            for video_type, path in video_paths.items():
                print(f"   📹 {video_type}: {path}")

            # Return the main combined video path
            if video_paths:
                return video_paths.get("combined", list(video_paths.values())[0])
            else:
                print("⚠️ No videos were generated successfully")
                raise ValueError(
                    "No videos could be generated - all frame rendering failed"
                )

        except Exception as e:
            print(f"❌ Enhanced video generation failed: {e}")
            raise

    def generate_enhanced_videos(self, predictions_file: str) -> Dict[str, str]:
        """
        Generate multiple enhanced videos from predictions.

        Args:
            predictions_file: Path to JSONL predictions file

        Returns:
            Dictionary of video paths by type
        """
        print("🎬 Starting Enhanced Video Generation Pipeline")
        print("=" * 50)

        # Validate inputs
        if not Path(predictions_file).exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Render frames in multiple formats
        print("📐 Step 1: Enhanced Frame Rendering")
        frame_sets = self._render_enhanced_frames(predictions_file)

        # Step 2: Generate multiple videos
        print("🎞️ Step 2: Multi-Video Generation")
        video_paths = self._generate_multiple_videos(frame_sets)

        # Step 3: Generate enhanced summary
        print("📊 Step 3: Enhanced Summary Generation")
        self._generate_enhanced_summary(predictions_file, frame_sets)

        print("✅ Enhanced video generation pipeline completed!")

        return video_paths

    def _render_all_frames(self, predictions_file: str) -> List[str]:
        """
        Render all frames from predictions with enhanced progress feedback.

        Args:
            predictions_file: Path to JSONL predictions file

        Returns:
            List of rendered frame paths
        """
        frame_paths = []

        # Count total frames first for progress tracking
        with open(predictions_file, "r") as f:
            total_frames = sum(1 for _ in f)

        print(f"🎨 Rendering {total_frames} frames...")
        if getattr(self.config, "verbose_logging", False):
            print(
                f"   Output resolution: {getattr(self.config, 'output_width', 1920)}x{getattr(self.config, 'output_height', 1080)}"
            )
            print(
                f"   Smart labeling: {getattr(self.config, 'smart_label_positioning', False)}"
            )
            print(
                f"   Congestion detection: {getattr(self.config, 'congestion_detection', False)}"
            )

        start_time = time.time()

        with open(predictions_file, "r") as f:
            for line_num, line in enumerate(f):
                try:
                    frame_data = json.loads(line.strip())
                    timestamp = frame_data["timestamp"]

                    # Enhanced progress logging
                    if getattr(self.config, "verbose_logging", False):
                        nuclei_count = len(frame_data.get("predictions", {}))
                        print(
                            f"   🎭 Frame {timestamp}: {nuclei_count} nuclei detected"
                        )

                    # Render frame
                    frame_start = time.time()
                    frame_path = self.frame_renderer.render_frame(timestamp, frame_data)
                    frame_time = time.time() - frame_start

                    frame_paths.append(frame_path)

                    # Progress update with ETA
                    progress = (line_num + 1) / total_frames
                    if (line_num + 1) % 5 == 0 or getattr(
                        self.config, "verbose_logging", False
                    ):
                        elapsed = time.time() - start_time
                        if (
                            getattr(self.config, "estimated_time_remaining", True)
                            and progress > 0
                        ):
                            eta = (elapsed / progress) - elapsed
                            eta_str = (
                                f" | ETA: {eta/60:.1f}min"
                                if eta > 60
                                else f" | ETA: {eta:.1f}s"
                            )
                        else:
                            eta_str = ""

                        print(
                            f"   📊 Progress: {line_num + 1}/{total_frames} ({progress*100:.1f}%) | "
                            f"Frame time: {frame_time:.2f}s{eta_str}"
                        )

                    # Memory usage monitoring
                    if (
                        getattr(self.config, "show_processing_stats", False)
                        and (line_num + 1) % 20 == 0
                    ):
                        try:
                            import psutil

                            memory_usage = psutil.virtual_memory().percent
                            print(f"   💾 Memory usage: {memory_usage:.1f}%")
                        except ImportError:
                            pass

                except Exception as e:
                    print(f"⚠️ Failed to render frame at line {line_num + 1}: {e}")
                    continue

        total_render_time = time.time() - start_time
        avg_frame_time = total_render_time / len(frame_paths) if frame_paths else 0

        print(
            f"✅ Rendered {len(frame_paths)} frames in {total_render_time/60:.1f} minutes"
        )
        print(f"   Average frame time: {avg_frame_time:.2f}s")

        return frame_paths

    def _assemble_video(self, frame_paths: List[str]) -> str:
        """
        Assemble frames into final video.

        Args:
            frame_paths: List of frame image paths

        Returns:
            Path to assembled video
        """
        if not frame_paths:
            raise ValueError("No frames to assemble into video")

        # Sort frame paths to ensure correct order
        frame_paths.sort()

        if self.config.video_format.lower() == "gif":
            return self._create_gif(frame_paths)
        else:
            return self._create_mp4(frame_paths)

    def _create_mp4(self, frame_paths: List[str]) -> str:
        """Create MP4 video using OpenCV."""
        # Get first frame to determine dimensions
        first_frame = cv2.imread(frame_paths[0])
        height, width = first_frame.shape[:2]

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(self.video_path), fourcc, self.config.fps, (width, height)
        )

        print(f"   Creating MP4: {width}x{height} @ {self.config.fps} FPS")

        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)

            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(frame_paths)} frames...")

        out.release()

        # Try to use ffmpeg for better compression if available
        temp_path = str(self.video_path)
        final_path = str(self.video_path).replace(".mp4", "_compressed.mp4")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    temp_path,
                    "-c:v",
                    "libx264",
                    "-crf",
                    "23",
                    "-preset",
                    "medium",
                    final_path,
                ],
                check=True,
                capture_output=True,
            )

            # Replace original with compressed version
            os.remove(temp_path)
            os.rename(final_path, temp_path)
            print("   Applied ffmpeg compression")

        except (subprocess.CalledProcessError, FileNotFoundError):
            print("   ffmpeg not available, using OpenCV output")

        return str(self.video_path)

    def _create_gif(self, frame_paths: List[str]) -> str:
        """Create GIF animation using PIL."""
        images = []

        print(f"   Creating GIF with {len(frame_paths)} frames...")

        for i, frame_path in enumerate(frame_paths):
            try:
                img = Image.open(frame_path)
                images.append(img)

                if (i + 1) % 20 == 0:
                    print(f"   Loaded {i + 1}/{len(frame_paths)} frames...")

            except Exception as e:
                print(f"⚠️ Failed to load frame {frame_path}: {e}")
                continue

        if not images:
            raise ValueError("No valid frames loaded for GIF creation")

        # Calculate duration per frame in milliseconds
        duration = int(1000 / self.config.fps)

        # Save GIF
        images[0].save(
            str(self.video_path),
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )

        return str(self.video_path)

    def _generate_summary(self, predictions_file: str, frame_count: int):
        """Generate summary statistics and metadata."""
        summary_data = {
            "video_info": {
                "video_path": str(self.video_path),
                "frame_count": frame_count,
                "fps": self.config.fps,
                "duration_seconds": frame_count / self.config.fps,
                "resolution": f"{self.config.output_width}x{self.config.output_height}",
                "format": self.config.video_format,
            },
            "model_info": {
                "model_type": self.config.model_type,
                "model_path": self.config.model_path,
                "batch_size": self.config.batch_size,
            },
            "dataset_info": {
                "raw_data_path": self.config.raw_data_path,
                "label_data_path": self.config.label_data_path,
                "total_frames_processed": frame_count,
            },
            "processing_stats": self._calculate_processing_stats(predictions_file),
        }

        # Save summary
        summary_path = self.output_dir / "video_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        print(f"📋 Summary saved: {summary_path}")

        # Print key statistics
        stats = summary_data["processing_stats"]
        print(f"📊 Processing Statistics:")
        print(f"   Total nuclei analyzed: {stats['total_nuclei']}")
        print(f"   Average nuclei per frame: {stats['avg_nuclei_per_frame']:.1f}")
        print(f"   Class distribution: {stats['class_distribution']}")

    def _calculate_processing_stats(self, predictions_file: str) -> Dict[str, Any]:
        """Calculate processing statistics from predictions."""
        total_nuclei = 0
        frame_counts = []
        class_counts = {}

        with open(predictions_file, "r") as f:
            for line in f:
                frame_data = json.loads(line.strip())
                nuclei_count = frame_data["nuclei_count"]
                frame_counts.append(nuclei_count)
                total_nuclei += nuclei_count

                # Count classes
                for nucleus_data in frame_data["predictions"].values():
                    class_name = nucleus_data["class_name"]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return {
            "total_nuclei": total_nuclei,
            "avg_nuclei_per_frame": np.mean(frame_counts) if frame_counts else 0,
            "max_nuclei_per_frame": max(frame_counts) if frame_counts else 0,
            "min_nuclei_per_frame": min(frame_counts) if frame_counts else 0,
            "class_distribution": class_counts,
        }

    def cleanup_temporary_files(self, keep_frames: bool = False):
        """Clean up temporary files after video generation."""
        if not keep_frames:
            print("🧹 Cleaning up temporary frame files...")
            frame_files = list(self.frames_dir.glob("*.png"))
            for frame_file in frame_files:
                frame_file.unlink()
            print(f"   Removed {len(frame_files)} frame files")

        # Clean up cache files
        cache_dir = Path(self.config.cache_dir)
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.tmp"))
            for cache_file in cache_files:
                cache_file.unlink()

    @classmethod
    def quick_video(
        cls,
        raw_data_path: str,
        label_data_path: str,
        model_path: str,
        output_dir: str,
        **kwargs,
    ) -> str:
        """
        Quick video generation with minimal configuration.

        Args:
            raw_data_path: Path to raw image data
            label_data_path: Path to label data
            model_path: Path to trained model
            output_dir: Output directory for video
            **kwargs: Additional configuration options

        Returns:
            Path to generated video
        """
        # Create configuration
        config = ConfigPresets.create_quick_preview()

        # Update paths
        config.raw_data_path = raw_data_path
        config.label_data_path = label_data_path
        config.model_path = model_path
        config.output_dir = output_dir

        # Apply any custom settings
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Generate video
        generator = cls(config)
        return generator.generate_video()

    def _render_enhanced_frames(self, predictions_file: str) -> Dict[str, List[str]]:
        """
        Render frames in multiple enhanced formats.

        Args:
            predictions_file: Path to JSONL predictions file

        Returns:
            Dictionary of frame sets by type
        """
        frame_sets = {"combined": [], "raw_only": [], "dual_channel": []}

        # Add per-class frame sets if enabled
        if getattr(self.config, "generate_per_class_videos", False):
            frame_sets.update(
                {"stable": [], "mitotic": [], "new_daughter": [], "death": []}
            )

        # Count total frames first for progress tracking
        with open(predictions_file, "r") as f:
            total_frames = sum(1 for _ in f)

        print(f"🎨 Rendering {total_frames} frames in multiple formats...")
        print(f"   📹 Formats: {list(frame_sets.keys())}")

        if getattr(self.config, "verbose_logging", False):
            print(
                f"   📺 Resolution: {getattr(self.config, 'output_width', 1920)}x{getattr(self.config, 'output_height', 1080)}"
            )
            print(
                f"   🎯 Probability visualization: {getattr(self.config, 'probability_visualization', False)}"
            )
            print(
                f"   📊 Dual channel: {getattr(self.config, 'dual_channel_display', False)}"
            )

        start_time = time.time()

        with open(predictions_file, "r") as f:
            for line_num, line in enumerate(f):
                try:
                    frame_data = json.loads(line.strip())
                    timestamp = frame_data["timestamp"]

                    # Enhanced progress logging
                    if getattr(self.config, "verbose_logging", False):
                        nuclei_count = len(frame_data.get("predictions", {}))
                        class_counts = self._count_classes_in_frame(frame_data)
                        print(
                            f"   🎭 Frame {timestamp}: {nuclei_count} nuclei | Classes: {class_counts}"
                        )

                    frame_start = time.time()

                    # Render different frame types
                    rendered_frames = self._render_frame_variants(timestamp, frame_data)

                    # Add to appropriate frame sets
                    for frame_type, frame_path in rendered_frames.items():
                        if frame_type in frame_sets:
                            frame_sets[frame_type].append(frame_path)

                    frame_time = time.time() - frame_start

                    # Progress update with ETA
                    progress = (line_num + 1) / total_frames
                    if (line_num + 1) % 5 == 0 or getattr(
                        self.config, "verbose_logging", False
                    ):
                        elapsed = time.time() - start_time
                        if (
                            getattr(self.config, "estimated_time_remaining", True)
                            and progress > 0
                        ):
                            eta = (elapsed / progress) - elapsed
                            eta_str = (
                                f" | ETA: {eta/60:.1f}min"
                                if eta > 60
                                else f" | ETA: {eta:.1f}s"
                            )
                        else:
                            eta_str = ""

                        print(
                            f"   📊 Progress: {line_num + 1}/{total_frames} ({progress*100:.1f}%) | "
                            f"Frame time: {frame_time:.2f}s{eta_str}"
                        )

                except Exception as e:
                    print(f"⚠️ Failed to render frame at line {line_num + 1}: {e}")
                    continue

        total_render_time = time.time() - start_time
        total_frames_rendered = sum(len(frames) for frames in frame_sets.values())

        print(f"✅ Enhanced rendering complete!")
        print(f"   Total time: {total_render_time/60:.1f} minutes")
        print(f"   Frame sets generated: {len(frame_sets)}")
        for frame_type, frames in frame_sets.items():
            print(f"   📹 {frame_type}: {len(frames)} frames")

        return frame_sets

    def _render_frame_variants(
        self, timestamp: int, frame_data: Dict
    ) -> Dict[str, str]:
        """
        Render multiple variants of a single frame.

        Args:
            timestamp: Frame timestamp
            frame_data: Frame prediction data

        Returns:
            Dictionary of frame paths by variant type
        """
        rendered_frames = {}

        # 1. Combined view (enhanced version of original)
        if getattr(self.config, "generate_combined_video", True):
            rendered_frames["combined"] = self.frame_renderer.render_enhanced_frame(
                timestamp, frame_data, mode="combined"
            )

        # 2. Raw data only
        rendered_frames["raw_only"] = self.frame_renderer.render_enhanced_frame(
            timestamp, frame_data, mode="raw_only"
        )

        # 3. Dual channel (raw + labels side by side)
        if getattr(self.config, "dual_channel_display", True):
            rendered_frames["dual_channel"] = self.frame_renderer.render_enhanced_frame(
                timestamp, frame_data, mode="dual_channel"
            )

        # 4. Per-class videos (if enabled)
        if getattr(self.config, "generate_per_class_videos", False):
            for class_name in ["stable", "mitotic", "new_daughter", "death"]:
                rendered_frames[class_name] = self.frame_renderer.render_enhanced_frame(
                    timestamp, frame_data, mode="single_class", class_filter=class_name
                )

        return rendered_frames

    def _count_classes_in_frame(self, frame_data: Dict) -> Dict[str, int]:
        """Count nuclei by class in a frame."""
        class_counts = {}
        for nucleus_data in frame_data.get("predictions", {}).values():
            class_name = nucleus_data.get("class_name", "unknown")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

    def _generate_multiple_videos(
        self, frame_sets: Dict[str, List[str]]
    ) -> Dict[str, str]:
        """
        Generate multiple video files from different frame sets.

        Args:
            frame_sets: Dictionary of frame lists by type

        Returns:
            Dictionary of video paths by type
        """
        video_paths = {}

        for video_type, frame_paths in frame_sets.items():
            if not frame_paths:
                continue

            print(f"   🎬 Creating {video_type} video ({len(frame_paths)} frames)...")

            # Create video-specific output path
            video_name = f"{self.config.video_name}_{video_type}"
            video_path = self.output_dir / f"{video_name}.{self.config.video_format}"

            # Temporarily update config for this video
            original_video_path = self.video_path
            self.video_path = video_path

            try:
                if self.config.video_format.lower() == "gif":
                    final_path = self._create_gif(frame_paths)
                else:
                    final_path = self._create_mp4(frame_paths)

                video_paths[video_type] = final_path

            finally:
                # Restore original video path
                self.video_path = original_video_path

        return video_paths

    def _generate_enhanced_summary(
        self, predictions_file: str, frame_sets: Dict[str, List[str]]
    ):
        """Generate enhanced summary with multiple video information."""
        # Calculate processing stats
        processing_stats = self._calculate_processing_stats(predictions_file)

        # Create enhanced summary
        summary_data = {
            "video_info": {
                "total_videos_generated": len(frame_sets),
                "video_types": list(frame_sets.keys()),
                "fps": self.config.fps,
                "format": self.config.video_format,
                "resolution": f"{self.config.output_width}x{self.config.output_height}",
            },
            "enhancement_features": {
                "probability_visualization": getattr(
                    self.config, "probability_visualization", False
                ),
                "dual_channel_display": getattr(
                    self.config, "dual_channel_display", False
                ),
                "per_class_videos": getattr(
                    self.config, "generate_per_class_videos", False
                ),
                "raw_data_enhancement": getattr(
                    self.config, "raw_data_enhancement", False
                ),
            },
            "frame_info": {
                frame_type: {
                    "frame_count": len(frames),
                    "duration_seconds": len(frames) / self.config.fps,
                }
                for frame_type, frames in frame_sets.items()
            },
            "model_info": {
                "model_type": self.config.model_type,
                "model_path": self.config.model_path,
                "batch_size": self.config.batch_size,
            },
            "dataset_info": {
                "raw_data_path": self.config.raw_data_path,
                "label_data_path": self.config.label_data_path,
            },
            "processing_stats": processing_stats,
        }

        # Save enhanced summary
        summary_path = self.output_dir / "enhanced_video_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        print(f"📋 Enhanced summary saved: {summary_path}")

        # Print key statistics
        print(f"📊 Enhanced Processing Statistics:")
        print(f"   🎬 Videos generated: {len(frame_sets)}")
        print(f"   🔬 Total nuclei analyzed: {processing_stats['total_nuclei']}")
        print(
            f"   📊 Average nuclei per frame: {processing_stats['avg_nuclei_per_frame']:.1f}"
        )
        print(f"   🎭 Class distribution: {processing_stats['class_distribution']}")

        # Show video durations
        for video_type, frames in frame_sets.items():
            duration = len(frames) / self.config.fps
            print(f"   📹 {video_type}: {len(frames)} frames, {duration:.1f}s duration")
