"""
Model Interface for Video Generation
Handles loading and running 3ncnn/4ncnn models for nucleus classification
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import importlib.util
from pathlib import Path
from typing import List, Tuple, Optional, Union, TYPE_CHECKING
import glob

if TYPE_CHECKING:
    from .config import VideoConfig

# Project root for model paths
PROJECT_ROOT = Path(__file__).parent.parent


class ModelInferenceEngine:
    """
    Unified interface for loading and running 3D CNN models.
    Supports both 3ncnn and 4ncnn architectures with GPU optimization.
    """

    def __init__(self, config: "VideoConfig"):
        """
        Initialize the model inference engine.

        Args:
            config: VideoConfig object containing model settings
        """
        self.config = config
        self.model_type = config.model_type
        self.batch_size = config.batch_size
        self.device = self._setup_device(config.device)
        self.model = None
        self.class_names = None
        self.preprocessor = None

        # Load model and setup
        self.model_path = config.model_path or self._find_best_model()
        self._load_model()
        self._setup_preprocessor()

        print(f"✅ ModelInferenceEngine initialized:")
        print(f"   Model type: {self.model_type}")
        print(f"   Device: {self.device}")
        print(f"   Model path: {self.model_path}")
        print(f"   Model Type: {self.model_type}")
        print(f"   Model Path: {self.model_path}")
        print(f"   Device: {self.device}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Classes: {self.class_names}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device with smart selection."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _find_best_model(self) -> str:
        """Auto-detect the best available trained model."""
        model_dir = PROJECT_ROOT / "model" / self.model_type / "training_outputs"

        if not model_dir.exists():
            raise FileNotFoundError(f"No training outputs found for {self.model_type}")

        # Find all model files
        model_files = glob.glob(str(model_dir / "*" / "best_model.pth"))

        if not model_files:
            raise FileNotFoundError(f"No trained models found in {model_dir}")

        # Return the most recent model (by directory name timestamp)
        latest_model = sorted(model_files)[-1]
        print(f"🔍 Auto-detected model: {latest_model}")
        return latest_model

    def _load_model(self):
        """Load the appropriate model architecture and weights."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if self.model_type == "3ncnn":
            self._load_3ncnn_model()
        elif self.model_type == "4ncnn":
            self._load_4ncnn_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Enable optimization for inference
        if self.device.type == "cuda":
            self.model = torch.jit.script(self.model)

        print(f"🚀 Model loaded and optimized for {self.device}")

    def _load_3ncnn_model(self):
        """Load 3-channel CNN model with isolated imports."""
        model_3ncnn_path = PROJECT_ROOT / "model" / "3ncnn"

        # Save current sys.path and working directory
        original_sys_path = sys.path.copy()
        original_cwd = os.getcwd()

        try:
            # Temporarily modify sys.path to prioritize 3ncnn directory
            sys.path.insert(0, str(model_3ncnn_path))

            # Change to 3ncnn directory
            os.chdir(str(model_3ncnn_path))

            # Import cnn_model from 3ncnn directory
            import cnn_model

            # Create the model
            self.model = cnn_model.Simple3DCNN()
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            self.class_names = ["mitotic", "new_daughter", "stable", "death"]
            self.input_channels = 3

            # Try to get input shape from model if available
            if hasattr(cnn_model, "HPARAMS"):
                self.input_shape = (
                    cnn_model.HPARAMS["input_depth"],
                    cnn_model.HPARAMS["input_height"],
                    cnn_model.HPARAMS["input_width"],
                )
            else:
                # Default shape if not available
                self.input_shape = (16, 64, 64)

            print(f"✅ 3ncnn model loaded successfully")
            print(f"   Classes: {self.class_names}")
            print(f"   Input shape: {self.input_shape}")

        except Exception as e:
            print(f"❌ Failed to load 3ncnn model: {e}")
            import traceback

            traceback.print_exc()
            raise ImportError(f"Failed to import 3ncnn model: {e}")

        finally:
            # Always restore original state
            sys.path[:] = original_sys_path
            os.chdir(original_cwd)

    def _load_4ncnn_model(self):
        """Load 4-channel CNN model with isolated imports."""
        model_4ncnn_path = PROJECT_ROOT / "model" / "4ncnn"

        # Save current sys.path and working directory
        original_sys_path = sys.path.copy()
        original_cwd = os.getcwd()

        try:
            # Temporarily modify sys.path to prioritize 4ncnn directory
            # This ensures 'from config import' finds the 4ncnn config first
            sys.path.insert(0, str(model_4ncnn_path))

            # Change to 4ncnn directory to ensure relative imports work
            os.chdir(str(model_4ncnn_path))

            # Import model_utils which will now find the correct config
            import model_utils

            # Create the model
            self.model = model_utils.Simple3DCNN()
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            self.class_names = model_utils.CLASS_NAMES
            self.input_channels = 4
            self.input_shape = (
                model_utils.HPARAMS["input_depth"],
                model_utils.HPARAMS["input_height"],
                model_utils.HPARAMS["input_width"],
            )

            print(f"✅ 4ncnn model loaded successfully")
            print(f"   Classes: {self.class_names}")
            print(f"   Input shape: {self.input_shape}")

        except Exception as e:
            print(f"❌ Failed to load 4ncnn model: {e}")
            import traceback

            traceback.print_exc()
            raise ImportError(f"Failed to import 4ncnn model: {e}")

        finally:
            # Always restore original state
            sys.path[:] = original_sys_path
            os.chdir(original_cwd)

    def _setup_preprocessor(self):
        """Setup model-specific preprocessing."""
        if self.model_type == "3ncnn":
            self.preprocessor = self._preprocess_3ncnn
        elif self.model_type == "4ncnn":
            self.preprocessor = self._preprocess_4ncnn

    def _preprocess_3ncnn(self, nucleus_sequence: dict) -> torch.Tensor:
        """
        Preprocess nucleus data for 3ncnn model.
        Expects sequence with keys: 't-1', 't', 't+1'
        """
        try:
            volumes = []
            for timepoint in ["t-1", "t", "t+1"]:
                if timepoint in nucleus_sequence:
                    volume = nucleus_sequence[timepoint].astype(np.float32)
                    # Simple normalize and resize
                    volume = (volume - volume.mean()) / (volume.std() + 1e-8)
                    volumes.append(volume)
                else:
                    volumes.append(np.zeros(self.input_shape, dtype=np.float32))

            # Stack into [3, D, H, W] tensor and ensure correct shape
            volume_stack = np.stack(volumes, axis=0)

            # Resize to model input shape if needed
            if volume_stack.shape[1:] != self.input_shape:
                volume_stack = np.resize(volume_stack, (3,) + self.input_shape)

            return torch.from_numpy(volume_stack).float()

        except Exception as e:
            print(f"⚠️ 3ncnn preprocessing failed for nucleus: {e}")
            # Return a blank tensor with correct shape
            blank_tensor = np.zeros((3,) + self.input_shape, dtype=np.float32)
            return torch.from_numpy(blank_tensor).float()

    def _preprocess_4ncnn(self, nucleus_sequence: dict) -> torch.Tensor:
        """
        Preprocess nucleus data for 4ncnn model.
        Expects sequence with keys: 't-1', 't', 't+1', 'label'
        """
        try:
            # Import center_crop_or_pad from data_utils
            from model.4ncnn.data_utils import center_crop_or_pad

            # Collect all available volumes for min-max normalization
            raw_volumes = []
            for timepoint in ["t-1", "t", "t+1"]:
                if timepoint in nucleus_sequence:
                    raw_volumes.append(nucleus_sequence[timepoint].astype(np.float32))

            # Compute global min/max for normalization
            if raw_volumes:
                stacked = np.concatenate([v.flatten() for v in raw_volumes])
                v_min = stacked.min()
                v_max = stacked.max()
                if v_max <= v_min:
                    v_min, v_max = 0.0, 1.0
            else:
                v_min, v_max = 0.0, 1.0

            # Center crop/pad and normalize each volume
            volumes = []
            for timepoint in ["t-1", "t", "t+1"]:
                if timepoint in nucleus_sequence:
                    volume = nucleus_sequence[timepoint].astype(np.float32)
                    processed = center_crop_or_pad(volume, self.input_shape)
                    # Min-max normalization
                    if v_max > v_min:
                        processed = (processed - v_min) / (v_max - v_min)
                    else:
                        processed = processed
                    volumes.append(processed)
                else:
                    volumes.append(np.zeros(self.input_shape, dtype=np.float32))

            # Process label channel
            if "label" in nucleus_sequence:
                label_volume = nucleus_sequence["label"].astype(np.float32)
                processed_label = center_crop_or_pad(label_volume, self.input_shape)
                # Make binary mask (0/1)
                unique_vals = np.unique(processed_label)
                if set(unique_vals).issubset({0, 1}):
                    processed_label = processed_label.astype(np.float32)
                else:
                    processed_label = (processed_label > 0).astype(np.float32)
                volumes.append(processed_label)
            else:
                volumes.append(np.zeros(self.input_shape, dtype=np.float32))

            # Stack into [4, D, H, W] tensor
            volume_stack = np.stack(volumes, axis=0)
            return torch.from_numpy(volume_stack).float()

        except Exception as e:
            print(f"⚠️ 4ncnn preprocessing failed for nucleus: {e}")
            blank_tensor = np.zeros((4,) + self.input_shape, dtype=np.float32)
            return torch.from_numpy(blank_tensor).float()

    def preprocess_nucleus(self, nucleus_sequence: dict) -> torch.Tensor:
        """
        Preprocess a single nucleus sequence for model input.

        Args:
            nucleus_sequence: Dict containing volume data for different timepoints

        Returns:
            Preprocessed tensor ready for model input
        """
        return self.preprocessor(nucleus_sequence)

    def batch_predict_gpu(self, nucleus_batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Run GPU-accelerated batch prediction on multiple nuclei.

        Args:
            nucleus_batch: List of preprocessed nucleus tensors

        Returns:
            Tensor of shape [batch_size, num_classes] with softmax probabilities
        """
        if not nucleus_batch:
            return torch.empty(0, len(self.class_names))

        with torch.no_grad():
            # Stack individual nucleus tensors into batch
            batch_tensor = torch.stack(nucleus_batch).to(self.device)

            # Run model inference
            raw_outputs = self.model(batch_tensor)

            # Convert to probabilities
            probabilities = torch.softmax(raw_outputs, dim=1)

            return probabilities.cpu()  # Move back to CPU for further processing

    def predict_single(self, nucleus_sequence: dict) -> Tuple[int, str, float]:
        """
        Convenience method for predicting a single nucleus.

        Args:
            nucleus_sequence: Dict containing volume data for different timepoints

        Returns:
            Tuple of (predicted_index, predicted_class, confidence)
        """
        # Preprocess
        processed_tensor = self.preprocess_nucleus(nucleus_sequence)

        # Predict
        probabilities = self.batch_predict_gpu([processed_tensor])

        # Extract results
        confidence, predicted_idx = torch.max(probabilities[0], 0)
        predicted_class = self.class_names[predicted_idx.item()]

        return predicted_idx.item(), predicted_class, confidence.item()

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "class_names": self.class_names,
            "input_channels": self.input_channels,
            "input_shape": self.input_shape,
            "device": str(self.device),
            "batch_size": self.batch_size,
        }


# Convenience functions for easy import
def load_3ncnn_model(
    model_path: Optional[str] = None, **kwargs
) -> ModelInferenceEngine:
    """Load a 3ncnn model."""
    return ModelInferenceEngine(model_type="3ncnn", model_path=model_path, **kwargs)


def load_4ncnn_model(
    model_path: Optional[str] = None, **kwargs
) -> ModelInferenceEngine:
    """Load a 4ncnn model."""
    return ModelInferenceEngine(model_type="4ncnn", model_path=model_path, **kwargs)


if __name__ == "__main__":
    # Test the model interface
    print("Testing ModelInferenceEngine...")

    # Test 3ncnn model
    try:
        engine_3d = load_3ncnn_model()
        print("✅ 3ncnn model loaded successfully")
        print(f"Model info: {engine_3d.get_model_info()}")
    except Exception as e:
        print(f"❌ Failed to load 3ncnn model: {e}")

    # Test 4ncnn model
    try:
        engine_4d = load_4ncnn_model()
        print("✅ 4ncnn model loaded successfully")
        print(f"Model info: {engine_4d.get_model_info()}")
    except Exception as e:
        print(f"❌ Failed to load 4ncnn model: {e}")
