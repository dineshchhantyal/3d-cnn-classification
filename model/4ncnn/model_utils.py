# model_utils.py
# Shared model utilities for training and prediction

import torch
import torch.nn as nn
import os
from config import HPARAMS, DEVICE, CLASS_NAMES


class Simple3DCNN(nn.Module):
    """
    A 3D CNN model that accepts a 4-channel input:
    [t-1, t, t+1, binary_segmentation_mask]
    """

    def __init__(
        self,
        in_channels=HPARAMS["num_input_channels"],
        num_classes=HPARAMS["num_classes"],
    ):
        super(Simple3DCNN, self).__init__()
        self.cnn_encoder = nn.Sequential(
            # First convolutional layer accepts 4 input channels
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.cnn_encoder(x)
        # Flatten the features for the classifier
        features = features.view(features.size(0), -1)
        return self.classifier(features)


def load_model(model_path: str, verbose: bool = True):
    """
    Load trained model from file.
    
    Args:
        model_path: Path to the .pth model file
        verbose: Whether to print loading message
    
    Returns:
        torch.nn.Module: Loaded model in eval mode
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    model = Simple3DCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    if verbose:
        print(f"🚀 Model loaded from {model_path} and running on {DEVICE}.")
    
    return model


def run_inference(model, input_tensor: torch.Tensor):
    """
    Run inference on input tensor using pre-loaded model.
    
    Args:
        model: Pre-loaded model in eval mode
        input_tensor: Input tensor of shape [1, 4, D, H, W]
    
    Returns:
        tuple: (predicted_index, predicted_class, confidence)
    """
    with torch.no_grad():
        input_tensor = input_tensor.to(DEVICE)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    return predicted_idx.item(), predicted_class, confidence.item()
