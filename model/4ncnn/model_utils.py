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


def run_inference(model, input_tensor: torch.Tensor, save_analysis: bool = False,
                 analysis_output_dir: str = None, sample_name: str = None):
    """
    Run inference on input tensor using pre-loaded model.
    
    Args:
        model: Pre-loaded model in eval mode
        input_tensor: Input tensor of shape [1, 4, D, H, W]
        save_analysis: Whether to save model analysis visualizations
        analysis_output_dir: Directory to save analysis outputs
        sample_name: Name of the sample for organizing outputs
    
    Returns:
        tuple: (predicted_index, predicted_class, confidence)
    """
    with torch.no_grad():
        input_tensor = input_tensor.to(DEVICE)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    
    # Save model analysis if requested
    if save_analysis and analysis_output_dir and sample_name:
        try:
            from visualization_utils import (
                create_output_structure, save_model_activations, save_prediction_analysis
            )
            
            # Create output structure
            dirs = create_output_structure(analysis_output_dir, sample_name)
            print(f"🧠 Saving model analysis to: {dirs['model_analysis']}")
            
            # Save activation maps
            save_model_activations(
                model, 
                input_tensor, 
                os.path.join(dirs['model_analysis'], 'activation_maps.png'),
                device=DEVICE
            )
            
            # Save prediction analysis
            save_prediction_analysis(
                output,
                predicted_class,
                confidence.item(),
                os.path.join(dirs['model_analysis'], 'prediction_analysis.png')
            )
            
            print(f"✅ Model analysis saved successfully!")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save model analysis: {e}")
    
    return predicted_idx.item(), predicted_class, confidence.item()
