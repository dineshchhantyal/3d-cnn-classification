# model_utils.py
# Architecture for a Spatio-Temporal ConvLSTM model

import torch
import torch.nn as nn
import os
from config import HPARAMS, DEVICE, CLASS_NAMES


# --- ConvLSTM Cell ---
# This is the core building block. It's like a regular LSTM cell, but its
# internal matrix multiplications are replaced with convolutional operations.
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels

        # Padding to ensure the output has the same spatial dimensions as the input
        padding = kernel_size // 2

        # A single convolutional layer to compute all gates at once for efficiency
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,  # 4 gates: input, forget, output, cell
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state

        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_cur], dim=1)

        # Compute all gates at once
        gates = self.conv(combined)

        # Split the gates
        in_gate, forget_gate, out_gate, cell_gate = torch.chunk(gates, 4, dim=1)

        # Apply activations
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        # Update the cell state
        c_next = (forget_gate * c_cur) + (in_gate * cell_gate)

        # Update the hidden state
        h_next = out_gate * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        # Initialize hidden and cell states with zeros
        return (
            torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


# --- Spatial Encoder ---
# This CNN processes a single 3D frame to extract a 2D feature map.
# We use AdaptiveAvgPool3d to flatten the depth dimension.
class EncoderCNN(nn.Module):
    def __init__(self, in_channels=1):
        super(EncoderCNN, self).__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            # This layer collapses the depth dimension, producing a 2D feature map
            # Input: (B, 32, D, H, W) -> Output: (B, 32, 1, H, W)
            nn.AdaptiveAvgPool3d((1, None, None)),
        )

    def forward(self, x):
        features_3d = self.cnn_encoder(x)
        # Squeeze the depth dimension to get a 2D feature map for the ConvLSTM
        features_2d = features_3d.squeeze(2)
        return features_2d


# --- Main Spatio-Temporal Model ---
class SpatioTemporalModel(nn.Module):
    def __init__(
        self,
        num_classes=HPARAMS["num_classes"],
        dropout_rate=HPARAMS.get("dropout_rate", 0.5),
    ):
        super(SpatioTemporalModel, self).__init__()
        self.encoder = EncoderCNN(in_channels=1)  # Processes one raw channel at a time

        # The ConvLSTM cell will process the feature maps from the encoder
        self.conv_lstm_cell = ConvLSTMCell(
            in_channels=32, hidden_channels=64, kernel_size=3
        )

        # Global average pooling to get a final feature vector
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input x has shape: (Batch, Time, Channels, Depth, Height, Width)
        # e.g., (16, 3, 1, 32, 32, 32)

        batch_size, _, _, _, height, width = x.size()

        # Initialize the hidden and cell states for the ConvLSTM
        # The image size for the hidden state is the downsampled size after the encoder
        h, c = self.conv_lstm_cell.init_hidden(batch_size, (height // 4, width // 4))

        # Iterate through the time sequence
        for t in range(x.size(1)):
            # Get the input for the current time step and pass it through the encoder
            # The encoder expects a 5D tensor (B, C, D, H, W)
            frame_features = self.encoder(x[:, t, :, :, :, :])

            # Update the hidden and cell states with the new features
            h, c = self.conv_lstm_cell(frame_features, (h, c))

        # After the loop, 'h' is the final hidden state, summarizing the sequence
        # Pool the spatial dimensions of the final hidden state
        pooled_features = self.avg_pool(h)

        # Flatten for the classifier
        flattened = pooled_features.view(batch_size, -1)

        # Apply dropout and classify
        dropped = self.dropout(flattened)
        output = self.classifier(dropped)

        return output


def load_model(model_path: str, verbose: bool = True):
    """
    Load trained SpatioTemporalModel from file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    model = SpatioTemporalModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    if verbose:
        print(
            f"🚀 SpatioTemporalModel loaded from {model_path} and running on {DEVICE}."
        )

    return model


def run_inference(
    model,
    input_tensor: torch.Tensor,
    save_analysis: bool = False,
    analysis_output_dir: str = None,
    sample_name: str = None,
):
    """
    Run inference on input tensor using pre-loaded model.
    NOTE: This function is a placeholder and would need to be adapted
    for the new model's input shape and analysis tools.
    """
    with torch.no_grad():
        input_tensor = input_tensor.to(DEVICE)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]

    print(
        f"Inference complete. Predicted: {predicted_class} with confidence {confidence.item():.4f}"
    )

    return predicted_idx.item(), predicted_class, confidence.item()
