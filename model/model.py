import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
import tifffile
from scipy.ndimage import zoom  # For resizing: pip install scipy
import random
from scipy.ndimage import rotate


# --- Configuration ---
# Adjust these parameters based on your specific data and resources.
# Data params
# The model requires every input to have the exact same dimensions.
INPUT_DEPTH = 64  # Target depth after resizing/padding
INPUT_HEIGHT = 64  # Target height after resizing/padding
INPUT_WIDTH = 64  # Target width after resizing/padding
NUM_TIMESTEPS = 3  # We have 3 timestamps
NUM_CLASSES = 3  # [mitotic, new_daughter, stable, death]

# Training params
LEARNING_RATE = 0.00001
BATCH_SIZE = 4
NUM_EPOCHS = 300  # Increase for real training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Config:
    """
    Configuration class to hold global parameters.
    This can be extended to include more parameters as needed.
    """

    def __init__(self):
        self.input_depth = INPUT_DEPTH
        self.input_height = INPUT_HEIGHT
        self.input_width = INPUT_WIDTH
        self.num_timesteps = NUM_TIMESTEPS
        self.num_classes = NUM_CLASSES
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.num_epochs = NUM_EPOCHS
        self.device = DEVICE


# --- Model Definition: 3D Conv-RNN (Convolutional Recurrent Neural Network) ---

class ConvRNN(nn.Module): 
    """
    A SIMPLER version of the hybrid model.
    """
    # <<< CHANGE: Reduced default hidden_size and num_layers
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=128, num_lstm_layers=1):
        super(ConvRNN, self).__init__()

        # --- Part 1: SIMPLER 3D CNN Encoder ---
        self.cnn_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(16),
            
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            nn.AdaptiveAvgPool3d(1),
        )

        # <<< CHANGE: The output from the CNN is now 32
        cnn_output_features = 32

        # --- Part 2: SIMPLER LSTM Decoder ---
        self.rnn_decoder = nn.LSTM(
            input_size=cnn_output_features,
            hidden_size=hidden_size,       # Now defaults to 128
            num_layers=num_lstm_layers, # Now defaults to 1
            batch_first=True,
            # Dropout is automatically 0 if num_layers is 1
            dropout=0.5 if num_lstm_layers > 1 else 0,
        )

        # --- Part 3: Final classifier layer ---
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # The forward pass logic remains exactly the same
        batch_size, timesteps, C, D, H, W = x.size()
        cnn_features_list = []
        for t in range(timesteps):
            volume_t = x[:, t, :, :, :, :]
            features = self.cnn_encoder(volume_t)
            features = features.view(batch_size, -1)
            cnn_features_list.append(features)
        cnn_sequence = torch.stack(cnn_features_list, dim=1)
        lstm_out, (h_n, c_n) = self.rnn_decoder(cnn_sequence)
        last_hidden_state = h_n[-1, :, :]
        output = self.classifier(last_hidden_state)
        return output

# --- Data Loading and Preprocessing ---


def resize_volume(volume, target_depth, target_height, target_width):
    """
    Resize a 3D volume to a target shape using interpolation.
    """
    d, h, w = volume.shape
    depth_factor = target_depth / d
    height_factor = target_height / h
    width_factor = target_width / w
    return zoom(volume, (depth_factor, height_factor, width_factor), order=1)


class NucleusDataset(Dataset):
    """
    Loads nucleus data from the specified directory structure.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # self.classes = ["mitotic", "new_daughter", "stable", "death"]
        self.classes = ["mitotic", "new_daughter", "stable"]
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue

            for sample_name in os.listdir(class_dir):
                sample_path = os.path.join(class_dir, sample_name)
                # Ensure it's a directory
                if os.path.isdir(sample_path):
                    # Check if it contains the required subfolders
                    if all(
                        os.path.isdir(os.path.join(sample_path, f))
                        for f in ["previous", "current", "next"]
                    ):
                        item = (sample_path, self.class_to_idx[class_name])
                        samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]

        # Define paths for the three time points
        time_points = ["previous", "current", "next"]
        volume_files = [
            os.path.join(sample_path, tp, "raw_original.tif") for tp in time_points
        ]

        volumes = []
        for f_path in volume_files:
            try:
                # Load the 3D TIF file
                vol = tifffile.imread(f_path)

                # IMPORTANT: Resize volume to a uniform size
                vol_resized = resize_volume(vol, INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH)
                volumes.append(vol_resized)
            except FileNotFoundError:
                print(f"Error: File not found {f_path}. Skipping sample.")
                # Return a dummy sample or handle error appropriately
                return self.__getitem__((idx + 1) % len(self))

        # Stack the volumes along the time dimension
        # Shape: (time, depth, height, width)
        sequence = np.stack(volumes, axis=0).astype(np.float32)

        # Normalize the intensity values to [0, 1]
        # This is a simple normalization. You might need a more robust method.
        min_val, max_val = sequence.min(), sequence.max()
        if max_val > min_val:
            sequence = (sequence - min_val) / (max_val - min_val)

        # Add a channel dimension for the model
        # Shape: (time, channels, depth, height, width)
        sequence = np.expand_dims(sequence, axis=1)

        # Apply transforms (data augmentation) if any
        if self.transform:
            sequence = self.transform(sequence)

        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)


class RandomAugmentation3D:
    """
    Applies random 3D augmentations to a sequence of volumes.
    """
    def __call__(self, sequence):
        # The input `sequence` has shape (time, channels, depth, height, width)
        
        # 50% chance of flipping along the z-axis (depth)
        if random.random() > 0.5:
            sequence = np.flip(sequence, axis=2).copy()
            
        # 50% chance of flipping along the y-axis (height)
        if random.random() > 0.5:
            sequence = np.flip(sequence, axis=3).copy()

        # Apply a random rotation between -15 and 15 degrees
        # We apply the same rotation to each volume in the sequence
        angle = random.uniform(-15, 15)
        
        # Reshape=False and order=1 (bilinear) are important to prevent size changes and artifacts
        # We must loop through the time dimension to rotate each volume
        for t in range(sequence.shape[0]):
             # Note: axes=(1, 2) corresponds to the height and width dimensions for rotation
            sequence[t, 0] = rotate(sequence[t, 0], angle, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0)

        return sequence

# --- Training Loop ---


def main():
    DATA_ROOT_DIR = (
        "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset"  #
    )

    if not os.path.isdir(DATA_ROOT_DIR):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE the 'DATA_ROOT_DIR' variable in main() !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    print(f"Using device: {DEVICE}")

    # 1. Initialize Model, Loss, and Optimizer
    model = ConvRNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 2. Prepare DataLoaders
    # This now uses the real dataset loader
    full_dataset = NucleusDataset(root_dir=DATA_ROOT_DIR)

    # Create a train/validation split (e.g., 80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    print(f"Found {len(full_dataset)} total samples.")
    print(
        f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples."
    )
    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters."
    )

    # 3. Training
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # 4. Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        epoch_duration = time.time() - start_time

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}% | "
            f"Duration: {epoch_duration:.2f}s"
        )

    # 6. Final model save
    final_model_path = "final_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
