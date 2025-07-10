import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import time
import os
import tifffile
from scipy.ndimage import zoom, rotate
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
import platform

# --- Configuration ---
# Group all hyperparameters into a dictionary for easy access.
HPARAMS = {
    "input_depth": 64,
    "input_height": 64,
    "input_width": 64,
    "num_classes": 3,
    "classes_names": ["mitotic", "new_daughter", "stable"],
    "learning_rate": 1e-5,
    "batch_size": 16,
    "num_epochs": 300,
    "num_input_channels": 4,  # Updated from 3 to 4 channels: [t-1, t, t+1, segmentation_mask]
    "max_samples_per_class": 216,
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT_DIR = "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2"
# Define a directory to save all outputs
OUTPUT_DIR = "four-channels"


# --- Model Definition: 3D CNN for 3-Channel Input ---
class Simple3DCNN(nn.Module):
    """
    A 3D CNN model that accepts a 3-channel input, where each channel represents
    a different time point (t-1, t, t+1).
    """

    def __init__(
        self,
        in_channels=HPARAMS["num_input_channels"],
        num_classes=HPARAMS["num_classes"],
    ):
        super(Simple3DCNN, self).__init__()
        self.cnn_encoder = nn.Sequential(
            # The first convolutional layer now accepts 3 input channels.
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


# --- Data Augmentation for Temporal Data ---
class RandomAugmentation3D:
    """
    Conservative augmentation that maintains spatial consistency between 
    temporal channels and binary segmentation mask.
    """

    def __call__(self, volume_stack):
        # volume_stack shape: (4, D, H, W) = [t-1, t, t+1, binary_mask]
        
        # 1. Safe transformations - apply to all channels identically
        
        # Horizontal flip (safe)
        if random.random() > 0.5:
            volume_stack = np.flip(volume_stack, axis=2).copy()  # Flip Height
        
        # Vertical flip (safe)  
        if random.random() > 0.5:
            volume_stack = np.flip(volume_stack, axis=3).copy()  # Flip Width
        
        # Depth flip (safe for 3D)
        if random.random() > 0.5:
            volume_stack = np.flip(volume_stack, axis=1).copy()  # Flip Depth
        
        # 2. Conservative rotation - smaller angles to minimize interpolation issues
        if random.random() > 0.5:  # Only rotate 50% of the time
            angle = random.uniform(-5, 5)  # Reduced from -15,15 to -5,5 degrees
            
            # Apply same rotation to all channels
            for i in range(volume_stack.shape[0]):
                if i == 3:  # Binary mask - use nearest neighbor
                    volume_stack[i] = rotate(
                        volume_stack[i],
                        angle,
                        axes=(1, 2),  # H, W plane
                        reshape=False,
                        order=0,  # Nearest neighbor
                        mode="constant",
                        cval=0,
                    )
                else:  # Raw volumes - use linear interpolation
                    volume_stack[i] = rotate(
                        volume_stack[i],
                        angle,
                        axes=(1, 2),
                        reshape=False,
                        order=1,  # Linear interpolation
                        mode="constant",
                        cval=0,
                    )
        
        # 3. Intensity augmentation - ONLY for raw channels (0,1,2), NOT binary mask
        if random.random() > 0.3:  # Apply to 70% of samples
            # Brightness adjustment
            brightness_factor = random.uniform(0.8, 1.2)
            volume_stack[:3] = np.clip(volume_stack[:3] * brightness_factor, 0, 1)
        
        if random.random() > 0.3:  # Apply to 70% of samples  
            # Contrast adjustment
            contrast_factor = random.uniform(0.8, 1.2)
            for i in range(3):  # Only raw channels
                mean_val = volume_stack[i].mean()
                volume_stack[i] = np.clip(
                    (volume_stack[i] - mean_val) * contrast_factor + mean_val, 0, 1
                )
        
        return volume_stack


# --- Data Loading ---
class NucleusDataset(Dataset):
    """
    Dataset class that loads t-1, t, and t+1 volumes for each sample.
    The key logic is that resizing and padding for all three volumes are determined
    by the dimensions of the 't' volume to ensure spatial consistency.
    """

    def __init__(self, root_dir, transform=None, max_samples_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class or HPARAMS.get("max_samples_per_class")
        self.classes = HPARAMS.get("classes_names", [])
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = self._make_dataset()
        
        # Print class distribution after limiting
        self._print_class_distribution()

    def _make_dataset(self):
        samples = []
        class_counts = {}
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                class_counts[class_name] = 0
                continue
                
            # Collect all valid samples for this class
            class_samples = []
            for sample_name in os.listdir(class_dir):
                sample_path = os.path.join(class_dir, sample_name)
                if os.path.isdir(sample_path) and os.path.exists(
                    os.path.join(sample_path, "t", "raw_cropped.tif")
                ):
                    class_samples.append((sample_path, self.class_to_idx[class_name]))
            
            # Apply per-class sample limiting with random selection
            if self.max_samples_per_class is not None and len(class_samples) > self.max_samples_per_class:
                # Randomly select up to max_samples_per_class samples
                import random
                random.seed(42)  # For reproducibility
                class_samples = random.sample(class_samples, self.max_samples_per_class)
            
            samples.extend(class_samples)
            class_counts[class_name] = len(class_samples)
        
        self.class_counts = class_counts
        return samples

    def _print_class_distribution(self):
        """Print the number of samples per class after limiting."""
        print(f"Dataset loaded with {len(self.samples)} total samples:")
        for class_name in self.classes:
            count = self.class_counts.get(class_name, 0)
            print(f"  - {class_name}: {count} samples")
        
        if self.max_samples_per_class is not None:
            print(f"  Applied per-class limit: {self.max_samples_per_class} samples/class")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]

        target_shape = (
            HPARAMS["input_depth"],
            HPARAMS["input_height"],
            HPARAMS["input_width"],
        )

        # --- Step 1: Load t volume to determine transformation params ---
        t_path = os.path.join(sample_path, "t", "raw_cropped.tif")
        t_volume = tifffile.imread(t_path).astype(np.float32)

        # Calculate resize factor and padding based *only* on the t volume
        original_shape = t_volume.shape
        ratios = np.array(target_shape) / np.array(original_shape)
        resize_factor = np.min(ratios)

        resized_shape = (np.array(original_shape) * resize_factor).astype(int)
        start_indices = (np.array(target_shape) - resized_shape) // 2

        # --- Step 2: Extract nucleus ID from folder name ---
        folder_name = os.path.basename(sample_path)
        # Extract nucleus ID from folder name pattern like "230212_stack6_frame_194_nucleus_077_count_11"
        nucleus_id = None
        parts = folder_name.split('_')
        for i, part in enumerate(parts):
            if part == 'nucleus' and i + 1 < len(parts):
                try:
                    nucleus_id = int(parts[i + 1])
                    break
                except ValueError:
                    continue
    
        if nucleus_id is None:
            print(f"Warning: Could not extract nucleus ID from folder name: {folder_name}")

        # --- Step 3: Define a reusable transformation function ---
        def transform_and_pad(volume, is_label=False, target_nucleus_id=None):
            # Resize using the factor from the t volume
            resized = zoom(volume, resize_factor, order=0 if is_label else 1, mode="constant", cval=0.0)
            padded = np.zeros(target_shape, dtype=np.float32)

            # Create slices to embed the resized volume centrally
            rz_shape = resized.shape
            slices = tuple(
                slice(start, start + size)
                for start, size in zip(start_indices, rz_shape)
            )
            rz_slices = tuple(slice(0, s.stop - s.start) for s in slices)

            padded[slices] = resized[rz_slices]

            # Handle different volume types
            if not is_label:
                # Normalize raw volumes
                min_val, max_val = padded.min(), padded.max()
                if max_val > min_val:
                    padded = (padded - min_val) / (max_val - min_val)
            else:
                # For labels, create binary mask for specific nucleus
                if target_nucleus_id is not None:
                    padded = (padded == target_nucleus_id).astype(np.float32)
                else:
                    # Fallback: any non-zero value becomes 1
                    padded = (padded > 0).astype(np.float32)
        
            return padded

        # --- Step 4: Apply the same transformation to all volumes ---
        all_volumes = []
        time_points = ["t-1", "t", "t+1"]

        for tp in time_points:
            file_path = os.path.join(sample_path, tp, "raw_cropped.tif")
            if os.path.exists(file_path):
                vol_to_process = (
                    t_volume
                    if tp == "t"
                    else tifffile.imread(file_path).astype(np.float32)
                )
                processed_vol = transform_and_pad(vol_to_process, is_label=False)
                all_volumes.append(processed_vol)
            else:
                # Append a blank, correctly shaped volume if the file is missing
                all_volumes.append(np.zeros(target_shape, dtype=np.float32))
                print(f"The timestamp {tp} for {sample_path} is missing")

        # --- Step 5: Load and process binary segmentation mask for current timestamp (t) ---
        label_path = os.path.join(sample_path, "t", "label_cropped.tif")
        if os.path.exists(label_path):
            label_volume = tifffile.imread(label_path).astype(np.float32)
            # Create binary mask for the specific nucleus
            processed_label = transform_and_pad(label_volume, is_label=True, target_nucleus_id=nucleus_id)
        else:
            # Create empty mask if label file is missing
            processed_label = np.zeros(target_shape, dtype=np.float32)
            print(f"Label file missing for {sample_path}")

        # Add the binary segmentation mask as the 4th channel
        all_volumes.append(processed_label)

        # --- Step 6: Stack and apply augmentation ---
        volume_processed = np.stack(
            all_volumes, axis=0
        )  # Stacks in [t-1, t, t+1, binary_nucleus_mask] order

        if self.transform:
            volume_processed = self.transform(volume_processed)

        return torch.from_numpy(volume_processed.copy()), torch.tensor(
            label, dtype=torch.long
        )


# --- Training & Validation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch and returns loss and accuracy."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), train_acc


def validate(model, dataloader, criterion, device):
    """Validates the model and returns loss, labels, and predictions."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(dataloader), all_labels, all_preds


# --- Plotting & Artifact Generation ---
def save_final_plots(history, val_labels, val_preds, class_names, output_dir):
    """Generates and saves final plots to the output directory."""
    fig_metrics, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(history["train_loss"], label="Train Loss", color="blue")
    ax1.plot(history["val_loss"], label="Validation Loss", color="orange")
    ax1.set_title("Loss Over Epochs", fontsize=16)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend()
    ax1.grid(True)

    ax2.plot(
        history["train_accuracy"],
        label="Train Accuracy",
        color="purple",
        linestyle="--",
    )
    ax2.plot(history["val_accuracy"], label="Validation Accuracy", color="green")
    ax2.plot(history["val_f1"], label="Validation F1-Score (Macro)", color="red")
    ax2.set_title("Performance Over Epochs", fontsize=16)
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    metrics_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(metrics_path)
    plt.close(fig_metrics)
    print(f"📈 Saved training metrics plot to {metrics_path}")

    fig_cm, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(val_labels, val_preds, labels=range(len(class_names)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Final Confusion Matrix", fontsize=16)
    cm_path = os.path.join(output_dir, "final_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close(fig_cm)
    print(f"📊 Saved final confusion matrix to {cm_path}")


# --- Main Execution ---
def main():
    """Main function to orchestrate the training and validation process."""
    if not os.path.isdir(DATA_ROOT_DIR):
        print(f"!!! ERROR: Data directory not found at '{DATA_ROOT_DIR}'")
        print("!!! PLEASE UPDATE the 'DATA_ROOT_DIR' variable in the script. !!!")
        return

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_output_dir = os.path.join(OUTPUT_DIR, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"🚀 Starting Run. All outputs will be saved to: {run_output_dir}")

    print(f"⚙️ Using device: {DEVICE}")
    print(f"Hyperparameters: {HPARAMS}")

    model = Simple3DCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=HPARAMS["learning_rate"])

    print("\nLoading datasets...")
    train_full_dataset = NucleusDataset(
        root_dir=DATA_ROOT_DIR, transform=RandomAugmentation3D()
    )
    val_full_dataset = NucleusDataset(root_dir=DATA_ROOT_DIR, transform=None)

    labels = [sample[1] for sample in train_full_dataset.samples]
    indices = list(range(len(train_full_dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)

    num_workers = 2 if platform.system() == "Linux" else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=HPARAMS["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=HPARAMS["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )

    print(
        f"\nTraining on {len(train_dataset)} samples, validating on {len(val_dataset)} samples."
    )
    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters."
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }
    best_val_f1 = -1

    print("\n--- Starting Training ---")
    for epoch in range(HPARAMS["num_epochs"]):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_labels, val_preds = validate(model, val_loader, criterion, DEVICE)

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"].append(val_f1)

        epoch_duration = time.time() - start_time

        print(
            f"Epoch [{epoch+1:03d}/{HPARAMS['num_epochs']}] | Duration: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_path = os.path.join(run_output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(
                f"🎉 New best model found! F1-Score: {best_val_f1:.4f}. Saved to {model_path}"
            )

    print("\n✅ Training finished.")

    print("\n💾 Saving final artifacts...")
    class_names = val_full_dataset.classes
    model.load_state_dict(torch.load(os.path.join(run_output_dir, "best_model.pth")))
    _, final_labels, final_preds = validate(model, val_loader, criterion, DEVICE)

    save_final_plots(history, final_labels, final_preds, class_names, run_output_dir)

    report = classification_report(
        final_labels, final_preds, target_names=class_names, zero_division=0
    )
    report_path = os.path.join(run_output_dir, "final_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Hyperparameters ---\n")
        for key, value in HPARAMS.items():
            f.write(f"{key}: {value}\n")
        f.write("\n--- Environment ---\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Python Version: {platform.python_version()}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write("\n--- Final Classification Report (from best model) ---\n")
        f.write(report)
    print(f"📝 Saved final classification report to {report_path}")
    print("\n✨ Run complete.")


if __name__ == "__main__":
    main()
