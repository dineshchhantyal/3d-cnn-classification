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
    balanced_accuracy_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
import platform
from collections import Counter, defaultdict

# --- Configuration ---
# Group all hyperparameters into a dictionary for easy access.
HPARAMS = {
    "input_depth": 64,
    "input_height": 64,
    "input_width": 64,
    "num_classes": 4,
    "learning_rate": 1e-5,
    "batch_size": 16,
    "num_epochs": 300,
    "num_input_channels": 3,  # Updated for t-1, t, t+1 volumes
    "max_samples_per_class": 230,  # Add this parameter
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT_DIR = (
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2"
)
# Define a directory to save all outputs
OUTPUT_DIR = "training_outputs"


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
    Applies the same random spatial augmentations consistently across a stack
    of 3D volumes (e.g., t-1, t, t+1).
    """

    def __call__(self, volume_stack):
        # volume_stack is a numpy array of shape (C, D, H, W), e.g., (3, 64, 64, 64)

        # Apply the same horizontal flip to all volumes in the stack
        if random.random() > 0.5:
            volume_stack = np.flip(volume_stack, axis=2).copy()  # Flip Height

        # Apply the same vertical flip to all volumes in the stack
        if random.random() > 0.5:
            volume_stack = np.flip(volume_stack, axis=3).copy()  # Flip Width

        # Apply the same rotation to all volumes in the stack
        angle = random.uniform(-15, 15)
        # Rotate each volume (channel) in the stack identically
        for i in range(volume_stack.shape[0]):
            volume_stack[i] = rotate(
                volume_stack[i],
                angle,
                axes=(1, 2),
                reshape=False,
                order=1,
                mode="constant",
                cval=0,
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
        self.classes = ["mitotic", "new_daughter", "stable", "death"]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = self._make_dataset()

        print(f"Found {len(self.samples)} samples before limiting.")

        # Apply class limits if specified
        if max_samples_per_class is not None:
            self.samples = apply_class_limits(
                self.samples, max_samples_per_class, self.classes
            )
            print(f"After limiting: {len(self.samples)} samples.")
        else:
            print(f"No class limits applied. Using all {len(self.samples)} samples.")

    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for sample_name in os.listdir(class_dir):
                sample_path = os.path.join(class_dir, sample_name)
                if os.path.isdir(sample_path) and os.path.exists(
                    os.path.join(sample_path, "t", "raw_cropped.tif")
                ):
                    samples.append((sample_path, self.class_to_idx[class_name]))
        return samples

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

        # --- Step 2: Define a reusable transformation function ---
        def transform_and_pad(volume):
            # Resize using the factor from the t volume
            resized = zoom(volume, resize_factor, order=1, mode="constant", cval=0.0)
            padded = np.zeros(target_shape, dtype=np.float32)

            # Create slices to embed the resized volume centrally
            rz_shape = resized.shape
            slices = tuple(
                slice(start, start + size)
                for start, size in zip(start_indices, rz_shape)
            )
            rz_slices = tuple(slice(0, s.stop - s.start) for s in slices)

            padded[slices] = resized[rz_slices]

            # Normalize the final padded volume
            min_val, max_val = padded.min(), padded.max()
            if max_val > min_val:
                padded = (padded - min_val) / (max_val - min_val)
            return padded

        # --- Step 3: Apply the same transformation to all volumes ---
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
                processed_vol = transform_and_pad(vol_to_process)
                all_volumes.append(processed_vol)
            else:
                # Append a blank, correctly shaped volume if the file is missing
                all_volumes.append(np.zeros(target_shape, dtype=np.float32))
                print(
                    f"Warning: Missing {tp} volume for sample '{sample_path}'. Using blank volume."
                )

        # --- Step 4: Stack and apply augmentation ---
        volume_processed = np.stack(
            all_volumes, axis=0
        )  # Stacks in [t-1, t, t+1] order

        if self.transform:
            volume_processed = self.transform(volume_processed)

        return torch.from_numpy(volume_processed.copy()), torch.tensor(
            label, dtype=torch.long
        )


# --- Focal Loss Definition ---
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha=None, gamma=2, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


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


def validate_with_detailed_metrics(model, dataloader, criterion, device, class_names):
    """Enhanced validation with per-class metrics"""
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

    # Calculate detailed metrics
    val_accuracy = accuracy_score(all_labels, all_preds)  # Add this line
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"Validation Accuracy: {val_accuracy:.4f}")  # Add this line
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    for i, class_name in enumerate(class_names):
        if i < len(per_class_f1):
            print(f"{class_name} F1: {per_class_f1[i]:.4f}")

    return (
        total_loss / len(dataloader),
        all_labels,
        all_preds,
        balanced_acc,
        macro_f1,
        val_accuracy,
    )  # Add val_accuracy to return


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
    ax2.plot(
        history["val_macro_f1"], label="Validation F1-Score (Macro)", color="red"
    )  # Fixed: use val_macro_f1 instead of val_f1
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


# --- Class Weight Calculation ---
def calculate_class_weights(samples):
    """Calculate inverse frequency weights for imbalanced classes"""

    # Count samples per class
    class_counts = Counter([sample[1] for sample in samples])
    total_samples = len(samples)
    num_classes = len(class_counts)

    # Calculate weights (inverse frequency)
    weights = {}
    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * count)

    # Convert to tensor in correct order
    weight_tensor = torch.zeros(num_classes)
    for class_idx, weight in weights.items():
        weight_tensor[class_idx] = weight

    print(
        f"Class counts: {dict(zip(['mitotic', 'new_daughter', 'stable', 'death'], [class_counts[i] for i in range(num_classes)]))}"
    )
    print(
        f"Class weights: {dict(zip(['mitotic', 'new_daughter', 'stable', 'death'], weight_tensor.tolist()))}"
    )
    return weight_tensor


def apply_class_limits(samples, max_samples_per_class, class_names):
    """Apply max samples per class limit"""
    from collections import defaultdict
    import random

    # Set random seed for reproducibility
    random.seed(42)  # Add this line for reproducible sampling

    # Group samples by class
    class_samples = defaultdict(list)
    for sample in samples:
        class_samples[sample[1]].append(sample)

    print(f"\nApplying class limits (max {max_samples_per_class} per class):")

    limited_samples = []
    for class_idx, samples_list in class_samples.items():
        original_count = len(samples_list)

        if original_count > max_samples_per_class:
            # Randomly sample without replacement
            selected_samples = random.sample(samples_list, max_samples_per_class)
            limited_samples.extend(selected_samples)
            print(
                f"  {class_names[class_idx]}: {original_count} -> {max_samples_per_class} (limited)"
            )
        else:
            limited_samples.extend(samples_list)
            print(f"  {class_names[class_idx]}: {original_count} (no change)")

    # Shuffle the final sample list
    random.shuffle(limited_samples)
    return limited_samples


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

    print("\nLoading datasets...")
    # Pass max_samples_per_class to the dataset
    train_full_dataset = NucleusDataset(
        root_dir=DATA_ROOT_DIR,
        transform=RandomAugmentation3D(),
        max_samples_per_class=HPARAMS["max_samples_per_class"],
    )
    val_full_dataset = NucleusDataset(
        root_dir=DATA_ROOT_DIR,
        transform=None,
        max_samples_per_class=HPARAMS["max_samples_per_class"],
    )

    # Calculate class weights AFTER applying limits
    class_weights = calculate_class_weights(train_full_dataset.samples)
    alpha = class_weights / class_weights.sum()  # Normalize

    # Use Focal Loss instead of CrossEntropyLoss
    criterion = FocalLoss(
        alpha=alpha.to(DEVICE), gamma=2, weight=class_weights.to(DEVICE)
    )
    optimizer = optim.Adam(model.parameters(), lr=HPARAMS["learning_rate"])

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
        "val_balanced_accuracy": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
    }
    best_val_metric = -1  # Use balanced accuracy or macro F1

    print("\n--- Starting Training ---")
    for epoch in range(HPARAMS["num_epochs"]):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        (
            val_loss,
            val_labels,
            val_preds,
            val_balanced_acc,
            val_macro_f1,
            val_accuracy,
        ) = validate_with_detailed_metrics(  # Updated to unpack val_accuracy
            model,
            val_loader,
            criterion,
            DEVICE,
            train_full_dataset.classes,  # Fixed: use train_full_dataset.classes
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_balanced_accuracy"].append(val_balanced_acc)
        history["val_macro_f1"].append(val_macro_f1)

        epoch_duration = time.time() - start_time

        print(
            f"Epoch [{epoch+1:03d}/{HPARAMS['num_epochs']}] | Duration: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val Balanced Acc: {val_balanced_acc:.4f} | Val Macro F1: {val_macro_f1:.4f}"
        )

        # Use balanced accuracy as primary metric for model selection
        current_metric = val_balanced_acc  # or val_macro_f1

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            model_path = os.path.join(run_output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(
                f"🎉 New best model found! Metric: {best_val_metric:.4f}. Saved to {model_path}"
            )

    print("\n✅ Training finished.")

    print("\n💾 Saving final artifacts...")
    class_names = (
        train_full_dataset.classes
    )  # Fixed: use train_full_dataset instead of val_full_dataset
    model.load_state_dict(torch.load(os.path.join(run_output_dir, "best_model.pth")))
    _, final_labels, final_preds, _, _, _ = (
        validate_with_detailed_metrics(  # Updated to handle new return value
            model, val_loader, criterion, DEVICE, class_names
        )
    )

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
