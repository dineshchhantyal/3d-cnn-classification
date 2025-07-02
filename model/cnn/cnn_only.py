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
import sys

# --- Configuration ---
# Group all hyperparameters into a dictionary for easy access.
HPARAMS = {
    "input_depth": 64,
    "input_height": 64,
    "input_width": 64,
    "num_classes": 3,
    "learning_rate": 1e-5,
    "batch_size": 4,
    "num_epochs": 200,
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT_DIR = "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset"
# Define a directory to save outputs
OUTPUT_DIR = "training_outputs"


# --- Model Definition: Simple 3D CNN ---
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=HPARAMS["num_classes"]):
        super(Simple3DCNN, self).__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
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
        features = features.view(features.size(0), -1)
        return self.classifier(features)


# --- Data Augmentation ---
class RandomAugmentation3D:
    def __call__(self, volume):
        if random.random() > 0.5:
            volume = np.flip(volume, axis=1).copy()
        if random.random() > 0.5:
            volume = np.flip(volume, axis=2).copy()
        angle = random.uniform(-15, 15)
        volume[0] = rotate(
            volume[0],
            angle,
            axes=(1, 2),
            reshape=False,
            order=1,
            mode="constant",
            cval=0,
        )
        return volume


# --- Data Loading ---
def resize_volume(volume, d, h, w):
    original_shape = volume.shape
    target_shape = np.array([d, h, w])
    ratios = target_shape / np.array(original_shape)
    factor = np.min(ratios)
    resized_volume = zoom(volume, factor, order=1, mode="constant", cval=0.0)
    padded_volume = np.zeros((d, h, w), dtype=volume.dtype)
    rz_shape = resized_volume.shape
    start_indices = (np.array(target_shape) - np.array(rz_shape)) // 2
    padded_volume[
        start_indices[0] : start_indices[0] + rz_shape[0],
        start_indices[1] : start_indices[1] + rz_shape[1],
        start_indices[2] : start_indices[2] + rz_shape[2],
    ] = resized_volume
    return padded_volume


class NucleusDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["mitotic", "new_daughter", "stable"]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for sample_name in os.listdir(class_dir):
                sample_path = os.path.join(class_dir, sample_name)
                if os.path.isdir(sample_path) and os.path.exists(
                    os.path.join(sample_path, "current", "raw_original.tif")
                ):
                    samples.append((sample_path, self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]
        file_path = os.path.join(sample_path, "current", "raw_original.tif")
        volume = tifffile.imread(file_path).astype(np.float32)
        volume_resized = resize_volume(
            volume,
            HPARAMS["input_depth"],
            HPARAMS["input_height"],
            HPARAMS["input_width"],
        )
        min_val, max_val = volume_resized.min(), volume_resized.max()
        if max_val > min_val:
            volume_resized = (volume_resized - min_val) / (max_val - min_val)
        volume_processed = np.expand_dims(volume_resized, axis=0)
        if self.transform:
            volume_processed = self.transform(volume_processed)
        return torch.from_numpy(volume_processed.copy()), torch.tensor(
            label, dtype=torch.long
        )


# --- Training & Validation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
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
    # Plotting Training History
    fig_metrics, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(history["train_loss"], label="Train Loss", color="blue")
    ax1.plot(history["val_loss"], label="Validation Loss", color="orange")
    ax1.set_title("Loss Over Epochs", fontsize=16)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend()
    ax1.grid(True)
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
    print(f"\n📈 Saved training metrics plot to {metrics_path}")

    # Plotting Confusion Matrix
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
    if not os.path.isdir(DATA_ROOT_DIR):
        print(f"!!! ERROR: Data directory not found at '{DATA_ROOT_DIR}'")
        print("!!! PLEASE UPDATE the 'DATA_ROOT_DIR' variable in the script. !!!")
        return

    # Create a unique directory for this run's outputs
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_output_dir = os.path.join(OUTPUT_DIR, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"🚀 Starting Run. All outputs will be saved to: {run_output_dir}")

    print(f"Using device: {DEVICE}")

    model = Simple3DCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=HPARAMS["learning_rate"])

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

    train_loader = DataLoader(
        train_dataset, batch_size=HPARAMS["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=HPARAMS["batch_size"], shuffle=False, num_workers=4
    )

    print(f"Found {len(train_full_dataset.samples)} total samples.")
    print(
        f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples."
    )
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} total parameters.")

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}
    best_val_f1 = -1

    for epoch in range(HPARAMS["num_epochs"]):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_labels, val_preds = validate(model, val_loader, criterion, DEVICE)

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"].append(val_f1)

        epoch_duration = time.time() - start_time

        print(
            f"\n--- Epoch [{epoch+1}/{HPARAMS['num_epochs']}] | Duration: {epoch_duration:.2f}s ---"
        )
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}"
        )

        # Save the model checkpoint if it's the best one so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_path = os.path.join(run_output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(
                f"🎉 New best model found! F1-Score: {best_val_f1:.4f}. Saved to {model_path}"
            )

    print("\n✅ Training finished.")

    # Save final artifacts
    print("\n💾 Saving final artifacts...")
    class_names = val_full_dataset.classes
    save_final_plots(history, val_labels, val_preds, class_names, run_output_dir)

    # Save classification report
    report = classification_report(
        val_labels, val_preds, target_names=class_names, zero_division=0
    )
    report_path = os.path.join(run_output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Hyperparameters ---\n")
        for key, value in HPARAMS.items():
            f.write(f"{key}: {value}\n")
        f.write("\n--- Environment ---\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Python Version: {platform.python_version()}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write("\n--- Classification Report ---\n")
        f.write(report)
    print(f"📝 Saved final classification report to {report_path}")


if __name__ == "__main__":
    main()
