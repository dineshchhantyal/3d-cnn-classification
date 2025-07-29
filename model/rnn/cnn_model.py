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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
import matplotlib.pyplot as plt
import platform

# Import shared modules
from config import HPARAMS, CLASS_NAMES, DEVICE
from data_utils import preprocess_sample

# Import the new SpatioTemporalModel
from model_utils import SpatioTemporalModel

DATA_ROOT_DIR = (
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3"
)
# Define a directory to save all outputs
OUTPUT_DIR = "training_outputs"


# --- Data Augmentation for Temporal Data ---
# This class is currently unused with the new model but kept for reference.
class RandomAugmentation3D:
    def __call__(self, volume_stack):
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
            # Contrast adjustment
            contrast_factor = random.uniform(0.8, 1.2)
            for i in range(3):  # Only raw channels
                mean_val = volume_stack[i].mean()
                volume_stack[i] = np.clip(
                    (volume_stack[i] - mean_val) * contrast_factor + mean_val, 0, 1
                )
        # 4. Gaussian noise - ONLY for raw channels (0,1,2), NOT binary mask
        if random.random() > 0.3:
            for i in range(3):
                noise = np.random.normal(0, 0.02, size=volume_stack[i].shape)
                volume_stack[i] = np.clip(volume_stack[i] + noise, 0, 1)

        # # 5. Brightness adjustment - ONLY for raw channels (0,1,2), NOT binary mask
        if random.random() > 0.3:
            shift = random.uniform(-0.1, 0.1)
            for i in range(3):
                volume_stack[i] = np.clip(volume_stack[i] + shift, 0, 1)

        return volume_stack


# --- Data Loading ---
class NucleusDataset(Dataset):
    """
    Dataset class that loads and reshapes data for the SpatioTemporalModel.
    """

    def __init__(self, root_dir, transform=None, max_samples_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class or HPARAMS.get(
            "max_samples_per_class"
        )
        self.classes = HPARAMS.get("classes_names", [])
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = self._make_dataset()
        self._print_class_distribution()

    def _make_dataset(self):
        # This method is unchanged
        samples = []
        class_counts = {}
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_samples = []
            for sample_name in os.listdir(class_dir):
                sample_path = os.path.join(class_dir, sample_name)
                if os.path.isdir(sample_path) and os.path.exists(
                    os.path.join(sample_path, "t", "raw_cropped.tif")
                ):
                    class_samples.append((sample_path, self.class_to_idx[class_name]))
            if self.max_samples_per_class is not None:
                limit = self.max_samples_per_class.get(class_name, len(class_samples))
                if len(class_samples) > limit:
                    random.seed(42)
                    class_samples = random.sample(class_samples, limit)
            samples.extend(class_samples)
            class_counts[class_name] = len(class_samples)
        self.class_counts = class_counts
        return samples

    def _print_class_distribution(self):
        # This method is unchanged
        print(f"Dataset loaded with {len(self.samples)} total samples:")
        for class_name, count in self.class_counts.items():
            print(f"  - {class_name}: {count} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]

        # preprocess_sample returns a tensor of shape (1, 4, D, H, W)
        volume_processed = preprocess_sample(folder_path=sample_path, for_training=True)

        # --- CRITICAL FIX: Squeeze the batch dimension added by preprocess_sample ---
        # This changes the shape from (1, 4, D, H, W) to (4, D, H, W)
        volume_processed = volume_processed.squeeze(0)

        # We only need the raw channels (t-1, t, t+1)
        # Shape: (3, D, H, W)
        raw_volumes = volume_processed[:3, :, :, :]

        # The ConvLSTM expects a sequence: (Time, Channels, D, H, W)
        # Reshape from (3, D, H, W) to (3, 1, D, H, W)
        sequence_tensor = raw_volumes.unsqueeze(1)

        # NOTE: Augmentation is disabled for this version.
        if self.transform:
            pass

        return sequence_tensor, torch.tensor(label, dtype=torch.long)


# --- Training & Validation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
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
    # This function is unchanged
    fig_metrics, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(history["train_loss"], label="Train Loss", color="blue")
    ax1.plot(history["val_loss"], label="Validation Loss", color="orange")
    ax1.set_title("Loss Over Epochs", fontsize=16)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
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
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close(fig_metrics)
    print(f"📈 Saved training metrics plot.")
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
    plt.savefig(os.path.join(output_dir, "final_confusion_matrix.png"))
    plt.close(fig_cm)
    print(f"📊 Saved final confusion matrix.")


# --- Main Execution ---
def main():
    if not os.path.isdir(DATA_ROOT_DIR):
        print(f"!!! ERROR: Data directory not found at '{DATA_ROOT_DIR}'")
        return

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_output_dir = os.path.join(OUTPUT_DIR, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"🚀 Starting Run. All outputs will be saved to: {run_output_dir}")

    print(f"⚙️ Using device: {DEVICE}")
    print(f"Hyperparameters: {HPARAMS}")

    model = SpatioTemporalModel().to(DEVICE)

    class_weights = torch.tensor(HPARAMS["class_weights"], dtype=torch.float32).to(
        DEVICE
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=HPARAMS["learning_rate"],
        weight_decay=HPARAMS.get("weight_decay", 1e-5),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10)

    print("\nLoading datasets...")
    train_full_dataset = NucleusDataset(root_dir=DATA_ROOT_DIR, transform=None)
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
    epochs_no_improve = 0
    early_stopping_patience = HPARAMS.get("early_stopping_patience", 25)

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
        scheduler.step(val_loss)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            model_path = os.path.join(run_output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"🎉 New best model found! F1-Score: {best_val_f1:.4f}. Saved.")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print(
                f"\n🛑 Early stopping triggered after {early_stopping_patience} epochs."
            )
            break
    print("\n✅ Training finished.")
    print("\n💾 Saving final artifacts...")
    class_names = val_full_dataset.classes
    best_model_path = os.path.join(run_output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path} for final evaluation.")
    _, final_labels, final_preds = validate(model, val_loader, criterion, DEVICE)
    save_final_plots(history, final_labels, final_preds, class_names, run_output_dir)
    report = classification_report(
        final_labels, final_preds, target_names=class_names, zero_division=0
    )
    report_path = os.path.join(run_output_dir, "final_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Hyperparameters ---\n")
        f.write(str(HPARAMS))
        f.write("\n--- Final Classification Report ---\n")
        f.write(report)
    print(f"📝 Saved final classification report.")
    print("\n✨ Run complete.")


if __name__ == "__main__":
    main()
