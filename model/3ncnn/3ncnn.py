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
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration ---
# Group all hyperparameters into a dictionary for easy access.
HPARAMS = {
    "input_depth": 64,
    "input_height": 64,
    "input_width": 64,
    "num_classes": 4,
    "learning_rate": 1e-5,  # Consider lower for imbalanced datasets
    "batch_size": 32,  # Reduced batch size for better minority class representation
    "num_epochs": 300,  # Increased epochs for minority class learning
    "num_input_channels": 3, # Updated for t-1, t, t+1 volumes
    "patience": 50,  # Early stopping patience
    "max_samples_per_class": 2000,  # Conservative limit for majority classes
    "preserve_minority_classes": [0, 3],  # Always keep all mitotic and death samples
    "random_seed": 42,  # For reproducible sampling
    "use_gpu_augmentation": True,  # Use GPU-optimized augmentation when available
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT_DIR = "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2"
# Define a directory to save all outputs
OUTPUT_DIR = "training_outputs"


# --- Model Definition: 3D CNN for 3-Channel Input ---
class Simple3DCNN(nn.Module):
    """
    A 3D CNN model that accepts a 3-channel input, where each channel represents
    a different time point (t-1, t, t+1).
    """
    def __init__(self, in_channels=HPARAMS["num_input_channels"], num_classes=HPARAMS["num_classes"]):
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
    Enhanced for imbalanced datasets with more aggressive augmentation for minority classes.
    """
    def __init__(self, minority_classes=[0, 3], augment_prob=0.8):  # mitotic=0, death=3
        self.minority_classes = minority_classes
        self.augment_prob = augment_prob
        
    def __call__(self, volume_stack, label=None):
        # volume_stack is a numpy array of shape (C, D, H, W), e.g., (3, 64, 64, 64)
        
        # More aggressive augmentation for minority classes
        if label is not None and label in self.minority_classes:
            augment_prob = self.augment_prob
        else:
            augment_prob = 0.5
        
        # Apply the same horizontal flip to all volumes in the stack
        if random.random() > (1 - augment_prob):
            volume_stack = np.flip(volume_stack, axis=2).copy()  # Flip Height
            
        # Apply the same vertical flip to all volumes in the stack
        if random.random() > (1 - augment_prob):
            volume_stack = np.flip(volume_stack, axis=3).copy()  # Flip Width
            
        # Apply depth flip for minority classes
        if label is not None and label in self.minority_classes and random.random() > 0.5:
            volume_stack = np.flip(volume_stack, axis=1).copy()  # Flip Depth
            
        # Apply the same rotation to all volumes in the stack
        if random.random() > (1 - augment_prob):
            angle = random.uniform(-20, 20) if label in self.minority_classes else random.uniform(-15, 15)
            # Rotate each volume (channel) in the stack identically
            for i in range(volume_stack.shape[0]):
                 volume_stack[i] = rotate(volume_stack[i], angle, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0)
        
        # Add noise for minority classes
        if label is not None and label in self.minority_classes and random.random() > 0.7:
            noise = np.random.normal(0, 0.02, volume_stack.shape).astype(np.float32)
            volume_stack = volume_stack + noise
            volume_stack = np.clip(volume_stack, 0, 1)
             
        return volume_stack


# --- GPU-Optimized Data Augmentation ---
class TensorAugmentation3D:
    """
    GPU-optimized data augmentation using PyTorch tensors.
    Performs augmentations on GPU for better performance.
    """
    def __init__(self, minority_classes=[0, 3], augment_prob=0.8, device='cuda'):
        self.minority_classes = minority_classes
        self.augment_prob = augment_prob
        self.device = device
        
    def __call__(self, volume_tensor, label=None):
        # volume_tensor should already be on the correct device when called
        # This will be called from the training loop, not in DataLoader workers
        
        # More aggressive augmentation for minority classes
        if label is not None and label in self.minority_classes:
            augment_prob = self.augment_prob
        else:
            augment_prob = 0.5
        
        # Apply horizontal flip
        if torch.rand(1).item() > (1 - augment_prob):
            volume_tensor = torch.flip(volume_tensor, dims=[2])  # Flip Height
            
        # Apply vertical flip
        if torch.rand(1).item() > (1 - augment_prob):
            volume_tensor = torch.flip(volume_tensor, dims=[3])  # Flip Width
            
        # Apply depth flip for minority classes
        if label is not None and label in self.minority_classes and torch.rand(1).item() > 0.5:
            volume_tensor = torch.flip(volume_tensor, dims=[1])  # Flip Depth
            
        # Add noise for minority classes (GPU accelerated)
        if label is not None and label in self.minority_classes and torch.rand(1).item() > 0.7:
            noise = torch.normal(0, 0.02, volume_tensor.shape, device=volume_tensor.device)
            volume_tensor = volume_tensor + noise
            volume_tensor = torch.clamp(volume_tensor, 0, 1)
             
        return volume_tensor


# --- Data Loading ---
class NucleusDataset(Dataset):
    """
    Dataset class that loads t-1, t, and t+1 volumes for each sample.
    The key logic is that resizing and padding for all three volumes are determined
    by the dimensions of the 't' volume to ensure spatial consistency.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ["mitotic", "new_daughter", "stable", "death"]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = self._make_dataset()
        print(f"\n✅ Dataset ready with {len(self.samples)} balanced samples.")

    def _make_dataset(self):
        samples = []
        class_counts = {class_name: 0 for class_name in self.classes}
        
        # Set random seed for reproducible sampling
        np.random.seed(HPARAMS["random_seed"])
        
        # First pass: collect all samples by class
        samples_by_class = {class_name: [] for class_name in self.classes}
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir): 
                continue
                
            class_samples = []
            for sample_name in os.listdir(class_dir):
                sample_path = os.path.join(class_dir, sample_name)
                if os.path.isdir(sample_path) and os.path.exists(os.path.join(sample_path, "t", "raw_image_cropped.tif")):
                    class_samples.append((sample_path, self.class_to_idx[class_name]))
            
            samples_by_class[class_name] = class_samples
            class_counts[class_name] = len(class_samples)
        
        print("\n📊 Original class distribution:")
        for class_name in self.classes:
            print(f"  {class_name}: {class_counts[class_name]:,} samples")
        
        # Second pass: apply sampling limits
        final_samples = []
        final_counts = {class_name: 0 for class_name in self.classes}
        
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            available_samples = samples_by_class[class_name]
            
            # Check if this is a minority class that should be preserved
            if class_idx in HPARAMS["preserve_minority_classes"]:
                # Keep all samples from minority classes
                selected_samples = available_samples
                print(f"  ✅ {class_name}: Preserving all {len(available_samples)} samples (minority class)")
            else:
                # Apply limit to majority classes
                max_samples = min(len(available_samples), HPARAMS["max_samples_per_class"])
                if len(available_samples) > max_samples:
                    # Randomly sample from available samples using proper indexing
                    indices = np.random.choice(
                        len(available_samples), 
                        size=max_samples, 
                        replace=False
                    )
                    selected_samples = [available_samples[i] for i in indices]
                    print(f"  📉 {class_name}: Limited from {len(available_samples):,} to {max_samples:,} samples")
                else:
                    selected_samples = available_samples
                    print(f"  ✅ {class_name}: Keeping all {len(available_samples)} samples (under limit)")
            
            final_samples.extend(selected_samples)
            final_counts[class_name] = len(selected_samples)
        
        print("\n📊 Final balanced class distribution:")
        total_samples = sum(final_counts.values())
        for class_name in self.classes:
            percentage = (final_counts[class_name] / total_samples) * 100
            print(f"  {class_name}: {final_counts[class_name]:,} samples ({percentage:.1f}%)")
        
        print(f"\n🎯 Total samples reduced from {sum(class_counts.values()):,} to {total_samples:,} "
              f"({((total_samples/sum(class_counts.values()))*100):.1f}% of original)")
        
        return final_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]
        
        target_shape = (HPARAMS["input_depth"], HPARAMS["input_height"], HPARAMS["input_width"])

        # --- Step 1: Load t volume to determine transformation params ---
        t_path = os.path.join(sample_path, "t", "raw_image_cropped.tif")
        t_volume = tifffile.imread(t_path).astype(np.float32)
        
        # Calculate resize factor and padding based *only* on the t volume
        original_shape = t_volume.shape
        ratios = np.array(target_shape) / np.array(original_shape)
        resize_factor = np.min(ratios)
        
        resized_shape = (np.array(original_shape) * resize_factor).astype(int)
        start_indices = (np.array(target_shape) - resized_shape) // 2

        # --- Step 2: Define a reusable transformation function (optimized) ---
        def transform_and_pad(volume):
            # Convert to tensor early for potential GPU acceleration
            volume_tensor = torch.from_numpy(volume).float()
            
            # Resize using zoom (still CPU-bound but optimized)
            volume_np = volume_tensor.numpy()
            resized = zoom(volume_np, resize_factor, order=1, mode='constant', cval=0.0)
            
            # Convert back to tensor for padding operations
            resized_tensor = torch.from_numpy(resized).float()
            padded = torch.zeros(target_shape, dtype=torch.float32)
            
            # Create slices to embed the resized volume centrally
            rz_shape = resized_tensor.shape
            slices = tuple(slice(start, start + size) for start, size in zip(start_indices, rz_shape))
            rz_slices = tuple(slice(0, s.stop - s.start) for s in slices)
            
            padded[slices] = resized_tensor[rz_slices]
            
            # Normalize using tensor operations
            min_val, max_val = padded.min(), padded.max()
            if max_val > min_val:
                padded = (padded - min_val) / (max_val - min_val)
            
            return padded.numpy()  # Return numpy for stacking, will convert to tensor later

        # --- Step 3: Apply the same transformation to all volumes ---
        all_volumes = []
        time_points = ["t-1", "t", "t+1"]
        
        for tp in time_points:
            file_path = os.path.join(sample_path, tp, "raw_image_cropped.tif")
            if os.path.exists(file_path):
                vol_to_process = t_volume if tp == "t" else tifffile.imread(file_path).astype(np.float32)
                processed_vol = transform_and_pad(vol_to_process)
                all_volumes.append(processed_vol)
            else:
                # Append a blank, correctly shaped volume if the file is missing
                all_volumes.append(np.zeros(target_shape, dtype=np.float32))

        # --- Step 4: Stack and convert to tensor early ---
        volume_stack = np.stack(all_volumes, axis=0)  # Shape: (C, D, H, W)
        volume_tensor = torch.from_numpy(volume_stack.copy()).float()

        # Apply augmentation (only CPU-based in DataLoader to avoid CUDA conflicts)
        if self.transform and not hasattr(self.transform, 'device'):
            # CPU-based augmentation (safe for DataLoader workers)
            volume_np = self.transform(volume_stack, label)
            volume_tensor = torch.from_numpy(volume_np.copy()).float()
        else:
            # No augmentation in DataLoader - will be done in training loop
            volume_tensor = torch.from_numpy(volume_stack.copy()).float()
            
        return volume_tensor, torch.tensor(label, dtype=torch.long)


# --- Focal Loss for Imbalanced Data ---
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses learning on hard-to-classify examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# --- Training & Validation Functions ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, gpu_augmentation=None):
    """Trains the model for one epoch and returns loss and accuracy."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for inputs, labels in dataloader:
        # Ensure tensors are on the correct device
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Apply GPU augmentation if available (after moving to GPU)
        if gpu_augmentation is not None:
            # Apply augmentation to each sample in the batch
            augmented_inputs = []
            for i in range(inputs.shape[0]):
                augmented_sample = gpu_augmentation(inputs[i], labels[i].item())
                augmented_inputs.append(augmented_sample)
            inputs = torch.stack(augmented_inputs)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Move predictions to CPU for metrics calculation
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Clear cache if using GPU
        if device == "cuda":
            torch.cuda.empty_cache()

    train_acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), train_acc


def validate(model, dataloader, criterion, device):
    """Validates the model and returns loss, labels, and predictions."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
    
    ax2.plot(history['train_accuracy'], label='Train Accuracy', color='purple', linestyle='--')
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
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

    # Calculate class weights for imbalanced dataset
    train_full_dataset = NucleusDataset(root_dir=DATA_ROOT_DIR, transform=None)
    labels = [sample[1] for sample in train_full_dataset.samples]
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(labels), 
        y=labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    print(f"Class distribution: {np.bincount(labels)}")
    print(f"Class weights: {class_weights}")

    model = Simple3DCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # Alternative: Use Focal Loss for even better handling of class imbalance
    # criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=HPARAMS["learning_rate"])

    print("\nLoading datasets...")
    # Choose augmentation strategy based on configuration
    if HPARAMS["use_gpu_augmentation"] and DEVICE == "cuda":
        print("🚀 Using GPU-accelerated augmentation (applied in training loop)")
        cpu_augmentation = RandomAugmentation3D()  # Fallback for DataLoader
        gpu_augmentation = TensorAugmentation3D(device=DEVICE)
    else:
        print("💻 Using CPU-based augmentation")
        cpu_augmentation = RandomAugmentation3D()
        gpu_augmentation = None
        
    train_full_dataset = NucleusDataset(root_dir=DATA_ROOT_DIR, transform=cpu_augmentation)
    val_full_dataset = NucleusDataset(root_dir=DATA_ROOT_DIR, transform=None)

    labels = [sample[1] for sample in train_full_dataset.samples]
    indices = list(range(len(train_full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)

    num_workers = 0  # Disable multiprocessing to avoid CUDA conflicts
    train_loader = DataLoader(train_dataset, batch_size=HPARAMS["batch_size"], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=HPARAMS["batch_size"], shuffle=False, num_workers=num_workers)

    print(f"\nTraining on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [], "val_f1": []}
    best_val_f1 = -1
    patience_counter = 0

    print("\n--- Starting Training ---")
    for epoch in range(HPARAMS["num_epochs"]):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, gpu_augmentation)
        val_loss, val_labels, val_preds = validate(model, val_loader, criterion, DEVICE)

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_f1"].append(val_f1)

        epoch_duration = time.time() - start_time

        print(f"Epoch [{epoch+1:03d}/{HPARAMS['num_epochs']}] | Duration: {epoch_duration:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model_path = os.path.join(run_output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"🎉 New best model found! F1-Score: {best_val_f1:.4f}. Saved to {model_path}")
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= HPARAMS["patience"]:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs (patience: {HPARAMS['patience']})")
            print(f"   Best F1-Score: {best_val_f1:.4f}")
            break

    print("\n✅ Training finished.")

    print("\n💾 Saving final artifacts...")
    class_names = val_full_dataset.classes
    model.load_state_dict(torch.load(os.path.join(run_output_dir, "best_model.pth")))
    _, final_labels, final_preds = validate(model, val_loader, criterion, DEVICE)
    
    save_final_plots(history, final_labels, final_preds, class_names, run_output_dir)

    report = classification_report(final_labels, final_preds, target_names=class_names, zero_division=0)
    report_path = os.path.join(run_output_dir, "final_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Hyperparameters ---\n")
        for key, value in HPARAMS.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\n--- Class Distribution (Final) ---\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Validation samples: {len(val_dataset)}\n")
        f.write(f"Best F1-Score: {best_val_f1:.4f}\n")
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
