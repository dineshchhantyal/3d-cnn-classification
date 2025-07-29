import os
from datetime import datetime
import json
import numpy as np

# --- Import Shared Modules ---
from config import HPARAMS, CLASS_NAMES, DEVICE
from utils.data_utils import preprocess_sample
from utils.model_utils import load_model, run_inference


def get_true_label_from_path(folder_path):
    """
    Extract the true class label from the directory path.

    Args:
        folder_path (str): Path to the sample folder

    Returns:
        str: True class label, or None if cannot be determined
    """
    # Split path and look for class directory names
    path_parts = folder_path.replace("\\", "/").split("/")

    # Mapping from directory names to model class names
    class_mapping = {
        "death": None,  # Death not in 3-class model
        "mitotic": "mitotic",
        "new_daughter": "new_daughter",
        "stable": "stable",
        "stable2": "stable",  # stable2 maps to stable
    }

    # Look for class directory in path (typically second-to-last or third-to-last)
    for part in reversed(path_parts):
        if part in class_mapping:
            return class_mapping[part]

    # If no known class found, return None
    return None


def crop_volume_by_bbox3d(volumes, bbox):
    """
    Crop the 3D volume based on the provided bounding box.

    Args:
        volumes (list): List of volume vectors [t-1, t, t+1, mask] --entire timestamp.
        bbox (tuple): Bounding box coordinates (z_min, z_max, y_min, y_max, x_min, x_max).

    Returns:
        list: Cropped volumes for each time point.
    """
    return [
        vol[bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]] for vol in volumes
    ]


def get_volumes_by_nuclei_ids_from_full_volumes(volumes, nuclei_ids):
    """
    Get the cropped volumes for specific nuclei IDs.

    Args:
        volumes (list): List of volume vectors [t-1, t, t+1, mask] --entire timestamp.
        nuclei_ids (list): List of nuclei IDs to filter by.

    Returns:
        dict: Dictionary with nuclei IDs as keys and cropped volumes as values.
    """
    if len(volumes) != 4:
        raise ValueError("Expected 4 volume vectors: [t-1, t, t+1, mask]")

    nucleus_cropped_volume = {}

    for nucleus_id in nuclei_ids:
        # Find the bounding box for this nucleus ID
        bbox = find_nucleus_bounding_box(
            volumes[3], nucleus_id
        )  # (z_min, z_max, y_min, y_max, x_min, x_max)
        if bbox is None:
            print(f"Warning: Nucleus ID {nucleus_id} not found in segmentation mask.")
            continue
        centroid = (
            (bbox[0] + bbox[1]) // 2,
            (bbox[2] + bbox[3]) // 2,
            (bbox[4] + bbox[5]) // 2,
        )
        # Get the cropped volume for this nucleus ID
        bbox = get_fixed_size_bbox(
            centroid,
            [
                HPARAMS.get("input_depth", 32),
                HPARAMS.get("input_height", 32),
                HPARAMS.get("input_width", 32),
            ],
            volumes[3].shape,
        )  # (z_min, z_max, y_min, y_max, x_min, x_max)

        # Get only the coordinate that is safe i.e within the volume
        bbox = (
            max(bbox[0], 0),
            min(bbox[1], volumes[0].shape[0]),
            max(bbox[2], 0),
            min(bbox[3], volumes[0].shape[1]),
            max(bbox[4], 0),
            min(bbox[5], volumes[0].shape[2]),
        )

        # Crop the volume for this nucleus ID
        cropped_volume = crop_volume_by_bbox3d(volumes, bbox)

        # Mask the mask binary
        if len(cropped_volume) == 4:
            cropped_volume[3] = (cropped_volume[3] == nucleus_id).astype(
                cropped_volume[3].dtype
            )

        nucleus_cropped_volume[nucleus_id] = {
            "t-1": cropped_volume[0],
            "t": cropped_volume[1],
            "t+1": cropped_volume[2],
            "mask": cropped_volume[3] if len(cropped_volume) == 4 else None,
        }

    return nucleus_cropped_volume


def find_nucleus_bounding_box(label_volume, nucleus_id, padding_factor=2.0):
    """
    Find 3D bounding box around a nucleus with optional padding

    Args:
        label_volume: 3D label array
        nucleus_id: Target nucleus ID
        padding_factor: Factor to expand the bounding box (e.g., 2.0 = 200% padding)

    Returns:
        tuple: (z_min, z_max, y_min, y_max, x_min, x_max) or None if not found
    """
    # Find all positions where nucleus exists
    positions = np.where(label_volume == nucleus_id)

    if len(positions[0]) == 0:
        return None

    # Get bounding box coordinates
    z_min, z_max = positions[0].min(), positions[0].max()
    y_min, y_max = positions[1].min(), positions[1].max()
    x_min, x_max = positions[2].min(), positions[2].max()

    # Apply padding
    if padding_factor > 1.0:
        z_size = z_max - z_min + 1
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1

        z_pad = int((z_size * (padding_factor - 1)) / 2)
        y_pad = int((y_size * (padding_factor - 1)) / 2)
        x_pad = int((x_size * (padding_factor - 1)) / 2)

        z_min = max(0, z_min - z_pad)
        z_max = min(label_volume.shape[0] - 1, z_max + z_pad)
        y_min = max(0, y_min - y_pad)
        y_max = min(label_volume.shape[1] - 1, y_max + y_pad)
        x_min = max(0, x_min - x_pad)
        x_max = min(label_volume.shape[2] - 1, x_max + x_pad)

    return (z_min, z_max, y_min, y_max, x_min, x_max)


def get_fixed_size_bbox(centroid, output_size, volume_shape):
    """
    Given a centroid (z, y, x), output_size [dz, dy, dx], and volume_shape,
    return a bounding box (z_min, z_max, y_min, y_max, x_min, x_max)
    centered on the centroid and clipped to the volume boundaries.
    """
    zc, yc, xc = [int(round(c)) for c in centroid]
    dz, dy, dx = output_size
    sz, sy, sx = volume_shape

    z_min = max(zc - dz // 2, 0)
    z_max = min(z_min + dz, sz)
    if z_max - z_min < dz and z_min > 0:
        z_min = max(z_max - dz, 0)

    y_min = max(yc - dy // 2, 0)
    y_max = min(y_min + dy, sy)
    if y_max - y_min < dy and y_min > 0:
        y_min = max(y_max - dy, 0)

    x_min = max(xc - dx // 2, 0)
    x_max = min(x_min + dx, sx)
    if x_max - x_min < dx and x_min > 0:
        x_min = max(x_max - dx, 0)

    return (z_min, z_max, y_min, y_max, x_min, x_max)
