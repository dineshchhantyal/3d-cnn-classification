from pathlib import Path
import tifffile


def get_volume_by_timestamp(base_dir, event_frame):
    """
    Get the volume of nodes at a specific timestamp.

    Args:
        base_dir (str): Base directory of the dataset
        event_frame (int): The timestamp to get the volume for

    Returns:
        dict: 'registered_image' and 'label_image' 3d numpy arrays
    """
    file_paths = get_file_paths(base_dir, event_frame)

    if not file_paths["label"] or not file_paths["registered"]:
        return {"registered_image": None, "label_image": None}

    label_volume = tifffile.imread(file_paths["label"][0])
    registered_volume = tifffile.imread(file_paths["registered"][0])

    return {"registered_image": registered_volume, "label_image": label_volume}


def check_if_frame_exists(base_dir, event_frame):
    """
    Check if a specific frame exists in the dataset.

    Args:
        base_dir (str): Base directory of the dataset
        event_frame (int): The timestamp to check

    Returns:
        bool: True if the frame exists, False otherwise
    """
    file_paths = get_file_paths(base_dir, event_frame)
    return bool(file_paths["label"] and file_paths["registered"])


# Extract common utilities
def get_file_paths(base_dir, frame_num):
    """Get paths for registered and label files"""
    label_dir = Path(base_dir) / "registered_label_images"
    registered_dir = Path(base_dir) / "registered_images"
    return {
        "label": list(label_dir.glob(f"label_reg8_{frame_num}.tif")),
        "registered": list(registered_dir.glob(f"nuclei_reg8_{frame_num}.tif")),
    }


def generate_frame_label(frame_num, event_frame):
    """Generate frame label (t, t-1, t+1, etc.)"""
    offset = frame_num - event_frame
    if offset == 0:
        return "t"
    elif offset < 0:
        return f"t{offset}"
    else:
        return f"t+{offset}"


def safe_bounds(volume_shape, bbox):
    """Safely calculate bounds within volume limits"""
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    vol_z, vol_y, vol_x = volume_shape

    return {
        "z": (max(0, min(z_min, vol_z - 1)), max(z_min + 1, min(z_max + 1, vol_z))),
        "y": (max(0, min(y_min, vol_y - 1)), max(y_min + 1, min(y_max + 1, vol_y))),
        "x": (max(0, min(x_min, vol_x - 1)), max(x_min + 1, min(x_max + 1, vol_x))),
    }
