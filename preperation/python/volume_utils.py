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
