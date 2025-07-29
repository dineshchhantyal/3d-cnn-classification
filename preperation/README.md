# Nucleus Extraction Pipeline

## Overview

The extraction pipeline is designed to extract 3D nucleus volumes from time-series microscopy data with their corresponding classification labels (death, mitotic, stable, new_daughter). The pipeline processes multiple time frames around classification events to create training datasets for 3D CNN classification models.

This document outlines the workflow for extracting nucleus frames with classified labels.

## Data Sources

### Input Data Structure

```
raw-data/230212_stack6/
├── LineageGraph.json          # Complete cell lineage and classification data
├── registered_images/         # Raw fluorescence images
└── registered_label_images/   # Segmented nucleus labels
```

### LineageGraph.json Data Format

The `LineageGraph.json` file contains comprehensive cell tracking and classification information:

## Command-Line Usage

The extraction pipeline can be run directly from the command line using `process_dataset.py`.

### Example Command

```bash
# Basic usage
python process_dataset.py \
  --dataset 220321_stack11 \
  --timeframe 1 \
  --output_dir /mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/220321_stack11_extracted \
  --max_samples 20 \
  --cube_max_size 64 \
  --verbose
```

### Argument Reference

-   `--dataset` : Dataset name (must match one in the script/config)
-   `--timeframe` : Number of frames before/after the event to extract
-   `--output_dir` : Output directory for extracted nuclei
-   `--max_samples` : (Optional) Maximum samples per classification
-   `--cube_max_size` : (Optional) Maximum size of the cube for extraction
-   `--verbose` : (Optional) Enable verbose output
-   `--dataset_dir` : (Required if dataset is not in the predefined list) Custom base directory for the dataset
-   `--lineage_file` : (Optional) Custom path to the lineage file

> You can use this command directly in a terminal or inside a SLURM batch script for cluster processing.

---

## Extraction Workflow

### 1. Configuration Setup

```python
from lineage_tree import read_json_file
from nucleus_lineage_to_classification import nucleus_extractor


# Example: Extract nuclei from a loaded lineage forest
print(f"📖 Loading lineage data from: {lineage_file}")
forest = read_json_file(lineage_file)

print(f"🔬 Starting nucleus extraction for {dataset_name}...")
results = nucleus_extractor(
  forest=forest,
  timeframe=1,
  base_dir="raw-data/datasetname", # must contains `LineageGraph.json`, `registered_images/` and `registered_label_images/`
  output_dir="data/nuclei_state_dataset",
  max_samples=20, # Maximum samples per classification type
  fixed_size=[64, 64, 64], # Fixed size for extracted volumes
)
print(f"✅ Extraction completed.")

if isinstance(results, dict):
  for classification, count in results.items():
      if isinstance(count, (int, float)):
          print(f"   📊 {classification}: {count} samples")
```

### 2. Data Loading and Initialization

-   Load cell lineage and classification data from `LineageGraph.json`
-   Parse node information to extract cell states and temporal relationships
-   Initialize paths to registered images and label images
-   Validate data integrity and file existence

### 3. Nucleus Extraction Process

#### A. Time Frame Selection

For each classified nucleus from LineageGraph.json:

-   **Event Frame (t)**: The frame where the nucleus is classified
-   **Previous Frame (t-1)**: Event frame - 1 (if exists)
-   **Next Frame (t+1)**: Event frame + 1 (if exists)
-   **Flexible Direction**: Can extract in any temporal direction based on data availability
-   **Conditional Extraction**: If certain frames don't exist, no data is written for those frames

#### B. Spatial Cropping

1. **Locate the nucleus in the event frame using segmentation labels.**
2. **Calculate the centroid of the nucleus.**
3. **Determine cropping strategy:**
    - If a fixed size is specified:
        - Crop a volume of the specified size centered at the nucleus centroid.
    - If a fixed size is not specified:
        - Calculate the bounding box around the nucleus.
        - Apply a padding factor (default 2x) to the bounding box to capture surrounding context, especially if the nucleus is off-center.
4. **Extract the corresponding regions from all three time frames using the determined crop or bounding box.**

### 4. Output Data Structure

#### Directory Organization

```

data/nuclei_state_dataset/
├── death/ # Death event nuclei
├── mitotic/ # Mitotic event nuclei
├── stable/ # Stable nuclei
└── new_daughter/ # Newly formed daughter cells

```

#### Directory and File Structure

Each nucleus extraction creates a dedicated directory with the following naming convention:

**Directory Name Format:**

```

{dataset}_frame_{frame}_nucleus_{id}_count_{total_nuclei_in_frame}/

```

**Files Inside Each Directory:**
For each time frame (t-1, t, t+1) when available, a timestamp subdirectory is created containing:

```

t-1/ # Previous frame subdirectory (if available)
├── raw_cropped.tif # Cropped raw fluorescence image
├── label_cropped.tif # Cropped segmentation labels
└── metadata.json # Frame-specific metadata and extraction parameters

t/ # Event frame subdirectory
├── raw_cropped.tif # Cropped raw fluorescence image
├── label_cropped.tif # Cropped segmentation labels
├── binary_label_cropped.tif # Binary mask of target nucleus
├── raw_image_cropped.tif # Raw image cropped using label (only target nucleus visible)
└── metadata.json # Frame-specific metadata and extraction parameters

t+1/ # Next frame subdirectory (if available)
├── raw_cropped.tif # Cropped raw fluorescence image
├── label_cropped.tif # Cropped segmentation labels
└── metadata.json # Frame-specific metadata and extraction parameters

```

#### Metadata Structure

The pipeline generates metadata at two levels:

1. **Main Nucleus Metadata** (`metadata.json` in the nucleus directory root)
2. **Frame-Specific Metadata** (`metadata.json` in each timestamp subdirectory)

**Main Nucleus Metadata Structure:**

```json
{
    "nucleus_summary": {
        "dataset_name": "230212_stack6",
        "nucleus_id": "023",
        "event_frame": 45,
        "total_nuclei_in_frame": 47,
        "classification": "death",
        "extraction_date": "2025-07-03T10:30:45Z",
        "available_frames": ["t-1", "t", "t+1"],
        "missing_frames": []
    },
    "lineage_info": {
        "parent_cells": ["044_015"],
        "daughter_cells": [],
        "division_events": 0,
        "death_frame": 45,
        "cell_fate": "apoptotic_death"
    },
    "extraction_config": {
        "crop_padding": 2.0,
        "time_window": 1,
        "min_object_size": 20,
        "hole_filling_enabled": true
    },
    "quality_metrics": {
        "extraction_success": true,
        "frame_consistency": 0.95,
        "volume_variance": 0.12,
        "signal_to_noise_ratio": 8.5
    }
}
```

**Frame-Specific Metadata Structure:**
Each timestamp subdirectory contains a `metadata.json` file with detailed frame information:

```json
{
    "extraction_info": {
        "dataset_name": "230212_stack6",
        "frame_number": 45,
        "nucleus_id": "023",
        "timestamp": "t",
        "total_nuclei_in_frame": 47,
        "extraction_date": "2025-07-03T10:30:45Z"
    },
    "nucleus_properties": {
        "centroid_x": 123.4,
        "centroid_y": 456.7,
        "centroid_z": 12.3,
        "bounding_box": {
            "min_x": 100,
            "max_x": 150,
            "min_y": 430,
            "max_y": 480,
            "min_z": 10,
            "max_z": 15
        },
        "volume_pixels": 1250,
        "fluorescence_markers": {
            "histone_raw": 42.3,
            "gata6_raw": 15.7,
            "nanog_raw": 8.9,
            "histone_norm": 0.52,
            "gata6_norm": 0.31,
            "nanog_norm": 0.18
        }
    },
    "extraction_parameters": {
        "crop_padding": 2.0,
        "min_object_size": 20,
        "hole_filling_enabled": true,
        "image_dimensions": {
            "width": 100,
            "height": 100,
            "depth": 1
        }
    },
    "classification": {
        "category": "death",
        "confidence": 0.95,
        "lineage_analysis": {
            "parent_cells": ["044_015"],
            "daughter_cells": [],
            "division_events": 0,
            "death_frame": 45
        }
    }
}
```
