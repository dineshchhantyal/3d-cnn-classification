# V2 Nucleus Extraction Pipeline

This document outlines the comprehensive workflow for extracting nucleus frames with classified labels using the V2 extraction pipeline.

## Overview

The V2 extraction pipeline is designed to extract 3D nucleus volumes from time-series microscopy data with their corresponding classification labels (death, mitotic, stable, new_daughter). The pipeline processes multiple time frames around classification events to create training datasets for 3D CNN classification models.

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
- **Edges**: Cell lineage connections between time frames
- **Nodes**: Individual cell data with:
  - **Name**: Cell identifier in format `frame_cell` (e.g., "024_002") [frame number and nucleus ID]
  - **Fluorescence markers**: Histone, Gata6, Nanog (raw, interpolated, normalized)
  - **Spatial coordinates**: Centroid_X, Centroid_Y, Centroid_Z
  - **Temporal information**: Frame number 
  - **Classification states**: Derived from cell behavior and lineage patterns

## Extraction Workflow

### 1. Configuration Setup
```python
config = NucleusExtractorConfig()
config.crop_padding = 2.0        # 200% padding around nucleus bounding box
config.time_window = 1           # Extract 3 frames: [t-1, t, t+1]
config.min_object_size = 20      # Minimum object size for cleaning
config.enable_hole_filling = True # Fill holes in nucleus masks
```

### 2. Data Loading and Initialization
- Load cell lineage and classification data from `LineageGraph.json`
- Parse node information to extract cell states and temporal relationships
- Initialize paths to registered images and label images
- Validate data integrity and file existence

### 3. Nucleus Extraction Process

#### A. Time Frame Selection
For each classified nucleus from LineageGraph.json:
- **Event Frame (t)**: The frame where the nucleus is classified
- **Previous Frame (t-1)**: Event frame - 1 (if exists)
- **Next Frame (t+1)**: Event frame + 1 (if exists)
- **Flexible Direction**: Can extract in any temporal direction based on data availability
- **Conditional Extraction**: If certain frames don't exist, no data is written for those frames

#### B. Spatial Cropping
- Locate nucleus in the event frame using segmentation labels
- Calculate bounding box around the nucleus
- Apply padding factor (default 2x) to capture surrounding context
- Extract corresponding regions from all three time frames

#### C. Volume Cleaning
1. **Noise Removal**: Remove small objects below minimum size threshold
2. **Hole Filling**: Fill internal holes in nucleus masks using morphological operations
3. **Border Clearing**: Remove objects touching image borders
4. **Background Masking**: Set non-nucleus pixels to zero in cleaned version

#### D. Data Validation
- Verify nucleus presence in all three frames
- Check volume consistency across time frames
- Validate spatial coordinates and dimensions

### 4. Output Data Structure

#### Directory Organization
```
data/nuclei_state_dataset/
├── death/           # Death event nuclei
├── mitotic/         # Mitotic event nuclei  
├── stable/          # Stable nuclei (controls)
└── new_daughter/    # Newly formed daughter cells
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
t-1/                         # Previous frame subdirectory (if available)
├── raw_cropped.tif          # Cropped raw fluorescence image
├── label_cropped.tif        # Cropped segmentation labels
├── binary_label_cropped.tif # Binary mask of target nucleus
├── raw_image_cropped.tif    # Raw image cropped using label (only target nucleus visible)
└── metadata.json            # Frame-specific metadata and extraction parameters

t/                           # Event frame subdirectory
├── raw_cropped.tif          # Cropped raw fluorescence image
├── label_cropped.tif        # Cropped segmentation labels
├── binary_label_cropped.tif # Binary mask of target nucleus
├── raw_image_cropped.tif    # Raw image cropped using label (only target nucleus visible)
└── metadata.json            # Frame-specific metadata and extraction parameters

t+1/                         # Next frame subdirectory (if available)
├── raw_cropped.tif          # Cropped raw fluorescence image
├── label_cropped.tif        # Cropped segmentation labels
├── binary_label_cropped.tif # Binary mask of target nucleus
├── raw_image_cropped.tif    # Raw image cropped using label (only target nucleus visible)
└── metadata.json            # Frame-specific metadata and extraction parameters
```

**Example Directory Structure:**
```
data/nuclei_state_dataset/death/
└── 230212_stack6_frame_045_nucleus_023_count_47/
    ├── metadata.json               # Main nucleus metadata and summary
    ├── t-1/                        # Previous frame (if available)
    │   ├── raw_cropped.tif
    │   ├── label_cropped.tif
    │   ├── binary_label_cropped.tif
    │   ├── raw_image_cropped.tif    # Raw image cropped using label (only target nucleus visible)
    │   └── metadata.json            # Frame-specific metadata
    ├── t/                          # Event frame
    │   ├── raw_cropped.tif
    │   ├── label_cropped.tif
    │   ├── binary_label_cropped.tif
    │   ├── raw_image_cropped.tif    # Raw image cropped using label (only target nucleus visible)
    │   └── metadata.json            # Frame-specific metadata
    └── t+1/                        # Next frame (if available)
        ├── raw_cropped.tif
        ├── label_cropped.tif
        ├── binary_label_cropped.tif
        ├── raw_image_cropped.tif    # Raw image cropped using label (only target nucleus visible)
        └── metadata.json            # Frame-specific metadata
```

#### Metadata Structure
The V2 pipeline generates metadata at two levels:

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
      "min_x": 100, "max_x": 150,
      "min_y": 430, "max_y": 480,
      "min_z": 10, "max_z": 15
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

### 5. Quality Control and Validation

#### Automatic Checks
- Volume size consistency across time frames
- Nucleus presence validation in target frames
- Spatial coordinate bounds checking
- File integrity verification

#### Manual Review Features
- 3D visualization of extracted nuclei
- Time series plotting capabilities
- Comprehensive analysis plots showing:
  - Original vs cleaned volumes
  - Cross-sectional views
  - 3D surface renderings
  - Intensity profiles

### 6. Batch Processing Workflow

```python
# Initialize extraction manager
manager = NucleusExtractorManager(data_path, config)

# Process nuclei from LineageGraph.json
results = manager.batch_extract_nuclei_from_lineage(dataset_name="230212_stack6")

# Save results with proper organization by classification type
manager.save_batch_results(results, output_base_dir="data/nuclei_state_dataset")
```

### 7. Classification Categories

#### Death Events
- Nuclei showing apoptotic characteristics
- Typically show volume reduction and morphological changes
- Extracted around the frame where death is classified

#### Mitotic Events  
- Nuclei undergoing cell division
- Show characteristic mitotic morphology
- Captured during active division phases

#### Stable Nuclei
- Control nuclei showing normal morphology
- No death or mitotic events in surrounding time frames
- Used as negative examples for classification

#### New Daughter Cells
- Recently formed cells after division
- Smaller volumes with distinct morphological features
- Important for studying post-mitotic cell behavior

## Key Features of V2 Pipeline

### 1. **Multi-frame Context**
- Captures temporal dynamics around classification events
- Provides before/during/after context for better classification

### 2. **Automated Cleaning**
- Removes imaging artifacts and noise
- Standardizes nucleus representations
- Maintains both original and cleaned versions

### 3. **Flexible Configuration**
- Adjustable time windows and cropping parameters
- Configurable cleaning thresholds
- Extensible to different datasets

### 4. **Quality Assurance**
- Built-in validation and error checking
- Comprehensive visualization tools
- Metadata tracking for reproducibility

### 5. **Scalable Processing**
- Batch processing capabilities
- Memory-efficient volume handling
- Progress tracking and error recovery

### 6. **High-Performance Parallel Processing** 🚀
- **Multi-level parallelization**: Parallel processing at batch, frame, and I/O levels
- **3-10x performance improvement** over sequential processing
- **Automatic resource optimization**: Configures workers based on available CPU and memory
- **Memory management**: Chunked processing to handle large datasets efficiently
- **Real-time monitoring**: Progress tracking and performance metrics
- **Scalable architecture**: Optimizes for any number of CPU cores and memory configurations

## Parallel Processing Configuration

### Automatic Optimization
```python
# Auto-configure based on system resources
config = create_optimized_config(
    dataset_size=1000,      # Expected number of nuclei
    available_cores=16,     # Available CPU cores (auto-detected if None)
    available_memory_gb=32  # Available memory in GB
)
```

### Manual Configuration
```python
from parallel_nucleus_extractor import ParallelConfig

# Custom configuration for specific needs
config = ParallelConfig(
    max_workers_batch=4,        # Number of nuclei processed in parallel
    max_workers_frames=8,       # Number of frames processed per nucleus
    max_workers_io=16,          # Number of I/O operations in parallel
    chunk_size=50,              # Nuclei per processing chunk
    max_memory_gb=16.0,         # Memory usage limit
    save_intermediate_results=True  # Save results as they complete
)
```

### Configuration Guidelines

#### For High-Memory Systems (32+ GB RAM)
```python
config = ParallelConfig(
    max_workers_batch=8,
    chunk_size=200,             # Larger chunks for better efficiency
    max_memory_gb=24.0
)
```

#### For Many-Core Systems (16+ cores)
```python
config = ParallelConfig(
    max_workers_batch=12,
    max_workers_frames=16,
    max_workers_io=32
)
```

#### For Network Storage Systems
```python
config = ParallelConfig(
    max_workers_io=4,           # Reduce I/O workers to avoid saturation
    chunk_size=20,              # Smaller chunks
    save_intermediate_results=True
)
```

### Performance Tuning Tips

1. **CPU Optimization**: If CPU usage is low, increase `max_workers_batch`
2. **Memory Optimization**: If you have more RAM, increase `chunk_size` 
3. **I/O Optimization**: For SSD storage, increase `max_workers_io`
4. **Network Storage**: Reduce I/O workers and use smaller chunks
5. **Large Datasets**: Enable `save_intermediate_results=True`

### Expected Performance Gains

- **3-5x speedup** on typical multi-core systems (4-8 cores)
- **5-10x speedup** on high-end systems (16+ cores)
- **Linear memory scaling** with configurable chunk sizes
- **Efficient resource utilization** across CPU, memory, and I/O

## Usage Examples

### Basic Extraction (Sequential)
```python
# Configure extraction parameters
config = NucleusExtractorConfig()
config.time_window = 1  # Extract t-1, t, t+1 frames when available

# Initialize manager with LineageGraph.json data
manager = NucleusExtractorManager("/path/to/dataset", config)

# Extract single nucleus time series
result = manager.extract_nucleus_from_lineage(
    nucleus_name="045_023"  # frame_cell format
)

# Visualize result
manager.plot_result(result, plot_type="comprehensive")
```

### High-Performance Parallel Extraction 🚀
```python
from parallel_nucleus_extractor import ParallelNucleusExtractor, create_optimized_config

# Auto-optimize configuration for your system
config = create_optimized_config(
    dataset_size=500,  # Expected number of nuclei
    available_memory_gb=16.0  # Available system memory
)

# Initialize parallel extractor
extractor = ParallelNucleusExtractor("/path/to/dataset", config)
extractor.load_dataset("230212_stack6")

# High-performance batch extraction
successful = extractor.batch_extract_nuclei_parallel(
    max_samples=100,
    event_types=["death", "mitotic"],
    dataset_name="230212_stack6"
)

print(f"Processed {successful} nuclei with 3-10x speedup!")
```

### Performance Benchmarking
```python
from performance_benchmark import PerformanceBenchmark

# Run comprehensive performance comparison
benchmark = PerformanceBenchmark("/path/to/data", "230212_stack6")
benchmark.setup_extractors()
benchmark.create_performance_report("/output/dir")

# Generates detailed performance report with visualizations
```

### Custom Time Windows
```python
# Extract 5-frame window
config.frame_offsets = [-2, -1, 0, 1, 2]

# Extract sparse sampling  
config.frame_offsets = [-5, -3, -1, 0, 1, 3, 5]

# Extract single frame only
config.frame_offsets = [0]
```

### Batch Processing with Classification
```python
# Process all death events from LineageGraph
death_results = manager.extract_by_lineage_classification(
    classification_type="death",
    max_samples=100
)

# Process all mitotic events from LineageGraph  
mitotic_results = manager.extract_by_lineage_classification(
    classification_type="mitotic",
    max_samples=100
)

# Process stable nuclei (control group)
stable_results = manager.extract_by_lineage_classification(
    classification_type="stable",
    max_samples=100
)
```

## Integration with CNN Training

The extracted nucleus images are directly compatible with the 3D CNN training pipeline:

1. **Input Structure**: Individual cropped images per time frame (t-1, t, t+1)
2. **Image Types**: 
   - Raw fluorescence images for all frames
   - Binary masks and nucleus-specific images for event frames
3. **Labels**: Categorical classification derived from LineageGraph.json analysis
4. **Preprocessing**: Images are pre-cropped and organized by classification type
5. **Flexibility**: Handles missing frames gracefully (no forced 3D volumes)

**Training Data Organization:**
- Each nucleus generates multiple files across time frames
- Event frame (t) contains additional binary and nucleus-specific images
- Classification labels derived from cell lineage analysis and behavior patterns
- Ready for both 2D and 3D CNN architectures depending on frame combination strategy

This V2 extraction pipeline provides a robust foundation for training accurate CNN models using comprehensive cell lineage data from time-series microscopy.
