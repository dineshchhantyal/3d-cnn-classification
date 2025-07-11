# 3D CNN Video Generation System

A comprehensive pipeline for generating videos from 3D CNN nucleus classification predictions. This system processes time-series microscopy data, applies trained models for nucleus classification, and creates beautiful visualizations showing the evolution of cellular states over time.

## 🚀 Features

-   **Multi-Model Support**: Works with both 3ncnn and 4ncnn architectures
-   **GPU Acceleration**: Optimized batch processing with CUDA support
-   **Parallel Processing**: CPU parallelization for data extraction
-   **Flexible Visualization**: Multiple projection views, customizable styling
-   **Video Formats**: Export as MP4 or GIF
-   **Memory Efficient**: Sliding window processing for large datasets
-   **Configuration Presets**: Quick setup for different use cases

## 📁 System Architecture

```
video_generation/
├── __init__.py              # Package initialization
├── config.py               # Configuration management system
├── model_interface.py      # Model loading and inference engine
├── data_pipeline.py        # Sliding window data processing
├── renderer.py             # Frame visualization and rendering
├── video_generator.py      # Main orchestration pipeline
├── examples.py             # Usage examples and demos
└── README.md               # This documentation
```

## 🛠️ Installation & Setup

### Prerequisites

```bash
# Required packages
pip install torch torchvision torchaudio  # PyTorch with CUDA support
pip install opencv-python pillow matplotlib tifffile
pip install numpy scipy scikit-learn
pip install pathlib dataclasses typing

# Optional for video compression
# Install ffmpeg for better video compression
sudo apt install ffmpeg  # Linux
# brew install ffmpeg    # macOS
```

### Quick Start

1. **Clone and navigate to the video generation directory:**

```bash
cd /mnt/home/dchhantyal/3d-cnn-classification/video_generation
```

2. **Test model loading:**

```bash
python examples.py test_model
```

3. **Generate your first video:**

```bash
python examples.py quick
```

## 🎬 Usage Examples

### Quick Video Generation

The fastest way to create a video:

```python
from video_generator import VideoGenerator

# One-liner video generation
video_path = VideoGenerator.quick_video(
    raw_data_path="/path/to/raw/data",
    label_data_path="/path/to/labels",
    model_path="/path/to/model",
    output_dir="./output",
    fps=15,
    video_format="mp4"
)
```

### Custom Configuration

For more control over the output:

```python
from config import VideoConfig
from video_generator import VideoGenerator

# Create custom configuration
config = VideoConfig()

# Data paths
config.raw_data_path = "/path/to/raw/data"
config.label_data_path = "/path/to/labels"
config.model_path = "/path/to/model"
config.output_dir = "./output"

# Video settings
config.fps = 30
config.video_format = "mp4"
config.output_width = 1920
config.output_height = 1080

# Visualization options
config.show_multiple_projections = True
config.show_confidence_scores = True
config.show_class_labels = True
config.dark_theme = True

# Performance settings
config.use_gpu = True
config.batch_size = 32
config.use_parallel_processing = True

# Generate video
generator = VideoGenerator(config)
video_path = generator.generate_video()
```

### Configuration Presets

Use predefined configurations for common scenarios:

```python
from config import ConfigPresets
from video_generator import VideoGenerator

# Quick preview (fast, lower quality)
config = ConfigPresets.create_quick_preview()

# High quality (balanced speed/quality)
config = ConfigPresets.create_high_quality()

# Publication ready (best quality)
config = ConfigPresets.create_publication_ready()

# Update paths and generate
config.raw_data_path = "/path/to/data"
config.model_path = "/path/to/model"
generator = VideoGenerator(config)
video_path = generator.generate_video()
```

### Command Line Usage

Generate videos directly from the command line:

```bash
# Quick video with default settings
python video_generator.py \
    --raw-data "/path/to/raw" \
    --label-data "/path/to/labels" \
    --model "/path/to/model" \
    --output "./output" \
    --preset quick

# High quality video with custom settings
python video_generator.py \
    --raw-data "/path/to/raw" \
    --label-data "/path/to/labels" \
    --model "/path/to/model" \
    --output "./output" \
    --preset high_quality \
    --fps 30 \
    --format mp4 \
    --cleanup
```

## ⚙️ Configuration Options

### Data Configuration

-   `raw_data_path`: Path to raw microscopy images
-   `label_data_path`: Path to nucleus label masks
-   `model_path`: Path to trained CNN model
-   `cache_dir`: Directory for temporary files

### Video Output

-   `output_dir`: Output directory for video and frames
-   `video_name`: Base name for output video
-   `video_format`: "mp4" or "gif"
-   `fps`: Frames per second
-   `output_width/height`: Video resolution

### Visualization Options

-   `show_raw_data`: Display raw microscopy data
-   `show_nucleus_outlines`: Draw nucleus boundaries
-   `show_nucleus_centers`: Mark nucleus centers
-   `show_class_labels`: Display classification labels
-   `show_confidence_scores`: Show prediction confidence
-   `show_multiple_projections`: Multiple viewing angles
-   `projection_axis`: Primary projection direction ("x", "y", "z")
-   `dark_theme`: Use dark background
-   `visualization_style`: Rendering style

### Performance Settings

-   `use_gpu`: Enable CUDA acceleration
-   `batch_size`: Number of nuclei processed together
-   `use_parallel_processing`: Enable CPU parallelization
-   `num_workers`: Number of parallel workers

## 🔧 Model Integration

### Supported Models

The system automatically detects and supports:

1. **3ncnn Models**: 3-channel CNN (t-1, t, t+1)

    - Input: 3 consecutive time frames
    - Located in: `model/3ncnn/`

2. **4ncnn Models**: 4-channel CNN (t-1, t, t+1, label)
    - Input: 3 time frames + current label mask
    - Located in: `model/4ncnn/`

### Model Auto-Detection

```python
# The system automatically detects model type
config = VideoConfig()
config.model_path = "/path/to/model/directory"  # Auto-detects 3ncnn vs 4ncnn
config.model_path = "/path/to/specific/model.pth"  # Uses specific model file
```

### Custom Model Integration

To use your own model:

```python
from model_interface import ModelInferenceEngine

# Custom model configuration
config = VideoConfig()
config.model_path = "/path/to/your/model.pth"
config.model_type = "3ncnn"  # or "4ncnn"

# The engine will load your model
engine = ModelInferenceEngine(config)
```

## 📊 Data Pipeline

### Input Data Structure

Expected directory structure:

```
raw_data/
├── frame_001.tif
├── frame_002.tif
└── ...

label_data/
├── frame_001.tif
├── frame_002.tif
└── ...
```

Alternative structures supported:

```
data/
├── 001.tif, 002.tif, ...
└── frame_001/, frame_002/, ...
```

### Sliding Window Processing

The system uses a memory-efficient sliding window approach:

1. **Load 3 consecutive frames** [t-1, t, t+1]
2. **Extract nuclei** from frame t
3. **Classify each nucleus** using temporal context
4. **Render visualization** for frame t
5. **Slide window** to next frame

### Memory Management

-   Only 3 frames kept in memory at once
-   Automatic cleanup of old frames
-   GPU memory management for batch processing
-   Configurable cache directory for temporary files

## 🎨 Visualization Features

### Projection Views

-   **Single Projection**: Focus on one viewing angle
-   **Multi-Projection**: Side-by-side X, Y, Z views
-   **Configurable Axis**: Choose primary projection direction

### Visual Elements

-   **Raw Data Overlay**: Show original microscopy images
-   **Nucleus Outlines**: Boundary visualization
-   **Classification Labels**: Color-coded class names
-   **Confidence Scores**: Prediction certainty values
-   **Temporal Tracking**: Consistent nucleus identification

### Styling Options

-   **Dark/Light Themes**: Professional appearance
-   **Color Schemes**: Customizable class colors
-   **Typography**: Scalable fonts and labels
-   **Layout**: Flexible arrangement of visual elements

## 🚀 Performance Optimization

### GPU Acceleration

```python
config = VideoConfig()
config.use_gpu = True           # Enable CUDA
config.batch_size = 32          # Larger batches for GPU
config.compile_model = True     # TorchScript optimization
```

### Parallel Processing

```python
config.use_parallel_processing = True
config.num_workers = 8          # Match CPU cores
```

### Memory Optimization

```python
config.batch_size = 16          # Reduce if memory limited
config.cache_dir = "/fast/ssd"  # Use fast storage
```

## 📋 Output Files

### Generated Files

```
output_directory/
├── video_name.mp4              # Final video
├── frames/                     # Individual frames (optional)
│   ├── frame_000001.png
│   ├── frame_000002.png
│   └── ...
├── video_summary.json          # Processing statistics
└── predictions.jsonl           # Raw prediction data
```

### Summary Statistics

The `video_summary.json` contains:

-   Video metadata (resolution, duration, fps)
-   Model information (type, path, parameters)
-   Processing statistics (total nuclei, class distribution)
-   Performance metrics (processing time, average nuclei per frame)

## 🔍 Troubleshooting

### Common Issues

1. **GPU Out of Memory**

```python
# Reduce batch size
config.batch_size = 8
# Or disable GPU
config.use_gpu = False
```

2. **Model Loading Errors**

```python
# Specify model type explicitly
config.model_type = "3ncnn"
# Or provide specific model file
config.model_path = "/path/to/specific/model.pth"
```

3. **Data Loading Issues**

```python
# Check data paths
print(f"Raw data exists: {Path(config.raw_data_path).exists()}")
print(f"Label data exists: {Path(config.label_data_path).exists()}")
```

4. **Video Generation Fails**

```python
# Test individual components
python examples.py test_model    # Test model loading
python examples.py presets       # Check configuration
```

### Debug Mode

Enable verbose output:

```python
config = VideoConfig()
config.debug_mode = True         # Enable debug prints
config.save_intermediate = True  # Save intermediate files
```

## 🤝 Contributing

### Adding New Features

1. **New Visualization**: Extend `renderer.py`
2. **Model Support**: Update `model_interface.py`
3. **Output Formats**: Modify `video_generator.py`
4. **Configuration**: Add options to `config.py`

### Testing

```bash
# Test individual components
python examples.py test_model
python examples.py presets

# Test full pipeline with minimal data
python examples.py quick
```

## 📚 API Reference

### Core Classes

#### VideoConfig

Configuration management with validation and presets.

#### ModelInferenceEngine

Model loading, preprocessing, and batch inference.

#### SlidingWindowProcessor

Memory-efficient temporal data processing.

#### FrameRenderer

High-quality frame visualization and rendering.

#### VideoGenerator

Main orchestration pipeline.

### Key Methods

```python
# Quick generation
VideoGenerator.quick_video(raw_data_path, label_data_path, model_path, output_dir)

# Custom generation
generator = VideoGenerator(config)
video_path = generator.generate_video()

# Model testing
engine = ModelInferenceEngine(config)
prediction = engine.predict_single(nucleus_sequence)
```

## 📄 License

This project is part of the 3D CNN classification system. Please refer to the main project license.

## 🙏 Acknowledgments

Built on top of:

-   PyTorch for deep learning
-   OpenCV for video processing
-   Matplotlib for visualization
-   tifffile for microscopy data handling

---

For questions or issues, please check the examples in `examples.py` or refer to the main project documentation.
