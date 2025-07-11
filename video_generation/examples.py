"""
Example Usage Scripts for Video Generation System
Demonstrates different ways to use the video generation pipeline
"""

import os
import sys
from pathlib import Path

# Add video_generation to path
sys.path.append(str(Path(__file__).parent))

from config import VideoConfig, ConfigPresets
from video_generator import VideoGenerator
from model_interface import ModelInferenceEngine


def example_quick_video():
    """Quick video generation example."""
    print("🚀 Example 1: Quick Video Generation")
    
    # Set your data paths here
    raw_data_path = "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/raw"
    label_data_path = "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/labels"
    model_path = "/mnt/home/dchhantyal/3d-cnn-classification/model/3ncnn/3ncnn.py"  # Will auto-detect best model
    output_dir = "./output/quick_video"
    
    try:
        video_path = VideoGenerator.quick_video(
            raw_data_path=raw_data_path,
            label_data_path=label_data_path,
            model_path=model_path,
            output_dir=output_dir,
            fps=15,
            video_format="mp4"
        )
        print(f"✅ Quick video saved: {video_path}")
        
    except Exception as e:
        print(f"❌ Quick video generation failed: {e}")


def example_custom_configuration():
    """Custom configuration example."""
    print("🚀 Example 2: Custom Configuration")
    
    # Create custom configuration
    config = VideoConfig()
    
    # Set data paths
    config.raw_data_path = "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/raw"
    config.label_data_path = "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/labels"
    config.model_path = "/mnt/home/dchhantyal/3d-cnn-classification/model/4ncnn"  # Use 4ncnn model
    config.output_dir = "./output/custom_video"
    
    # Customize video settings
    config.fps = 20
    config.video_format = "gif"
    config.output_width = 1280
    config.output_height = 720
    
    # Customize visualization
    config.show_multiple_projections = True
    config.show_confidence_scores = True
    config.show_class_labels = True
    config.dark_theme = True
    
    # Performance settings
    config.batch_size = 16
    config.use_gpu = True
    config.use_parallel_processing = True
    config.num_workers = 4
    
    try:
        generator = VideoGenerator(config)
        video_path = generator.generate_video()
        print(f"✅ Custom video saved: {video_path}")
        
    except Exception as e:
        print(f"❌ Custom video generation failed: {e}")


def example_publication_ready():
    """Publication-ready video example."""
    print("🚀 Example 3: Publication-Ready Video")
    
    # Use publication-ready preset
    config = ConfigPresets.create_publication_ready()
    
    # Update data paths
    config.raw_data_path = "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/raw"
    config.label_data_path = "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/labels"
    config.model_path = "/mnt/home/dchhantyal/3d-cnn-classification/model/notebooks/final_model.pth"  # Use final model
    config.output_dir = "./output/publication_video"
    
    # Publication-specific settings
    config.video_name = "nucleus_classification_analysis"
    config.fps = 30
    config.video_format = "mp4"
    
    try:
        generator = VideoGenerator(config)
        video_path = generator.generate_video()
        
        # Clean up temporary files for publication
        generator.cleanup_temporary_files(keep_frames=False)
        
        print(f"✅ Publication video saved: {video_path}")
        
    except Exception as e:
        print(f"❌ Publication video generation failed: {e}")


def example_test_model_inference():
    """Test model inference without full video generation."""
    print("🚀 Example 4: Test Model Inference")
    
    # Create minimal config for testing
    config = VideoConfig()
    config.model_path = "/mnt/home/dchhantyal/3d-cnn-classification/model/3ncnn"
    config.model_type = "3ncnn"  # Explicitly set model type
    
    try:
        # Initialize model engine
        model_engine = ModelInferenceEngine(config)
        print(f"✅ Model loaded successfully: {model_engine.model_type}")
        print(f"   Classes: {model_engine.class_names}")
        print(f"   Device: {model_engine.device}")
        
        # Test with dummy data
        import numpy as np
        dummy_sequence = {
            't-1': np.random.rand(32, 32, 32).astype(np.float32),
            't': np.random.rand(32, 32, 32).astype(np.float32),
            't+1': np.random.rand(32, 32, 32).astype(np.float32)
        }
        
        # Test preprocessing
        tensor = model_engine.preprocess_nucleus(dummy_sequence)
        print(f"✅ Preprocessing successful: {tensor.shape}")
        
        # Test prediction
        prediction = model_engine.predict_single(dummy_sequence)
        print(f"✅ Prediction successful: {prediction}")
        
    except Exception as e:
        print(f"❌ Model inference test failed: {e}")


def example_batch_processing():
    """Example of batch processing multiple datasets."""
    print("🚀 Example 5: Batch Processing Multiple Datasets")
    
    # Define multiple datasets
    datasets = [
        {
            "name": "230212_stack6",
            "raw_path": "/mnt/home/dchhantyal/3d-cnn-classification/raw-data/230212_stack6",
            "label_path": "/mnt/home/dchhantyal/3d-cnn-classification/data/labels",
            "output": "./output/230212_stack6_video"
        },
        # Add more datasets as needed
    ]
    
    # Base configuration
    base_config = ConfigPresets.create_high_quality()
    base_config.model_path = "/mnt/home/dchhantyal/3d-cnn-classification/model/3ncnn"
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset['name']}")
        
        # Update config for this dataset
        config = base_config.copy()
        config.raw_data_path = dataset['raw_path']
        config.label_data_path = dataset['label_path']
        config.output_dir = dataset['output']
        config.video_name = f"{dataset['name']}_analysis"
        
        try:
            generator = VideoGenerator(config)
            video_path = generator.generate_video()
            print(f"✅ {dataset['name']} video saved: {video_path}")
            
        except Exception as e:
            print(f"❌ Failed to process {dataset['name']}: {e}")
            continue


def example_configuration_presets():
    """Demonstrate different configuration presets."""
    print("🚀 Example 6: Configuration Presets")
    
    presets = {
        "Quick Preview": ConfigPresets.create_quick_preview(),
        "High Quality": ConfigPresets.create_high_quality(),
        "Publication Ready": ConfigPresets.create_publication_ready()
    }
    
    for name, config in presets.items():
        print(f"\n{name} Configuration:")
        print(f"  Resolution: {config.output_width}x{config.output_height}")
        print(f"  FPS: {config.fps}")
        print(f"  Format: {config.video_format}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Multiple projections: {config.show_multiple_projections}")
        print(f"  Show confidence: {config.show_confidence_scores}")


def main():
    """Run examples based on command line argument."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Generation Examples")
    parser.add_argument("example", choices=[
        "quick", "custom", "publication", "test_model", "batch", "presets"
    ], help="Which example to run")
    
    args = parser.parse_args()
    
    examples = {
        "quick": example_quick_video,
        "custom": example_custom_configuration,
        "publication": example_publication_ready,
        "test_model": example_test_model_inference,
        "batch": example_batch_processing,
        "presets": example_configuration_presets
    }
    
    example_func = examples[args.example]
    example_func()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Run all examples if no argument provided
        print("Running all examples...\n")
        
        # Run examples that don't require heavy computation
        example_configuration_presets()
        example_test_model_inference()
        
        print("\n" + "="*50)
        print("To run specific examples:")
        print("python examples.py quick")
        print("python examples.py custom")
        print("python examples.py publication")
        print("python examples.py test_model")
        print("python examples.py batch")
        print("python examples.py presets")
        
    else:
        main()
