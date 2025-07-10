# predict.py
# Refactored to use shared modules - eliminates code duplication
# Batch processing for multiple samples with single model loading

import torch
import os
import argparse

# --- Import Shared Modules ---
from config import HPARAMS, CLASS_NAMES, DEVICE
from data_utils import preprocess_sample
from model_utils import load_model, run_inference


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D CNN prediction on nucleus state data.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pth file).")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--folder_path", nargs='+', help="One or more paths to sample folders with t-1, t, t+1 subdirs.")
    input_group.add_argument("--volumes", nargs='+', help="3-4 space-separated paths: t-1, t, t+1 .tif files, and optionally label file.")

    args = parser.parse_args()

    try:
        # Validate input arguments
        if args.volumes:
            if len(args.volumes) < 3:
                raise ValueError("--volumes requires at least 3 paths (t-1, t, t+1)")
            elif len(args.volumes) > 4:
                raise ValueError("--volumes accepts maximum 4 paths (t-1, t, t+1, label)")
            
            # Process single sample with volume paths
            folder_paths = None
            volume_paths = args.volumes
        else:
            # Process one or more folder paths
            folder_paths = args.folder_path
            volume_paths = None

        # Load model once
        print("Loading model...")
        model = load_model(args.model_path)
        
        if folder_paths:
            # Batch processing for multiple folders
            total_samples = len(folder_paths)
            print(f"\n🚀 Processing {total_samples} sample(s)...")
            print("=" * 50)
            
            results = []
            for i, folder_path in enumerate(folder_paths):
                sample_name = os.path.basename(folder_path)
                print(f"\n📁 Sample {i+1}/{total_samples}: {sample_name}")
                
                try:
                    # Preprocess single sample using shared function
                    input_tensor = preprocess_sample(folder_path=folder_path, for_training=False)
                    print(f"   Input tensor shape: {input_tensor.shape}")
                    
                    # Run prediction using shared function
                    pred_index, pred_class, pred_confidence = run_inference(model, input_tensor)
                    
                    # Store result
                    result = {
                        'sample': sample_name,
                        'index': pred_index,
                        'class': pred_class,
                        'confidence': pred_confidence
                    }
                    results.append(result)
                    
                    print(f"   Predicted: {pred_class.upper()} ({pred_confidence:.2%})")
                    
                except Exception as e:
                    print(f"   ❌ Error processing {sample_name}: {e}")
                    results.append({
                        'sample': sample_name,
                        'error': str(e)
                    })
            
            # Summary
            print(f"\n{'='*50}")
            print("🎉 BATCH PROCESSING COMPLETE")
            print(f"{'='*50}")
            for i, result in enumerate(results):
                if 'error' not in result:
                    print(f"{i+1}. {result['sample'][:40]:40} → {result['class'].upper():12} ({result['confidence']:.1%})")
                else:
                    print(f"{i+1}. {result['sample'][:40]:40} → ERROR: {result['error']}")
        
        else:
            # Single sample processing with volume paths
            print("Preprocessing input data...")
            input_tensor = preprocess_sample(volume_paths=volume_paths, for_training=False)
            print(f"Input tensor created with shape: {input_tensor.shape}")

            pred_index, pred_class, pred_confidence = run_inference(model, input_tensor)

            print("\n--- Prediction Result ---")
            print(f"Predicted Class Index: {pred_index}")
            print(f"Predicted Class Name:  {pred_class.upper()}")
            print(f"Confidence:            {pred_confidence:.2%}")
            print("-------------------------\n")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")