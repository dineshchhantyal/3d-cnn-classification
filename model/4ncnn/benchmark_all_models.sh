#!/bin/bash
# benchmark_all_models.sh
# Clean script to benchmark all trained models

# Configuration
# Configuration
DEFAULT_SAMPLES=(
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/new_daughter/David4EPI_frame_175_nucleus_034_count_64"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/new_daughter/David4EPI_frame_085_nucleus_023_count_25"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/new_daughter/David4EPI_frame_138_nucleus_040_count_42"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_060_nucleus_007_count_17"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_086_nucleus_018_count_25"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_040_nucleus_004_count_10"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_075_nucleus_002_count_17"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_065_nucleus_007_count_17"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_065_nucleus_015_count_17"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_093_nucleus_007_count_31"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_064_nucleus_010_count_17"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_068_nucleus_007_count_17"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_084_nucleus_019_count_24"
)


# Handle multiple sample paths
if [ $# -eq 0 ]; then
    # No arguments - use default
    SAMPLE_PATHS=("${DEFAULT_SAMPLES[@]}")
else
    # Use all provided arguments as sample paths
    SAMPLE_PATHS=("$@")
fi

echo "🚀 Benchmarking all models..."
echo "Sample paths (${#SAMPLE_PATHS[@]}):"
for path in "${SAMPLE_PATHS[@]}"; do
    echo "  - $path"
done
echo "=================================================="

source ~/venvs/jupyter-gpu/bin/activate

for model_dir in training_outputs/*/; do
  if [ ! -d "$model_dir" ]; then continue; fi
  
  model_path="$model_dir/best_model.pth"
  model_name=$(basename "$model_dir")
  
  if [ -f "$model_path" ]; then
    echo ""
    echo "🔬 Testing model: $model_name"
    echo "----------------------------------------"
    
    # Batch processing - all samples with single model load
    time srun python predict.py \
      --model_path "$model_path" \
      --folder_path "${SAMPLE_PATHS[@]}" \
      --save_analysis \
      --output_dir "analysis_results/$model_name" \
      --analysis_level "full" \
    
    echo "✅ Completed: $model_name"
  else
    echo "❌ Model not found: $model_name"
  fi
done

echo "🎉 All benchmarks completed!"

