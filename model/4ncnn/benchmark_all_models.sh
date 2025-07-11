#!/bin/bash
# benchmark_all_models.sh
# Clean script to benchmark all trained models

# Configuration
# Configuration
DEFAULT_SAMPLES=(
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/250709_stack6_frame_190_nucleus_202_count_079"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/250709_stack6_frame_201_nucleus_016_count_094"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/250709_stack6_frame_39_nucleus_005_count_017"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/250709_stack6_frame_153_nucleus_024_count_064"


    # "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/death/David4EPI_frame_078_nucleus_018_count_3"
    # "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/death/David4EPI_frame_082_nucleus_022_count_6"
    # "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/death/David4EPI_frame_084_nucleus_024_count_4"
    # "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/death/David4EPI_frame_092_nucleus_023_count_9"

    # "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/mitotic/David4EPI_frame_039_nucleus_009_count_1"
    # "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/mitotic/David4EPI_frame_040_nucleus_008_count_2"
    # "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/mitotic/David4EPI_frame_041_nucleus_008_count_1"
    # "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/mitotic/David4EPI_frame_044_nucleus_010_count_1"

    # "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/new_daughter/David4EPI_frame_040_nucleus_009_count_1"
    # "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/new_daughter/David4EPI_frame_040_nucleus_010_count_1"
    # "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/new_daughter/David4EPI_frame_041_nucleus_009_count_1"
    # "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/new_daughter/David4EPI_frame_041_nucleus_010_count_3"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_088_nucleus_005_count_10"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_090_nucleus_020_count_4"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_102_nucleus_017_count_6"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted/stable/David4EPI_frame_069_nucleus_011_count_4"
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
      --folder_path "${SAMPLE_PATHS[@]}"
    
    echo "✅ Completed: $model_name"
  else
    echo "❌ Model not found: $model_name"
  fi
done

echo "🎉 All benchmarks completed!"

