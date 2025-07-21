#!/bin/bash
# benchmark_all_models.sh
# Clean script to benchmark all trained models

# Configuration
DEFAULT_SAMPLES=(
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/death/stack_19_channel_1_obj_left_frame_106_nucleus_122_count_64"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/death/stack_19_channel_1_obj_left_frame_096_nucleus_016_count_66"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/mitotic/stack_19_channel_1_obj_left_frame_099_nucleus_060_count_66"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/mitotic/stack_19_channel_1_obj_left_frame_050_nucleus_019_count_40"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/mitotic/stack_19_channel_1_obj_left_frame_093_nucleus_061_count_63"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/mitotic/stack_19_channel_1_obj_left_frame_018_nucleus_008_count_29"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/mitotic/stack_19_channel_1_obj_left_frame_103_nucleus_019_count_67"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/mitotic/stack_19_channel_1_obj_left_frame_062_nucleus_007_count_54"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/mitotic/stack_19_channel_1_obj_left_frame_089_nucleus_017_count_62"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/mitotic/stack_19_channel_1_obj_left_frame_007_nucleus_009_count_20"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/mitotic/stack_19_channel_1_obj_left_frame_089_nucleus_005_count_62"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/mitotic/stack_19_channel_1_obj_left_frame_061_nucleus_017_count_52"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/new_daughter/stack_19_channel_1_obj_left_frame_103_nucleus_068_count_67"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/new_daughter/stack_19_channel_1_obj_left_frame_062_nucleus_030_count_54"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/new_daughter/stack_19_channel_1_obj_left_frame_059_nucleus_051_count_52"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/new_daughter/stack_19_channel_1_obj_left_frame_012_nucleus_012_count_25"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/new_daughter/stack_19_channel_1_obj_left_frame_014_nucleus_024_count_28"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/new_daughter/stack_19_channel_1_obj_left_frame_059_nucleus_052_count_52"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/new_daughter/stack_19_channel_1_obj_left_frame_090_nucleus_043_count_64"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/new_daughter/stack_19_channel_1_obj_left_frame_100_nucleus_054_count_67"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/new_daughter/stack_19_channel_1_obj_left_frame_053_nucleus_045_count_48"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/new_daughter/stack_19_channel_1_obj_left_frame_018_nucleus_027_count_29"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/stable/stack_19_channel_1_obj_left_frame_098_nucleus_023_count_65"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/stable/stack_19_channel_1_obj_left_frame_070_nucleus_044_count_56"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/stable/stack_19_channel_1_obj_left_frame_033_nucleus_013_count_32"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/stable/stack_19_channel_1_obj_left_frame_093_nucleus_057_count_63"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/stable/stack_19_channel_1_obj_left_frame_096_nucleus_063_count_66"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/stable/stack_19_channel_1_obj_left_frame_074_nucleus_036_count_57"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/stable/stack_19_channel_1_obj_left_frame_089_nucleus_043_count_62"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/stable/stack_19_channel_1_obj_left_frame_085_nucleus_023_count_61"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/stable/stack_19_channel_1_obj_left_frame_002_nucleus_004_count_17"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted/stable/stack_19_channel_1_obj_left_frame_100_nucleus_044_count_67"
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

