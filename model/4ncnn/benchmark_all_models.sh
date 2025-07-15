#!/bin/bash
# benchmark_all_models.sh
# Clean script to benchmark all trained models

# Configuration
# Configuration
DEFAULT_SAMPLES=(
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_232_nucleus_105_count_8"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/221016_FUCCI_Nanog_stack_3_frame_201_nucleus_097_count_2"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_199_nucleus_035_count_11"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_214_nucleus_411_count_8"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_215_nucleus_423_count_12"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_235_nucleus_095_count_10"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_194_nucleus_077_count_11"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_199_nucleus_084_count_4"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_181_nucleus_067_count_5"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_201_nucleus_087_count_6"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/230212_stack6_frame_203_nucleus_045_count_13"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/221016_FUCCI_Nanog_stack_3_frame_121_nucleus_045_count_5"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/221016_FUCCI_Nanog_stack_3_frame_158_nucleus_044_count_1"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/221016_FUCCI_Nanog_stack_3_frame_123_nucleus_036_count_10"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/230212_stack6_frame_201_nucleus_090_count_9"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/221016_FUCCI_Nanog_stack_3_frame_116_nucleus_022_count_2"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/221016_FUCCI_Nanog_stack_3_frame_022_nucleus_009_count_1"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/230212_stack6_frame_065_nucleus_005_count_4"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/221016_FUCCI_Nanog_stack_3_frame_172_nucleus_007_count_3"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/230212_stack6_frame_117_nucleus_027_count_7"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/221016_FUCCI_Nanog_stack_3_frame_200_nucleus_094_count_9"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/230212_stack6_frame_066_nucleus_016_count_1"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/230212_stack6_frame_208_nucleus_366_count_4"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/221016_FUCCI_Nanog_stack_3_frame_119_nucleus_023_count_3"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/230212_stack6_frame_196_nucleus_063_count_4"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/221016_FUCCI_Nanog_stack_3_frame_063_nucleus_012_count_4"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/230212_stack6_frame_224_nucleus_032_count_8"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/230212_stack6_frame_207_nucleus_073_count_5"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/221016_FUCCI_Nanog_stack_3_frame_070_nucleus_025_count_5"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/Eszter1_frame_051_nucleus_027_count_2"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/Eszter1_frame_038_nucleus_016_count_1"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/221016_FUCCI_Nanog_stack_3_frame_182_nucleus_078_count_2"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/230212_stack6_frame_202_nucleus_066_count_8"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/230212_stack6_frame_192_nucleus_076_count_2"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/230212_stack6_frame_197_nucleus_042_count_9"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/221016_FUCCI_Nanog_stack_3_frame_103_nucleus_028_count_2"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/230212_stack6_frame_189_nucleus_009_count_1"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/221016_FUCCI_Nanog_stack_3_frame_196_nucleus_023_count_7"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/221016_FUCCI_Nanog_stack_3_frame_159_nucleus_021_count_1"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/221016_FUCCI_Nanog_stack_3_frame_202_nucleus_003_count_5"
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

