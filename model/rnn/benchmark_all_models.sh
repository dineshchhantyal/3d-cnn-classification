#!/bin/bash
# benchmark_all_models.sh
# Clean script to benchmark all trained models

# Configuration
   
DEFAULT_SAMPLES=(
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/death/230212_stack6_frame_196_nucleus_013_count_89"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/death/stack_19_channel_1_obj_left_frame_106_nucleus_122_count_64"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/death/230212_stack6_frame_201_nucleus_087_count_94"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/death/230212_stack6_frame_210_nucleus_090_count_98"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/death/230212_stack6_frame_199_nucleus_035_count_91"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/death/230212_stack6_frame_232_nucleus_105_count_105"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/death/230212_stack6_frame_199_nucleus_084_count_91"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/death/230212_stack6_frame_161_nucleus_015_count_65"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/death/230212_stack6_frame_233_nucleus_006_count_105"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/death/230212_stack6_frame_197_nucleus_082_count_89"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/stack_19_channel_1_obj_left_frame_052_nucleus_014_count_44"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_069_nucleus_023_count_24"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_218_nucleus_023_count_103"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/stack_19_channel_1_obj_left_frame_005_nucleus_013_count_19"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_179_nucleus_013_count_71"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/David4EPI_frame_127_nucleus_022_count_34"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/David4EPI_frame_126_nucleus_017_count_33"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/230212_stack6_frame_194_nucleus_045_count_84"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/David4EPI_frame_045_nucleus_007_count_13"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/mitotic/stack_19_channel_1_obj_left_frame_055_nucleus_048_count_49"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/new_daughter/230212_stack6_frame_067_nucleus_022_count_24"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/new_daughter/230212_stack6_frame_131_nucleus_049_count_49"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/new_daughter/230212_stack6_frame_204_nucleus_084_count_94"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/new_daughter/stack_19_channel_1_obj_left_frame_104_nucleus_048_count_68"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/new_daughter/stack_19_channel_1_obj_left_frame_013_nucleus_026_count_26"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/new_daughter/230212_stack6_frame_128_nucleus_025_count_47"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/new_daughter/230212_stack6_frame_176_nucleus_050_count_70"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/new_daughter/230212_stack6_frame_208_nucleus_022_count_101"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/new_daughter/David4EPI_frame_170_nucleus_057_count_64"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/new_daughter/Eszter1_frame_103_nucleus_023_count_15"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/stable/David4EPI_frame_070_nucleus_014_count_17"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/stable/David4EPI_frame_163_nucleus_002_count_59"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/stable/stack_19_channel_1_obj_left_frame_089_nucleus_058_count_62"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/stable/David4EPI_frame_151_nucleus_001_count_50"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/stable/stack_19_channel_1_obj_left_frame_024_nucleus_020_count_31"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/stable/Eszter1_frame_129_nucleus_051_count_63"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/stable/Eszter1_frame_128_nucleus_028_count_65"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/stable/stack_19_channel_1_obj_left_frame_072_nucleus_015_count_57"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/stable/Eszter1_frame_134_nucleus_001_count_46"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v3/stable/stack_19_channel_1_obj_left_frame_099_nucleus_045_count_66"
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

