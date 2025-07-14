#!/bin/bash
# benchmark_all_models.sh
# Clean script to benchmark all trained models

# Configuration
# Configuration
DEFAULT_SAMPLES=(
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_210_nucleus_090_count_3"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_232_nucleus_105_count_8"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/221016_FUCCI_Nanog_stack_3_frame_159_nucleus_133_count_10"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_218_nucleus_058_count_4"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_189_nucleus_075_count_5"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_197_nucleus_082_count_6"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/221016_FUCCI_Nanog_stack_3_frame_201_nucleus_097_count_2"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_235_nucleus_095_count_10"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_228_nucleus_018_count_5"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/death/230212_stack6_frame_214_nucleus_411_count_8"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/221016_FUCCI_Nanog_stack_3_frame_182_nucleus_019_count_5"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/230212_stack6_frame_066_nucleus_011_count_8"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/221016_FUCCI_Nanog_stack_3_frame_191_nucleus_015_count_9"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/230212_stack6_frame_125_nucleus_039_count_6"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/230212_stack6_frame_124_nucleus_031_count_5"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/230212_stack6_frame_193_nucleus_017_count_4"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/221016_FUCCI_Nanog_stack_3_frame_182_nucleus_037_count_1"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/230212_stack6_frame_065_nucleus_018_count_4"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/Eszter1_frame_054_nucleus_014_count_8"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/mitotic/230212_stack6_frame_196_nucleus_005_count_4"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/Eszter1_frame_048_nucleus_019_count_4"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/230212_stack6_frame_221_nucleus_098_count_5"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/221016_FUCCI_Nanog_stack_3_frame_159_nucleus_055_count_1"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/Eszter1_frame_051_nucleus_025_count_5"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/221016_FUCCI_Nanog_stack_3_frame_122_nucleus_042_count_8"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/230212_stack6_frame_122_nucleus_029_count_6"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/230212_stack6_frame_128_nucleus_028_count_3"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/221016_FUCCI_Nanog_stack_3_frame_079_nucleus_027_count_5"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/230212_stack6_frame_128_nucleus_040_count_9"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/new_daughter/Eszter1_frame_116_nucleus_028_count_2"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/221016_FUCCI_Nanog_stack_3_frame_190_nucleus_035_count_68"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/221016_FUCCI_Nanog_stack_3_frame_141_nucleus_004_count_1"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/221016_FUCCI_Nanog_stack_3_frame_165_nucleus_047_count_1"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/230212_stack6_frame_195_nucleus_028_count_3"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/230212_stack6_frame_173_nucleus_024_count_5"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/230212_stack6_frame_216_nucleus_001_count_3"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/Eszter1_frame_086_nucleus_012_count_7"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/221016_FUCCI_Nanog_stack_3_frame_178_nucleus_004_count_7"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/221016_FUCCI_Nanog_stack_3_frame_207_nucleus_040_count_8"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/221016_FUCCI_Nanog_stack_3_frame_153_nucleus_051_count_2"

    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable2/250709_stack6_frame_226_nucleus_091_count_105"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable2/250709_stack6_frame_234_nucleus_081_count_103"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable2/250709_stack6_frame_178_nucleus_057_count_070"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable2/250709_stack6_frame_129_nucleus_022_count_047"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable2/250709_stack6_frame_153_nucleus_024_count_064"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable2/250709_stack6_frame_217_nucleus_074_count_097"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable2/250709_stack6_frame_213_nucleus_093_count_098"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable2/250709_stack6_frame_180_nucleus_390_count_071"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable2/250709_stack6_frame_188_nucleus_178_count_079"
    "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2/stable2/250709_stack6_frame_224_nucleus_022_count_102"
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

