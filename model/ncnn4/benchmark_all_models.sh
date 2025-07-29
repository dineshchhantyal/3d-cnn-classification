#!/bin/bash
# benchmark_all_models.sh
# Clean script to benchmark all trained models

# Configuration
DEFAULT_SAMPLES=(
   
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
    
    echo "✅ Completed: $model_name"
  else
    echo "❌ Model not found: $model_name"
  fi
done

echo "🎉 All benchmarks completed!"


