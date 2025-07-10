for d in training_outputs/*; do
  model="$d/best_model.pth"
  log_name="$(basename "$d")_benchmark.log"
  if [ -f "$model" ]; then
    echo "== Running benchmark for $model ==" | tee "$log_name"
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 1 > "$log_name.nvidia" &
    smi_pid=$!
    
    { time srun -p gpu --gpus 1 \
      python predict.py \
      --folder_path ~/3d-cnn-classification/data/nuclei_state_dataset/v2/stable/230212_stack6_frame_204_nucleus_069_count_1 \
      --model_path "$model"; } 2>&1 | tee -a "$log_name"

    kill $smi_pid
  else
    echo "Model not found in $d"
  fi
done

