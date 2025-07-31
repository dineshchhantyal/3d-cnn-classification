#!/bin/bash -l

#SBATCH --job-name=dataset_processing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/DATASET="220321_stack11_%j.out"
#SBATCH --error=logs/DATASET="220321_stack11_%j.err"

# Load required modules
module purge
module load python

# Activate your Python environment
source ~/venvs/cellstate3d-env/bin/activate

# ====== USER CONFIGURABLE SECTION ======
# Pick one of: 230212_stack6, 220321_stack11, 221016_FUCCI_Nanog_stack_3, David4EPI, 230101_Gata6Nanog_stack_19
DATASET="220321_stack11"
TIMEFRAME=1
OUTPUT_DIR="/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/${DATASET}_extracted"
MAX_SAMPLES=    # Leave empty for unlimited, or set a number
# =======================================

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "============================="
echo "🚀 Starting $DATASET dataset processing"
echo "📅 Job started: $(date)"
echo "🏷️  Job ID: $SLURM_JOB_ID"
echo "🖥️  Node: $SLURMD_NODENAME"
echo "📁 Dataset: $DATASET"
echo "⏱️  Timeframe: ±$TIMEFRAME"
echo "📂 Output: $OUTPUT_DIR"
echo "📊 Max samples: ${MAX_SAMPLES:-unlimited}"
echo "========================================="

# Build the command
CMD="python process_dataset.py --dataset \"$DATASET\" --timeframe \"$TIMEFRAME\" --output_dir \"$OUTPUT_DIR\" --verbose"
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# Run the processing script
eval $CMD

EXIT_STATUS=$?

echo "========================================="
echo "📅 Job completed: $(date)"
echo "🎯 Exit status: $EXIT_STATUS"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ Processing completed successfully!"
    echo "📂 Results saved to: $OUTPUT_DIR"
    echo "📊 Output summary:"
    ls -lh "$OUTPUT_DIR" 2>/dev/null || echo "   No output directory found"
else
    echo "❌ Processing failed with exit code: $EXIT_STATUS"
fi
echo "========================================="

exit $EXIT_STATUS