#!/bin/bash -l

#SBATCH --job-name=epi_processing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # Changed from 2
#SBATCH --cpus-per-task=1    # Changed from 4
#SBATCH --mem=16G            # This was perfect, keep it.
#SBATCH --time=02:00:00      # Reduced from 24h to 2h (32 min runtime + buffer)
#SBATCH --output=logs/david4epi_%j.out
#SBATCH --error=logs/david4epi_%j.err

# Load required modules
module purge
module load python

# Activate your Python environment (adjust path as needed)
source ~/venvs/jupyter-gpu/bin/activate

# Set up variables
DATASET="David4EPI"
TIMEFRAME=1
OUTPUT_DIR="/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/david4epi_extracted"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Print job information
echo "============================="
echo "🚀 Starting David4EPI dataset processing"
echo "📅 Job started: $(date)"
echo "🏷️  Job ID: $SLURM_JOB_ID"
echo "🖥️  Node: $SLURMD_NODENAME"
echo "📁 Dataset: $DATASET"
echo "⏱️  Timeframe: ±$TIMEFRAME"
echo "📂 Output: $OUTPUT_DIR"
echo "========================================="

# Run the processing script with verbose output
python process_dataset.py \
    --dataset "$DATASET" \
    --timeframe "$TIMEFRAME" \
    --output_dir "$OUTPUT_DIR" \
    --verbose

# Capture exit status
EXIT_STATUS=$?

# Print completion information
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
