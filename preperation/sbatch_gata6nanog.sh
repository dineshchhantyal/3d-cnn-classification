#!/bin/bash -l

#SBATCH --job-name=dataset_processing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # Conservative CPU allocation
#SBATCH --mem=16G  # Conservative memory allocation
#SBATCH --time=24:00:00  # 24 hours should be sufficient
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load required modules
module purge
module load python

# Activate your Python environment (adjust path as needed)
source ~/venvs/jupyter-gpu/bin/activate

# Set up variables
DATASET="230101_Gata6Nanog_stack_19"
TIMEFRAME=1
OUTPUT_DIR="/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/gata6nanog_extracted"
MAX_SAMPLES=1500  # Conservative limit to control processing time and storage

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Print job information
echo "============================="
echo "🚀 Starting 230101_Gata6Nanog_stack_19 dataset processing"
echo "📅 Job started: $(date)"
echo "🏷️  Job ID: $SLURM_JOB_ID"
echo "🖥️  Node: $SLURMD_NODENAME"
echo "📁 Dataset: $DATASET"
echo "⏱️  Timeframe: ±$TIMEFRAME"
echo "📂 Output: $OUTPUT_DIR"
echo "📊 Max samples: $MAX_SAMPLES"
echo "========================================="

# Run the processing script with verbose output
python process_dataset.py \
    --dataset "$DATASET" \
    --timeframe "$TIMEFRAME" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples "$MAX_SAMPLES" \
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
