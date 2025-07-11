#!/bin/bash -l

#SBATCH --job-name=enhanced_video_gen_4ncnn_stack3
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=3
#SBATCH --mem=16G

# Script for generating ENHANCED video from 4ncnn model predictions on 221016_FUCCI_Nanog_stack_3 dataset
# Created: July 11, 2025 - Enhanced with congested nuclei handling and improved visualization
# Dataset: 221016_FUCCI_Nanog_stack_3 (210 frames)
# Model: 4ncnn best_model.pth from training_outputs/20250710-131550
# 
# ENHANCEMENTS:
# - Slower FPS (3 instead of 15) for better observation
# - Smart label positioning to handle overlapping nuclei
# - Congestion detection and visual indicators
# - Enhanced text size and contrast for better readability
# - Detailed progress logging with ETA estimates
# - Higher resolution output (2560x1440)

echo "=================================================="
echo "ENHANCED Video Generation Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Enhancements: Smart labeling, congestion detection, better visuals"
echo "=================================================="

# Load required modules following cluster pattern
module purge
module load python
module load cuda cudnn nccl

# Activate virtual environment
source ~/venvs/jupyter-gpu/bin/activate

# Set up environment with optimized threading
export PYTHONPATH="/mnt/home/dchhantyal/3d-cnn-classification:$PYTHONPATH"

# Optimize threading for better CPU utilization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMBA_NUM_THREADS=4

# Navigate to project directory
cd /mnt/home/dchhantyal/3d-cnn-classification

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p video_generation/output

# Dataset and model paths
RAW_DATA_PATH="/mnt/home/dchhantyal/3d-cnn-classification/raw-data/221016_FUCCI_Nanog_stack_3/registered_images"
LABEL_DATA_PATH="/mnt/home/dchhantyal/3d-cnn-classification/raw-data/221016_FUCCI_Nanog_stack_3/registered_label_images"
MODEL_PATH="/mnt/home/dchhantyal/3d-cnn-classification/model/4ncnn/training_outputs/20250710-131550/best_model.pth"
OUTPUT_DIR="/mnt/home/dchhantyal/3d-cnn-classification/video_generation/output/stack3_4ncnn_$(date +%Y%m%d_%H%M%S)"

echo "Data Configuration:"
echo "  Raw data: $RAW_DATA_PATH"
echo "  Label data: $LABEL_DATA_PATH" 
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo ""

# Verify paths exist
if [ ! -d "$RAW_DATA_PATH" ]; then
    echo "❌ ERROR: Raw data path does not exist: $RAW_DATA_PATH"
    exit 1
fi

if [ ! -d "$LABEL_DATA_PATH" ]; then
    echo "❌ ERROR: Label data path does not exist: $LABEL_DATA_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model file does not exist: $MODEL_PATH"
    exit 1
fi

echo "✅ All paths verified"

# Count available frames
RAW_FRAME_COUNT=$(ls "$RAW_DATA_PATH"/nuclei_reg8_*.tif 2>/dev/null | wc -l)
LABEL_FRAME_COUNT=$(ls "$LABEL_DATA_PATH"/label_reg8_*.tif 2>/dev/null | wc -l)

echo "Frame Analysis:"
echo "  Raw frames found: $RAW_FRAME_COUNT"
echo "  Label frames found: $LABEL_FRAME_COUNT"

if [ "$RAW_FRAME_COUNT" -eq 0 ] || [ "$LABEL_FRAME_COUNT" -eq 0 ]; then
    echo "❌ ERROR: No frames found with expected naming pattern"
    echo "Expected patterns: nuclei_reg8_*.tif and label_reg8_*.tif"
    exit 1
fi

if [ "$RAW_FRAME_COUNT" -ne "$LABEL_FRAME_COUNT" ]; then
    echo "⚠️ WARNING: Frame count mismatch between raw and label data"
fi

echo "✅ Frame verification complete"
echo ""

# Print system information
echo "System Information:"
echo "  SLURM_JOB_ID: $SLURM_JOB_ID"
echo "  SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "  SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "  SLURM_CPUS_PER_GPU: $SLURM_CPUS_PER_GPU"
echo "  Node: $SLURMD_NODENAME"
echo "  Memory allocated: $SLURM_MEM_PER_NODE MB"
echo ""

# Print resource limits
echo "Resource Information:"
echo "  CPU cores available: $(nproc)"
echo "  Memory available: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "  Disk space: $(df -h /mnt/home/dchhantyal | tail -1 | awk '{print $4}') available"
echo ""

# Check GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "GPU not available"
echo ""

# Python environment check
echo "Python Environment:"
python --version
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'false')"
echo ""

# Start video generation
echo "🚀 Starting video generation..."
echo "=================================================="

# Run specialized script for stack3 dataset using srun
srun python video_generation/run_stack3_4ncnn.py

EXIT_CODE=$?

echo "=================================================="
echo "Video Generation Job Completed: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Enhanced video generation completed successfully!"
    echo ""
    echo "🎉 ENHANCEMENTS APPLIED:"
    echo "   • Slower FPS (3) for better nucleus observation"
    echo "   • Smart label positioning to avoid overlaps"
    echo "   • Congestion detection with visual indicators" 
    echo "   • Enhanced text size and contrast"
    echo "   • Higher resolution output (2560x1440)"
    echo "   • Detailed progress logging"
    echo ""
    
    # Display output information if directory was created
    OUTPUT_DIR_ACTUAL=$(ls -d /mnt/home/dchhantyal/3d-cnn-classification/video_generation/output/stack3_4ncnn_* 2>/dev/null | tail -1)
    if [ -n "$OUTPUT_DIR_ACTUAL" ] && [ -d "$OUTPUT_DIR_ACTUAL" ]; then
        echo "📁 Output Files:"
        ls -la "$OUTPUT_DIR_ACTUAL"/*.mp4 2>/dev/null || echo "No MP4 files found"
        
        # Show file sizes
        echo ""
        echo "📊 File Sizes:"
        du -h "$OUTPUT_DIR_ACTUAL"/* 2>/dev/null | head -10
        
        # Show summary if available
        if [ -f "$OUTPUT_DIR_ACTUAL/video_summary.json" ]; then
            echo ""
            echo "📈 Summary Statistics:"
            python -c "
import json
with open('$OUTPUT_DIR_ACTUAL/video_summary.json', 'r') as f:
    summary = json.load(f)
    print(f\"  🎬 Video duration: {summary['video_info']['duration_seconds']:.1f} seconds\")
    print(f\"  🖼️ Total frames: {summary['video_info']['frame_count']}\")
    print(f\"  🔬 Total nuclei: {summary['processing_stats']['total_nuclei']}\")
    print(f\"  📊 Avg nuclei/frame: {summary['processing_stats']['avg_nuclei_per_frame']:.1f}\")
" 2>/dev/null || echo "Could not parse summary file"
        fi
        
        echo ""
        echo "🚀 NEXT STEPS:"
        echo "   1. Download the video file for review"
        echo "   2. Check congested regions marked with yellow circles"
        echo "   3. Verify nucleus IDs help distinguish overlapping cells"
        echo "   4. Observe slower pacing allows better analysis"
    fi
    
else
    echo "❌ FAILED: Video generation failed with exit code $EXIT_CODE"
    echo "Check error logs above for details"
fi

echo "=================================================="

# Cleanup if requested and successful
if [ $EXIT_CODE -eq 0 ] && [ "$1" = "--cleanup-temps" ]; then
    echo "🧹 Cleaning up temporary files..."
    if [ -n "$OUTPUT_DIR_ACTUAL" ] && [ -d "$OUTPUT_DIR_ACTUAL" ]; then
        rm -rf "$OUTPUT_DIR_ACTUAL/frames" 2>/dev/null
        rm -rf "$OUTPUT_DIR_ACTUAL"/*.tmp 2>/dev/null
        echo "✅ Cleanup complete"
    fi
fi

exit $EXIT_CODE
