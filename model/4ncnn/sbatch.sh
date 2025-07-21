#!/bin/bash -l

#SBATCH --job-name=3classes-no-mask
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
# This is the physical number of GPUs per node
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G

module purge
module load python
module load cuda cudnn nccl

source ~/venvs/jupyter-gpu/bin/activate

master_node=$SLURMD_NODENAME

srun python $(which torchrun) \
        --nnodes $SLURM_JOB_NUM_NODES \
        --nproc_per_node $SLURM_GPUS_PER_NODE \
        --rdzv_id $SLURM_JOB_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint $master_node:29500 \
        cnn_model.py
