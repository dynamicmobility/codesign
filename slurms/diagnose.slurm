#!/bin/bash
#SBATCH --job-name=diagnose
#SBATCH --account=gts-mtucker34
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=njanwani@gatech.edu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source ~/.bashrc
module load anaconda3
conda activate codesign

cd /storage/home/hcoda1/3/njanwani3/Documents/codesign

SCRATCH=/storage/home/hcoda1/3/njanwani3/scratch

mkdir -p "$SCRATCH/runs"
mkdir -p "$SCRATCH/wandb"
mkdir -p "$SCRATCH/.cache/wandb"
mkdir -p logs

module load cuda

export WANDB_DIR="$SCRATCH/wandb"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"

nvidia-smi                      # driver version + max CUDA version supported
python -c "import jax; print(jax.__version__)"   # or torch/tensorflow