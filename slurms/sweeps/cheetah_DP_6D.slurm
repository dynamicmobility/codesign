#!/bin/bash
#SBATCH --job-name=moco_hyper_sweep_trial
#SBATCH --account=gts-mtucker34
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=njanwani@gatech.edu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source ~/.bashrc
module load anaconda3
conda activate codesign-scratch

cd /storage/home/hcoda1/3/njanwani3/Documents/codesign

SCRATCH=/storage/home/hcoda1/3/njanwani3/scratch

mkdir -p "$SCRATCH/runs"
mkdir -p "$SCRATCH/wandb"
mkdir -p "$SCRATCH/.cache/wandb"
mkdir -p logs

module load cuda

export WANDB_DIR="$SCRATCH/wandb"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"

python3 -m scripts.wandb_sweep \
--config config/mo_design_predictor_hypernetwork/cheetah6D.yaml \
--sweep config/mo_design_predictor_hypernetwork/sweeps/cheetah6D.yaml \
--num_trials 40 \
--pace True