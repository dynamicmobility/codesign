#!/bin/bash

# Run a wandb sweep inside a tmux session.
#
# Usage:
#   bash scripts/sweep.sh config/ppo/sweeps/cheetah_callouts.yaml config/ppo/cheetah.yaml 6
#
# The agent runs NUM_TRIALS arms of the sweep, so a grid sweep needs at least as many
# trials as it has arms to cover them all.

SESSION_NAME="CODESIGN_SWEEP"
PYTHON_SCRIPT="wandb_sweep"
MODULE_NAME="scripts"
CONDA_ENV_NAME="codesign"
SWEEP_FILE=$1
YAML_FILE=$2
NUM_TRIALS=${3:-3}

if [ -z "$SWEEP_FILE" ] || [ -z "$YAML_FILE" ]; then
    echo "Usage: bash scripts/sweep.sh <sweep_yaml> <config_yaml> [num_trials]"
    exit 1
fi

tmux new-session -d -s "$SESSION_NAME"

tmux send-keys -t "$SESSION_NAME" "source .env" C-m
tmux send-keys -t "$SESSION_NAME" "conda activate $CONDA_ENV_NAME" C-m
tmux send-keys -t "$SESSION_NAME" "wandb login" C-m
tmux send-keys -t "$SESSION_NAME" "python3 -m $MODULE_NAME.$PYTHON_SCRIPT --sweep $SWEEP_FILE --config $YAML_FILE --num_trials $NUM_TRIALS" C-m

tmux attach-session -t "$SESSION_NAME"
