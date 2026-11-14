#!/bin/bash
# One run split across GPUs on one node with DDP (torchrun).
# The global batch stays at the paper's value and is split across GPUs:
#   -ps small: 128 total, so GPUS must divide 128
#   -ps large: 2 total, so GPUS can be at most 2
#
#   sbatch slurm/amos_ddp.sbatch 0 LocBAM small
#
# Build the PersistentDataset cache once beforehand (one single-GPU run), so the
# processes don't all write it at the same time on the first epoch.
#SBATCH --job-name=locbam-ddp
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
##SBATCH --partition=<your-gpu-partition>

set -euo pipefail
FOLD=${1:-0}
STRAT=${2:-LocBAM}
PS=${3:-small}
GPUS=2   # keep equal to --gres=gpu:N above

DATA_DIR=/vol/nasciam/datasets/amos_bpr/
CACHE_DIR=/vol/ciamspace/datasets/amos_bpr/cache
VENV=$HOME/venvs/locbam/bin/activate

source "$VENV"
cd "$SLURM_SUBMIT_DIR/BasicUNet"
torchrun --standalone --nproc_per_node="$GPUS" amos_loc_context.py \
    -fold "$FOLD" -strategy "$STRAT" -ps "$PS" \
    -data_dir "$DATA_DIR" -cache_dir "$CACHE_DIR"
