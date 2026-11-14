#!/bin/bash
# One single-GPU job per (fold, strategy, patch size): 5 x 3 x 2 = 30 independent runs.
# No code changes needed; this is the cheapest way to use many GPUs.
#
#   sbatch slurm/amos_grid.sbatch                  # all 30
#   sbatch --array=10-14 slurm/amos_grid.sbatch    # e.g. only LocBAM, small patches
#
# Edit DATA_DIR / CACHE_DIR / VENV and the partition line for your cluster.
#SBATCH --job-name=locbam-amos
#SBATCH --array=0-29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%A_%a.out
##SBATCH --partition=<your-gpu-partition>

set -euo pipefail
DATA_DIR=/vol/nasciam/datasets/amos_bpr/
CACHE_DIR=/vol/ciamspace/datasets/amos_bpr/cache   # delete after changing deterministic transforms
VENV=$HOME/venvs/locbam/bin/activate               # Python 3.10/3.11 + requirements.txt

FOLDS=(0 1 2 3 4)
STRATS=(Baseline XYZ LocBAM)
PS=(small large)
i=$SLURM_ARRAY_TASK_ID
fold=${FOLDS[$((i % 5))]}
strat=${STRATS[$(((i / 5) % 3))]}
ps=${PS[$((i / 15))]}

source "$VENV"
cd "$SLURM_SUBMIT_DIR/BasicUNet"
echo "fold=$fold strategy=$strat ps=$ps"
python amos_loc_context.py -fold "$fold" -strategy "$strat" -ps "$ps" \
    -data_dir "$DATA_DIR" -cache_dir "$CACHE_DIR"
