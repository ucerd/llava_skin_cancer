#!/bin/bash
#SBATCH --job-name=derma
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.out
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [ -n "${DERMAATTR_VENV:-}" ]; then
    source "$DERMAATTR_VENV/bin/activate"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
: "${CFG:?}"; : "${SEED:=0}"; GEN="${GEN:-0}"
echo "[$(hostname)] START ${CFG} s${SEED} gen=${GEN} $(date)"
GENFLAG=""; [ "$GEN" = "1" ] && GENFLAG="--gen"
python -u run_config.py --config "$CFG" --seed "$SEED" --epochs 3 \
    --per_image_cap 6 --accum 8 --gen_n 250 $GENFLAG
echo "[$(hostname)] DONE ${CFG} s${SEED} rc=$? $(date)"
