#!/bin/bash
#SBATCH --job-name=derma_llava
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
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
echo "[$(hostname)] llava baselines START $(date)"
python -u llava_baselines.py
echo "[$(hostname)] llava baselines DONE rc=$? $(date)"
