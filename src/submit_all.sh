#!/bin/bash
# Submit all training/evaluation jobs in parallel across GPU nodes.
# Each config trains and evaluates in one job (slurm_run.sh); LLaVA prompt
# baselines run in a separate job (slurm_llava.sh).
set -u
cd "$(dirname "$0")"
mkdir -p logs
for seed in 0 1 2; do
  for cfg in full no_attr no_rec no_gate no_dist oracle shuffled; do
    gen=0; [ "$cfg" = full ] && gen=1
    jid=$(sbatch --parsable --job-name="d_${cfg}_s${seed}" \
      --export=ALL,CFG=$cfg,SEED=$seed,GEN=$gen slurm_run.sh)
    echo "submitted ${cfg} seed ${seed} -> job ${jid}"
  done
done
jid=$(sbatch --parsable --job-name=d_llava slurm_llava.sh)
echo "submitted llava baselines -> job ${jid}"
echo "Track with: squeue -u \$USER"
echo "After completion: python stats.py full_s0 resnet50 && python make_figures.py"
