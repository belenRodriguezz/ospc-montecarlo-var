#!/bin/bash
# Ray strong scaling: fixed N, workers 1..32, 5 repetitions. Ray only (serial runs are done by Person 5).
#   N=100000000 ASSETS=50 sbatch jobs/ray_scaling.sbatch
#SBATCH --job-name=var-ray
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=results/logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/pc55/ospc-ray-venv/bin/python
N=${N:-100000000}
ASSETS=${ASSETS:-50}
CSV=results/strong_ray.csv

echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK version=ray N=$N assets=$ASSETS"
for rep in 1 2 3 4 5; do
  for w in 1 2 4 8 16 32; do
    $PY run.py --version ray --workers $w --n $N --assets $ASSETS --rep $rep --csv $CSV
  done
done
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,TotalCPU,MaxRSS,NodeList

