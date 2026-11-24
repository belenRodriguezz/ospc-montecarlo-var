#!/bin/bash
# Calibration: how long does the SERIAL baseline take for growing N? One core, one rep each.
# Local helper, run from the repo root:  mkdir -p results/logs && sbatch calibrate.sbatch
#SBATCH --job-name=calib
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR"
export PATH=/data/software/envs/ospc/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID"
for N in 100000000 200000000 400000000; do
  echo "=== N=$N"
  /usr/bin/time -v python run.py --version serial --workers 1 --n $N --rep 1 --csv results/calib_serial.csv
done
