#!/bin/bash
# Person 3 - oversubscription (nested parallelism) experiment.
#   sbatch jobs/oversub.sbatch
# Each pool worker also starts BLAS_THREADS OpenBLAS threads.
# 32 workers x 4 BLAS threads = 128 threads on 32 physical cores.
#SBATCH --job-name=var-oversub
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=results/logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR"
export PATH=/data/software/envs/ospc/bin:$PATH

N=${N:-100000000}
ASSETS=${ASSETS:-50}

echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK N=$N assets=$ASSETS"
cat /proc/loadavg

for rep in 1 2 3 4 5; do
  for bt in 1 2 4; do
    for w in 16 32; do
      # run.py copies BLAS_THREADS into OMP/OPENBLAS/MKL_NUM_THREADS before
      # importing numpy, and every pool worker inherits it.
      # One CSV per BLAS-thread setting: the file name says which.
      BLAS_THREADS=$bt python run.py --version pool --workers $w --n $N \
             --assets $ASSETS --rep $rep --csv results/oversub_bt${bt}.csv
    done
  done
done

sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,TotalCPU,MaxRSS,NodeList
