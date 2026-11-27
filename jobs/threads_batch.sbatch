#!/bin/bash
# Person 3 - why do threads stop scaling after 8-16 workers?
#   sbatch jobs/threads_batch.sbatch
# Same N, three batch sizes. If bigger batches help at 16-32 threads, the
# cost is per batch (Python code between NumPy calls holds the GIL, array
# allocation in one shared process). If not, look at memory bandwidth.
# Memory: each thread holds ~2 x batch x assets x 8 bytes (400k -> 320 MB).
#SBATCH --job-name=var-thbatch
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=results/logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR"
export PATH=/data/software/envs/ospc/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

N=${N:-100000000}
ASSETS=${ASSETS:-50}

echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK N=$N assets=$ASSETS"
cat /proc/loadavg

for rep in 1 2 3 4 5; do
  for bs in 25000 100000 400000; do
    for w in 8 16 32; do
      python run.py --version threads --workers $w --n $N --assets $ASSETS \
             --batch-size $bs --rep $rep --csv results/threads_batch.csv
    done
  done
done

sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,TotalCPU,MaxRSS,NodeList
