#!/bin/bash
# Person 3 - strong scaling of the BLAS version (parallelism inside OpenBLAS).
#   sbatch jobs/blas_scaling.sbatch
#SBATCH --job-name=var-blas
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=01:45:00
#SBATCH --output=results/logs/%x_%j.out

cd "$SLURM_SUBMIT_DIR"
export PATH=/data/software/envs/ospc/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

N=${N:-100000000}
ASSETS=${ASSETS:-50}

echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK N=$N assets=$ASSETS"
cat /proc/loadavg
python -c "import numpy, threadpoolctl; print(threadpoolctl.threadpool_info())"

for rep in 1 2 3 4 5; do
  # serial baseline on the same node, for the speedup
  python run.py --version serial --workers 1 --n $N --assets $ASSETS --rep $rep \
         --csv results/blas_serial.csv
  # BLAS version, team batch size
  for w in 1 2 4 8 16 32 64; do
    python run.py --version blas --workers $w --n $N --assets $ASSETS \
           --batch-size 100000 --rep $rep --csv results/blas_strong.csv
  done
  # bigger batches: do bigger matrices help the BLAS threads?
  for w in 1 8 32; do
    python run.py --version blas --workers $w --n $N --assets $ASSETS \
           --batch-size 1000000 --rep $rep --csv results/blas_strong.csv
  done
done

sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,TotalCPU,MaxRSS,NodeList
