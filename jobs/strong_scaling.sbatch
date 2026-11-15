#!/bin/bash
# Strong scaling: fixed N, workers 1..32, 5 repetitions. One row per run in the CSV.
#   sbatch jobs/strong_scaling.sbatch pool
#   sbatch jobs/strong_scaling.sbatch threads
#
# 64 CPUs in Slurm = 32 physical cores x 2 hardware threads (our max).
#SBATCH --job-name=var-strong
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=results/logs/%x_%j.out

# Activate the course Python environment here if needed (see Blackboard instructions)

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

VERSION=${1:-pool}
N=${N:-100000000}        # ~75 s on one core; raise to reach >= 10 min for the final runs
ASSETS=40
CSV=results/strong_${VERSION}.csv

echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK version=$VERSION N=$N"
lscpu | head -20

# Repetitions in the OUTER loop, so a temporary spike in node load
# affects one point of every worker count instead of all reps of one point.
for rep in 1 2 3 4 5; do
  python run.py --version serial --workers 1 --n $N --assets $ASSETS --rep $rep \
         --csv results/strong_serial_${SLURM_JOB_ID}.csv
  for w in 1 2 4 8 16 32; do
    python run.py --version $VERSION --workers $w --n $N --assets $ASSETS \
           --rep $rep --csv $CSV
  done
done

# CPU time, elapsed time and peak memory of this job, for the characterization
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,TotalCPU,MaxRSS,NodeList
