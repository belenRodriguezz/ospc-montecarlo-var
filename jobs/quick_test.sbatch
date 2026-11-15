#!/bin/bash
# Small job (~1 min): runs the correctness tests + one tiny run of each version.
#   sbatch jobs/quick_test.sbatch
#SBATCH --job-name=var-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=results/logs/%x_%j.out

# Activate the course Python environment here if needed (see Blackboard instructions)
# e.g.  source activate <env>   or   module load <python module>

cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"
python tests.py
for v in serial threads pool; do
  python run.py --version $v --workers 4 --n 4000000
done
