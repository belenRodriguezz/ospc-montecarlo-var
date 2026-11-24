#!/bin/bash
#SBATCH --job-name=strace
#SBATCH --partition=cpu
#SBATCH --nodelist=julia
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=results/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
PY=/data/software/envs/ospc/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -f run.py
test -x "$PY"
test -x /usr/bin/strace
OUT="results/strace_$SLURM_JOB_ID"
mkdir "$OUT"
echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID out=$OUT"
date; uptime
/usr/bin/time -v strace -c -f -o "$OUT/strace_serial_10M.txt" "$PY" run.py --version serial --workers 1 --n 10000000 --rep 1 --csv "$OUT/strace_serial.csv"
test -s "$OUT/strace_serial_10M.txt"
head -25 "$OUT/strace_serial_10M.txt"
date; uptime
echo "done ok"
