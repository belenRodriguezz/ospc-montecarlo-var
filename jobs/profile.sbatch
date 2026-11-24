#!/bin/bash
#SBATCH --job-name=prof
#SBATCH --partition=cpu
#SBATCH --nodelist=julia
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
PY=/data/software/envs/ospc/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -f run.py
test -x "$PY"
OUT="results/prof_$SLURM_JOB_ID"
mkdir "$OUT"
echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID out=$OUT"
date; uptime
"$PY" -m cProfile -o "$OUT/prof_serial_100M.prof" run.py --version serial --workers 1 --n 100000000 --rep 1 --csv "$OUT/prof_serial.csv"
test -s "$OUT/prof_serial_100M.prof"
test -s "$OUT/prof_serial.csv"
"$PY" -c "import pstats,sys; pstats.Stats(sys.argv[1]).sort_stats('cumulative').print_stats(12)" "$OUT/prof_serial_100M.prof"
"$PY" -c "import pstats,sys; pstats.Stats(sys.argv[1]).sort_stats('tottime').print_stats(8)" "$OUT/prof_serial_100M.prof"
date; uptime
echo "done ok"
