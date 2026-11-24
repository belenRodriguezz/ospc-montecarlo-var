#!/bin/bash
#SBATCH --job-name=pool8htop
#SBATCH --partition=cpu
#SBATCH --nodelist=julia
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=results/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
PY=/data/software/envs/ospc/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -f run.py
test -x "$PY"
OUT="results/pool8htop_$SLURM_JOB_ID"
mkdir "$OUT"
N=600000000
echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID out=$OUT n=$N assets=50 workers=8"
date
taskset -cp $$ || true
echo "waiting 30 s so htop can be attached"
sleep 30
date
/usr/bin/time -v "$PY" run.py --version pool --workers 8 --n "$N" --assets 50 --rep 1 --csv "$OUT/pool8_600M.csv" 2> "$OUT/time_pool8_600M.txt" &
POOL_PID=$!
echo "pool started pid=$POOL_PID"
# text snapshots of the busiest processes while it runs (fallback for the htop screenshot)
for t in 10 25 40; do
  sleep $(( t == 10 ? 10 : 15 ))
  echo "=== top snapshot at about ${t} s"
  COLUMNS=200 top -b -n 1 -u "$USER" | head -22 || true
done
wait "$POOL_PID"
echo "pool exit=0"
grep -E "Elapsed|Maximum resident|Percent of CPU|User time|System time" "$OUT/time_pool8_600M.txt"
test -s "$OUT/pool8_600M.csv"
cat "$OUT/pool8_600M.csv"
date
echo "done ok"
