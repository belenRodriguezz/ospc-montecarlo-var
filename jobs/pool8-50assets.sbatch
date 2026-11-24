#!/bin/bash
#SBATCH --job-name=pool8
#SBATCH --partition=cpu
#SBATCH --nodelist=julia
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
PY=/data/software/envs/ospc/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -f run.py
test -x "$PY"
OUT="results/pool8_$SLURM_JOB_ID"
mkdir "$OUT"
N=100000000
A=50
echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID out=$OUT n=$N assets=$A workers=8"
date; uptime
taskset -cp $$ || true
scontrol show job "$SLURM_JOB_ID" | grep -E "NumCPUs|CPUs/Task|AllocTRES" || true
# pool run with 8 workers under GNU time
/usr/bin/time -v "$PY" run.py --version pool --workers 8 --n "$N" --assets "$A" --rep 1 --csv "$OUT/pool8.csv" 2> "$OUT/time_pool8.txt" &
POOL_PID=$!
sleep 12
# text snapshot of the processes while the pool runs (not an htop screenshot)
ps -u "$USER" -o pid,ppid,stat,psr,pcpu,rss,args | grep -E "run.py|PID" | grep -v grep > "$OUT/ps_during.txt" || true
cat "$OUT/ps_during.txt"
wait "$POOL_PID"
grep -E "Elapsed|Maximum resident|Minor|Major|Percent of CPU|User time|System time|context" "$OUT/time_pool8.txt"
test -s "$OUT/pool8.csv"
cat "$OUT/pool8.csv"
date; uptime
sleep 20
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,TotalCPU,MaxRSS,NodeList || true
echo "done ok"
