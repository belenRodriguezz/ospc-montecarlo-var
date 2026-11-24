#!/bin/bash
#SBATCH --job-name=stage
#SBATCH --partition=cpu
#SBATCH --nodelist=julia
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=results/logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
PY=/data/software/envs/ospc/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -f run.py
test -x "$PY"
echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID"
date; uptime
for REP in 1 2 3; do
echo "=== rep $REP"
"$PY" - <<'PYEOF'
import time
import numpy as np
from mcvar.common import load_params, make_batches, PORTFOLIO_VALUE
n_batches = 100
mu, cov, w, source = load_params(40)
chol = np.linalg.cholesky(cov)
batches = make_batches(n_batches * 100_000, 100_000, 2026)
t_rng = t_mm = t_vec = 0.0
for seed_seq, size in batches:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed_seq)
    z = rng.standard_normal((size, mu.shape[0]))
    t1 = time.perf_counter()
    returns = mu + z @ chol.T
    t2 = time.perf_counter()
    losses = -PORTFOLIO_VALUE * (returns @ w)
    t3 = time.perf_counter()
    t_rng += t1 - t0
    t_mm += t2 - t1
    t_vec += t3 - t2
tot = t_rng + t_mm + t_vec
print(f"source={source} batches={n_batches} scenarios={n_batches*100_000:,} numpy={np.__version__}")
print(f"rng+standard_normal {t_rng:7.3f} s {100*t_rng/tot:5.1f}%")
print(f"mu + z @ chol.T     {t_mm:7.3f} s {100*t_mm/tot:5.1f}%")
print(f"returns @ w         {t_vec:7.3f} s {100*t_vec/tot:5.1f}%")
print(f"total               {tot:7.3f} s ({1000*tot/n_batches:.1f} ms/batch)")
PYEOF
done
date; uptime
echo "done ok"
