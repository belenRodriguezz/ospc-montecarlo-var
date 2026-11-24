"""Monte Carlo VaR / ES of a Euro Stoxx 50 portfolio.

Usage examples:
    python run.py --version serial  --n 1000000
    python run.py --version pool    --workers 8  --n 100000000
    python run.py --version threads --workers 8  --n 100000000

Every version simulates the SAME batches (same seeds), so they must return
the same VaR and ES as the serial baseline.
"""
# --- BLAS threads: set BEFORE numpy is imported (Matteo's point 3) ---------
# Our parallelism comes from the workers, not from OpenBLAS/MKL. If we did
# not set this, each worker would also start its own BLAS threads and they
# would all compete for the same cores. Worker processes inherit these.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = os.environ.get("BLAS_THREADS", "1")

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from mcvar.common import (Timer, append_csv, load_params, make_batches,
                          run_info, simulate_batch, var_es)

# ---------------------------------------------------------------- versions
# Each takes the list of batches and returns the list of loss arrays,
# IN BATCH ORDER (so the order in which workers finish cannot change it).


def run_serial(batches, mu, chol, w, workers):
    return [simulate_batch(s, n, mu, chol, w) for s, n in batches]


def run_threads(batches, mu, chol, w, workers):
    # NumPy releases the GIL inside the RNG and the matrix product,
    # so threads may scale even though this is Python.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda b: simulate_batch(b[0], b[1], mu, chol, w),
                           batches))


# process pool: parameters are sent ONCE to each worker by the initializer
_P = {}


def _init_worker(mu, chol, w):
    _P["mu"], _P["chol"], _P["w"] = mu, chol, w


def _pool_task(batch):
    s, n = batch
    return simulate_batch(s, n, _P["mu"], _P["chol"], _P["w"])


def run_pool(batches, mu, chol, w, workers):
    ctx = mp.get_context("fork")   # Linux; explicit so it is reproducible
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_init_worker,
                             initargs=(mu, chol, w)) as ex:
        # chunksize=1: batches are already large units of work
        return list(ex.map(_pool_task, batches, chunksize=1))

# Ray: tasks run in worker processes started by Ray. The shared inputs go into
# Ray's object store ONCE; every task receives a reference, not a copy.
def run_ray(batches, mu, chol, w, workers):
    import ray  # imported here so the other versions work without Ray installed

    @ray.remote
    def ray_task(seed_seq, size, mu_ref, chol_ref, w_ref):
        return simulate_batch(seed_seq, size, mu_ref, chol_ref, w_ref)

    ray.init(num_cpus=workers, include_dashboard=False, logging_level="ERROR")
    try:
        mu_ref, chol_ref, w_ref = ray.put(mu), ray.put(chol), ray.put(w)
        futures = [ray_task.remote(s, n, mu_ref, chol_ref, w_ref)
                   for s, n in batches]
        return ray.get(futures)  # submission order = batch order
    finally:
        ray.shutdown()
        
VERSIONS = {"serial": run_serial, "threads": run_threads, "pool": run_pool,
            "ray": run_ray}



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=VERSIONS, default="serial")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--n", type=int, default=1_000_000, help="number of scenarios")
    p.add_argument("--assets", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--alpha", type=float, default=0.99)
    p.add_argument("--method", choices=["partition", "sort"], default="partition")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--csv", default=None, help="append one row per run")
    a = p.parse_args()

    with Timer() as t_setup:              # serial: parameters + Cholesky + seeds
        mu, cov, w, source = load_params(a.assets)
        chol = np.linalg.cholesky(cov)
        batches = make_batches(a.n, a.batch_size, a.seed)

    with Timer() as t_sim:                # the parallel part
        parts = VERSIONS[a.version](batches, mu, chol, w, a.workers)

    with Timer() as t_q:                  # serial: gather + quantile
        losses = np.concatenate(parts)
        var, es = var_es(losses, a.alpha, a.method)

    total = t_setup.elapsed + t_sim.elapsed + t_q.elapsed
    print(f"{a.version:8s} workers={a.workers:<3d} n={a.n:,} assets={a.assets} "
          f"({source})  VaR={var:,.2f}  ES={es:,.2f}  "
          f"sim={t_sim.elapsed:.2f}s  quantile={t_q.elapsed:.2f}s  total={total:.2f}s")

    if a.csv:
        row = dict(version=a.version, workers=a.workers, n_scenarios=a.n,
                   n_assets=a.assets, batch_size=a.batch_size, rep=a.rep,
                   method=a.method, t_setup=f"{t_setup.elapsed:.4f}",
                   t_simulate=f"{t_sim.elapsed:.4f}", t_quantile=f"{t_q.elapsed:.4f}",
                   t_total=f"{total:.4f}", var=repr(var), es=repr(es), **run_info())
        append_csv(a.csv, row)


if __name__ == "__main__":
    main()
