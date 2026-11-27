"""Correctness checks. Run:  python tests.py

1. The baseline matches the exact formula for normal returns.
2. Every parallel version returns the same VaR/ES as the serial baseline.
3. The sort and partition methods give identical answers.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

from mcvar.common import (closed_form_normal, load_params, make_batches,
                          simulate_batch, var_es)
from run import run_blas, run_pool, run_serial, run_threads

N, ASSETS, BATCH, SEED = 2_000_000, 40, 100_000, 7


def losses_of(version, workers):
    mu, cov, w, _ = load_params(ASSETS)
    chol = np.linalg.cholesky(cov)
    batches = make_batches(N, BATCH, SEED)
    return np.concatenate(version(batches, mu, chol, w, workers))


def test_closed_form():
    mu, cov, w, _ = load_params(ASSETS)
    var_mc, es_mc = var_es(losses_of(run_serial, 1))
    var_ex, es_ex = closed_form_normal(mu, cov, w)
    # Monte Carlo error of a 99% quantile with 2M scenarios is ~0.2%; allow 1%.
    assert abs(var_mc / var_ex - 1) < 0.01, (var_mc, var_ex)
    assert abs(es_mc / es_ex - 1) < 0.01, (es_mc, es_ex)
    print(f"OK closed form: VaR MC={var_mc:,.2f} exact={var_ex:,.2f} | "
          f"ES MC={es_mc:,.2f} exact={es_ex:,.2f}")


def test_versions_equal():
    ref = var_es(losses_of(run_serial, 1))
    for name, fn in [("threads", run_threads), ("pool", run_pool), ("blas", run_blas)]:
        for workers in (1, 2, 4):
            got = var_es(losses_of(fn, workers))
            # Same seeds, same batches, same order -> expect identical numbers.
            # Tolerance only guards against BLAS differences across threads.
            assert np.allclose(got, ref, rtol=1e-12, atol=0), (name, workers, got, ref)
        print(f"OK versions equal: serial/threads/pool/blas with 1,2,4 workers -> "
          f"VaR={ref[0]:,.2f} ES={ref[1]:,.2f}")


def test_sort_vs_partition():
    losses = losses_of(run_serial, 1)
    v_s, es_s = var_es(losses, method="sort")
    v_p, es_p = var_es(losses, method="partition")
    # VaR is the same element -> exactly equal. ES is a mean of the same tail
    # values, but summed in a different order (partition leaves the tail
    # unsorted), so floating-point rounding can differ in the last digits.
    assert v_s == v_p, (v_s, v_p)
    assert np.isclose(es_s, es_p, rtol=1e-12, atol=0), (es_s, es_p)
    print(f"OK sort vs partition: VaR identical, ES differs by "
          f"{abs(es_s - es_p) / es_s:.1e} (relative, summation order)")


if __name__ == "__main__":
    test_closed_form()
    test_versions_equal()
    test_sort_vs_partition()
    print("ALL TESTS PASSED")
