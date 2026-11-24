"""Check that the Ray version gives the same VaR/ES as the serial baseline.
Run: python test_ray.py   (small, separate from the team's tests.py)"""
import numpy as np
from mcvar.common import load_params, make_batches, var_es
from run import run_ray, run_serial

N, ASSETS, BATCH, SEED = 2_000_000, 40, 100_000, 7

def losses_of(version, workers):
    mu, cov, w, _ = load_params(ASSETS)
    chol = np.linalg.cholesky(cov)
    batches = make_batches(N, BATCH, SEED)
    return np.concatenate(version(batches, mu, chol, w, workers))

if __name__ == "__main__":
    ref = var_es(losses_of(run_serial, 1))
    for workers in (1, 2):
        got = var_es(losses_of(run_ray, workers))
        # same seeds, same batches, same order -> identical numbers expected
        assert np.allclose(got, ref, rtol=1e-12, atol=0), (workers, got, ref)
    print(f"OK ray equals serial with 1,2 workers: VaR={ref[0]:,.2f} ES={ref[1]:,.2f}")

