"""Code shared by every version: model parameters, batches, simulation, VaR/ES.

The versions (serial, threads, pool, ...) differ ONLY in how they run
`simulate_batch` over the list of batches. Everything else lives here.
"""
import os
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "params.npz"
PORTFOLIO_VALUE = 1_000_000.0  # EUR


# ---------------------------------------------------------------- parameters
def synthetic_params(n_assets, seed=12345):
    """Realistic but synthetic daily parameters (used when no calibrated file
    exists, or when we ask for more assets than we have data for, e.g. weak
    scaling in the number of assets)."""
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.15, 0.40, n_assets) / np.sqrt(252)      # daily vol
    mu = rng.uniform(0.00, 0.10, n_assets) / 252                # daily drift
    # one-factor correlation structure: rho_ij = b_i * b_j
    beta = rng.uniform(0.5, 0.85, n_assets)
    corr = np.outer(beta, beta)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(vol, vol)
    return mu, cov


def load_params(n_assets):
    """Return (mu, cov, weights, source). Uses data/params.npz (calibrated on
    Euro Stoxx 50 prices by get_data.py) when it has enough assets."""
    if DATA_FILE.exists():
        d = np.load(DATA_FILE, allow_pickle=True)
        if d["mu"].shape[0] >= n_assets:
            mu = d["mu"][:n_assets]
            cov = d["cov"][:n_assets, :n_assets]
            source = "euro_stoxx_50"
        else:
            mu, cov = synthetic_params(n_assets)
            source = "synthetic"
    else:
        mu, cov = synthetic_params(n_assets)
        source = "synthetic"
    weights = np.full(n_assets, 1.0 / n_assets)                 # equal weights
    return mu, cov, weights, source


# ------------------------------------------------------------------- batches
def make_batches(n_scenarios, batch_size, seed):
    """Split the work into batches, each with its OWN random stream.

    The streams depend only on (seed, batch index), never on the number of
    workers, so every version simulates exactly the same scenarios."""
    sizes = [batch_size] * (n_scenarios // batch_size)
    if n_scenarios % batch_size:
        sizes.append(n_scenarios % batch_size)
    children = np.random.SeedSequence(seed).spawn(len(sizes))
    return list(zip(children, sizes))


def simulate_batch(seed_seq, size, mu, chol, weights, value=PORTFOLIO_VALUE):
    """Simulate `size` one-day scenarios and return only the portfolio losses.

    asset returns R = mu + Z @ L^T  with Z ~ N(0, I)  (multivariate normal)
    loss = -value * (R @ w)
    """
    rng = np.random.default_rng(seed_seq)
    z = rng.standard_normal((size, mu.shape[0]))
    returns = mu + z @ chol.T          # matrix product (BLAS)
    return -value * (returns @ weights)


# ------------------------------------------------------------- risk measures
def var_es(losses, alpha=0.99, method="partition"):
    """VaR = alpha-quantile of the losses, ES = mean of the losses >= VaR.

    method="sort": full O(n log n) sort.
    method="partition": np.partition, O(n) (Matteo's point 2).
    Both use the same index, so they return identical numbers."""
    n = losses.shape[0]
    k = int(np.ceil(alpha * n)) - 1
    if method == "sort":
        ordered = np.sort(losses)
        tail = ordered[k:]
        var = ordered[k]
    elif method == "partition":
        part = np.partition(losses, k)
        tail = part[k:]
        var = part[k]
    else:
        raise ValueError(method)
    return float(var), float(tail.mean())


def closed_form_normal(mu, cov, weights, alpha=0.99, value=PORTFOLIO_VALUE):
    """Exact VaR and ES of a linear portfolio with normal returns (test case)."""
    from statistics import NormalDist
    m = value * float(mu @ weights)
    s = value * float(np.sqrt(weights @ cov @ weights))
    z = NormalDist().inv_cdf(alpha)
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return -m + s * z, -m + s * pdf / (1 - alpha)


# ------------------------------------------------------------------- logging
CSV_HEADER = ("version,workers,n_scenarios,n_assets,batch_size,rep,method,"
              "t_setup,t_simulate,t_quantile,t_total,var,es,"
              "node,job_id,load1,date\n")


def run_info():
    """Node, Slurm job ID and node load, recorded for every run."""
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = float("nan")
    return {
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "load1": load1,
        "date": datetime.now().isoformat(timespec="seconds"),
    }


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a") as f:
        if new:
            f.write(CSV_HEADER)
        f.write(",".join(str(row[c]) for c in CSV_HEADER.strip().split(",")) + "\n")


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
