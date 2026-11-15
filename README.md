# Monte Carlo VaR / ES of a Euro Stoxx 50 portfolio — OSPC Group Project (New Group 2)

What is the 99% one-day Value-at-Risk and Expected Shortfall of an equally
weighted portfolio of Euro Stoxx 50 stocks (EUR 1M)?

## Layout
```
mcvar/common.py          shared code: parameters, batches + seeds, simulation, VaR/ES
run.py                   command line entry point; versions: serial, threads, pool
tests.py                 correctness: closed form (normal case), all versions equal, sort vs partition
get_data.py              calibrate on Euro Stoxx 50 prices -> data/params.npz (run on laptop/Colab)
jobs/quick_test.sbatch   small test job (~1 min)
jobs/strong_scaling.sbatch   strong scaling, workers 1..32, 5 repetitions
analysis/plot_scaling.py tables + plots from results/*.csv
results/                 raw CSVs (one row per run), logs, plots
```

## Quick start on the cluster (< 5 minutes)
```bash
git clone <repo-url> && cd <repo>
mkdir -p results/logs
python tests.py                                      # correctness checks
python run.py --version pool --workers 2 --n 2000000 # tiny run
sbatch jobs/quick_test.sbatch                        # same, as a batch job
```

## Experiments
```bash
sbatch jobs/strong_scaling.sbatch pool
sbatch jobs/strong_scaling.sbatch threads
squeue -u $USER                       # watch the queue
python analysis/plot_scaling.py       # after the jobs finish -> results/strong_scaling.png
```
Change the problem size with `N=200000000 sbatch jobs/strong_scaling.sbatch pool`.

## Design notes
- **Random streams**: the scenarios are split into batches of `--batch-size`;
  each batch gets its own stream from `SeedSequence(seed).spawn(...)`. The
  streams depend on the batch index, not on the worker, so all versions and
  all worker counts simulate exactly the same scenarios. Results are gathered
  in batch order, so the order in which workers finish does not matter.
- **BLAS threads** are set to 1 in every version (`run.py`, top of file, and
  in the job scripts), so worker threads/processes and OpenBLAS threads do not
  oversubscribe the cores.
- **Memory**: workers return only the loss vector (8 bytes per scenario).
  Peak per worker ≈ 2 × batch_size × assets × 8 bytes (64 MB with defaults).
- **Timing**: setup (parameters, Cholesky, seeds), simulation (parallel part)
  and quantile (gather + VaR/ES, serial part) are timed separately.
- **Quantile**: `--method partition` (np.partition, O(n), default) or
  `--method sort` (O(n log n)). VaR is identical; ES can differ in the last
  digits because the tail is summed in a different order.

## Generative AI
Parts of the initial code were drafted with Claude (Anthropic) and reviewed /
tested by the team. (Complete in the format required by the syllabus.)
