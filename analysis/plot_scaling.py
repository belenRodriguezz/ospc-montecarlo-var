"""Strong scaling plots from results/*.csv:  python analysis/plot_scaling.py

Produces results/strong_scaling.png (run time and speedup) and prints a table
with median, min and max of every configuration.
"""
import glob
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"pool": "#2a78d6", "threads": "#eb6834", "dask": "#1baf7a", "ray": "#eda100"}

df = pd.concat(pd.read_csv(f) for f in glob.glob(str(ROOT / "results" / "strong_*.csv")))
t_serial = df[df.version == "serial"].t_total.median()

stats = (df[df.version != "serial"]
         .groupby(["version", "workers"]).t_total
         .agg(median="median", tmin="min", tmax="max", runs="count")
         .reset_index())
stats["speedup"] = t_serial / stats["median"]
stats["efficiency"] = stats["speedup"] / stats["workers"]
print(f"serial baseline median: {t_serial:.2f} s "
      f"({(df.version == 'serial').sum()} runs)\n")
print(stats.round(3).to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for v, g in stats.groupby("version"):
    c = COLORS.get(v, "#555555")
    err = [g["median"] - g.tmin, g.tmax - g["median"]]
    ax1.errorbar(g.workers, g["median"], yerr=err, marker="o", ms=6, lw=2,
                 capsize=3, color=c, label=v)
    s_err = [g.speedup - t_serial / g.tmax, t_serial / g.tmin - g.speedup]
    ax2.errorbar(g.workers, g.speedup, yerr=s_err, marker="o", ms=6, lw=2,
                 capsize=3, color=c, label=v)

w = sorted(stats.workers.unique())
ax2.plot(w, w, ls="--", lw=1, color="#999999", label="ideal")
for ax in (ax1, ax2):
    ax.set_xscale("log", base=2)
    ax.set_xticks(w, [str(x) for x in w])
    ax.set_xlabel("workers")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
ax1.set_yscale("log")
ax1.set_ylabel("run time (s), median, bars = min-max")
ax1.set_title("Run time")
ax2.set_yscale("log", base=2)
ax2.set_ylabel("speedup vs serial baseline")
ax2.set_title("Speedup")
ax2.legend(frameon=False)
for ax in (ax1, ax2):
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(ScalarFormatter())
n = int(df.n_scenarios.iloc[0])
fig.suptitle(f"Strong scaling, N = {n:,} scenarios, {int(df.n_assets.iloc[0])} assets")
fig.tight_layout()
out = ROOT / "results" / "strong_scaling.png"
fig.savefig(out, dpi=150)
print(f"\nsaved {out}")
