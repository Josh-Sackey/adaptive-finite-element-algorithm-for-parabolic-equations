"""Comparison plots and summary table for the Allen-Cahn convergence campaign.

Usage:  python plot_results.py [container-output directory]
Writes figures/*.png and figures/summary.csv inside that directory.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "container-output")
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# Colour = mesh resolution, line style = time step (secondary encoding).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CASES = {
    "baseline":     dict(label="baseline (coarse, Δt=3e-4)",       color=BLUE,   ls="-"),
    "time-fine":    dict(label="time-fine (coarse, Δt=1.5e-4)",    color=BLUE,   ls="--"),
    "spatial-fine": dict(label="spatial-fine (fine, Δt=3e-4)",     color=ORANGE, ls="-"),
    "production":   dict(label="production (fine, Δt=1.5e-4)",     color=ORANGE, ls="--"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.labelcolor": INK2,
    "lines.linewidth": 1.6, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load(case):
    d = ROOT / case
    sim = pd.read_csv(d / "simulation-history.csv")
    nn = pd.read_csv(d / "neural-training.csv")
    newton = pd.read_csv(d / "newton-history.csv")
    return sim, nn, newton


data = {c: load(c) for c in CASES if (ROOT / c / "simulation-history.csv").exists()}


def style(ax, xlabel="time t", ylabel=""):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=0)


def shared_legend(fig, axes, ncol=4):
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.0))


# ---------------------------------------------------------------- Figure 1
fig, axs = plt.subplots(2, 2, figsize=(10, 7))
axs = axs.ravel()
for c, s in CASES.items():
    if c not in data:
        continue
    sim = data[c][0]
    kw = dict(color=s["color"], ls=s["ls"], label=s["label"])
    axs[0].plot(sim.time, sim.energy, **kw)
    axs[1].plot(sim.time, sim.dofs, **kw)
    overshoot = (sim[["max_solution"]].values.ravel() - 1).clip(min=0)
    undershoot = (-1 - sim[["min_solution"]].values.ravel()).clip(min=0)
    axs[2].plot(sim.time, 100 * pd.Series(overshoot).combine(pd.Series(undershoot), max), **kw)
    axs[3].plot(sim.time, sim.elapsed_seconds, **kw)
axs[0].set_title("Discrete energy")
style(axs[0], ylabel="energy")
axs[1].set_title("Degrees of freedom")
style(axs[1], ylabel="DoFs")
axs[2].set_title("Overshoot beyond [-1, 1]")
style(axs[2], ylabel="max(|u|) − 1   (%)")
axs[3].set_title("Cumulative wall time")
style(axs[3], ylabel="seconds")
shared_legend(fig, axs)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "fig1-campaign-comparison.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
for c, s in CASES.items():
    if c not in data:
        continue
    sim, nn, newton = data[c]
    kw = dict(color=s["color"], ls=s["ls"], label=s["label"])
    later = nn.iloc[1:]  # first fit is the long 3000-iteration cold start
    axs[0].plot(later.time, later.final_mse, **kw)
    axs[1].plot(later.time, later.seconds, **kw)
    per_step = newton.groupby("step").iteration.max()
    times = newton.groupby("step").time.first()
    axs[2].plot(times, per_step, **kw, drawstyle="steps-mid")
axs[0].set_yscale("log")
axs[0].set_title("NN fit error (warm-started fits)")
style(axs[0], ylabel="final MSE")
axs[1].set_title("NN training time per fit")
style(axs[1], ylabel="seconds")
axs[2].set_title("Newton corrections per time step")
style(axs[2], ylabel="max iterations over adaptive solves")
shared_legend(fig, axs)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(OUT / "fig2-neural-and-newton.png")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3
repro = {
    "baseline":                ("campaign run, 4 threads", BLUE, "-"),
    "repro-baseline-threads4": ("repeat, 4 threads", AQUA, ":"),
    "repro-baseline-threads1": ("repeat, 1 thread", ORANGE, "-"),
}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for c, (label, color, ls) in repro.items():
    p = ROOT / c / "simulation-history.csv"
    if not p.exists():
        continue
    sim = pd.read_csv(p)
    lw = 2.6 if ls == ":" else 1.6
    axs[0].plot(sim.time, sim.energy, color=color, ls=ls, lw=lw, label=label)
    axs[1].plot(sim.time, sim.dofs, color=color, ls=ls, lw=lw, label=label)
axs[0].scatter([0.03], [141.020], marker="D", s=40, color=INK, zorder=5,
               label="Joshua's reference (T=0.03)")
axs[1].scatter([0.03], [685], marker="D", s=40, color=INK, zorder=5,
               label="Joshua's reference (T=0.03)")
axs[0].set_title("Baseline reproducibility: energy")
style(axs[0], ylabel="energy")
axs[1].set_title("Baseline reproducibility: DoFs")
style(axs[1], ylabel="DoFs")
shared_legend(fig, axs)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(OUT / "fig3-reproducibility.png")
plt.close(fig)

# ---------------------------------------------------------------- Summary table
rows = []
for c in ["smoke", *CASES, "repro-baseline-threads1", "repro-baseline-threads4"]:
    if not (ROOT / c / "simulation-history.csv").exists():
        continue
    sim, nn, newton = load(c)
    last = sim.iloc[-1]
    rows.append({
        "case": c,
        "time_steps": int(last.step),
        "final_time": last.time,
        "final_cells": int(last.active_cells),
        "final_dofs": int(last.dofs),
        "max_dofs": int(sim.dofs.max()),
        "final_energy": round(last.energy, 4),
        "final_mass": round(last.mass, 5),
        "min_solution": round(sim.min_solution.min(), 5),
        "max_solution": round(sim.max_solution.max(), 5),
        "energy_increases": int((sim.energy.diff() > 0).sum()),
        "max_newton_iterations": int(newton.groupby("step").iteration.max().max()),
        "nn_fits": len(nn),
        "nn_initial_mse": nn.final_mse.iloc[0],
        "nn_max_mse": nn.final_mse.max(),
        "nn_final_mse": nn.final_mse.iloc[-1],
        "nn_training_seconds": round(nn.seconds.sum(), 2),
        "wall_seconds": round(last.elapsed_seconds, 2),
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "summary.csv", index=False)
print(summary.to_string(index=False))
print(f"\nFigures written to {OUT}")
