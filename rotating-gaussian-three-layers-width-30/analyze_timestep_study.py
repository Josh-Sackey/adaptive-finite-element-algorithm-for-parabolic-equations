#!/usr/bin/env python3
"""Combine classical and NN histories from the timestep sweep."""

import csv
import sys
from pathlib import Path

METHODS = (
    ("classical", "Classical SolutionTransfer"),
    ("nn-once-per-time", "NN once per time level"),
)


def dt_slug(dt):
    return f"dt-{dt:.5f}".replace(".", "p")


def read_history(root, dt, method):
    path = root / dt_slug(dt) / method / "simulation-history.csv"
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["phase"] == "time_step"]
    return {round(float(row["time"]), 10): row for row in rows}


def training_totals(root, dt, method):
    if method == "classical":
        return 0, 0.0, 0.0
    path = root / dt_slug(dt) / method / "neural-training.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    seconds = [float(row["seconds"]) for row in rows]
    return len(rows), sum(seconds), seconds[-1] if seconds else 0.0


def write_svg(root, histories, dts):
    import math

    width, height = 1200, 560
    panels = ((60, "l2_error", "L2 error"),
              (630, "h1_seminorm", "H1 seminorm error"))
    colors = {0.01: "#1f77b4", 0.005: "#ff7f0e", 0.0025: "#2ca02c"}
    dashes = {"classical": "", "nn-once-per-time": "8,5"}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:sans-serif;fill:#222}.title{font-size:16px;font-weight:bold}.label{font-size:12px}</style>']
    for x0, metric, title in panels:
        left, right, top, bottom = x0 + 70, x0 + 530, 55, 390
        series = []
        for dt in dts:
            for method, label in METHODS:
                history = histories[(dt, method)]
                points = [(t, float(history[t][metric])) for t in sorted(history)]
                series.append((dt, method, label, points))
        xs = [x for _, _, _, points in series for x, _ in points]
        ys = [math.log10(max(y, 1e-300)) for _, _, _, points in series for _, y in points]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        sx = lambda x: left + (x - xmin) / (xmax - xmin) * (right - left)
        sy = lambda y: bottom - (math.log10(max(y, 1e-300)) - ymin) / (ymax - ymin) * (bottom - top)
        svg += [f'<text class="title" x="{(left + right) / 2}" y="25" text-anchor="middle">{title}</text>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
                f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>']
        for dt, method, label, points in series:
            coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
            dash = f' stroke-dasharray="{dashes[method]}"' if dashes[method] else ""
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{colors[dt]}" stroke-width="2.5"{dash}/>')
        svg.append(f'<text class="label" x="{(left + right) / 2}" y="{bottom + 30}" text-anchor="middle">Physical time</text>')

    # Place one shared legend below both panels so it cannot obscure data.
    svg.append('<text class="title" x="600" y="455" text-anchor="middle">Legend</text>')
    for method_index, (method, label) in enumerate(METHODS):
        column_x = 170 + 560 * method_index
        for dt_index, dt in enumerate(dts):
            legend_y = 480 + 22 * dt_index
            dash = f' stroke-dasharray="{dashes[method]}"' if dashes[method] else ""
            svg += [f'<line x1="{column_x}" y1="{legend_y}" x2="{column_x + 34}" y2="{legend_y}" stroke="{colors[dt]}" stroke-width="2.5"{dash}/>',
                    f'<text class="label" x="{column_x + 42}" y="{legend_y + 4}">{label}, dt={dt:g}</text>']
    svg.append('</svg>')
    (root / "timestep-errors.svg").write_text("\n".join(svg))
    print(f"Wrote {root / 'timestep-errors.svg'}")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/early-time")
    dts = [float(value) for value in sys.argv[2:]] or [0.01, 0.005, 0.0025]
    histories = {(dt, method): read_history(root, dt, method)
                 for dt in dts for method, _ in METHODS}

    fields = ["time_step", "time", "method", "active_cells",
              "degrees_of_freedom", "l2_error", "h1_seminorm",
              "elapsed_seconds"]
    with (root / "timestep-history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for dt in dts:
            for method, _ in METHODS:
                for time, row in sorted(histories[(dt, method)].items()):
                    writer.writerow({
                        "time_step": dt,
                        "time": time,
                        "method": method,
                        "active_cells": row["active_cells"],
                        "degrees_of_freedom": row["degrees_of_freedom"],
                        "l2_error": row["l2_error"],
                        "h1_seminorm": row["h1_seminorm"],
                        "elapsed_seconds": row["elapsed_seconds"],
                    })

    summary_fields = ["time_step", "method", "time_steps", "training_fits",
                      "training_seconds", "total_elapsed_seconds",
                      "final_l2_error", "final_h1_seminorm", "final_dofs"]
    with (root / "timestep-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields,
                                lineterminator="\n")
        writer.writeheader()
        for dt in dts:
            for method, _ in METHODS:
                history = histories[(dt, method)]
                final = history[max(history)]
                fits, training_seconds, final_fit_seconds = training_totals(root, dt, method)
                writer.writerow({
                    "time_step": dt,
                    "method": method,
                    "time_steps": len(history),
                    "training_fits": fits,
                    "training_seconds": f"{training_seconds:.6f}",
                    "total_elapsed_seconds":
                        f"{float(final['elapsed_seconds']) + final_fit_seconds:.6f}",
                    "final_l2_error": final["l2_error"],
                    "final_h1_seminorm": final["h1_seminorm"],
                    "final_dofs": final["degrees_of_freedom"],
                })

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        write_svg(root, histories, dts)
        return

    colors = {0.01: "#1f77b4", 0.005: "#ff7f0e", 0.0025: "#2ca02c"}
    lines = {"classical": "-", "nn-once-per-time": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for dt in dts:
        for method, label in METHODS:
            history = histories[(dt, method)]
            times = sorted(history)
            for axis, metric in zip(axes, ("l2_error", "h1_seminorm")):
                axis.semilogy(times,
                              [float(history[t][metric]) for t in times],
                              lines[method], color=colors.get(dt),
                              label=f"{label}, dt={dt:g}")
    for axis, ylabel in zip(axes, ("L2 error", "H1 seminorm error")):
        axis.set(xlabel="Physical time", ylabel=ylabel)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("Rotating Gaussian: early-time timestep refinement")
    fig.savefig(root / "timestep-errors.png", dpi=180)
    print(f"Wrote {root / 'timestep-history.csv'}")
    print(f"Wrote {root / 'timestep-summary.csv'}")
    print(f"Wrote {root / 'timestep-errors.png'}")


if __name__ == "__main__":
    main()
