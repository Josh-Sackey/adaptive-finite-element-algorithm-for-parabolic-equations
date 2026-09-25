#!/usr/bin/env python3
"""Combine and plot matched Figure 8 classical/NN benchmark histories."""

import argparse
import csv
import math
from pathlib import Path


def read_history(path: Path, method: str):
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["phase"] == "time_step"]
    for row in rows:
        row["method"] = method
        for key in ("time", "l2_error", "h1_seminorm", "elapsed_seconds"):
            row[key] = float(row[key])
        for key in ("index", "active_cells", "degrees_of_freedom"):
            row[key] = int(row[key])
    return rows


def write_svg(root: Path, datasets):
    width, height = 1200, 800
    panels = [
        (60, 45, 510, 300, "time", "l2_error", True, False,
         "L2 error versus time", "Physical time", "L2 error", datasets),
        (650, 45, 510, 300, "time", "h1_seminorm", True, False,
         "H1 error versus time", "Physical time", "H1 seminorm error", datasets),
        (60, 445, 510, 300, "time", "degrees_of_freedom", False, False,
         "Spatial work versus time", "Physical time", "Degrees of freedom", datasets),
        (650, 445, 510, 300, "time", "elapsed_seconds", True, False,
         "Cumulative runtime versus time", "Physical time", "Elapsed seconds", datasets),
    ]
    colors = {"classical": "#1f77b4", "nn": "#d62728"}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:sans-serif;fill:#222}.title{font-size:18px;'
           'font-weight:bold}.label{font-size:13px}.tick{font-size:11px}</style>']
    for (x0, y0, pw, ph, xkey, ykey, logy, logx, title,
         xlabel, ylabel, panel_data) in panels:
        left, right, top, bottom = x0 + 65, x0 + pw - 20, y0 + 35, y0 + ph - 45
        all_rows = [row for _, rows in panel_data for row in rows]
        tx = lambda value: math.log10(max(value, 1e-300)) if logx else value
        ty = lambda value: math.log10(max(value, 1e-300)) if logy else value
        xs, ys = [tx(row[xkey]) for row in all_rows], [ty(row[ykey]) for row in all_rows]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        if xmax == xmin: xmax = xmin + 1.0
        if ymax == ymin: ymax = ymin + 1.0
        sx = lambda value: left + (tx(value) - xmin) / (xmax - xmin) * (right - left)
        sy = lambda value: bottom - (ty(value) - ymin) / (ymax - ymin) * (bottom - top)
        svg += [f'<text class="title" x="{x0 + pw/2}" y="{y0 + 20}" '
                f'text-anchor="middle">{title}</text>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
                f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>']
        for i in range(6):
            gx = left + i * (right-left)/5
            gy = top + i * (bottom-top)/5
            xv = xmin + i * (xmax-xmin)/5
            yv = ymax - i * (ymax-ymin)/5
            x_label = 10 ** xv if logx else xv
            y_label = 10 ** yv if logy else yv
            svg += [f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{bottom}" stroke="#ddd"/>',
                    f'<line x1="{left}" y1="{gy:.1f}" x2="{right}" y2="{gy:.1f}" stroke="#ddd"/>',
                    f'<text class="tick" x="{gx:.1f}" y="{bottom+17}" text-anchor="middle">{x_label:.2g}</text>',
                    f'<text class="tick" x="{left-7}" y="{gy+4:.1f}" text-anchor="end">{y_label:.2g}</text>']
        svg += [f'<text class="label" x="{(left+right)/2}" y="{y0+ph-8}" text-anchor="middle">{xlabel}</text>',
                f'<text class="label" x="{x0+14}" y="{(top+bottom)/2}" text-anchor="middle" '
                f'transform="rotate(-90 {x0+14} {(top+bottom)/2})">{ylabel}</text>']
        for method, rows in panel_data:
            points = " ".join(f"{sx(row[xkey]):.2f},{sy(row[ykey]):.2f}" for row in rows)
            if len(rows) > 1:
                svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[method]}" '
                           f'stroke-width="2.5"/>')
            for row in rows:
                radius = 7 if len(rows) == 1 else 2.5
                svg.append(f'<circle cx="{sx(row[xkey]):.2f}" cy="{sy(row[ykey]):.2f}" r="{radius}" '
                           f'fill="{colors[method]}"/>')
                if len(rows) == 1:
                    svg.append(f'<text class="label" x="{sx(row[xkey])+10:.2f}" '
                               f'y="{sy(row[ykey])-8:.2f}">{method.upper()}</text>')
        svg += [f'<line x1="{right-125}" y1="{top+12}" x2="{right-95}" y2="{top+12}" '
                f'stroke="{colors["classical"]}" stroke-width="3"/>',
                f'<text class="label" x="{right-90}" y="{top+17}">CLASSICAL</text>',
                f'<line x1="{right-125}" y1="{top+32}" x2="{right-95}" y2="{top+32}" '
                f'stroke="{colors["nn"]}" stroke-width="3"/>',
                f'<text class="label" x="{right-90}" y="{top+37}">NN</text>']
    svg.append('</svg>')
    (root / "comparison.svg").write_text("\n".join(svg))
    print(f"Wrote {root / 'comparison.svg'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="runs/comparison")
    args = parser.parse_args()
    root = Path(args.root)
    classical = read_history(root / "classical" / "simulation-history.csv", "classical")
    neural = read_history(root / "nn" / "simulation-history.csv", "nn")
    classical_by_time = {round(row["time"], 10): row for row in classical}
    neural_by_time = {round(row["time"], 10): row for row in neural}
    common_times = sorted(classical_by_time.keys() & neural_by_time.keys())
    fields = [
        "index", "time",
        "classical_active_cells", "nn_active_cells",
        "classical_dofs", "nn_dofs", "nn_to_classical_dof_ratio",
        "classical_l2_error", "nn_l2_error", "l2_error_improvement_classical_over_nn",
        "classical_h1_error", "nn_h1_error", "h1_error_improvement_classical_over_nn",
        "classical_elapsed_seconds", "nn_elapsed_seconds",
        "nn_to_classical_runtime_ratio",
    ]
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for time in common_times:
            c = classical_by_time[time]
            n = neural_by_time[time]
            writer.writerow({
                "index": c["index"],
                "time": time,
                "classical_active_cells": c["active_cells"],
                "nn_active_cells": n["active_cells"],
                "classical_dofs": c["degrees_of_freedom"],
                "nn_dofs": n["degrees_of_freedom"],
                "nn_to_classical_dof_ratio": n["degrees_of_freedom"] / c["degrees_of_freedom"],
                "classical_l2_error": c["l2_error"],
                "nn_l2_error": n["l2_error"],
                "l2_error_improvement_classical_over_nn": c["l2_error"] / n["l2_error"],
                "classical_h1_error": c["h1_seminorm"],
                "nn_h1_error": n["h1_seminorm"],
                "h1_error_improvement_classical_over_nn": c["h1_seminorm"] / n["h1_seminorm"],
                "classical_elapsed_seconds": c["elapsed_seconds"],
                "nn_elapsed_seconds": n["elapsed_seconds"],
                "nn_to_classical_runtime_ratio": n["elapsed_seconds"] / c["elapsed_seconds"],
            })

    datasets = (("classical", classical), ("nn", neural))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        write_svg(root, datasets)
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    styles = {"classical": ("#1f77b4", "-"), "nn": ("#d62728", "--")}
    for method, data in datasets:
        color, line = styles[method]
        time = [row["time"] for row in data]
        axes[0, 0].semilogy(time, [row["l2_error"] for row in data], line,
                           color=color, label=method.upper())
        axes[0, 1].semilogy(time, [row["h1_seminorm"] for row in data], line,
                           color=color, label=method.upper())
        axes[1, 0].plot(time, [row["degrees_of_freedom"] for row in data], line,
                        color=color, label=method.upper())
        axes[1, 1].semilogy(time, [row["elapsed_seconds"] for row in data], line,
                           color=color, label=method.upper())

    labels = [
        (axes[0, 0], "Time", r"$L^2$ error", r"$L^2$ error versus time"),
        (axes[0, 1], "Time", r"$H^1$ seminorm error", r"$H^1$ error versus time"),
        (axes[1, 0], "Physical time", "Degrees of freedom", "Spatial work versus time"),
        (axes[1, 1], "Physical time", "Elapsed seconds",
         "Cumulative runtime versus time"),
    ]
    for axis, xlabel, ylabel, title in labels:
        axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    fig.savefig(root / "comparison.png", dpi=180)
    print(f"Wrote {root / 'comparison.csv'}")
    print(f"Wrote {root / 'comparison.png'}")


if __name__ == "__main__":
    main()
