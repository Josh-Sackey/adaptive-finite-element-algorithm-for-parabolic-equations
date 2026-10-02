#!/usr/bin/env python3
"""Create a matched three-way transfer-strategy comparison."""

import csv
import sys
from pathlib import Path

METHODS = (
    ("classical", "Classical SolutionTransfer"),
    ("nn-once-per-time", "NN once per time level"),
    ("nn-every-crossing", "NN every crossing"),
)


def read_history(root, method):
    path = root / method / "simulation-history.csv"
    with path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["phase"] == "time_step"]
    return {round(float(r["time"]), 10): r for r in rows}


def training_totals(root, method):
    if method == "classical":
        return {"fits": 0, "seconds": 0.0, "evaluations": 0}
    path = root / method / "neural-training.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "fits": len(rows),
        "seconds": sum(float(r["seconds"]) for r in rows),
        "evaluations": sum(int(r["closure_evaluations"]) for r in rows),
    }


def write_svg(root, histories, common_times):
    width, height = 1200, 800
    panels = (
        (55, 45, "l2_error", "L2 error", True),
        (625, 45, "h1_seminorm", "H1 seminorm error", True),
        (55, 425, "degrees_of_freedom", "Degrees of freedom", False),
        (625, 425, "elapsed_seconds", "Cumulative runtime (s)", True),
    )
    colors = {"classical": "#1f77b4", "nn-once-per-time": "#2ca02c",
              "nn-every-crossing": "#d62728"}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:sans-serif;fill:#222}.t{font-size:16px;font-weight:bold}'
           '.l{font-size:12px}</style>']
    import math
    for x0, y0, metric, title, logy in panels:
        left, right, top, bottom = x0 + 65, x0 + 530, y0 + 35, y0 + 320
        vals = [float(histories[m][t][metric]) for m, _ in METHODS for t in common_times]
        transformed = [math.log10(max(v, 1e-300)) if logy else v for v in vals]
        ymin, ymax = min(transformed), max(transformed)
        if ymin == ymax:
            ymax += 1.0
        xmin, xmax = min(common_times), max(common_times)
        if xmin == xmax:
            xmax += 1.0
        sx = lambda t: left + (t - xmin) / (xmax - xmin) * (right - left)
        sy = lambda v: bottom - ((math.log10(max(v, 1e-300)) if logy else v) - ymin) / (ymax - ymin) * (bottom - top)
        svg += [f'<text class="t" x="{(left+right)/2}" y="{y0+18}" text-anchor="middle">{title}</text>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
                f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>']
        for method, label in METHODS:
            points = " ".join(f"{sx(t):.2f},{sy(float(histories[method][t][metric])):.2f}" for t in common_times)
            svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[method]}" stroke-width="2.5"/>')
        for i, (method, label) in enumerate(METHODS):
            ly = top + 16 + 18 * i
            svg += [f'<line x1="{left+8}" y1="{ly}" x2="{left+35}" y2="{ly}" stroke="{colors[method]}" stroke-width="3"/>',
                    f'<text class="l" x="{left+40}" y="{ly+4}">{label}</text>']
        svg.append(f'<text class="l" x="{(left+right)/2}" y="{bottom+28}" text-anchor="middle">Physical time</text>')
    svg.append('</svg>')
    (root / "comparison.svg").write_text("\n".join(svg))
    print(f"Wrote {root / 'comparison.svg'}")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/comparison")
    histories = {method: read_history(root, method) for method, _ in METHODS}
    common_times = sorted(set.intersection(*(set(rows) for rows in histories.values())))
    fields = ["index", "time"]
    metrics = ("active_cells", "degrees_of_freedom", "l2_error",
               "h1_seminorm", "elapsed_seconds")
    for method, _ in METHODS:
        fields.extend(f"{method}_{metric}" for metric in metrics)
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for time in common_times:
            first = histories["classical"][time]
            out = {"index": first["index"], "time": time}
            for method, _ in METHODS:
                row = histories[method][time]
                for metric in metrics:
                    out[f"{method}_{metric}"] = row[metric]
            writer.writerow(out)

    with (root / "training-summary.csv").open("w", newline="") as handle:
        fields = ["method", "training_fits", "closure_evaluations",
                  "training_seconds", "final_elapsed_seconds",
                  "final_l2_error", "final_h1_seminorm", "final_dofs"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method, _ in METHODS:
            final = histories[method][common_times[-1]]
            totals = training_totals(root, method)
            writer.writerow({
                "method": method,
                "training_fits": totals["fits"],
                "closure_evaluations": totals["evaluations"],
                "training_seconds": totals["seconds"],
                "final_elapsed_seconds": final["elapsed_seconds"],
                "final_l2_error": final["l2_error"],
                "final_h1_seminorm": final["h1_seminorm"],
                "final_dofs": final["degrees_of_freedom"],
            })

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        write_svg(root, histories, common_times)
        return

    styles = {
        "classical": ("#1f77b4", "-"),
        "nn-once-per-time": ("#2ca02c", "--"),
        "nn-every-crossing": ("#d62728", ":"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    plots = (
        (axes[0, 0], "l2_error", True, "L2 error"),
        (axes[0, 1], "h1_seminorm", True, "H1 seminorm error"),
        (axes[1, 0], "degrees_of_freedom", False, "Degrees of freedom"),
        (axes[1, 1], "elapsed_seconds", True, "Cumulative runtime (s)"),
    )
    for method, label in METHODS:
        rows = [histories[method][t] for t in common_times]
        color, line = styles[method]
        for axis, metric, log_scale, ylabel in plots:
            values = [float(r[metric]) for r in rows]
            draw = axis.semilogy if log_scale else axis.plot
            draw(common_times, values, line, color=color, label=label)
            axis.set(xlabel="Physical time", ylabel=ylabel)
            axis.grid(True, which="both", alpha=0.3)
    for axis, _, _, _ in plots:
        axis.legend(fontsize=8)
    fig.suptitle("Rotating Gaussian: transfer-strategy comparison")
    fig.savefig(root / "comparison.png", dpi=180)
    print(f"Wrote {root / 'comparison.csv'}")
    print(f"Wrote {root / 'training-summary.csv'}")
    print(f"Wrote {root / 'comparison.png'}")


if __name__ == "__main__":
    main()
