#!/usr/bin/env python3
"""Render a benchmark_threads.py report as a standalone SVG (requires matplotlib)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.report.read_text())
    labels = {"rust": "ragisa (Rust)", "python312": "nagisa 0.3.0 · CPython 3.12 (GIL)",
              "python314t_gil": "nagisa 0.3.0 · CPython 3.14t (-X gil=1)",
              "python314t_nogil": "nagisa 0.3.0 · CPython 3.14t (-X gil=0)"}
    colors = ("#bd4b2f", "#3c8f68", "#797387", "#286e9f")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "ragisa-threads"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, mode, title in zip(axes, ("words", "tagging"), ("Word segmentation", "Words + POS"), strict=True):
        for (engine, label), color in zip(labels.items(), colors, strict=True):
            rows = sorted((row for row in report["summary"] if row["mode"] == mode and row["engine"] == engine),
                          key=lambda row: row["threads"])
            if not rows:
                continue
            x = [row["threads"] for row in rows]
            y = [row["median_lines_s"] for row in rows]
            ax.fill_between(x, [row["min_lines_s"] for row in rows],
                            [row["max_lines_s"] for row in rows], color=color, alpha=0.10)
            ax.plot(x, y, marker="o", color=color, label=label,
                    linestyle="--" if engine == "python314t_gil" else "-")
        ax.set(title=title, xlabel="Worker threads", ylabel="Throughput (lines / second)", xticks=x)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Shared-model thread scaling · {report['environment']['cpu']}", fontsize=16, weight="bold")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.045), frameon=False)
    runs = sum(run["mode"] == "words" for run in report["runs"])
    fig.text(0.5, 0.025, f"{len(report['request']['indices']):,} total lines per batch · median of {runs} runs · "
             "shading: observed min–max · all outputs verified", ha="center", color="#555555", fontsize=9)
    fig.tight_layout(rect=(0, 0.15, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Date": None})


if __name__ == "__main__":
    main()
