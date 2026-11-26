"""Render the held-out model comparison from the saved evaluation report."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    plt.switch_backend("Agg")
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "results/evaluation.json").read_text())
    rows = {
        name: scores
        for name, scores in report["outer_summary"].items()
        if name != "Selection procedure"
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    fig.patch.set_facecolor("#f4f7fa")
    fig.suptitle("BatteryLife AI  |  Held-out Android devices", fontsize=15, weight="bold")
    for axis, key, title, color in zip(
        axes,
        ["mae", "soc_median_mae"],
        ["Drain MAE (pp/h)", "Median sequence SoC MAE (pp)"],
        ["#16746e", "#42689d"],
        strict=True,
    ):
        values = [scores[key] for scores in rows.values()]
        bars = axis.barh(list(rows), values, color=color)
        axis.invert_yaxis()
        axis.set_title(title, loc="left", fontsize=11)
        axis.set_xlim(0, max(values) * 1.2)
        axis.bar_label(bars, fmt="%.2f", padding=4)
        axis.grid(axis="x", alpha=0.16)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(left=False)
    fig.savefig(root / "results/model_comparison.png", dpi=165)


if __name__ == "__main__":
    main()
