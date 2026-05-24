"""
Visualization for Simple Linear Regression
===========================================

Generates two charts:
1. Scatter plot of (X, Y) with the fitted regression line
2. Elasticity coefficient by observation

Charts are saved as PNG files in the ./images/ directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from main import (
    load_data,
    estimate_coefficients,
    elasticity,
)

ROOT = Path(__file__).parent
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def plot_regression(
    x: np.ndarray, y: np.ndarray,
    b0: float, b1: float,
    save_path: Path,
):
    x_line = np.linspace(x.min() - 1, x.max() + 1, 100)
    y_line = b0 + b1 * x_line

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    ax.scatter(
        x, y,
        s=80, color="#1f77b4", edgecolor="white", linewidth=1.5,
        label="Actual data (Y)", zorder=3,
    )
    ax.plot(
        x_line, y_line,
        color="#ff7f0e", linewidth=2.5,
        label=f"Regression line: Y = {b0:.3f} + {b1:.3f}·X", zorder=2,
    )

    ax.set_title(
        "Correlation Field and Regression Line",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path.relative_to(ROOT)}")


def plot_elasticity(x: np.ndarray, e: np.ndarray, save_path: Path):
    order = np.argsort(x)
    x_sorted = x[order]
    e_sorted = e[order]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    ax.fill_between(
        x_sorted, e_sorted,
        color="#2ca02c", alpha=0.18, zorder=1,
    )
    ax.plot(
        x_sorted, e_sorted,
        color="#2ca02c", linewidth=2.5,
        marker="o", markersize=7, markerfacecolor="white", markeredgewidth=2,
        label="Elasticity E", zorder=2,
    )
    ax.axhline(
        e.mean(),
        color="#d62728", linestyle="--", linewidth=1.5,
        label=f"Mean E = {e.mean():.3f}", zorder=3,
    )

    ax.set_title(
        "Elasticity Coefficient",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Elasticity", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path.relative_to(ROOT)}")


def main():
    x, y = load_data(str(ROOT / "data.json"))
    b0, b1 = estimate_coefficients(x, y)
    e = elasticity(b1, x, y)

    plot_regression(x, y, b0, b1, IMAGES_DIR / "regression_line.png")
    plot_elasticity(x, e, IMAGES_DIR / "elasticity.png")

    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    main()