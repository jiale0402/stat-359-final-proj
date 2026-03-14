"""Build reward-curve figures from the report's published run summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPORT_DIR = Path(__file__).resolve().parent


def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "axes.facecolor": "#f8f6f1",
            "figure.facecolor": "#f4f1ea",
            "savefig.facecolor": "#f4f1ea",
            "grid.color": "#d9d2c3",
            "grid.alpha": 0.6,
            "axes.edgecolor": "#8c806a",
            "font.family": "DejaVu Sans",
        }
    )


def _save(fig: plt.Figure, filename: str) -> None:
    path = REPORT_DIR / filename
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def build_accuracy_curves() -> None:
    steps = np.array([0, 3, 7, 11, 15, 19, 23, 27, 31, 35], dtype=float)
    h1 = np.array([0.1406, 0.153, 0.176, 0.205, 0.236, 0.268, 0.300, 0.325, 0.336, 0.340])
    h2 = np.array([0.1406, 0.241, 0.300, 0.308, 0.301, 0.291, 0.284, 0.286, 0.288, 0.287])
    h3 = np.array([0.1406, 0.233, 0.300, 0.314, 0.325, 0.334, 0.339, 0.343, 0.341, 0.342])

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.plot(steps, h1, marker="o", linewidth=2.6, color="#355070", label="H1: Binary only")
    ax.plot(
        steps,
        h2,
        marker="o",
        linewidth=2.6,
        color="#bc6c25",
        label="H2: Binary + process + format",
    )
    ax.plot(
        steps,
        h3,
        marker="o",
        linewidth=2.6,
        color="#588157",
        label="H3: Binary + distance",
    )
    ax.axhline(0.30, color="#6d6875", linewidth=1.4, linestyle="--", alpha=0.8)
    ax.annotate("Baseline = 0.1406", xy=(0, 0.1406), xytext=(1.2, 0.155), color="#4a4a4a")
    ax.annotate("H2/H3 reach 0.30 by step 7", xy=(7, 0.30), xytext=(10, 0.318), color="#4a4a4a")
    ax.annotate("H1 reaches 0.30 by step 23", xy=(23, 0.30), xytext=(24.5, 0.283), color="#4a4a4a")
    ax.set_title("Exact-Match Accuracy Across Reward Designs")
    ax.set_xlabel("GRPO Step")
    ax.set_ylabel("Validation Accuracy")
    ax.set_xlim(0, 35)
    ax.set_ylim(0.12, 0.36)
    ax.legend(frameon=True, facecolor="#fffdf8", edgecolor="#cfc7b6")
    _save(fig, "reward_accuracy_curves.png")


def build_h2_component_curves() -> None:
    steps = np.array([0, 7, 15, 23, 31, 35], dtype=float)
    exact_accuracy = np.array([0.1406, 0.300, 0.301, 0.284, 0.288, 0.287])
    process_reward = np.array([0.591, 0.690, 0.770, 0.830, 0.870, 0.890])
    format_reward = np.array([0.729, 0.820, 0.890, 0.950, 0.990, 1.000])

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.plot(
        steps,
        process_reward,
        marker="o",
        linewidth=2.5,
        color="#bc6c25",
        label="Process reward",
    )
    ax.plot(
        steps,
        format_reward,
        marker="o",
        linewidth=2.5,
        color="#dda15e",
        label="Format reward",
    )
    ax.plot(
        steps,
        exact_accuracy,
        marker="o",
        linewidth=2.5,
        color="#355070",
        label="Exact-match accuracy",
    )
    ax.annotate(
        "Proxy rewards rise while accuracy stalls",
        xy=(23, 0.284),
        xytext=(13, 0.55),
        color="#4a4a4a",
    )
    ax.set_title("H2 Reward Components vs. Exact Accuracy")
    ax.set_xlabel("GRPO Step")
    ax.set_ylabel("Metric Value")
    ax.set_xlim(0, 35)
    ax.set_ylim(0.10, 1.05)
    ax.legend(frameon=True, facecolor="#fffdf8", edgecolor="#cfc7b6")
    _save(fig, "h2_reward_component_curves.png")


def build_h3_component_curves() -> None:
    steps = np.array([0, 7, 15, 23, 31, 35], dtype=float)
    exact_accuracy = np.array([0.1406, 0.300, 0.325, 0.339, 0.341, 0.342])
    distance_reward = np.array([0.410, 0.535, 0.602, 0.648, 0.666, 0.672])

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.plot(
        steps,
        distance_reward,
        marker="o",
        linewidth=2.6,
        color="#588157",
        label="Distance reward",
    )
    ax.plot(
        steps,
        exact_accuracy,
        marker="o",
        linewidth=2.6,
        color="#355070",
        label="Exact-match accuracy",
    )
    ax.annotate(
        "Dense signal stays aligned with correctness",
        xy=(23, 0.339),
        xytext=(12, 0.47),
        color="#4a4a4a",
    )
    ax.set_title("H3 Distance Reward Tracks Task Progress")
    ax.set_xlabel("GRPO Step")
    ax.set_ylabel("Metric Value")
    ax.set_xlim(0, 35)
    ax.set_ylim(0.10, 0.72)
    ax.legend(frameon=True, facecolor="#fffdf8", edgecolor="#cfc7b6")
    _save(fig, "h3_reward_component_curves.png")


def main() -> None:
    _apply_style()
    build_accuracy_curves()
    build_h2_component_curves()
    build_h3_component_curves()


if __name__ == "__main__":
    main()
