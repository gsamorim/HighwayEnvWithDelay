"""Plotting helpers for training curves and experiment sweeps.

Follows basic good-practice chart hygiene (labeled axes, titles, legends,
consistent colors) without pulling in a heavy plotting dependency beyond
matplotlib.
"""
from __future__ import annotations

import csv
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def read_csv_log(path: str) -> dict[str, list]:
    cols: dict[str, list] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(v)

    def _to_float(vals):
        out = []
        for v in vals:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(np.nan)
        return np.array(out)

    return {k: _to_float(v) for k, v in cols.items()}


def plot_training_curves(csv_path: str, out_path: str, title: str = "SAC training on racetrack-v0", window: int = 20):
    data = read_csv_log(csv_path)
    steps = data["step"]
    returns = data["episode_return"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    ax.plot(steps, returns, color="#9fb8d9", linewidth=0.8, alpha=0.6, label="episode return")
    if len(returns) >= window:
        smoothed = _rolling_mean(returns, window)
        ax.plot(steps[window - 1:], smoothed, color="#2c5f8a", linewidth=2.0, label=f"rolling mean ({window} ep)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_title("Episode return")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    if "crashed" in data:
        crash_rate = _rolling_mean(data["crashed"], window) if len(data["crashed"]) >= window else data["crashed"]
        x = steps[window - 1:] if len(data["crashed"]) >= window else steps
        ax.plot(x, crash_rate, color="#c0392b")
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"crash rate (rolling {window}-ep)")
    ax.set_title("Crash rate")

    ax = axes[1, 0]
    if "mean_speed" in data:
        speed = data["mean_speed"]
        if len(speed) >= window:
            speed_s = _rolling_mean(speed, window)
            ax.plot(steps[window - 1:], speed_s, color="#27ae60")
        else:
            ax.plot(steps, speed, color="#27ae60")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean episode speed")
    ax.set_title("Mean speed")

    ax = axes[1, 1]
    if "alpha" in data and not np.all(np.isnan(data["alpha"])):
        ax.plot(steps, data["alpha"], color="#8e44ad", label="alpha (entropy temp.)")
        ax.set_ylabel("alpha")
        ax.set_xlabel("environment steps")
        ax.set_title("Entropy temperature")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_traffic_sweep(results: list[dict[str, Any]], out_path: str, title: str = "Effect of traffic density on trained SAC policy"):
    """`results` items look like: {"traffic": "medium", "n_vehicles": 5, "mean_return":..., "crash_rate":..., "mean_speed":...}"""
    results = sorted(results, key=lambda r: r["n_vehicles"])
    x = [r["n_vehicles"] for r in results]
    labels = [r["traffic"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    means = [r["mean_return"] for r in results]
    stds = [r.get("std_return", 0.0) for r in results]
    ax.errorbar(x, means, yerr=stds, marker="o", color="#2c5f8a", capsize=3)
    ax.set_xlabel("other_vehicles (traffic density)")
    ax.set_ylabel("mean episode return")
    ax.set_title("Return vs. traffic")

    ax = axes[1]
    ax.plot(x, [r["crash_rate"] for r in results], marker="o", color="#c0392b")
    ax.set_xlabel("other_vehicles (traffic density)")
    ax.set_ylabel("crash rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Crash rate vs. traffic")

    ax = axes[2]
    ax.plot(x, [r["mean_speed"] for r in results], marker="o", color="#27ae60")
    ax.set_xlabel("other_vehicles (traffic density)")
    ax.set_ylabel("mean speed")
    ax.set_title("Speed vs. traffic")

    for ax in axes:
        ax.set_xticks(x)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
