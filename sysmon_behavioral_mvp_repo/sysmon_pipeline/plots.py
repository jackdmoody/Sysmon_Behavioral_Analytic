"""Simple plots for the MVP."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_score_histogram(triage: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "score_histogram.png"
    plt.figure()
    triage["anomaly_score"].plot(kind="hist", bins=30)
    plt.title("Host anomaly score distribution")
    plt.xlabel("anomaly_score (higher = more suspicious)")
    plt.ylabel("count of hosts")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_top_hosts(triage: pd.DataFrame, out_dir: Path, top_n: int = 10) -> Path:
    path = out_dir / "top_hosts.png"
    top = triage.head(top_n).iloc[::-1]  # reverse for horizontal bar aesthetics
    plt.figure()
    top["anomaly_score"].plot(kind="barh")
    plt.title(f"Top {top_n} hosts by anomaly score")
    plt.xlabel("anomaly_score")
    plt.ylabel("host")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path
