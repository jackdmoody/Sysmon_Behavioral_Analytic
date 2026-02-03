from __future__ import annotations
from typing import Dict, Any, Optional, List
import pandas as pd

try:
    import ipywidgets as widgets
    from IPython.display import display
except Exception:  # pragma: no cover
    widgets = None
    display = None

from .visuals import (
    plot_host_timeline,
    plot_host_mitre_bar,
    plot_sankey_for_host,
    plot_host_markov_heatmap,
)

def get_host_list(df: pd.DataFrame, host_col: str = "host.fqdn"):
    return sorted(df[host_col].dropna().unique().tolist())

def host_dashboard(hostname: str):
    df_host = df_conn[df_conn["host.fqdn"] == hostname].copy()
    if df_host.empty:
        print(f"No events for host: {hostname}")
        return

    print(f"Host: {hostname} — {len(df_host)} events")

    # 1) Timeline
    plot_host_timeline(df_host)

    # 2) MITRE technique bar
    plot_host_mitre_bar(df_host)

    # 3) Sankey of event-ID transitions
    plot_host_sankey(df_host)

    # 4) Markov chain heatmap of transitions
    plot_host_markov_heatmap(df_host)

    # 5) Markov state network graph
    plot_host_markov_network(df_host)

    # 6) NEW: linear Markov chain (like your example image)
    plot_host_markov_linear(df_host)

def launch_dashboard():
    # Optional: show a global stationary distribution once at launch
    print("Global Markov view over all hosts:")
    plot_global_stationary_distribution(df_conn)

    hosts = get_host_list(df_conn)
    if not hosts:
        print("No hosts found in df_conn.")
        return

    @interact(hostname=Dropdown(options=hosts, description="Host"))
    def _interactive(hostname):
        host_dashboard(hostname)
