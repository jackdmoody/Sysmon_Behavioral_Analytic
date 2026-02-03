from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plotly is optional; functions will error if not installed when called
try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception:  # pragma: no cover
    go = None
    px = None

def plot_sankey_for_host(
    pairs_df: pd.DataFrame,
    host: str,
    host_col: str = "host.fqdn",
    source_col: str = "winlog.event_id_1",
    target_col: str = "winlog.event_id_2",
    value_name: str = "count",
):
    """
    Build and display a Sankey diagram for a single host.
   
    - Nodes are unique event IDs seen as either source or target.
    - Links are (event_id_1 -> event_id_2) pairs.
    - Link 'value' is the number of occurrences of that pair for this host.
    """
    # Filter to this host
    hdf = pairs_df[pairs_df[host_col] == host]
    if hdf.empty:
        print(f"No pairs found for host: {host}")
        return
   
    # Group by pair and count occurrences (this is the weight)
    df_sankey = (
        hdf.groupby([source_col, target_col])
           .size()
           .reset_index(name=value_name)
    )
   
    # Get all unique event IDs that appear in any source/target
    all_nodes = sorted(
        set(df_sankey[source_col].dropna().unique()) |
        set(df_sankey[target_col].dropna().unique())
    )
   
    # Map event ID -> node index for Sankey
    node_index = {eid: idx for idx, eid in enumerate(all_nodes)}
   
    # Build source/target/value arrays
    sources = df_sankey[source_col].map(node_index).tolist()
    targets = df_sankey[target_col].map(node_index).tolist()
    values  = df_sankey[value_name].tolist()
   
    # Labels as strings so they render nicely
    labels = [str(eid) for eid in all_nodes]
   
    # Optional hover text per link (e.g., "4688 → 10: 5 times")
    link_labels = [
        f"{int(s)} → {int(t)}: {c} times"
        for s, t, c in zip(df_sankey[source_col], df_sankey[target_col], values)
    ]
   
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=15,
            label=labels,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
        )
    )])
   
    fig.update_layout(
        title_text=f"Correlated Event Pairs – Sankey Diagram for {host}",
        font=dict(size=12)
    )
   
    fig.show()

def plot_top_sankeys(
    pairs_df: pd.DataFrame,
    host_col: str = "host.fqdn",
    top_n: int = 5,
):
    """
    Plot Sankey diagrams for the top `top_n` hosts ranked by
    number of correlated pairs (rows) in `pairs_df`.
    """
    # Rank hosts by how many pair rows they have
    host_counts = (
        pairs_df.groupby(host_col)
                .size()
                .sort_values(ascending=False)
    )
    top_hosts = host_counts.head(top_n).index.tolist()

    print(f"Top {len(top_hosts)} hosts by pair count:")
    print(host_counts.head(top_n))

    # Plot one Sankey per host
    for host in top_hosts:
        print(f"\n=== Sankey for host: {host} ===")
        plot_sankey_for_host(pairs_df, host=host, host_col=host_col)

def plot_transition_sankey(transitions_df: pd.DataFrame, min_count: int = 10):
    """Plot a Sankey diagram of event-to-event transitions."""
    df_filt = transitions_df[transitions_df["count"] >= min_count].copy()
    if df_filt.empty:
        print(f"No transitions with count >= {min_count}")
        return

    # Build label list and index mapping
    labels = sorted(set(df_filt["src_event"]).union(df_filt["dst_event"]))
    label_to_idx = {ev: i for i, ev in enumerate(labels)}

    sources = df_filt["src_event"].map(label_to_idx).tolist()
    targets = df_filt["dst_event"].map(label_to_idx).tolist()
    values = df_filt["count"].tolist()

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(width=0.5),
            label=[str(l) for l in labels],
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        ),
    )])

    fig.update_layout(title_text="Event-to-Event Transition Sankey", font_size=10)
    fig.show()

def plot_global_stationary_distribution(
    df: pd.DataFrame,
    event_col: str = "winlog.event_id",
    time_col: str = "_timestamp",
    n_iter: int = 100,
    title: str = "Approx. Stationary Distribution (Global Markov Chain)",
):
    """
    Builds a global Markov chain over ALL hosts/events and shows
    the approximate stationary distribution pi.
    """
    # Treat all events as one big sequence (ordered by time)
    dfg = df.dropna(subset=[event_col, time_col]).copy()
    if dfg.empty:
        print("No events to compute global stationary distribution.")
        return

    dfg = dfg.sort_values(time_col)
    events = dfg[event_col].astype("Int64").tolist()
    if len(events) < 2:
        print("Not enough events for global Markov chain.")
        return

    src = events[:-1]
    dst = events[1:]
    pairs_counts = (
        pd.DataFrame({"src": src, "dst": dst})
        .groupby(["src", "dst"])
        .size()
        .reset_index(name="count")
    )

    trans_probs, states = build_markov_from_pairs(pairs_counts)
    if trans_probs.size == 0:
        print("Global transition matrix is empty.")
        return

    # Power iteration for stationary distribution
    n = len(states)
    pi = np.ones(n) / n
    for _ in range(n_iter):
        pi = pi @ trans_probs

    df_pi = pd.DataFrame({
        "event": [str(s) for s in states],
        "pi": pi
    }).sort_values("pi", ascending=False)

    fig = px.bar(
        df_pi,
        x="event",
        y="pi",
        title=title,
        labels={"event": "Event ID", "pi": "Stationary Probability"},
    )
    fig.show()

def plot_host_markov_heatmap(
    df_host: pd.DataFrame,
    event_col: str = "winlog.event_id",
    time_col: str = "_timestamp",
):
    pairs_counts = compute_host_transitions(df_host, event_col=event_col, time_col=time_col)
    if pairs_counts.empty:
        print("Not enough events to build a Markov transition heatmap for this host.")
        return

    trans_probs, states = build_markov_from_pairs(pairs_counts)
    if trans_probs.size == 0:
        print("Transition matrix is empty.")
        return

    fig = px.imshow(
        trans_probs,
        x=[str(s) for s in states],
        y=[str(s) for s in states],
        labels=dict(x="Next Event ID", y="Current Event ID", color="P(Next | Current)"),
        title=f"Markov Transition Matrix for {df_host['host.fqdn'].iloc[0]}",
        aspect="auto",
    )
    fig.update_xaxes(side="top")
    fig.show()

def plot_host_markov_linear(
    df_host: pd.DataFrame,
    event_col: str = "winlog.event_id",
    time_col: str = "_timestamp",
    prob_threshold: float = 0.05,
):
    """
    Linear Markov visualization (similar to your example image):
      - States on a line
      - Dashed lines between states
      - Probabilities printed near the lines
    """
    pairs_counts = compute_host_transitions(df_host, event_col=event_col, time_col=time_col)
    if pairs_counts.empty:
        print("Not enough events to build a Markov chain for this host.")
        return

    trans_probs, states = build_markov_from_pairs(pairs_counts)
    if trans_probs.size == 0:
        print("Transition matrix is empty.")
        return

    # Lay states out on a line: x = 0..n-1, y = 0
    n = len(states)
    x = np.arange(n)
    y = np.zeros(n)

    # Build edge segments
    edge_x = []
    edge_y = []
    text_x = []
    text_y = []
    texts = []

    for i in range(n):
        for j in range(n):
            p = float(trans_probs[i, j])
            if p <= prob_threshold:
                continue

            # Line segment from state i to state j
            edge_x += [x[i], x[j], None]
            edge_y += [y[i], y[j], None]

            # Label position (slightly above the line)
            text_x.append((x[i] + x[j]) / 2.0)
            text_y.append(0.1)  # small offset above the line
            texts.append(f"{p:.2f}")

    if not edge_x:
        print(f"No transitions above threshold {prob_threshold:.2f} for this host.")
        return

    # Nodes
    node_trace = go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        text=[str(s) for s in states],
        textposition="bottom center",
        hoverinfo="text",
        marker=dict(size=14),
        showlegend=False,
    )

    # Edges as dashed lines
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(dash="dash", width=2),
        hoverinfo="none",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    # Add probability annotations
    for tx, ty, txt in zip(text_x, text_y, texts):
        fig.add_annotation(
            x=tx,
            y=ty,
            text=txt,
            showarrow=False,
            font=dict(size=10),
        )

    host = df_host["host.fqdn"].iloc[0]
    fig.update_layout(
        title=f"Linear Markov Chain View for {host}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.show()

def plot_host_markov_network(
    df_host: pd.DataFrame,
    event_col: str = "winlog.event_id",
    time_col: str = "_timestamp",
    prob_threshold: float = 0.05,
):
    """
    Circular graph: nodes = event IDs, directed edges = transitions
    thicker lines = higher transition probability.
    """
    pairs_counts = compute_host_transitions(df_host, event_col=event_col, time_col=time_col)
    if pairs_counts.empty:
        print("Not enough events to build a Markov network for this host.")
        return

    trans_probs, states = build_markov_from_pairs(pairs_counts)
    if trans_probs.size == 0:
        print("Transition matrix is empty.")
        return

    n = len(states)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)

    # Nodes
    node_trace = go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=[str(s) for s in states],
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=14)
    )

    # Edges (only above threshold)
    edge_traces = []
    for i in range(n):
        for j in range(n):
            p = float(trans_probs[i, j])
            if p <= prob_threshold:
                continue
            edge_traces.append(
                go.Scatter(
                    x=[xs[i], xs[j]],
                    y=[ys[i], ys[j]],
                    mode="lines",
                    line=dict(width=2 + 6 * p),
                    hoverinfo="text",
                    hovertext=f"{states[i]} → {states[j]}: {p:.2f}",
                    showlegend=False,
                )
            )

    host = df_host["host.fqdn"].iloc[0]
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=f"Markov State Graph for {host}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    fig.show()

def plot_host_mitre_bar(df_host: pd.DataFrame):
    if "technique" not in df_host.columns:
        print("No MITRE mapping present; run the MITRE mapping cell first.")
        return
    agg = (
        df_host
        .groupby(["tactic", "technique", "technique_name"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    if agg.empty:
        print("No MITRE-mapped events for this host.")
        return
    fig = px.bar(
        agg,
        x="technique",
        y="count",
        color="tactic",
        hover_data=["technique_name"],
        title=f"MITRE Technique Counts for {df_host['host.fqdn'].iloc[0]}",
    )
    fig.update_layout(xaxis_title="Technique", yaxis_title="Event Count")
    fig.show()

def plot_host_sankey(df_host: pd.DataFrame):
    pairs_counts = compute_host_transitions(df_host)
    if pairs_counts.empty:
        print("Not enough events to build a Sankey diagram for this host.")
        return

    unique_events = sorted(
        pd.unique(
            pairs_counts[["src", "dst"]].values.ravel("K")
        ).tolist()
    )
    idx_map = {ev: i for i, ev in enumerate(unique_events)}

    source_indices = pairs_counts["src"].map(idx_map).tolist()
    target_indices = pairs_counts["dst"].map(idx_map).tolist()
    values = pairs_counts["count"].tolist()
    labels = [str(ev) for ev in unique_events]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(width=0.5),
                    label=labels,
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                ),
            )
        ]
    )
    host = df_host["host.fqdn"].iloc[0]
    fig.update_layout(
        title_text=f"Event-ID Transition Sankey for {host}",
        font_size=10,
    )
    fig.show()

def plot_host_timeline(df_host: pd.DataFrame):
    if "_timestamp" not in df_host.columns:
        print("No _timestamp column found; cannot plot timeline.")
        return
    df_host_sorted = df_host.sort_values("_timestamp")
    fig = px.scatter(
        df_host_sorted,
        x="_timestamp",
        y="winlog.event_id",
        color="tactic",
        hover_data=["technique", "technique_name", "description"],
        title=f"Event Timeline for {df_host_sorted['host.fqdn'].iloc[0]}",
    )
    fig.show()
