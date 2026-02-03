from __future__ import annotations
from typing import Tuple
import pandas as pd
import networkx as nx

def build_graph_from_pairs(pairs_df: pd.DataFrame, *, src_col: str="src_event", dst_col: str="dst_event", weight_col: str="count") -> nx.DiGraph:
    G = nx.DiGraph()
    for _, r in pairs_df.iterrows():
        s, d, w = int(r[src_col]), int(r[dst_col]), float(r[weight_col])
        if G.has_edge(s,d):
            G[s][d]["weight"] += w
        else:
            G.add_edge(s,d,weight=w)
    return G

def compute_graph_metrics(G: nx.DiGraph) -> pd.DataFrame:
    # Basic, fast metrics (extend with centrality if desired)
    nodes = list(G.nodes())
    out = pd.DataFrame({
        "node": nodes,
        "in_degree": [G.in_degree(n, weight=None) for n in nodes],
        "out_degree": [G.out_degree(n, weight=None) for n in nodes],
        "in_weight": [G.in_degree(n, weight="weight") for n in nodes],
        "out_weight": [G.out_degree(n, weight="weight") for n in nodes],
    })
    return out.sort_values("out_weight", ascending=False)
