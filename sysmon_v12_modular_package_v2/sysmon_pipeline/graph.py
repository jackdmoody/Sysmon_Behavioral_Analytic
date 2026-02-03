from __future__ import annotations
import pandas as pd
import networkx as nx

def build_graph_from_pairs(pairs_df: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph from a DataFrame of event-ID pairs.

    Expected columns in `pairs_df`:
      - 'winlog.event_id_1' : source event ID
      - 'winlog.event_id_2' : destination event ID

    The function aggregates counts of each (event_id_1, event_id_2) pair
    and uses them as edge weights in the graph.
    """
    if pairs_df.empty:
        return nx.DiGraph()

    # Aggregate edge weights: how many times did we see each ordered pair?
    agg = (
        pairs_df
        .groupby(["winlog.event_id_1", "winlog.event_id_2"])
        .size()
        .reset_index(name="weight")
    )

    G = nx.DiGraph()
    for _, row in agg.iterrows():
        u = int(row["winlog.event_id_1"])
        v = int(row["winlog.event_id_2"])
        w = int(row["weight"])
        # If the edge already exists, accumulate weight
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    return G

def compute_graph_metrics(G: nx.DiGraph) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute node and edge importance metrics for the global event graph.

    Returns:
      - nodes_df: event_id, degree_centrality
      - edges_df: winlog.event_id_1, winlog.event_id_2, edge_betweenness
    """
    if G is None or len(G) == 0:
        return (
            pd.DataFrame(columns=["event_id", "degree_centrality"]),
            pd.DataFrame(
                columns=["winlog.event_id_1", "winlog.event_id_2", "edge_betweenness"]
            ),
        )

    # Node centrality
    deg_centrality = nx.degree_centrality(G)
    nodes_df = pd.DataFrame(
        [
            {"event_id": node, "degree_centrality": centrality}
            for node, centrality in deg_centrality.items()
        ]
    )

    # Edge betweenness (sample k for performance on big graphs)
    edge_bet = nx.edge_betweenness_centrality(
        G, k=min(500, len(G.edges())) if len(G.edges()) > 0 else None
    )
    edges_df = pd.DataFrame(
        [
            {
                "winlog.event_id_1": u,
                "winlog.event_id_2": v,
                "edge_betweenness": edge_bet[(u, v)],
            }
            for (u, v) in edge_bet.keys()
        ]
    )

    return nodes_df, edges_df
