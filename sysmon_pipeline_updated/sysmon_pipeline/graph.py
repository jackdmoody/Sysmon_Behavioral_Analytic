import pandas as pd

def build_transition_graph_edges(trans: pd.DataFrame, host: str | None = None) -> pd.DataFrame:
    """    Optional: produce weighted edges for graph-based explainability.

    Returns columns:
      - state, next_state, dt_bucket, weight

    TODO
    ----
    - Add graph metrics (PageRank, betweenness) for research features.
    """
    t = trans.copy()
    if host is not None:
        t = t[t["host"] == host]
    edges = (t.groupby(["state","next_state","dt_bucket"])["count"].sum()
              .reset_index()
              .rename(columns={"count":"weight"}))
    return edges
