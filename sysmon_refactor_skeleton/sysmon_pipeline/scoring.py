from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .divergence import kl_divergence_matrix, js_divergence_matrix

def compute_host_markov_scores(
    host_markov: Dict[str, np.ndarray],
    baseline_markov: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for host, P in host_markov.items():
        rows.append({
            "host": host,
            "markov_kl": kl_divergence_matrix(P, baseline_markov),
            "markov_js": js_divergence_matrix(P, baseline_markov),
        })
    return pd.DataFrame(rows).sort_values("markov_js", ascending=False)

def build_ranked_triage(
    *,
    pair_stats: pd.DataFrame,
    markov_scores: pd.DataFrame,
    host_col: str,
) -> pd.DataFrame:
    # Merge evidence layers into one table
    triage = pair_stats.merge(markov_scores, left_on=host_col, right_on="host", how="left")
    triage = triage.drop(columns=["host"])
    # Simple combined score (tune as needed)
    triage["combined_score"] = (
        triage.get("iforest_score", 0).fillna(0)
        + triage.get("markov_js", 0).fillna(0)
    )
    return triage.sort_values("combined_score", ascending=False)
