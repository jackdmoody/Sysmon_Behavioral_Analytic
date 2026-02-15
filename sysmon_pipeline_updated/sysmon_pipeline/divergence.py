import numpy as np
import pandas as pd
from typing import Dict
from .config import PipelineConfig

def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence between probability vectors."""
    p = np.clip(p, eps, 1); p = p / p.sum()
    q = np.clip(q, eps, 1); q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)

def fit_peer_baselines(
    trans: pd.DataFrame,
    host_roles: pd.DataFrame,
    cfg: PipelineConfig
) -> Dict[str, pd.DataFrame]:
    """    Fit peer baseline transition distributions per role_id.

    Fixes critique #3: baseline against *peer role groups* rather than a global baseline.

    Returns
    -------
    Dict[str, pd.DataFrame]
      role_id -> baseline rows:
        ['state','next_state','dt_bucket','p_baseline']

    TODO
    ----
    - Contamination-aware fitting: pick central hosts in role-feature space and fit on that subset.
    - Backoff baseline: also fit on token_coarse transitions and store for backoff scoring.
    """
    t = trans.merge(host_roles[["host","role_id"]], on="host", how="left")
    baselines: Dict[str, pd.DataFrame] = {}

    for role_id, g in t.groupby("role_id"):
        agg = (g.groupby(["state","next_state","dt_bucket"])["count"].sum()
                .reset_index())

        agg["total"] = agg.groupby(["state"])["count"].transform("sum")

        alpha = cfg.baseline.laplace_alpha
        k = agg.groupby("state")["count"].transform("count")  # number of observed outgoing edges
        agg["p_baseline"] = (agg["count"] + alpha) / (agg["total"] + alpha * k)

        baselines[str(role_id)] = agg[["state","next_state","dt_bucket","p_baseline"]].copy()

    return baselines

def score_sequence_divergence(
    trans: pd.DataFrame,
    host_roles: pd.DataFrame,
    baselines: Dict[str, pd.DataFrame],
    cfg: PipelineConfig
) -> pd.DataFrame:
    """    Compute per-host sequence anomaly score using JS divergence between:
      P_host(next,dt | state) and P_peer(next,dt | state)

    Outputs
    -------
    DataFrame: columns
      - host, role_id, S_seq, rare_transition_hits

    TODO
    ----
    - Implement backoff divergence (medium -> coarse) using cfg.baseline.backoff_lambdas.
    - Add multi-window scoring (15m/1h/6h/24h).
    """
    t = trans.merge(host_roles[["host","role_id"]], on="host", how="left")
    rows = []

    for (host, role_id), g in t.groupby(["host","role_id"]):
        base = baselines.get(str(role_id))
        if base is None or base.empty:
            rows.append((host, role_id, 0.0, 0))
            continue

        divs = []
        rare_hits = 0

        for state, hs in g.groupby("state"):
            hb = hs.groupby(["next_state","dt_bucket"])["count"].sum().reset_index()
            hb["p_host"] = hb["count"] / hb["count"].sum()

            pb = base[base["state"] == state].copy()
            if pb.empty:
                continue

            hb["key"] = hb["next_state"].astype(str) + "|" + hb["dt_bucket"].astype(str)
            pb["key"] = pb["next_state"].astype(str) + "|" + pb["dt_bucket"].astype(str)

            all_keys = sorted(set(hb["key"]).union(set(pb["key"])))
            p = np.array([hb.loc[hb["key"] == k, "p_host"].sum() for k in all_keys], dtype=float)
            q = np.array([pb.loc[pb["key"] == k, "p_baseline"].sum() for k in all_keys], dtype=float)

            divs.append(_js_divergence(p, q))

            # Rare transition hits: count transitions with low peer probability
            # (simple heuristic; refine with thresholds or quantiles per role)
            rare_hits += int((q[q > 0].min() if (q > 0).any() else 1.0) < 1e-4)

        S_seq = float(np.mean(divs)) if divs else 0.0
        rows.append((host, role_id, S_seq, rare_hits))

    return pd.DataFrame(rows, columns=["host","role_id","S_seq","rare_transition_hits"])
