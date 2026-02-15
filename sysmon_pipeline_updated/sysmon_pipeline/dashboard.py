import pandas as pd

def top_weird_transitions(trans: pd.DataFrame, baseline: pd.DataFrame, host: str, n: int = 15) -> pd.DataFrame:
    """Return host transitions with lowest peer baseline probability (good for explainability)."""
    h = trans[trans["host"] == host].copy()
    m = h.merge(baseline, on=["state","next_state","dt_bucket"], how="left")
    m["p_baseline"] = m["p_baseline"].fillna(0.0)
    return m.sort_values(["p_baseline","count"], ascending=[True, False]).head(n)

def score_breakdown_table(fused_scores: pd.DataFrame) -> pd.DataFrame:
    """Analyst triage view: channel scores + gating."""
    cols = ["host","role_id","score","gate_pass","gate_reason","S_seq","S_freq","S_ctx","S_drift","rare_transition_hits"]
    return fused_scores[[c for c in cols if c in fused_scores.columns]].copy()
