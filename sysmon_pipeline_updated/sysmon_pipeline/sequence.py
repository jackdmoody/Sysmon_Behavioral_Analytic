import pandas as pd
from .config import PipelineConfig

def assign_sessions(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """    Assign session_id per host using an inactivity gap threshold.

    Fixes critique #1 (temporal abstraction): prevents gluing '6 hours later' activity
    into the same chain as an initial user-driven sequence.

    TODO
    ----
    - Consider sessionization per (host,user) for workstation-heavy datasets.
    - Add process-tree reset boundaries if desired (more granular sessions).
    """
    out = df.sort_values(["host","ts"]).copy()
    gap = cfg.time.session_gap_seconds

    out["prev_ts"] = out.groupby("host")["ts"].shift(1)
    out["gap_s"] = (out["ts"] - out["prev_ts"]).dt.total_seconds()

    new_sess = out["gap_s"].isna() | (out["gap_s"] > gap)
    out["session_num"] = new_sess.groupby(out["host"]).cumsum()
    out["session_id"] = out["host"].astype(str) + ":" + out["session_num"].astype(int).astype(str)

    out = out.drop(columns=["prev_ts"])
    return out

def bucket_deltas(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Compute dt_s and dt_bucket within each session."""
    out = df.sort_values(["session_id","ts"]).copy()
    out["prev_ts_sess"] = out.groupby("session_id")["ts"].shift(1)
    out["dt_s"] = (out["ts"] - out["prev_ts_sess"]).dt.total_seconds().fillna(0.0)

    labels = []
    bounds = cfg.time.buckets
    for dt in out["dt_s"].to_numpy():
        for ub, lab in bounds:
            if dt <= ub:
                labels.append(lab)
                break

    out["dt_bucket"] = labels
    out = out.drop(columns=["prev_ts_sess"])
    return out

def build_transition_counts(df: pd.DataFrame, level: str = "token_medium") -> pd.DataFrame:
    """    Build time-aware transition counts: (state -> next_state, dt_bucket).

    Returns columns:
      - host, state, next_state, dt_bucket, count

    Notes
    -----
    Transition probabilities are later computed *per-role* with smoothing. Keep this step
    strictly about counts.

    TODO
    ----
    - Add windowed transitions (15m/1h/6h/24h) if you want multi-timescale scoring.
    """
    out = df.sort_values(["session_id","ts"]).copy()

    out["state"] = out[level].astype(str)
    out["next_state"] = out.groupby("session_id")[level].shift(-1)
    out["next_dt_bucket"] = out.groupby("session_id")["dt_bucket"].shift(-1)

    trans = out.dropna(subset=["next_state","next_dt_bucket"]).copy()

    trans = (trans.groupby(["host","state","next_state","next_dt_bucket"])
                  .size()
                  .reset_index(name="count")
                  .rename(columns={"next_dt_bucket":"dt_bucket"}))
    return trans
