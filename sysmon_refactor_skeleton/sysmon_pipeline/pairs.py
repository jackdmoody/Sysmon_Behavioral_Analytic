from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def ensure_pair_column(events: pd.DataFrame, event_id_col: str, pair_col: str="pair") -> pd.DataFrame:
    out = events.copy()
    out[pair_col] = out[event_id_col].astype("Int64").astype(str)
    return out

def correlate_critical_events_single_host(
    host_df: pd.DataFrame,
    *,
    ts_col: str,
    event_id_col: str,
    severity_col: str,
    window_seconds: int,
    critical_labels: Tuple[str, ...] = ("critical","high"),
) -> pd.DataFrame:
    # Build event pairs within a time window among critical events
    df = host_df.sort_values(ts_col).copy()
    df = df[df[severity_col].isin(critical_labels)].dropna(subset=[ts_col, event_id_col])
    if df.empty:
        return pd.DataFrame(columns=["src_event","dst_event","dt_seconds","count","pair"])

    times = df[ts_col].astype("int64") // 10**9
    evs = df[event_id_col].astype("Int64").astype(int).to_numpy()

    pairs = {}
    # O(n*w) naive; replace with two-pointer if needed
    for i in range(len(df)):
        t0 = times.iloc[i]
        j = i+1
        while j < len(df) and (times.iloc[j] - t0) <= window_seconds:
            key = (evs[i], evs[j])
            pairs[key] = pairs.get(key, 0) + 1
            j += 1

    rows = [{"src_event":k[0], "dst_event":k[1], "count":v, "pair":f"{k[0]}->{k[1]}"} for k,v in pairs.items()]
    out = pd.DataFrame(rows).sort_values("count", ascending=False)
    return out

def correlate_critical_events_by_host(
    df: pd.DataFrame,
    *,
    host_col: str,
    ts_col: str,
    event_id_col: str,
    severity_col: str,
    window_seconds: int,
) -> pd.DataFrame:
    all_rows = []
    for host, g in df.groupby(host_col, dropna=False):
        pairs = correlate_critical_events_single_host(
            g, ts_col=ts_col, event_id_col=event_id_col, severity_col=severity_col, window_seconds=window_seconds
        )
        if not pairs.empty:
            pairs.insert(0, host_col, host)
            all_rows.append(pairs)
    if not all_rows:
        return pd.DataFrame(columns=[host_col,"src_event","dst_event","count","pair"])
    return pd.concat(all_rows, ignore_index=True)

def compute_pair_stats(pairs_df: pd.DataFrame, *, host_col: str, count_col: str="count") -> pd.DataFrame:
    # host-level summary stats
    g = pairs_df.groupby(host_col)[count_col]
    stats = g.agg(["sum","mean","std","max","count"]).rename(columns={"count":"n_pairs"})
    stats = stats.reset_index()
    stats["std"] = stats["std"].fillna(0.0)
    return stats

def run_isolation_forest_on_hosts(stats_df: pd.DataFrame, *, feature_cols: List[str], contamination: float, random_state: int) -> pd.DataFrame:
    out = stats_df.copy()
    X = out[feature_cols].fillna(0.0).to_numpy()
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)
    out["iforest_score"] = -model.score_samples(X)  # higher = more anomalous
    out["iforest_label"] = model.predict(X)         # -1 outlier, 1 inlier
    return out
