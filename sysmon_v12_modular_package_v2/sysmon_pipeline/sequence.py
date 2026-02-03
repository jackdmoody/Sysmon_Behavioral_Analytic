from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

def ensure_sorted_events(df: pd.DataFrame, *, ts_col: str, host_col: str) -> pd.DataFrame:
    return df.sort_values([host_col, ts_col]).copy()

def build_state_map(df: pd.DataFrame, *, event_id_col: str) -> Dict[int,int]:
    uniq = pd.Series(df[event_id_col].dropna().astype("Int64").astype(int).unique()).sort_values()
    return {int(v): i for i, v in enumerate(uniq.tolist())}

def build_transition_counts(
    df: pd.DataFrame,
    *,
    host_col: str,
    ts_col: str,
    event_id_col: str,
    state_map: Dict[int,int],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    n = len(state_map)
    for host, g in df.groupby(host_col, dropna=False):
        g = g.sort_values(ts_col)
        seq = g[event_id_col].dropna().astype("Int64").astype(int).to_list()
        mat = np.zeros((n, n), dtype=float)
        for a, b in zip(seq, seq[1:]):
            if a in state_map and b in state_map:
                mat[state_map[a], state_map[b]] += 1.0
        out[str(host)] = mat
    return out

def normalize_rows(mat: np.ndarray, eps: float=1e-12) -> np.ndarray:
    rs = mat.sum(axis=1, keepdims=True)
    return mat / (rs + eps)

def build_host_markov_matrix(transition_counts: np.ndarray) -> np.ndarray:
    return normalize_rows(transition_counts)

def compute_baseline_markov_matrix(host_markov: Dict[str, np.ndarray], baseline_hosts: List[str]) -> np.ndarray:
    if not baseline_hosts:
        raise ValueError("baseline_hosts is empty")
    mats = [host_markov[h] for h in baseline_hosts if h in host_markov]
    if not mats:
        raise ValueError("no baseline hosts found in host_markov")
    avg = np.mean(mats, axis=0)
    return normalize_rows(avg)
