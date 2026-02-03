from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# hmmlearn is optional; functions will error if not installed when called
try:
    from hmmlearn import hmm
except Exception:  # pragma: no cover
    hmm = None

def encode_sequence_to_indices(sequence: List[int],
                               state_map: Dict[int, int]) -> np.ndarray:
    """
    Map raw event IDs into integer indices for the HMM.

    Unmapped IDs are skipped.
    """
    idxs = [state_map[e] for e in sequence if e in state_map]
    if not idxs:
        return np.empty((0, 1), dtype=int)
    return np.array(idxs, dtype=int).reshape(-1, 1)

def train_hmm_on_baseline(
    events_df: pd.DataFrame,
    state_map: Dict[int, int],
    num_states: int,
    host_col: str = "host.fqdn",
    event_col: str = "winlog.event_id",
    ts_col: str = "_timestamp",
    baseline_filter: Optional[pd.Series] = None,
    n_components: int = 8,
    max_iter: int = 200,
    random_state: int = 42,
) -> Optional["hmm.MultinomialHMM"]:
    """
    Train a Multinomial HMM on baseline hosts / time windows.

    n_components: number of hidden states (tune per environment).
    """
    if hmm is None:
        print("⚠️ hmmlearn is not available; cannot train HMM.")
        return None

    df = ensure_sorted_events(events_df, ts_col)
    if baseline_filter is not None:
        df = df[baseline_filter].copy()

    sequences = []
    lengths = []

    for host, hdf in df.groupby(host_col):
        seq = hdf[event_col].dropna().astype(int).tolist()
        X = encode_sequence_to_indices(seq, state_map)
        if X.shape[0] == 0:
            continue
        sequences.append(X)
        lengths.append(X.shape[0])

    if not sequences:
        print("⚠️ No baseline sequences to train HMM on.")
        return None

    X_concat = np.vstack(sequences)

    model = hmm.MultinomialHMM(
        n_components=n_components,
        n_iter=max_iter,
        random_state=random_state,
        verbose=False,
    )
    model.n_features = num_states
    model.fit(X_concat, lengths)
    return model

def score_hosts_with_hmm(
    events_df: pd.DataFrame,
    model: "hmm.MultinomialHMM",
    state_map: Dict[int, int],
    host_col: str = "host.fqdn",
    event_col: str = "winlog.event_id",
    ts_col: str = "_timestamp",
    min_len: int = 10,
) -> pd.DataFrame:
    """
    Compute HMM log-likelihood scores per host.

    Output:
      - host.fqdn
      - hmm_log_likelihood : higher = more normal, lower = more anomalous
      - seq_length         : number of events used
    """
    if model is None:
        raise RuntimeError("HMM model is None. Train it first with train_hmm_on_baseline().")

    df = ensure_sorted_events(events_df, ts_col)
    rows = []
    for host, hdf in df.groupby(host_col):
        seq = hdf[event_col].dropna().astype(int).tolist()
        X = encode_sequence_to_indices(seq, state_map)
        if X.shape[0] < min_len:
            continue
        log_prob = model.score(X)
        rows.append({
            "host.fqdn": host,
            "hmm_log_likelihood": float(log_prob),
            "seq_length": int(X.shape[0]),
        })

    result = pd.DataFrame(rows)
    # Sort ascending: most negative (worst fit) at the top = most suspicious
    result = result.sort_values("hmm_log_likelihood", ascending=True).reset_index(drop=True)
    return result

def build_hmm_hidden_state_features(
    events_df: pd.DataFrame,
    model: "hmm.MultinomialHMM",
    state_map: Dict[int, int],
    host_col: str = "host.fqdn",
    event_col: str = "winlog.event_id",
    ts_col: str = "_timestamp",
    min_len: int = 10,
) -> pd.DataFrame:
    """
    For each host:
      - Decode its sequence into HMM hidden states.
      - Count how often each state is visited.
      - Normalize to a probability per state (behavioral fingerprint).

    Output DataFrame:
      index  = host.fqdn
      cols   = hidden_state_0 ... hidden_state_{n_components-1}
    """
    if model is None:
        raise RuntimeError("HMM model is None. Train it first.")

    df = ensure_sorted_events(events_df, ts_col)
    n_components = model.n_components

    feature_rows = {}
    for host, hdf in df.groupby(host_col):
        seq = hdf[event_col].dropna().astype(int).tolist()
        X = encode_sequence_to_indices(seq, state_map)
        if X.shape[0] < min_len:
            continue

        hidden_states = model.predict(X)
        counts = np.bincount(hidden_states, minlength=n_components).astype(float)
        total = counts.sum()
        if total == 0:
            continue
        probs = counts / total
        feature_rows[host] = probs

    if not feature_rows:
        return pd.DataFrame()

    feats = pd.DataFrame.from_dict(
        feature_rows,
        orient="index",
        columns=[f"hidden_state_{i}" for i in range(n_components)],
    )
    feats.index.name = "host.fqdn"
    return feats

def run_isolation_forest_on_hmm_features(
    hmm_features: pd.DataFrame,
    contamination: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Apply IsolationForest to HMM hidden-state features.

    Output:
      - host.fqdn
      - iforest_label_hmm : -1 outlier, +1 normal
      - iforest_score_hmm : anomaly score (lower = more anomalous)
    """
    if hmm_features.empty:
        return pd.DataFrame()

    X = hmm_features.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
        n_jobs=-1,
    )
    labels = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)

    result = pd.DataFrame({
        "host.fqdn": hmm_features.index,
        "iforest_label_hmm": labels,
        "iforest_score_hmm": scores,
    }).reset_index(drop=True)

    result = result.sort_values("iforest_score_hmm", ascending=True)
    return result
