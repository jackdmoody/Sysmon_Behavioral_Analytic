from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
import numpy as np
import pandas as pd

from .config import SysmonConfig
from .schema import SchemaSpec, validate_schema, coerce_types
from .mapping import grade_events
from .pairs import correlate_critical_events_by_host, compute_pair_stats, run_isolation_forest_on_hosts
from .sequence import ensure_sorted_events, build_state_map, build_transition_counts, build_host_markov_matrix, compute_baseline_markov_matrix
from .scoring import compute_host_markov_scores, build_ranked_triage

@dataclass
class SysmonArtifacts:
    events: pd.DataFrame
    pairs: pd.DataFrame
    pair_stats: pd.DataFrame
    state_map: Dict[int,int]
    host_transition_counts: Dict[str, np.ndarray]
    host_markov: Dict[str, np.ndarray]
    baseline_markov: np.ndarray
    markov_scores: pd.DataFrame
    triage: pd.DataFrame

class SysmonPipeline:
    def __init__(self, cfg: SysmonConfig):
        self.cfg = cfg
        self._baseline_hosts: Optional[List[str]] = None
        self._state_map: Optional[Dict[int,int]] = None
        self._baseline_markov: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> "SysmonPipeline":
        cfg = self.cfg
        spec = SchemaSpec(required=(cfg.col_timestamp, cfg.col_host, cfg.col_event_id))
        validate_schema(df, spec)
        df2 = coerce_types(df, ts_col=cfg.col_timestamp, event_id_col=cfg.col_event_id, host_col=cfg.col_host)

        # basic enrich
        df2 = grade_events(df2, event_id_col=cfg.col_event_id, out_label_col=cfg.col_severity_label)

        # baseline hosts
        if cfg.baseline_host_allowlist is not None:
            self._baseline_hosts = [str(h) for h in cfg.baseline_host_allowlist]
        else:
            # naive inference: take most-common hosts as baseline
            self._baseline_hosts = df2[cfg.col_host].value_counts().head(20).index.astype(str).tolist()

        # markov baseline
        df_sorted = ensure_sorted_events(df2, ts_col=cfg.col_timestamp, host_col=cfg.col_host)
        self._state_map = build_state_map(df_sorted, event_id_col=cfg.col_event_id)
        tcounts = build_transition_counts(df_sorted, host_col=cfg.col_host, ts_col=cfg.col_timestamp, event_id_col=cfg.col_event_id, state_map=self._state_map)
        host_markov = {h: build_host_markov_matrix(m) for h, m in tcounts.items()}
        self._baseline_markov = compute_baseline_markov_matrix(host_markov, self._baseline_hosts)
        return self

    def score(self, df: pd.DataFrame) -> SysmonArtifacts:
        if self._state_map is None or self._baseline_markov is None or self._baseline_hosts is None:
            raise RuntimeError("Pipeline not fit(). Call fit(df_baseline) before score(df).")

        cfg = self.cfg
        spec = SchemaSpec(required=(cfg.col_timestamp, cfg.col_host, cfg.col_event_id))
        validate_schema(df, spec)
        events = coerce_types(df, ts_col=cfg.col_timestamp, event_id_col=cfg.col_event_id, host_col=cfg.col_host)
        events = grade_events(events, event_id_col=cfg.col_event_id, out_label_col=cfg.col_severity_label)

        # event pairs
        pairs = correlate_critical_events_by_host(
            events,
            host_col=cfg.col_host,
            ts_col=cfg.col_timestamp,
            event_id_col=cfg.col_event_id,
            severity_col=cfg.col_severity_label,
            window_seconds=cfg.window_seconds,
        )

        pair_stats = compute_pair_stats(pairs, host_col=cfg.col_host, count_col="count")
        # IsolationForest on host summary
        feature_cols = [c for c in ["sum","mean","std","max","n_pairs"] if c in pair_stats.columns]
        pair_stats = run_isolation_forest_on_hosts(
            pair_stats,
            feature_cols=feature_cols,
            contamination=cfg.iforest_contamination,
            random_state=cfg.random_state
        )

        # markov scores for hosts
        df_sorted = ensure_sorted_events(events, ts_col=cfg.col_timestamp, host_col=cfg.col_host)
        tcounts = build_transition_counts(df_sorted, host_col=cfg.col_host, ts_col=cfg.col_timestamp, event_id_col=cfg.col_event_id, state_map=self._state_map)
        host_markov = {h: build_host_markov_matrix(m) for h, m in tcounts.items()}
        markov_scores = compute_host_markov_scores(host_markov, self._baseline_markov)

        triage = build_ranked_triage(pair_stats=pair_stats, markov_scores=markov_scores, host_col=cfg.col_host)

        return SysmonArtifacts(
            events=events,
            pairs=pairs,
            pair_stats=pair_stats,
            state_map=self._state_map,
            host_transition_counts=tcounts,
            host_markov=host_markov,
            baseline_markov=self._baseline_markov,
            markov_scores=markov_scores,
            triage=triage,
        )
