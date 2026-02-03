from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Sequence
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
    pairs: Optional[pd.DataFrame] = None
    pair_stats: Optional[pd.DataFrame] = None
    state_map: Optional[Dict[int,int]] = None
    host_transition_counts: Optional[Dict[str, np.ndarray]] = None
    host_markov: Optional[Dict[str, np.ndarray]] = None
    baseline_markov: Optional[np.ndarray] = None
    markov_scores: Optional[pd.DataFrame] = None
    triage: Optional[pd.DataFrame] = None

class SysmonPipeline:
    """Stage-based pipeline.

    Typical usage:
        pipe = SysmonPipeline(cfg)
        pipe.fit(df_baseline)  # learns baseline markov
        art = pipe.run(df_new, stages=("enrich","pairs","markov","triage"))
    """

    def __init__(self, cfg: SysmonConfig):
        self.cfg = cfg
        self._baseline_hosts: Optional[List[str]] = None
        self._state_map: Optional[Dict[int,int]] = None
        self._baseline_markov: Optional[np.ndarray] = None

    # ---------------------------
    # Fit (learn baselines)
    # ---------------------------
    def fit(self, df: pd.DataFrame) -> "SysmonPipeline":
        cfg = self.cfg
        spec = SchemaSpec(required=(cfg.col_timestamp, cfg.col_host, cfg.col_event_id))
        validate_schema(df, spec)
        base = coerce_types(df, ts_col=cfg.col_timestamp, host_col=cfg.col_host, event_id_col=cfg.col_event_id)

        if cfg.drop_event_ids:
            base = base[~base[cfg.col_event_id].isin(list(cfg.drop_event_ids))]

        # enrich severity
        base = grade_events(
            base,
            event_id_col=cfg.col_event_id,
            out_score_col=cfg.col_severity_score,
            out_label_col=cfg.col_severity_label,
        )

        # choose baseline hosts
        if cfg.baseline_host_allowlist is not None:
            self._baseline_hosts = [str(h) for h in cfg.baseline_host_allowlist]
        else:
            self._baseline_hosts = (
                base[cfg.col_host].value_counts().head(cfg.baseline_top_n_hosts).index.astype(str).tolist()
            )

        # learn state map + baseline markov
        base_sorted = ensure_sorted_events(base, ts_col=cfg.col_timestamp, host_col=cfg.col_host)
        self._state_map = build_state_map(base_sorted, event_id_col=cfg.col_event_id)

        tcounts = build_transition_counts(
            base_sorted,
            host_col=cfg.col_host,
            ts_col=cfg.col_timestamp,
            event_id_col=cfg.col_event_id,
            state_map=self._state_map
        )
        host_markov = {h: build_host_markov_matrix(m) for h, m in tcounts.items()}
        self._baseline_markov = compute_baseline_markov_matrix(host_markov, self._baseline_hosts)

        return self

    # ---------------------------
    # Individual stages
    # ---------------------------
    def stage_enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        spec = SchemaSpec(required=(cfg.col_timestamp, cfg.col_host, cfg.col_event_id))
        validate_schema(df, spec)
        out = coerce_types(df, ts_col=cfg.col_timestamp, host_col=cfg.col_host, event_id_col=cfg.col_event_id)

        if cfg.drop_event_ids:
            out = out[~out[cfg.col_event_id].isin(list(cfg.drop_event_ids))]

        out = grade_events(
            out,
            event_id_col=cfg.col_event_id,
            out_score_col=cfg.col_severity_score,
            out_label_col=cfg.col_severity_label,
        )
        return out

    def stage_pairs(self, events: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        return correlate_critical_events_by_host(
            events,
            host_col=cfg.col_host,
            ts_col=cfg.col_timestamp,
            event_id_col=cfg.col_event_id,
            severity_col=cfg.col_severity_label,
            window_seconds=cfg.window_seconds,
            critical_labels=tuple(cfg.critical_labels),
        )

    def stage_pair_stats(self, pairs: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        stats = compute_pair_stats(pairs, host_col=cfg.col_host, count_col="count")
        feat_cols = [c for c in ["sum","mean","std","max","n_pairs"] if c in stats.columns]
        stats = run_isolation_forest_on_hosts(
            stats,
            feature_cols=feat_cols,
            contamination=cfg.iforest_contamination,
            random_state=cfg.random_state,
        )
        return stats

    def stage_markov(self, events: pd.DataFrame) -> Dict[str, Any]:
        if self._state_map is None or self._baseline_markov is None:
            raise RuntimeError("Markov stage requires fit() to be called first.")
        cfg = self.cfg
        sorted_df = ensure_sorted_events(events, ts_col=cfg.col_timestamp, host_col=cfg.col_host)
        tcounts = build_transition_counts(
            sorted_df,
            host_col=cfg.col_host,
            ts_col=cfg.col_timestamp,
            event_id_col=cfg.col_event_id,
            state_map=self._state_map,
        )
        host_markov = {h: build_host_markov_matrix(m) for h, m in tcounts.items()}
        scores = compute_host_markov_scores(host_markov, self._baseline_markov)
        return {
            "state_map": self._state_map,
            "baseline_markov": self._baseline_markov,
            "host_transition_counts": tcounts,
            "host_markov": host_markov,
            "markov_scores": scores,
        }

    def stage_triage(self, pair_stats: pd.DataFrame, markov_scores: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        return build_ranked_triage(pair_stats=pair_stats, markov_scores=markov_scores, host_col=cfg.col_host)

    # ---------------------------
    # Run selected stages
    # ---------------------------
    def run(self, df: pd.DataFrame, stages: Sequence[str]=("enrich","pairs","pair_stats","markov","triage")) -> SysmonArtifacts:
        """Run a subset of stages and return collected artifacts.

        Valid stages: enrich, pairs, pair_stats, markov, triage
        """
        cfg = self.cfg
        events = None
        pairs = None
        pair_stats = None
        state_map = None
        host_transition_counts = None
        host_markov = None
        baseline_markov = None
        markov_scores = None
        triage = None

        for st in stages:
            if st == "enrich":
                events = self.stage_enrich(df)
            elif st == "pairs":
                if events is None:
                    events = self.stage_enrich(df)
                pairs = self.stage_pairs(events)
            elif st == "pair_stats":
                if pairs is None:
                    if events is None:
                        events = self.stage_enrich(df)
                    pairs = self.stage_pairs(events)
                pair_stats = self.stage_pair_stats(pairs)
            elif st == "markov":
                if events is None:
                    events = self.stage_enrich(df)
                m = self.stage_markov(events)
                state_map = m["state_map"]
                baseline_markov = m["baseline_markov"]
                host_transition_counts = m["host_transition_counts"]
                host_markov = m["host_markov"]
                markov_scores = m["markov_scores"]
            elif st == "triage":
                if pair_stats is None:
                    if pairs is None:
                        if events is None:
                            events = self.stage_enrich(df)
                        pairs = self.stage_pairs(events)
                    pair_stats = self.stage_pair_stats(pairs)
                if markov_scores is None:
                    if events is None:
                        events = self.stage_enrich(df)
                    m = self.stage_markov(events)
                    state_map = m["state_map"]
                    baseline_markov = m["baseline_markov"]
                    host_transition_counts = m["host_transition_counts"]
                    host_markov = m["host_markov"]
                    markov_scores = m["markov_scores"]
                triage = self.stage_triage(pair_stats, markov_scores)
            else:
                raise ValueError(f"Unknown stage: {st}")

        if events is None:
            events = self.stage_enrich(df)

        return SysmonArtifacts(
            events=events,
            pairs=pairs,
            pair_stats=pair_stats,
            state_map=state_map,
            host_transition_counts=host_transition_counts,
            host_markov=host_markov,
            baseline_markov=baseline_markov,
            markov_scores=markov_scores,
            triage=triage,
        )
