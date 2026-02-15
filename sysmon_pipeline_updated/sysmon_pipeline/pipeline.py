import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

from .config import PipelineConfig
from .schema import coerce_sysmon_schema
from .mapping import build_tokens
from .sequence import assign_sessions, bucket_deltas, build_transition_counts
from .pairs import compute_rate_features, build_role_features
from .divergence import fit_peer_baselines, score_sequence_divergence
from .scoring import (
    fit_frequency_model, score_frequency, score_context, score_drift, fuse_scores
)

@dataclass
class FittedArtifacts:
    cfg: PipelineConfig
    host_roles: pd.DataFrame
    peer_baselines: Dict[str, pd.DataFrame]
    freq_model: object

def infer_roles(role_features: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """    Assign role_id per host.

    Fixes critique #3 by ensuring peer baselines are computed among comparable hosts.

    Current default: everything is 'default' role.

    TODO
    ----
    - Best: join from asset inventory / CMDB (DC/web/file/db/admin-workstation/etc.)
    - Fallback: clustering (HDBSCAN/KMeans) on role_features
    """
    out = role_features[["host"]].copy()
    out["role_id"] = "default"
    return out

class SysmonBehavioralPipeline:
    """    Orchestrates fit/score for the time-aware, peer-baselined multi-channel Sysmon analytic.

    Methodology alignment
    ---------------------
    - #1 Temporal abstraction: sessionization + dt_bucket Markov transitions
    - #2 Sparsity: multi-resolution tokens; fine detail used in context channel
    - #3 Baseline fallacy: peer baselines by role + drift channel
    - #4 Seq vs freq: explicit separate channels with fusion + gating
    """
    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.artifacts: Optional[FittedArtifacts] = None

    def fit(self, raw_df: pd.DataFrame) -> FittedArtifacts:
        df = coerce_sysmon_schema(raw_df)
        df = build_tokens(df)

        df = assign_sessions(df, self.cfg)
        df = bucket_deltas(df, self.cfg)

        rates = compute_rate_features(df, self.cfg)
        role_feats = build_role_features(rates, self.cfg)
        host_roles = infer_roles(role_feats, self.cfg)

        trans_med = build_transition_counts(df, level="token_medium")

        peer_baselines = fit_peer_baselines(trans_med, host_roles, self.cfg)

        freq_model = fit_frequency_model(
            rates.drop(columns=["host"], errors="ignore").join(rates[["host"]]),
            self.cfg
        )

        self.artifacts = FittedArtifacts(
            cfg=self.cfg,
            host_roles=host_roles,
            peer_baselines=peer_baselines,
            freq_model=freq_model
        )
        return self.artifacts

    def score(self, raw_df: pd.DataFrame, prior_window_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if self.artifacts is None:
            raise RuntimeError("Call fit() first.")

        df = coerce_sysmon_schema(raw_df)
        df = build_tokens(df)
        df = assign_sessions(df, self.cfg)
        df = bucket_deltas(df, self.cfg)

        host_roles = self.artifacts.host_roles

        trans_med = build_transition_counts(df, level="token_medium")
        seq_scores = score_sequence_divergence(trans_med, host_roles, self.artifacts.peer_baselines, self.cfg)

        rates = compute_rate_features(df, self.cfg)
        freq_scores = score_frequency(rates, self.artifacts.freq_model)

        ctx_scores = score_context(df)

        drift_scores = score_drift(trans_med, None)  # TODO: use prior_window_df transitions

        fused = fuse_scores(seq_scores, freq_scores, ctx_scores, drift_scores, self.cfg)
        fused = fused.merge(host_roles, on="host", how="left")

        return fused
