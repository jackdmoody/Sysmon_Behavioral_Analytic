from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class TimeBucketingConfig:
    """Configuration for time delta bucketing and sessionization."""
    # (upper_bound_seconds, label)
    buckets: List[Tuple[float, str]] = field(default_factory=lambda: [
        (10, "<10s"),
        (60, "10-60s"),
        (5 * 60, "1-5m"),
        (30 * 60, "5-30m"),
        (2 * 60 * 60, "30m-2h"),
        (float("inf"), ">2h"),
    ])
    session_gap_seconds: int = 30 * 60  # inactivity -> new session

@dataclass
class BaselineConfig:
    """Peer baseline fitting configuration (robust-by-design)."""
    # Robust fitting / contamination assumptions
    central_fraction: float = 0.7  # fit baseline on densest 70% in role group (TODO)
    laplace_alpha: float = 0.5     # smoothing for transition probs
    # Backoff weighting for probability estimation (medium vs coarse)
    backoff_lambdas: Tuple[float, float] = (0.7, 0.3)

@dataclass
class ScoringConfig:
    """Fusion weights and gating logic for multi-channel scoring."""
    # Fusion weights
    w_seq: float = 0.40
    w_freq: float = 0.30
    w_ctx: float = 0.20
    w_drift: float = 0.10

    # Gating: require agreement unless extreme
    extreme_seq: float = 0.95
    extreme_freq: float = 0.95
    extreme_ctx: float = 0.95
    require_two_channels: bool = True

@dataclass
class RoleConfig:
    """Host role inference configuration."""
    # Role clustering features to compute in pairs.py
    role_feature_cols: List[str] = field(default_factory=lambda: [
        "proc_rate_total",
        "script_rate",
        "office_rate",
        "lolbin_rate",
        "has_encoded_rate",
        "has_download_cradle_rate",
        "unique_users",
        "unique_parents",
    ])
    min_hosts_per_role: int = 10  # fallback if too small (TODO for per-role models)

@dataclass
class PipelineConfig:
    """Top-level config object for the analytic."""
    time: TimeBucketingConfig = field(default_factory=TimeBucketingConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    role: RoleConfig = field(default_factory=RoleConfig)
