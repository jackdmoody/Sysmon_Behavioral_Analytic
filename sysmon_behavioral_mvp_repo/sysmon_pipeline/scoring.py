"""Scoring functions for the MVP."""

from __future__ import annotations

import numpy as np
import pandas as pd
from .config import PipelineConfig


def score_hosts(features: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Return triage table with anomaly_score (higher = more suspicious).

    Methods:
      - zscore: sum of absolute z-scores across columns
      - iforest: IsolationForest anomaly score (scaled so higher = more suspicious)
    """
    X = features.copy()

    # Remove non-signal columns from scoring if desired (keep it simple; score everything numeric)
    # Ensure float for model math
    X = X.astype(float)

    if cfg.scoring_method.lower() == "zscore":
        mu = X.mean(axis=0)
        sd = X.std(axis=0).replace(0, np.nan)
        z = (X - mu) / sd
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        score = z.abs().sum(axis=1)

    elif cfg.scoring_method.lower() == "iforest":
        from sklearn.ensemble import IsolationForest

        clf = IsolationForest(
            n_estimators=300,
            contamination=cfg.iforest_contamination,
            random_state=cfg.random_seed,
        )
        clf.fit(X)
        # decision_function: higher = more normal. We want higher = more suspicious.
        score = -clf.decision_function(X)

    else:
        raise ValueError(f"Unknown scoring_method: {cfg.scoring_method}. Use 'zscore' or 'iforest'.")

    triage = pd.DataFrame(index=X.index)
    triage.index.name = "host"
    triage["anomaly_score"] = score.astype(float)

    # Add a couple helpful columns for interpretation
    triage["total_events"] = features.get("total_events", np.nan)
    triage["events_per_hour"] = features.get("events_per_hour", np.nan)
    triage = triage.sort_values("anomaly_score", ascending=False)

    return triage
