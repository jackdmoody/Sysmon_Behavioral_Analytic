import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from .config import PipelineConfig

def fit_frequency_model(df_rates: pd.DataFrame, cfg: PipelineConfig) -> IsolationForest:
    """    Fit Isolation Forest on per-host rate features.

    Fixes critique #4 (sequence vs frequency): this model is *only* for volumetric/rate anomalies.

    TODO
    ----
    - Robust-scale features (median/MAD) prior to fitting.
    - Fit per-role IF models if you have enough hosts per role.
    """
    X = df_rates.drop(columns=["host"], errors="ignore").fillna(0.0).to_numpy()
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)
    return model

def score_frequency(df_rates: pd.DataFrame, model: IsolationForest) -> pd.DataFrame:
    """Return S_freq in [0,1] where higher is more anomalous."""
    X = df_rates.drop(columns=["host"], errors="ignore").fillna(0.0).to_numpy()
    normality = model.decision_function(X)  # higher => more normal
    ranks = pd.Series(normality).rank(pct=True)
    S_freq = 1.0 - ranks.to_numpy()
    return pd.DataFrame({"host": df_rates["host"].values, "S_freq": S_freq})

def score_context(df: pd.DataFrame) -> pd.DataFrame:
    """    Context anomaly score based on compact 'fine token' flags.

    Fixes critique #2 (avoid sparse matrices): fine-grain detail contributes here, not in the main
    transition model.

    TODO
    ----
    - Add per-role suppression (common-but-weird tooling).
    - Add parent/child rarity modeling (e.g., java->powershell common on SIEM servers).
    - Add signature + integrity interactions (unsigned high-integrity LOLBIN etc.).
    """
    g = df.groupby("host").agg(
        encoded_hits=("has_encoded","sum"),
        download_hits=("has_download_cradle","sum"),
        lolbin_hits=("is_lolbin","sum"),
    ).reset_index()

    x = (0.6*g["encoded_hits"] + 0.3*g["download_hits"] + 0.2*g["lolbin_hits"]).to_numpy()
    S_ctx = 1.0 - np.exp(-x / 5.0)
    return pd.DataFrame({"host": g["host"].values, "S_ctx": S_ctx})

def score_drift(current_trans: pd.DataFrame, prior_trans: pd.DataFrame | None = None) -> pd.DataFrame:
    """    Optional drift score: behavior change vs the host's own recent history.

    Fixes critique #3 (baseline fallacy) by elevating *change* as a first-class signal.

    TODO
    ----
    - Implement per-host JS divergence between current and prior transition distributions.
    - If no prior window is provided, return zeros.
    """
    hosts = pd.Index(current_trans["host"].unique())
    if prior_trans is None or prior_trans.empty:
        return pd.DataFrame({"host": hosts.values, "S_drift": 0.0})
    # Placeholder
    return pd.DataFrame({"host": hosts.values, "S_drift": 0.0})

def fuse_scores(
    seq_scores: pd.DataFrame,
    freq_scores: pd.DataFrame,
    ctx_scores: pd.DataFrame,
    drift_scores: pd.DataFrame,
    cfg: PipelineConfig
) -> pd.DataFrame:
    """    Fuse explicit channels with gating to reduce false positives.

    Gate rule (default):
      - if any channel is extreme -> pass
      - else require at least two channels above 0.7

    Outputs
    -------
    DataFrame with:
      host, score, gate_pass, gate_reason, S_seq, S_freq, S_ctx, S_drift, rare_transition_hits
    """
    df = (seq_scores.merge(freq_scores, on="host", how="left")
                  .merge(ctx_scores, on="host", how="left")
                  .merge(drift_scores, on="host", how="left"))

    for c in ["S_seq","S_freq","S_ctx","S_drift"]:
        df[c] = df[c].fillna(0.0)

    df["score_raw"] = (
        cfg.scoring.w_seq*df["S_seq"]
        + cfg.scoring.w_freq*df["S_freq"]
        + cfg.scoring.w_ctx*df["S_ctx"]
        + cfg.scoring.w_drift*df["S_drift"]
    )
    df["score"] = df["score_raw"].rank(pct=True)

    extreme = (
        (df["S_seq"] >= cfg.scoring.extreme_seq) |
        (df["S_freq"] >= cfg.scoring.extreme_freq) |
        (df["S_ctx"] >= cfg.scoring.extreme_ctx)
    )

    if cfg.scoring.require_two_channels:
        high = (df[["S_seq","S_freq","S_ctx","S_drift"]] >= 0.7).sum(axis=1) >= 2
        df["gate_pass"] = extreme | high
        df["gate_reason"] = np.where(extreme, "extreme_channel",
                             np.where(high, "multi_channel", "low_support"))
    else:
        df["gate_pass"] = True
        df["gate_reason"] = "no_gating"

    return df.sort_values("score", ascending=False).reset_index(drop=True)
