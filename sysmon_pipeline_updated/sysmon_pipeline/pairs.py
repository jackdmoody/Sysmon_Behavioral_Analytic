import pandas as pd
from .config import PipelineConfig

def compute_rate_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """    Compute per-host rate features for the frequency channel + role inference.

    Fixes critique #4 by separating frequency modeling from sequencing.

    Produces (per host):
      - proc_rate_total (events/hour)
      - script_rate, office_rate, lolbin_rate
      - has_encoded_rate, has_download_cradle_rate
      - unique_users, unique_parents

    TODO
    ----
    - Add network-centric rates if Sysmon Event ID 3 is available.
    - Add per-user features for workstation contexts (e.g., user diversity, admin logons).
    """
    out = df.copy()

    host_span = out.groupby("host")["ts"].agg(["min","max"])
    host_span["hours"] = (host_span["max"] - host_span["min"]).dt.total_seconds() / 3600.0
    host_span["hours"] = host_span["hours"].clip(lower=1e-6)

    counts = out.groupby("host").size().rename("event_count").to_frame()
    feats = counts.join(host_span["hours"])
    feats["proc_rate_total"] = feats["event_count"] / feats["hours"]

    for cname, token in [("script_rate","SCRIPT"), ("office_rate","OFFICE"), ("lolbin_rate","LOLBIN")]:
        c = out[out["token_coarse"] == token].groupby("host").size().rename(cname)
        feats = feats.join(c, how="left")
        feats[cname] = feats[cname].fillna(0.0) / feats["hours"]

    for flag in ["has_encoded", "has_download_cradle"]:
        cname = f"{flag}_rate"
        c = out[out[flag] == True].groupby("host").size().rename(cname)
        feats = feats.join(c, how="left")
        feats[cname] = feats[cname].fillna(0.0) / feats["hours"]

    feats["unique_users"] = out.groupby("host")["user"].nunique(dropna=True)
    feats["unique_parents"] = out.groupby("host")["parent_image"].nunique(dropna=True)

    return feats.reset_index()

def build_role_features(df_rates: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Return the role feature matrix used for host role inference."""
    cols = ["host"] + cfg.role.role_feature_cols
    cols = [c for c in cols if c in df_rates.columns]
    return df_rates[cols].copy()
