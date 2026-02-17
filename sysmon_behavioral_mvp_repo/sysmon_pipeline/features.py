"""Feature engineering for the MVP.

Produces a host-level feature table suitable for simple anomaly scoring.

Feature blocks:
  - total_events
  - event_id counts (top-N most common event IDs in the dataset)
  - image counts (top-N most common process images)
  - unique_images
  - unique_users
  - events_per_hour (simple rate)
"""

from __future__ import annotations

import pandas as pd
from .config import PipelineConfig


def build_host_features(events: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    # Basic host aggregation
    g = events.groupby("host", dropna=False)

    out = pd.DataFrame(index=g.size().index)
    out.index.name = "host"

    out["total_events"] = g.size().astype(int)

    # Time span + rate
    tmin = g["timestamp"].min()
    tmax = g["timestamp"].max()
    hours = (tmax - tmin).dt.total_seconds() / 3600.0
    hours = hours.clip(lower=1e-6)  # avoid divide-by-zero
    out["window_hours"] = hours
    out["events_per_hour"] = out["total_events"] / out["window_hours"]

    # Unique counts
    out["unique_images"] = g["image"].nunique(dropna=True).astype(int)
    out["unique_users"] = g["user"].nunique(dropna=True).astype(int)

    # Top-N event IDs as one-hot-ish counts
    top_eids = events["event_id"].value_counts().head(cfg.top_event_ids).index.tolist()
    eid_counts = (
        events[events["event_id"].isin(top_eids)]
        .groupby(["host", "event_id"])
        .size()
        .unstack(fill_value=0)
    )
    eid_counts.columns = [f"eid_{int(c)}" for c in eid_counts.columns]
    out = out.join(eid_counts, how="left").fillna(0)

    # Top-N images as counts (normalize image string a bit)
    images = events["image"].fillna("").astype(str).str.lower().str.strip()
    events2 = events.copy()
    events2["image_norm"] = images

    top_imgs = events2["image_norm"].value_counts().head(cfg.top_images).index.tolist()
    img_counts = (
        events2[events2["image_norm"].isin(top_imgs)]
        .groupby(["host", "image_norm"])
        .size()
        .unstack(fill_value=0)
    )
    # keep feature names short-ish
    img_counts.columns = [f"img_{c[:60].replace('\\\', '').replace(' ', '_').replace(':','_')}" for c in img_counts.columns]
    out = out.join(img_counts, how="left").fillna(0)

    # Final: ensure numeric
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0)
    return out
