import pandas as pd

CANON_COLS = [
    "ts","host","user","event_id","image","parent_image","cmdline",
    "integrity_level","signed","hash_sha256","dest_ip","dest_port","protocol"
]

def coerce_sysmon_schema(df: pd.DataFrame) -> pd.DataFrame:
    """    Coerce raw Sysmon/WEC exports into the canonical schema used by the pipeline.

    Expected minimal columns after coercion:
      - ts (datetime64[ns], UTC recommended)
      - host (str)
      - user (str|None)
      - event_id (int)
      - image (str)
      - parent_image (str|None)
      - cmdline (str|None)

    Notes
    -----
    This function is intentionally conservative: it only standardizes types and ensures
    required columns exist. You should extend the rename-map section to match your export.

    TODO
    ----
    - Map your specific ingestion fields (e.g., Sysmon XML / JSON keys) to canonical names.
    - Normalize ts to UTC.
    - Normalize process images (case, basename, etc.) if needed.
    """
    out = df.copy()

    # TODO: rename columns from your raw export into canonical names:
    # out = out.rename(columns={
    #   "UtcTime": "ts",
    #   "Computer": "host",
    #   "User": "user",
    #   "EventID": "event_id",
    #   "Image": "image",
    #   "ParentImage": "parent_image",
    #   "CommandLine": "cmdline",
    # })

    if "ts" in out.columns:
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce", utc=True)
    else:
        out["ts"] = pd.NaT

    if "host" in out.columns:
        out["host"] = out["host"].astype(str)
    else:
        out["host"] = None

    # Optional columns: create if missing
    for c in CANON_COLS:
        if c not in out.columns:
            out[c] = None

    # Basic hygiene
    out["image"] = out["image"].fillna("").astype(str)
    out["parent_image"] = out["parent_image"].fillna("").astype(str)
    out["cmdline"] = out["cmdline"].fillna("").astype(str)

    out = out.dropna(subset=["ts","host"])
    out = out.sort_values(["host","ts"]).reset_index(drop=True)
    return out
