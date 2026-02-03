from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd

def score_to_label(score: float) -> str:
    # Keep consistent with your notebook semantics; adjust thresholds as desired.
    if score >= 0.85:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"

def grade_event(event_id: int) -> float:
    # Placeholder defaults. Replace with your notebook’s event_id->score mapping.
    # You likely had a dict-based mapping + fallback logic.
    # Return value in [0,1].
    defaults = {
        1: 0.2,     # Process Create
        3: 0.5,     # Network Connect
        7: 0.6,     # Image Loaded (example)
        11: 0.7,    # File Create
        13: 0.8,    # Registry Value Set
        22: 0.85,   # DNS Query
    }
    return float(defaults.get(int(event_id), 0.1))

def grade_events(df: pd.DataFrame, event_id_col: str, out_score_col: str="severity_score", out_label_col: str="severity_label") -> pd.DataFrame:
    out = df.copy()
    out[out_score_col] = out[event_id_col].astype("Int64").apply(lambda x: grade_event(x) if pd.notna(x) else 0.0)
    out[out_label_col] = out[out_score_col].apply(score_to_label)
    return out

def tag_pairs_with_mitre(pairs_df: pd.DataFrame, *, pair_col: str="pair", mapping: Optional[Dict[str,str]]=None) -> pd.DataFrame:
    # mapping: pair string -> MITRE technique ID/name label
    out = pairs_df.copy()
    if mapping is None:
        mapping = {}
    out["mitre_tag"] = out[pair_col].map(mapping).fillna("unmapped")
    return out
