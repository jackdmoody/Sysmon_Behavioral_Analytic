"""Sysmon Behavioral Analytic MVP runner.

Usage:
  1) Put Sysmon exports into data/sample_sysmon.csv (default) OR edit config.py.
  2) Run:
       python -m sysmon_pipeline.pipeline

Outputs (in output/ by default):
  - normalized_events.csv
  - host_features.csv
  - triage.csv
  - score_histogram.png (optional)
  - top_hosts.png (optional)
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

from .config import PipelineConfig
from .io import load_events, ensure_output_dir
from .schema import normalize_schema, validate_schema
from .features import build_host_features
from .scoring import score_hosts
from .plots import plot_score_histogram, plot_top_hosts


logger = logging.getLogger("sysmon_pipeline")


def run(cfg: PipelineConfig) -> dict[str, Path]:
    out_dir = ensure_output_dir(cfg)

    logger.info("Loading events from %s", cfg.input_path)
    raw = load_events(cfg)
    logger.info("Loaded %d rows with %d columns", len(raw), raw.shape[1])

    logger.info("Normalizing schema")
    events = normalize_schema(raw, cfg)
    validate_schema(events)
    logger.info("Normalized to %d rows", len(events))

    # Save normalized events (helpful for debugging)
    normalized_path = out_dir / "normalized_events.csv"
    events.to_csv(normalized_path, index=False)

    logger.info("Building host features")
    feats = build_host_features(events, cfg)
    features_path = out_dir / "host_features.csv"
    feats.to_csv(features_path)

    logger.info("Scoring hosts (%s)", cfg.scoring_method)
    triage = score_hosts(feats, cfg)
    triage_path = out_dir / "triage.csv"
    triage.to_csv(triage_path)

    artifacts: dict[str, Path] = {
        "normalized_events": normalized_path,
        "host_features": features_path,
        "triage": triage_path,
    }

    if cfg.make_plots:
        logger.info("Generating plots")
        artifacts["score_histogram"] = plot_score_histogram(triage, out_dir)
        artifacts["top_hosts"] = plot_top_hosts(triage, out_dir, top_n=10)

    logger.info("Done. Outputs in %s", out_dir)
    return artifacts


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = PipelineConfig()
    run(cfg)


if __name__ == "__main__":
    main()
