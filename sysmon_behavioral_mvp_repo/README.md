# Sysmon Behavioral Analytic — MVP

This is a **copy‑paste minimal working pipeline (MVP)** for a Sysmon behavioral analytic.

It implements a thin vertical slice:

1. Load Sysmon-like events from **CSV** or **JSONL**
2. Normalize to a consistent schema (`timestamp`, `host`, `event_id`, `image`, `command_line`, `user`)
3. Build **host-level features** (counts + simple rates + top event IDs / images)
4. Compute an **anomaly score** (default: z-score; optional: Isolation Forest)
5. Output a ranked **triage table** and basic plots

## Repo layout

```text
.
├── sysmon_pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── io.py
│   ├── schema.py
│   ├── features.py
│   ├── scoring.py
│   ├── plots.py
│   └── pipeline.py
├── data/
│   └── sample_sysmon.csv
├── output/
├── tests/
├── requirements.txt
└── .gitignore
```

## Quickstart

### 1) Create a venv and install deps

```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
# .venv\Scripts\activate  # windows

pip install -r requirements.txt
```

### 2) Put Sysmon data into `data/sample_sysmon.csv`

The MVP expects (at minimum) columns equivalent to:

- timestamp column: `UtcTime` (default; configurable in `PipelineConfig.timestamp_col`)
- host column: one of `Computer`, `Host`, `Hostname`, `host`
- event id column: one of `EventID`, `EventId`, `event_id`

Optional but recommended:
- `Image` (process image)
- `CommandLine`
- `User` / `UserName`

> If your export uses different names, update the candidates in `sysmon_pipeline/config.py`.

### 3) Run

```bash
python -m sysmon_pipeline.pipeline
```

Outputs will be written to `output/` by default.

## Outputs

- `output/normalized_events.csv` — normalized schema, good for debugging
- `output/host_features.csv` — host-level feature matrix
- `output/triage.csv` — ranked hosts by `anomaly_score`
- `output/score_histogram.png` — distribution of scores (if enabled)
- `output/top_hosts.png` — top 10 hosts (if enabled)

## Switching scoring methods

In `sysmon_pipeline/config.py`, set:

- `scoring_method="zscore"` (default) — sums absolute z-scores across feature columns
- `scoring_method="iforest"` — Isolation Forest anomaly score (requires scikit-learn)

## Next steps (easy upgrades)

- Add **event-pair** features (consecutive event transitions per host)
- Add a **Markov transition matrix** and **JS divergence** drift vs baseline
- Add peer baselines by **host role** (DC/WEC/server/workstation)
- Add MITRE/severity enrichment and evidence fusion

## Notes

- This MVP is designed to be small, deterministic, and easy to extend.
- It is not a detection guarantee — it's a baseline scoring/triage engine to support hunting.
