# Sysmon Behavioral Analytic (Time-Aware, Peer-Baselined, Multi-Channel)

This repository provides a deployable methodology and reference implementation for **behavioral anomaly detection** on Sysmon / Windows Event telemetry using:

- **Time-aware transition modeling** (sessionized, ∆t-bucketed Markov-style transitions)
- **Peer baselines** by host role (robust to “compromised-by-default” environments)
- **Explicit multi-channel scoring** that separates:
  - **Sequence** anomalies (structural weirdness)
  - **Frequency** anomalies (volumetric weirdness)
  - **Context** anomalies (fine-grained flags like EncodedCommand)
  - **Drift** anomalies (behavior change over time)

The pipeline is designed to directly address four common implementation critiques:

1. **Temporal abstraction ambiguity** (2 minutes vs 6 hours “same” chain)
2. **Granularity vs sparsity tradeoff** (sparse transition matrices causing FP storms)
3. **Baseline fallacy** (network already compromised → baseline contains attacker behavior)
4. **Sequence vs frequency mismatch** (Markov ≠ Isolation Forest; they detect different things)

---

## Methodology

### 1) Normalize → Enrich → Abstract
Raw Sysmon logs are coerced into a canonical schema (`schema.py`), then enriched into **multi-resolution tokens** (`mapping.py`):

- `token_coarse` (low sparsity): `OFFICE`, `SCRIPT`, `BROWSER`, `LOLBIN`, `PROC`
- `token_medium` (balanced): `OFFICE:WORD`, `SCRIPT:POWERSHELL`, ...
- `token_fine` (compact context): adds parent, key cmd flags, integrity, signature

**Key design choice:** Transition modeling uses `token_medium` (and optionally `token_coarse` backoff) to avoid state explosion, while `token_fine` contributes to **context scoring**, not the transition matrix.

### 2) Sessionization + ∆t bucketing (Fixes critique #1)
Events are split into sessions using an inactivity gap (`config.time.session_gap_seconds`), then time deltas are bucketed:

`P(next_state, dt_bucket | state)`

This separates “rapid kill-chain” behavior from delayed scheduled tasks that would otherwise look identical as pure sequences.

### 3) Peer baselines by role (Fixes critique #3)
Baselines are computed **within role groups** (server/web/DC/etc.) instead of global network baselines.

This repository includes a placeholder role assignment (`infer_roles()`), intended to be replaced by:
- asset inventory joins (best), or
- clustering on stable per-host role features (fallback)

### 4) Separate channels: sequence vs frequency vs context vs drift (Fixes critique #4)
Scoring uses explicit channels:

- **Sequence channel (S_seq)**: Jensen–Shannon divergence between a host’s time-aware transition distributions and its peer baseline (`divergence.py`)
- **Frequency channel (S_freq)**: Isolation Forest on per-host rate features (`pairs.py`, `scoring.py`)
- **Context channel (S_ctx)**: compact fine-token flags (encoded commands, download cradles, LOLBins)
- **Drift channel (S_drift)**: behavior change vs recent history (placeholder; intended for windowed operation)

Scores are fused with configurable weights and **gating** to reduce false positives (`scoring.fuse_scores()`).

---

## Repository Layout

```
sysmon_pipeline/
  __init__.py
  config.py        # thresholds, buckets, weights
  schema.py        # canonical schema coercion
  mapping.py       # tokens + context flags + (TODO) MITRE + severity
  sequence.py      # sessions + dt buckets + transition counts
  pairs.py         # per-host rate features (frequency channel + role features)
  divergence.py    # peer baselines + JS divergence scoring
  scoring.py       # IF model + context scoring + fusion + gating
  graph.py         # optional graph helpers
  dashboard.py     # explainability helpers
  pipeline.py      # fit/score orchestration
```

---

## Quickstart

### Install
This is plain Python. You’ll want:
- pandas
- numpy
- scikit-learn

### Use
```python
import pandas as pd
from sysmon_pipeline import SysmonBehavioralPipeline, PipelineConfig

raw = pd.read_parquet("sysmon.parquet")  # or your loader
pipe = SysmonBehavioralPipeline(PipelineConfig())

pipe.fit(raw)              # learns peer baselines + frequency model
scores = pipe.score(raw)   # returns triage table

print(scores.head(20))
```

---

## Data Contract

Minimum columns expected after coercion:

- `ts`, `host`, `event_id`, `image`, `parent_image`, `cmdline`, `user`

Optional but recommended:

- `integrity_level`, `signed`, `hash_sha256`, `dest_ip`, `dest_port`, `protocol`

See `schema.py` for the canonical column list. Extend the rename-map in `coerce_sysmon_schema()` to match your export.

---

## What to implement next (highest ROI)

1. **Role inference**: replace placeholder `infer_roles()` with inventory joins or clustering.
2. **Backoff divergence**: implement medium→coarse backoff using `config.baseline.backoff_lambdas`.
3. **Windowed scoring**: add 15m/1h/6h/24h windows and compute drift (S_drift) properly.
4. **Per-role suppression**: learn “common-but-weird” tooling patterns (e.g., java→powershell on SIEM servers) to reduce FPs.

---

## Notes for research write-up

This implementation is suitable as a methods appendix for an academic paper:
- The core novelty is **time-aware transition modeling** + **peer baselines** + **multi-channel fusion** designed for compromised-by-default DCO assumptions.
- Explainability hooks are included via `dashboard.py` and optional graph edges.

---

## License
Choose a license appropriate for your intended release (MIT/Apache-2.0 are common).
