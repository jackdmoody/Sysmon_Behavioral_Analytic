# Sysmon Behavioral Analytics Pipeline

A modular, end-to-end analytics pipeline for **behavior-based detection, sequence drift analysis, and host-centric triage** using Windows Sysmon telemetry.

This project refactors a previously monolithic research notebook into a **testable, reproducible, and extensible package** suitable for:
- threat hunting / SOC workflows  
- analytic prototyping and research  
- operational experimentation and evaluation  

---

## Conceptual Overview

At a high level, the pipeline answers the question:

> *“Which hosts are behaving abnormally, **why**, and based on what evidence?”*

It does this by layering **complementary analytic perspectives** over the same Sysmon event stream:

1. **Behavioral Correlation**  
   Identifies suspicious *sequences of events* by correlating critical Sysmon events occurring close together in time.

2. **Statistical Anomaly Detection**  
   Aggregates per-host behavioral summaries and flags hosts that deviate from the population using Isolation Forest.

3. **Sequential Drift Detection**  
   Models each host’s event stream as a Markov process and measures divergence from a learned baseline using KL and Jensen–Shannon divergence.

4. **Evidence Fusion & Triage**  
   Combines statistical outliers and sequential drift into a single ranked triage table designed for analyst review.

Rather than producing binary alerts, the system produces **ranked, evidence-backed leads**.

---
## Analytic Philosophy

This pipeline is intentionally:

- **Host-centric**  
  Behavior is modeled per endpoint, not per event.

- **Sequence-aware**  
  Order and transition structure matter, not just event counts.

- **Baseline-driven**  
  “Abnormal” is defined relative to peer behavior, not static thresholds.

- **Explainable**  
  Every score can be traced back to event pairs, transitions, or distributions.

---

## Repository Structure

```
sysmon_refactor/
├── sysmon_pipeline/
│   ├── __init__.py
│   ├── config.py
│   ├── schema.py
│   ├── mapping.py
│   ├── pairs.py
│   ├── graph.py
│   ├── sequence.py
│   ├── divergence.py
│   ├── scoring.py
│   ├── pipeline.py
│   └── dashboard.py
│
├── notebooks/
│   └── 01_run_pipeline.ipynb
│
├── scripts/
│   └── extract_from_notebook.py
│
├── tests/
│   ├── test_imports.py
│   └── test_end_to_end_synthetic.py
│
└── README.md
```


---

## Core Design Principles

- **Modular by design**  
  Each analytic layer is isolated and testable.

- **Stage-based execution**  

- **Deterministic & reproducible**  
  Centralized configuration, fixed random seeds, and explicit baselines.

- **Explainable outputs**  
  Every score maps back to interpretable evidence (event pairs, transitions).

- **Research-friendly**  
  Supports ablation, parameter sweeps, and comparative evaluation.

- **Operationally realistic**  
  Built around Sysmon semantics and host-centric triage.

---

## Data Expectations

The pipeline assumes a Sysmon-derived table with (at minimum):

| Column | Description |
|------|-------------|
| `_timestamp` | Event timestamp (coerced to UTC) |
| `host.fqdn` | Host identifier |
| `winlog.event_id` | Sysmon event ID |

Additional columns are preserved and may be used for enrichment.

Schema validation occurs early and will fail fast if required columns are missing.

---

## Quick Start

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn networkx
## Quick Start
```
### 2. Load Data

```python
import pandas as pd

df = pd.read_csv("sysmon.csv")  # or pd.read_excel(...) / pd.read_parquet(...)
```

---

### 3. Configure and Run

```python
from sysmon_pipeline import SysmonConfig, SysmonPipeline

cfg = SysmonConfig(
    window_seconds=60,
    iforest_contamination=0.02,
)

pipe = SysmonPipeline(cfg)

pipe.fit(df)            # learn baseline behavior
artifacts = pipe.score(df)
```

---

### 4. Inspect Results

```python
artifacts.triage.head(20)
```

---

## Pipeline Outputs (`SysmonArtifacts`)

The `score()` call returns a structured bundle of artifacts:

- **`events`** – cleaned and enriched Sysmon events  
- **`pairs`** – correlated critical event pairs by host  
- **`pair_stats`** – per-host behavioral statistics with Isolation Forest scores  
- **`state_map`** – event → Markov state mapping  
- **`host_markov`** – per-host Markov transition matrices  
- **`baseline_markov`** – learned baseline transition matrix  
- **`markov_scores`** – KL / Jensen–Shannon divergence per host  
- **`triage`** – final ranked host table  

This makes it easy to:

- build dashboards  
- export reports  
- perform retrospective analysis  

---

## Baseline Learning

By default, the pipeline:

- selects the most common hosts as a baseline, **or**
- accepts an explicit baseline allowlist via configuration  

This enables **role-aware or environment-specific baselines**, rather than relying on global assumptions.

---

## Stage Based Execution

You can run only selected analytic layers:

artifacts = pipe.run(
    df,
    stages=("enrich", "pairs", "pair_stats", "markov", "triage")
)

Valid stages:
- `enrich`
- `pairs`
- `pairstats`
- `markov`
- `triage`

This supports Hunter Mode (fast traige) and Research Mode (deep inspection)

---

## Refactoring Legacy Notebooks

If you started with a large exploratory notebook:

```bash
python scripts/extract_from_notebook.py Sysmon_v12_enhanced.ipynb
```

This script:

- detects duplicate function definitions  
- reports conflicts  
- dumps all functions into a single file for controlled migration  

This prevents silent overwrites and version drift during refactoring.

---

## Intended Use Cases

- Threat hunting and proactive detection  
- Behavioral malware analysis  
- Host-centric triage pipelines  
- Research on sequential drift and behavioral baselining  

---

## What This Is Not

- A signature-based IDS  
- A supervised classifier trained on labeled malware  
- A real-time streaming system (yet)  

This pipeline is optimized for **analytic insight and investigation**, not alert spam.

---

## Extensibility Ideas

- Replace Markov chains with HMMs or neural sequence models  
- Add role-based or peer-group baselines  
- Integrate MITRE ATT&CK scoring directly into triage  
- Export evidence bundles as JSON for case management systems  

---

## License / Disclaimer

This code is provided for **research and analytic purposes**.  
Validate results before operational deployment.
