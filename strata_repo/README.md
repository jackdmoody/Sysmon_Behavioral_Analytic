# STRATA-E

**Structural and Temporal Role-Aware Threat Analytics for Endpoint Telemetry**

A multi-channel behavioral anomaly detection pipeline for host-based threat hunting.
Designed for Sysmon/WEC telemetry and validated on DARPA Transparent Computing datasets.

---

## Architecture

```
Raw Sysmon/WEC CSV
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Schema Normalization  →  Token Mapping  →  Role Inference  │
└─────────────────────────────────────────────────────────────┘
       │                           │
       ▼                           ▼
┌────────────┐            ┌─────────────────┐
│  Sequence  │            │  Peer Baselines │  (Dirichlet shrinkage)
│  Channel   │◄───────────│  per role       │
│  (JSD)     │            └─────────────────┘
└────────────┘
       │
       ├── Frequency Channel  (IsolationForest on rate features)
       │
       ├── Context Channel    (severity + flags + semantic event pairs + TF-IDF)
       │
       └── Drift Channel      (JSD between current and prior window)
                │
                ▼
         ┌──────────────┐
         │ Borda Fusion │  + Corroboration Gate
         └──────────────┘
                │
                ▼
         Ranked Triage Table
         (host, score, top_tactic, MITRE tactic breadth, ...)
```

### Key Components

| Module | Description |
|---|---|
| `config.py` | Dataclass config with sub-configs for IO, time bucketing, baselines, roles, scoring, ablation |
| `schema.py` | Flexible column detection + canonical schema normalization |
| `mapping.py` | Multi-resolution token mapping, LOLBin/script detection, severity grading |
| `sequence.py` | Adaptive sessionization (KDE τgap), transition counting |
| `divergence.py` | KL/JS divergence, Dirichlet shrinkage peer baselines, JSD calibration, drift scoring |
| `pairs.py` | Rate features, role inference (KMeans), semantic event pair correlation (MITRE ATT&CK mapped) |
| `scoring.py` | Frequency channel, context channel, Borda fusion, corroboration gate |
| `graph.py` | Transition graph construction, centrality metrics for explainability |
| `visuals.py` | Matplotlib static plots + Plotly interactive (timeline, Sankey, heatmap) |
| `loaders.py` | DARPA TC JSON loader, generic Sysmon CSV loader, time-window splitter |
| `pipeline.py` | `StrataPipeline.fit()` / `.score()` orchestrator + CLI entry point |

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/strata-e.git
cd strata-e

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install
pip install -e .

# With optional HMM support
pip install -e ".[hmm]"

# With dev/test dependencies
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10

### 2. Run on synthetic data (no dataset needed)

```bash
# Quick smoke test — verifies the pipeline runs end-to-end
python run_experiments.py --synthetic --n-hosts 30 --n-attack 6 --output results/

# Run a single hypothesis
python run_experiments.py --synthetic --hypothesis H1 --output results/
```

Results are written to `results/`:
- `h1_variance_reduction.csv` — Shrinkage vs MLE JSD variance
- `h2_role_vs_global_recall.csv` — Role baselining Top-K recall
- `h3_pvalue_uniformity.csv` — KS test calibration
- `h4_channel_comparison.csv` — Per-channel vs fused recall
- `h5_gating_fpr_recall.csv` — FPR / recall with/without corroboration gate
- `results_summary.csv` — One-line-per-hypothesis paper table
- `*.png` — Figures

### 3. Run on your own Sysmon CSV

```bash
# Score a CSV export from your SIEM
strata --input data/sysmon_export.csv --output results/
```

Or from Python:

```python
from sysmon_pipeline import StrataPipeline, StrataConfig
from sysmon_pipeline.loaders import load_sysmon_csv, split_time_windows

df = load_sysmon_csv("data/sysmon_export.csv")
baseline_df, score_df = split_time_windows(df, baseline_days=7, score_days=1)

pipe = StrataPipeline(StrataConfig())
fitted = pipe.fit(baseline_df)
art    = pipe.score(score_df, fitted)

print(art.triage.head(20))
```

### 4. Run on DARPA Transparent Computing

```bash
# Download DARPA TC E3 data (requires Google Drive access)
# See: https://github.com/darpa-i2o/Transparent-Computing
# Extract CADETS to data/darpa/cadets/

python run_experiments.py \
    --dataset darpa \
    --data-dir data/darpa/cadets/ \
    --darpa-name cadets \
    --output results/
```

---

## CLI Reference

```bash
# Full pipeline on a CSV
strata --input data/sysmon.csv --output results/

# With a custom config JSON
strata --input data/sysmon.csv --config config.json

# Ablation conditions (for paper reproducibility)
strata --input data/sysmon.csv --ablation sequence_only
strata --input data/sysmon.csv --ablation no_shrinkage
strata --input data/sysmon.csv --ablation no_role_baselining
strata --input data/sysmon.csv --ablation no_drift

# Experiment runner
python run_experiments.py --help
```

---

## Configuration

Generate a default config JSON to customize:

```python
from sysmon_pipeline import StrataConfig
StrataConfig().to_json("config.json")
```

Key parameters to tune for your environment:

| Parameter | Default | Notes |
|---|---|---|
| `scoring.fusion_method` | `"borda"` | `"weighted_linear"` matches paper Eq. 15 |
| `scoring.iforest_contamination` | `0.02` | Expected attack prevalence in your fleet |
| `baseline.dirichlet_kappa` | `10.0` | Higher = more shrinkage toward role prior |
| `scoring.window_seconds` | `60` | Pair correlation window |
| `scoring.require_corroboration` | `True` | Multi-channel gate; set False to see all |
| `ablation.*` | all True | Toggle pipeline components for ablation |

---

## Ablation Study

All five ablation conditions from the paper run via config flags:

```python
from sysmon_pipeline import StrataConfig, AblationConfig, StrataPipeline

conditions = {
    "full":              AblationConfig.full_pipeline(),
    "sequence_only":     AblationConfig.sequence_only(),
    "no_shrinkage":      AblationConfig.no_shrinkage(),
    "no_role_baselining": AblationConfig.no_role_baselining(),
    "no_drift":          AblationConfig.no_drift(),
}

for name, ablation in conditions.items():
    cfg = StrataConfig(ablation=ablation)
    art = StrataPipeline(cfg).fit_score(df)
    # evaluate(art.triage, labels, condition=name)
```

---

## Tests

```bash
pytest tests/ -v
```

Tests cover: schema normalization, full pipeline smoke test, anomaly host ranking,
all ablation conditions, both fusion methods, drift channel, config serialization,
shrinkage evasion detection.

---

## Semantic Event Pairs (MITRE ATT&CK)

The context channel uses a curated list of 40 adversarially meaningful event-pair
transitions rather than generic co-occurrence. Each pair is mapped to a MITRE tactic
and assigned a confidence weight (0–1). High-weight examples:

| Pair | Weight | Tactic | Interpretation |
|---|---|---|---|
| 8→10 | 1.00 | credential_access | CreateRemoteThread → LSASS (Mimikatz) |
| 11→10 | 0.95 | credential_access | File drop → LSASS (dumper written to disk) |
| 7→10 | 0.90 | credential_access | DLL load → LSASS (reflective injection) |
| 4768→4769 | 0.85 | credential_access | TGT → Service Ticket (Kerberoasting) |
| 4624→7045 | 0.80 | lateral_movement | Logon → Service installed |
| 22→1 | 0.70 | execution | DNS → Process (download-and-execute) |

Triage output includes `top_tactic` and `n_tactics` columns for analyst explainability.

---

## Output: Triage Table

Key columns in `art.triage`:

| Column | Description |
|---|---|
| `host` | Hostname |
| `score` | Fused anomaly score (percentile rank, higher = more suspicious) |
| `triage_rank` | Rank (1 = most suspicious) |
| `gate_pass` | Passed corroboration gate (multi-channel confirmation) |
| `gate_reason` | `extreme_channel` / `multi_channel` / `low_support` |
| `S_seq` | Sequence channel score (JSD from peer baseline) |
| `S_freq` | Frequency channel score (IsolationForest) |
| `S_ctx` | Context channel score (severity + flags + pairs) |
| `S_drift` | Drift channel score (behavioral shift from prior window) |
| `top_tactic` | Dominant MITRE ATT&CK tactic |
| `n_tactics` | Number of distinct tactics observed |
| `max_pair_weight` | Confidence of highest-weight attack pair (1.0 = near-certain) |
| `weighted_score_sum` | Total weighted pair evidence |
| `evasion_signal` | Shrinkage anomaly flag (log suppression indicator) |

---

## Project Structure

```
strata-e/
├── sysmon_pipeline/
│   ├── __init__.py          # Public API
│   ├── config.py            # StrataConfig, AblationConfig
│   ├── schema.py            # Schema normalization
│   ├── mapping.py           # Token mapping, severity grading
│   ├── sequence.py          # Sessionization, transition counting
│   ├── divergence.py        # JSD, Dirichlet baselines, drift
│   ├── pairs.py             # Rate features, roles, semantic pairs
│   ├── scoring.py           # Channels, fusion, gate
│   ├── graph.py             # Transition graph metrics
│   ├── visuals.py           # Plots (matplotlib + plotly)
│   ├── loaders.py           # DARPA TC + Sysmon CSV loaders
│   └── pipeline.py          # StrataPipeline orchestrator + CLI
├── tests/
│   └── test_pipeline.py
├── run_experiments.py        # H1–H5 hypothesis tests
├── data/
│   └── README.txt           # Data placement instructions
├── results/                  # Generated outputs (gitignored)
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{strata-e-2026,
  title     = {{STRATA-E}: Structural and Temporal Role-Aware Threat Analytics
               for Endpoint Telemetry},
  author    = {Moody, Jack},
  booktitle = {Proc. IEEE ...},
  year      = {2026},
}
```

---

## License

MIT
