
# STRATA-(E)
## Structural and Temporal Role-Aware Threat Analytics for Endpoint Telemetry

STRATA-(E) is a hierarchical, statistically calibrated behavioral anomaly detection architecture for Windows Sysmon and Windows Event telemetry.

It is designed to answer the following operational question:

> **Which hosts are behaving abnormally, why, and based on what evidence?**

Rather than producing binary alerts, STRATA generates ranked, explainable host triage outputs using time-aware sequence modeling, hierarchical Bayesian role baselines, calibrated divergence scoring, and multi-channel anomaly fusion.

---

# Core Design Motivation

Enterprise endpoint anomaly detection faces four recurring challenges:

1. Temporal abstraction ambiguity  
2. Transition sparsity vs. semantic fidelity  
3. Baseline contamination in compromised-by-default environments  
4. Structural vs. volumetric anomaly mismatch  

STRATA-(E) addresses these challenges using principled probabilistic modeling and statistical calibration to produce stable, interpretable anomaly scores.

---

# Architectural Overview

STRATA operates as a staged behavioral modeling pipeline:

1. Ingest and Preprocessing 
2. Feature Construction
3. Bayesian Peer-Role Baselines 
4. Calibrated Multi-Channel Anomaly Scoring  
5. Evidence Fusion & Triage Output  
6. Statistical Validation and Evaluation Framework  

<img width="1102" height="1118" alt="STRATA_E_V2" src="https://github.com/user-attachments/assets/ee7cdd70-955d-41d6-8736-4118f80bba8c" />

---
# Ingest and Preprocessing

## Canonical Schema & Normalization

Raw Sysmon / Windows Event telemetry is coerced into a deterministic schema including:

- UTC timestamp  
- Host identifier  
- User identifier  
- Event ID  
- Process image & parent image  
- Command line  
- Network attributes (IP, port, protocol)  
- Integrity level  
- Signature metadata  

## Multi-Resolution Token Abstraction

Each event is mapped into three token resolutions:

### Coarse Tokens (Low Sparsity)
Example:
- OFFICE  
- SCRIPT  
- BROWSER  
- LOLBIN  

### Medium Tokens (Balanced Resolution)
Example:
- SCRIPT:POWERSHELL  
- OFFICE:WORD  

Used for transition modeling.

### Fine Tokens (Context-Rich)
Augments medium tokens with:
- Parent process  
- Encoded command flags  
- Integrity level  
- Signature state  

Fine tokens are excluded from the transition state space to prevent sparsity-driven false positives.

## Sessionization & Time-Aware Transition Modeling

Host event streams are segmented using an inactivity threshold (τgap).

Inter-event times (Δt) are discretized into buckets:

    β = bucket(Δt)

Transitions are modeled as:

    (z → z′, β)

The system estimates:

    P(z′, β | z)

This distinguishes rapid kill-chain behavior from delayed execution patterns.

---

# Bayesian Peer-Role Baselines

Hosts are assigned to peer roles (e.g., workstation, server, domain controller).

Rather than using naïve empirical estimates, STRATA models role and host transition distributions using a hierarchical Dirichlet framework:

- Role-level distribution θ_r  
- Host-level distribution θ_h  
- Shrinkage toward role baseline via concentration parameter κ  

Posterior host transition estimates use Dirichlet shrinkage:

    θ̂_h = (N_h + κ θ̂_r) / (n_h + κ)

This stabilizes divergence estimates in sparse windows while preserving meaningful host deviations.

Baselines include:
- Smoothed role-conditioned transition distributions  
- Role-specific volumetric feature norms  

---

# Calibrated Multi-Channel Anomaly Scoring

STRATA separates anomaly detection into four independent channels.

## Sequence Channel (Structural Anomalies – Calibrated)

Structural deviation is measured using Jensen–Shannon divergence:

    Sseq(h) = JS(θ̂_h || θ̂_r(h))

### Statistical Calibration

Raw divergence depends on window size.  
STRATA estimates a role-conditional null distribution via bootstrap:

    N ~ Multinomial(n_h, θ̂_r)

This produces:

- p-value  
- z-score  
- role-percentile  

Example interpretation:

"Host is a 99.9th percentile structural deviation for workstations."

Calibration significantly reduces false positives across heterogeneous window sizes.

---

## Frequency Channel (Volumetric Anomalies)

    Sfreq(h) = IsolationForest(xh)

Captures unusual activity volume and rate deviations independent of sequence structure.

---

## Context Channel (Fine-Grained Signals)

Aggregates high-signal indicators such as:

- Encoded commands  
- Suspicious parent-child relations  
- Integrity mismatches  
- Unsigned execution  
- Known LOLBin usage  

Produces:

    Sctx(h) = weighted flag aggregation

---

## Drift Channel (Behavioral Change Over Time)

    Sdrift(h) = JS(θ̂_h^cur || θ̂_h^hist)

Captures sustained behavioral shifts across rolling windows.  
Transition distributions are Dirichlet-smoothed for stability.

---

# Evidence Fusion & Gating

Channel scores are fused:

    Stotal(h) = w1 Sseq + w2 Sfreq + w3 Sctx + w4 Sdrift

Optional gating logic prioritizes corroborated anomalies across channels.

Outputs include:

- Ranked host triage tables  
- Top anomalous transitions  
- Divergence contributors  
- Outlier feature dimensions  
- Context flag summaries  
- Drift comparison artifacts  
- Calibrated percentile and significance indicators  

---

# Statistical Validation and Evaluation Framework

STRATA includes a structured evaluation framework to ensure robustness and publishability.

### Synthetic Attack Injection
Realistic attack chains (encoded PowerShell, LOLBin abuse, lateral movement, persistence) are injected into benign telemetry to measure sensitivity.

### Ablation Analysis
Incremental performance is evaluated with and without:
- Role baselining  
- Dirichlet shrinkage  
- JSD calibration  
- Context channel  
- Drift modeling  

### Calibration Testing
Under benign subsets, calibrated p-values are tested for approximate uniformity to verify false positive control.

### Operational Metrics
- Top-K recall  
- Precision–Recall AUC  
- False positive rate per host-day  
- Stability of ranked triage outputs  

---

# Analytic Philosophy

STRATA is intentionally:

Host-Centric – Behavior modeled per endpoint  
Time-Aware – Transitions incorporate discretized inter-event timing  
Role-Aware – Abnormality defined relative to peer-role baselines  
Statistically Calibrated – Divergence scores are significance-adjusted relative to role-conditioned null distributions  
Channel-Separated – Structural, volumetric, contextual, and drift anomalies detected independently  
Explainable – Every anomaly score is traceable to transitions, features, and calibrated percentiles  

---

# Intended Use Cases

- Threat hunting  
- SOC triage support  
- Research experimentation  
- Behavioral baseline evaluation  
- Academic study of endpoint telemetry modeling  

---

# Limitations

- Batch-oriented (not streaming-native)  
- Assumes reliable host role assignment  
- Dependent on Sysmon configuration quality  
- Unsupervised (no attribution modeling)  
- Calibration assumes multinomial null approximation  

---
