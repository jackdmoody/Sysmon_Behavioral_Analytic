# Contributions

This repository introduces a unified stochastic--structural framework
for host-level behavioral anomaly detection in endpoint telemetry.

While prior research has explored sequence modeling, graph-based
intrusion detection, anomaly detection algorithms, and concept drift
independently, there remains a lack of integrated systems that combine
these components into a cohesive, interpretable analytic pipeline
aligned with operational security workflows.

The primary contributions of this work are summarized below.

------------------------------------------------------------------------

## 1. Stochastic Transition Graph Representation of Host Behavior

This pipeline models endpoint telemetry using a dual representation:

-   A directed event-pair graph `G_h`
-   A host-specific Markov transition matrix `P_h`

This formulation treats host activity as a stochastic graph process,
enabling simultaneous analysis of:

-   Structural connectivity (graph topology)
-   Transition probabilities (temporal ordering)
-   Higher-order behavioral dynamics

Unlike prior sequence-only or graph-only approaches, this representation
unifies temporal and structural perspectives within a single analytic
framework.

------------------------------------------------------------------------

## 2. Information-Theoretic Behavioral Drift Measurement

This work applies information-theoretic divergence to quantify
behavioral deviation relative to peer baselines.

For each host:

-   `P_h` = host transition distribution
-   `P_b` = peer-group baseline transition distribution

Behavioral drift is measured using Jensen--Shannon divergence:

    D_JS(P_h || P_b)

This provides:

-   A principled distribution-level deviation metric
-   Symmetric and bounded divergence behavior
-   Robustness to sparse transition structures

Few operational endpoint analytics apply divergence measures directly to
transition matrices, making this integration a key structural
contribution.

------------------------------------------------------------------------

## 3. Multi-View Ensemble Evidence Fusion

The analytic fuses multiple independent behavioral signals:

-   Graph structural metrics (e.g., centrality measures)
-   Transition divergence scores
-   Isolation Forest anomaly scores
-   ATT&CK-based semantic severity weighting

The resulting host anomaly score can be conceptualized as:

    A_h = f(D_JS, C(G_h), IF_h, W_ATTACK)

This multi-view integration moves beyond single-model anomaly detection
and provides a structured decision-support--oriented ranking mechanism
for SOC triage workflows.

------------------------------------------------------------------------

## 4. Operational Semantic Integration via MITRE ATT&CK

Sysmon events are enriched with MITRE ATT&CK technique mappings.

This enables:

-   Technique-level weighting of anomalous behaviors
-   Tactic-aware anomaly prioritization
-   Alignment with detection engineering practices

Unlike purely statistical approaches, this framework grounds anomaly
detection within a structured adversarial ontology.

------------------------------------------------------------------------

## 5. Interpretability-Centric Behavioral Modeling

In contrast to deep graph neural network or opaque embedding-based
systems, this framework emphasizes interpretability through:

-   Explicit transition matrices
-   Transparent divergence metrics
-   Explainable structural graph features
-   Auditable ensemble scoring

This design supports high-stakes cybersecurity environments where
analyst trust and traceability are essential.

------------------------------------------------------------------------

## 6. Bridging Theory and Operational Security

This repository bridges multiple theoretical domains:

-   Markov chain modeling
-   Graph theory
-   Information theory
-   Unsupervised anomaly detection
-   Knowledge-based risk modeling

These components are integrated into a single modular pipeline for
endpoint telemetry analytics.

------------------------------------------------------------------------

# Summary of Novelty

The novelty of this framework does not lie in introducing a new
standalone algorithm. Rather, it lies in:

> The structured integration of stochastic transition modeling, graph
> topology analysis, information-theoretic drift detection, ensemble
> anomaly scoring, and semantic adversary mapping into a unified
> host-level behavioral analytic framework.

This integration addresses a methodological gap between theoretical
anomaly detection research and practical SOC decision-support systems.
