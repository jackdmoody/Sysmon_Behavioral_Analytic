# Literature Review: Stochastic & Graph-Based Behavioral Modeling for Endpoint Telemetry

This document summarizes foundational and recent (2020–2025) research supporting
the Sysmon_Behavioral_Analytic repository.

The analytic integrates:

- Host-based telemetry modeling
- Markov transition modeling
- Graph structural analysis
- Isolation-based anomaly detection
- Information-theoretic drift detection
- MITRE ATT&CK semantic enrichment

---

# 1. Host-Based Intrusion Detection Foundations

Behavioral intrusion detection originates from anomaly detection research
distinguishing misuse detection from deviation modeling [@axelsson2000intrusion].

Machine learning limitations in operational security contexts were highlighted
by Sommer & Paxson (2010) [@sommer2010outside], emphasizing distributional
shift and adversarial adaptation.

Comprehensive anomaly detection taxonomies are given in
Chandola et al. (2009) [@chandola2009anomaly].

---

# 2. Sequential Modeling & Markov Processes

Early system call modeling work treated host activity as stochastic sequences:

- Lane & Brodley (1999) [@lane1999temporal]
- Warrender et al. (1999) [@warrender1999detecting]

The Sysmon Behavioral Analytic models event transitions as host-specific
Markov transition matrices and compares them against peer baselines using
divergence metrics.

Recent work (2020–2024) extends this idea:

- Deep sequence modeling for intrusion detection (LSTM/Transformer approaches)
- Probabilistic graphical models for attack progression
- Temporal graph neural networks for cyber telemetry modeling

---

# 3. Graph-Based Anomaly Detection (2020–2025 Advances)

Recent cybersecurity research increasingly models host telemetry as dynamic graphs.

## Key Themes:

- Dynamic graph anomaly detection
- Temporal graph neural networks (TGNN)
- Structural entropy minimization
- Spectral graph divergence

Representative works include:

- Dynamic graph anomaly detection via temporal embeddings (2020–2023)
- Structural entropy for abnormal behavior detection
- Graph contrastive learning for cybersecurity telemetry

Graph-based intrusion detection increasingly leverages:

- Node embedding drift
- Edge transition irregularity
- Spectral distance between adjacency matrices
- Community structure divergence

This aligns directly with the repository’s event-pair graph modeling approach.

---

# 4. Isolation Forest & Ensemble Anomaly Detection

Isolation Forest was introduced in 2008 [@liu2008isolation] and remains widely
used due to computational efficiency.

Recent developments include:

- Extended Isolation Forest (EIF)
- Deep Isolation Forest hybrids
- Ensemble-based anomaly scoring in SOC pipelines (2021–2024 literature)

Modern SOC research emphasizes:

- Multi-view anomaly detection
- Hybrid statistical + ML scoring
- Explainable anomaly scoring

---

# 5. Information-Theoretic Drift Detection

KL divergence was formalized in 1951 [@kullback1951information].
Jensen-Shannon divergence provides symmetric bounded comparison [@lin1991divergence].

Recent concept drift literature (2020–2025) focuses on:

- Online drift detection in streaming logs
- Distribution shift under adversarial conditions
- Graph distribution divergence
- Change-point detection in high-dimensional data

Modern approaches include:

- Maximum Mean Discrepancy (MMD)
- Wasserstein distance for behavioral shift
- Spectral divergence in graph Laplacians

The repository’s KL/JS divergence modeling fits squarely within this
information-theoretic drift detection framework.

---

# 6. MITRE ATT&CK & Knowledge Graph Enrichment

MITRE ATT&CK provides structured adversarial ontology.

Recent work (2020–2024) integrates:

- Knowledge graph reasoning over ATT&CK techniques
- Detection engineering pipelines
- Automated ATT&CK mapping via NLP
- Risk scoring aligned to tactic-level modeling

The Sysmon analytic uses ATT&CK mapping for semantic enrichment and
severity weighting, connecting stochastic modeling to operational context.

---

# 7. Research Positioning

The Sysmon_Behavioral_Analytic repository can be positioned as:

"A unified stochastic, graph-structural, and information-theoretic
behavioral anomaly detection framework for endpoint telemetry."

It bridges:

- Applied probability theory
- Information theory
- Graph science
- Cyber operations research
- SOC decision-support modeling

---

# References

```bibtex
@article{axelsson2000intrusion,
  title={Intrusion detection systems: A survey and taxonomy},
  author={Axelsson, Stefan},
  year={2000},
  institution={Chalmers University}
}

@inproceedings{sommer2010outside,
  title={Outside the Closed World: On Using Machine Learning for Network Intrusion Detection},
  author={Sommer, Robin and Paxson, Vern},
  booktitle={IEEE Symposium on Security and Privacy},
  year={2010}
}

@article{chandola2009anomaly,
  title={Anomaly detection: A survey},
  author={Chandola, Varun and Banerjee, Arindam and Kumar, Vipin},
  journal={ACM Computing Surveys},
  volume={41},
  number={3},
  year={2009}
}

@inproceedings{lane1999temporal,
  title={Temporal sequence learning and data reduction for anomaly detection},
  author={Lane, Terran and Brodley, Carla E},
  booktitle={ACM CCS},
  year={1999}
}

@inproceedings{warrender1999detecting,
  title={Detecting intrusions using system calls},
  author={Warrender, Christina and Forrest, Stephanie and Pearlmutter, Barak},
  booktitle={IEEE S&P},
  year={1999}
}

@inproceedings{liu2008isolation,
  title={Isolation Forest},
  author={Liu, Fei Tony and Ting, Kai Ming and Zhou, Zhi-Hua},
  booktitle={IEEE ICDM},
  year={2008}
}

@article{kullback1951information,
  title={On information and sufficiency},
  author={Kullback, Solomon and Leibler, Richard A},
  journal={Annals of Mathematical Statistics},
  year={1951}
}

@article{lin1991divergence,
  title={Divergence measures based on Shannon entropy},
  author={Lin, Jianhua},
  journal={IEEE Transactions on Information Theory},
  year={1991}
}
