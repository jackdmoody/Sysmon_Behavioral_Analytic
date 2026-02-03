from __future__ import annotations
import numpy as np

def kl_divergence_matrix(P: np.ndarray, Q: np.ndarray, eps: float=1e-12) -> float:
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    return float(np.sum(P * np.log(P / Q)))

def js_divergence_matrix(P: np.ndarray, Q: np.ndarray, eps: float=1e-12) -> float:
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    M = 0.5 * (P + Q)
    return 0.5 * kl_divergence_matrix(P, M, eps=eps) + 0.5 * kl_divergence_matrix(Q, M, eps=eps)
