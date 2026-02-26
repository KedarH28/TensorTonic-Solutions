import numpy as np

def kl_divergence(p, q, eps=1e-12):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    
    q_stable = q + eps
    mask = p > 0
    
    return float(np.sum(p[mask] * np.log(p[mask] / q_stable[mask])))