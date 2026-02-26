import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    Z1 = np.array(Z1, dtype=float)
    Z2 = np.array(Z2, dtype=float)
    
    # Similarity matrix
    S = np.dot(Z1, Z2.T) / temperature
    
    # Numerical stability
    S_stable = S - np.max(S, axis=1, keepdims=True)
    
    exp_S = np.exp(S_stable)
    probs = exp_S / np.sum(exp_S, axis=1, keepdims=True)
    
    # Positive pairs are diagonal
    positive_probs = np.diag(probs)
    
    loss = -np.mean(np.log(positive_probs))
    
    return float(loss)