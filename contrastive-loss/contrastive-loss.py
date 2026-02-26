import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    a = np.array(a)
    b = np.array(b)
    y = np.array(y)
    
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    if y.ndim == 0:
        y = y.reshape(1)
    
    d = np.linalg.norm(a - b, axis=1)
    
    loss = y * (d ** 2) + (1 - y) * np.maximum(0, margin - d) ** 2
    
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        return loss