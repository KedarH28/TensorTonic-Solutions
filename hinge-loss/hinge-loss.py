import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    y_true = np.array(y_true, dtype=float)
    y_score = np.array(y_score, dtype=float)
    
    losses = np.maximum(0.0, margin - y_true * y_score)
    
    if reduction == "mean":
        return float(np.mean(losses))
    elif reduction == "sum":
        return float(np.sum(losses))
    else:
        return losses