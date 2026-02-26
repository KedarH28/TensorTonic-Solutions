import numpy as np

def dice_loss(p, y, eps=1e-8):
    p = np.array(p, dtype=float).flatten()
    y = np.array(y, dtype=float).flatten()
    
    intersection = np.sum(p * y)
    sum_p = np.sum(p)
    sum_y = np.sum(y)
    
    dice = (2 * intersection + eps) / (sum_p + sum_y + eps)
    
    return 1 - dice