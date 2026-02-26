import numpy as np

def focal_loss(p, y, gamma=2.0):
    p = np.array(p)
    y = np.array(y)
    
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    
    term1 = (1 - p) ** gamma * y * np.log(p)
    term2 = p ** gamma * (1 - y) * np.log(1 - p)
    
    loss = -(term1 + term2)
    
    return np.mean(loss)