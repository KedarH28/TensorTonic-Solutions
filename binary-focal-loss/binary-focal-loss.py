import math

def binary_focal_loss(predictions, targets, alpha, gamma):
    losses = []
    
    for p, y in zip(predictions, targets):
        if y == 1:
            p_t = p
        else:
            p_t = 1 - p
        
        loss = -alpha * ((1 - p_t) ** gamma) * math.log(p_t)
        losses.append(loss)
    
    return sum(losses) / len(losses)