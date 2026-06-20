import numpy as np
def categorical_ce_backward(p, t):
    p = np.array(p)
    t = np.array(t)

    N = p.shape[0]
    dlogits = p.copy()
    dlogits[np.arange(N), t] -= 1.0
    dlogits /= N
    return dlogits

probs = [[0.1, 0.9, 0.0],
         [0.8, 0.1, 0.1]]
targets = [1,0]

dx = categorical_ce_backward(probs,  targets)
print(dx)
