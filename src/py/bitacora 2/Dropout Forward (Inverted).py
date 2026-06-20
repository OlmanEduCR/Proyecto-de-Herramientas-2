import numpy as np

def dropout_forward(x, p, train=True, seed=None):

    if seed is not None:
        np.random.seed(seed)

    if not train:
        return x, None

    if p == 1:
        mask = np.zeros_like(x)
        out = np.zeros_like(x)
        return out, mask

    mask = np.random.binomial(1, 1 - p, size=x.shape)

    out = x * mask / (1 - p)

    return out, mask

#Ejemplo
x = np.array([[1.0, 2.0, 3.0, 4.0],
     [5.0, 6.0, 7.0, 8.0]])  
p = 0.5  
seed = 42
train = True

print(dropout_forward(x, p,seed=seed))