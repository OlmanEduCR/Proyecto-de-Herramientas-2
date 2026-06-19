import numpy as np

def dropout_backward(dout, mask, p, train=True):
    if train:
        dx = (dout * mask) / (1.0 - p)
    else:
        dx = dout
        
    return dx


dout = np.array([[0.1, 0.2, 0.3, 0.4],
                 [0.5, 0.6, 0.7, 0.8]])

mask = np.array([[1, 0, 1, 0],
                 [0, 1, 1, 0]])

p = 0.5


dx_train = dropout_backward(dout, mask, p, train=True)
print("Modo Entrenamiento (dx):\n", dx_train)

dx_eval = dropout_backward(dout, mask=None, p=p, train=False)
print("\nModo Evaluacion (dx):\n", dx_eval)