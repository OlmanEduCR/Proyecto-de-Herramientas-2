import numpy as np

def rmsprop_step(w, dw, lr, decay, eps, cache=None):
    # Inicializar cache si es la primera iteración
    if cache is None:
        cache = np.zeros_like(w)

    # Actualizar promedio móvil de gradientes cuadrados
    cache = decay * cache + (1 - decay) * (dw ** 2)

    # Actualizar pesos
    w = w - lr * dw / (np.sqrt(cache) + eps)

    return w, cache

#Ejemplo
w = np.array([[1.0, 2.0],
     [3.0, 4.0]])
dw = np.array([[0.5, -0.3],    # Current gradient
     [-0.2, 0.1]])
cache = np.array([[0.1, 0.05],   # Previous squared gradient average
         [0.02, 0.01]])
lr = 0.01
decay = 0.9
eps = 1e-8

print(rmsprop_step(w, dw, lr, decay, eps,cache))
