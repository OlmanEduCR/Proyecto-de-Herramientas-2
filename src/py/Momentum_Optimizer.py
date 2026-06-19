""" 
Ejercicio #11
Bitacora #2
Momentum Optimizer
"""

import numpy as np

def momentum_step(w, dw, v, lr, momentum):
    w = np.array(w)
    dw = np.array(dw)
    
    if v is None:
        v = np.zeros(np.shape(w))
    else:
        v = np.array(v)
    
    
    ## Actualización de la Velocidad
    v_nueva = momentum * v + dw
    
    ## Actualización de los pesos
    w_nueva = w - lr * v_nueva
    
    return v_nueva, w_nueva

print(momentum_step([[1.0, 2.0],[3.0, 4.0]], [[0.5, -0.3],[-0.2, 0.1]], [[0.1, -0.05],[-0.02, 0.01]], 0.01, 0.9))
print(momentum_step([[1.0, 2.0],[3.0, 4.0]], [[0.5, -0.3],[-0.2, 0.1]], None , 0.01, 0.9))
