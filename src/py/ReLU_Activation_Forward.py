""" 
Ejercicio #3
Bitacora #2
ReLU Activation Forward
"""

import numpy as np

def relu_forward(x):
    x = np.array(x)
    x = np.maximum(0, x)
    return x

print(relu_forward([-2.0, -1.0, 0.0, 1.0, 2.0, 5.0]))
print(relu_forward([[-1, 2, -3], [4, -5, 6]]))
