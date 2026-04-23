import numpy as np

def broadcast_ops(X, b, w): 
    suma = (X + b) 
    multipicador = w.reshape(-1, 1)
    Y = suma * multipicador
    return Y
    


#ejemplo 1
X = np.array([[1,2],
              [3,4]])
b = np.array([10,20])
w = np.array([0.5,2])

aplicar = broadcast_ops(X, b, w)
print(aplicar)
