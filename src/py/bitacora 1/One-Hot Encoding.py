"""
Ejercicio #12
One-Hot Encoding
"""

import numpy as np

def one_hot_encode(indices: np.ndarray, num_classes: int) -> np.ndarray:
    indices_np = np.array(indices) ## Convierte el arreglo A en un arreglo de NumPy
    N = len(indices)
    C = num_classes ## Cambio de variable por facilidad
    Resultado = [] ## Matriz vacía por caso básico
    
    if N <= C:
        Resultado = np.zeros((N, C))
        if (indices_np >= 0).all() >= 0 and (indices_np < C).all():
            Resultado[np.arange(N), indices] = 1
    else:
        print("Verifique los indices sean menores a la cantidad de clases o mayores o iguales a 0")
    return Resultado


print(one_hot_encode(indices = [0, 2, 1], num_classes = 3))

        
        