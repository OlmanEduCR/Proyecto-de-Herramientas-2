
#6) Vector Norms

import numpy as np


def compute_norms(x):
    resultado={}
    
    # L1
    # se debe aplicar valor absoluto a cada elemento de la matriz y luego sumar por fila, obteniendo una lista con la suma de cada fila
    suma_l1= np.sum(np.abs(x),axis=1) # se obtiene la lista con las normas 
    resultado["L1"] = suma_l1 # se agrega la lista de las normas al diccionario
    
    # L2
    # se debe elevar cada elemento al cuadrado, luego sumar cada fila, por último se aplica raíz a cada elemento de la lista de las sumas
    suma_l2= np.sqrt(np.sum(x**2,axis=1)) # se obtiene la lista con las normas 
    resultado["L2"]= suma_l2 # se agrega la lista de las normas al diccionario
    
    return resultado

# Ejemplos de prueba

x = np.array([[3, 4],
              [1, -1]])
print(compute_norms(x))

