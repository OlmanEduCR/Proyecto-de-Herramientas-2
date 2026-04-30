
# 14) Cross-Entropy Loss

import numpy as np

# Queremos calcular el cross-entropy loss
# Nos dan una matriz con probabilidades (probs) y un vector que indica cual es la prob correcta para cada fila (target)
# Para calcular el cross-entropy loss necesitamos
# Tomar el negativo del logaritmo de las probablidades correctas y luego promediar
# Asi que, primero hay que obtener las probabilidades correctas
# Luego se aplica logaritmo
# Por ultimo se promedian y se le aplica el negativo


def cross_entropy_loss(probs , targets):
    
    probs=np.array(probs) # convertir a array de numpy 
    targets=np.array(targets)
    N = probs.shape[0] # Número de filas de la matriz probs
    eps = 1e-9 # epsilon pequeño para evitar errores con log(0)
    
    # las probabilidades correctas se encuentran para cada fila i en el la columna target[i]
    # Por lo tanto, se pueden obtener recorriendo cada fila y buscando su valor de target respectivo
    correct_probs = probs[np.arange(N),targets] # arange genera una lista de 0 a N-1 que recorrer las filas y target permite seleccionar la probabilidad correcta de cada fila
    
    loss= -np.mean(np.log(correct_probs + eps)) # calculo final del cross-entropy loss

    return loss

# Ejemplo de prueba 

probs = [[0.1, 0.9],
         [0.8, 0.2]]

targets = [1, 0]

print(cross_entropy_loss(probs, targets))





























