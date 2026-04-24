import numpy as np
def lote_matmul(Q, K):
    return np.einsum("bhid,bhjd->bhij", Q, K)
     
#---- Ejemplo 1 -----
# Parametros
B, H, S, D = 2, 8, 128, 64
#Tensores de consulta
A1 = np.random.randn(B, H, S, D)
V1 = np.random.randn(B, H, S, D)
#Puntuacion de atencion
scores1 = lote_matmul(A1, V1)
#imprimir
print(scores1.shape) 