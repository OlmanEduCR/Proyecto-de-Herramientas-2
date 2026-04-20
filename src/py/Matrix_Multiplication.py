
# 2) Matrix Multiplication

import numpy as np

# 1.

def matmul_naive(A,B):
    C=[]
    
    for i in range(A.shape[0]):     # se usará para moverse de fila de A
        lista=[]    # lista para agragar las filas de la matriz final
        
        for j in range(B.shape[0]):     # se usará para moverse de columna de B
            num=0   # representa el resultado del producto punto de cada columna de B
            
            for c in range(A.shape[1]):     # se usará para recorrer los elemntos de la fila de A y columna de B
                num+=A[i][c]*B[c][j]    # producto punto
            lista.append(num)   # se agrega el producto punto a una lista que será una nueva fila para C
            
        C.append(lista)     # se agrega nueva fila a C
        
    return C

# Ejemplo de prueba

A = np.array([[1, 2],
     [3, 4]])
B = np.array([[5, 6],
     [7, 8]])

print(matmul_naive(A, B))


# 2.

def matmul_vectorized(A,B):
    C = A @ B # multiplicación de matrices de numpy
    return C

# Ejemplo de pruba

print(matmul_vectorized(A, B))