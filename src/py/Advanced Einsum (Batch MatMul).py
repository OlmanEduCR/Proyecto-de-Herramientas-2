# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:00:29 2026

@author: crish
"""
import numpy as np
def lote_matmul(Q, K):
    return np.einsum("bhid,bhjd->bhij", Q, K)
     
#---- Ejemplo 1 -----
B, H, S, D = 2, 8, 128, 64
query = np.random.randn(B, H, S, D)
key = np.random.randn(B, H, S, D)
scores = lote_matmul(query, key)
print(f"Forma de los scores: {scores.shape}") 
#---- Ejemplo 2 -----
B, H, S, D = 2, 8, 128, 64
A = np.random.rand(B, H, S, S)
V = np.random.randn(B, H, S, D)
context = np.einsum('bhij,bhjd->bhid', A, V)
print(context.shape) # Resultado: (2, 8, 128, 64)
