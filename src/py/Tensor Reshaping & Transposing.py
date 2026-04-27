"""
Ejercicio #4
Tensor Reshaping & Transposing
"""
import numpy as np
def reshape_and_transpose(x: np.ndarray, B: int, C: int, H: int, W: int):
    if len(x) == B * C * H * W: ## verifica si 
        x = np.reshape(x, (B, C, H, W))
        x = np.transpose(x, (0, 2, 3, 1))
    return x

print(reshape_and_transpose(
    x= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], 
    B=1, 
    C=2, 
    H=3, 
    W=4))
