"""
Ejercicio #4
Tensor Reshaping & Transposing
"""
import numpy as np
def reshape_and_transpose(x: np.ndarray, B: int, C: int, H: int, W: int):
    if len(x) == B * C * H * W: ## Verifica si la cantidad de los elementos de x corresponde al producto de B,C,H,W
        x = np.reshape(x, (B, C, H, W)) ## le da una nueva forma con respecto a la tupla (B, C, H, W)
        x = np.transpose(x, (0, 2, 3, 1)) ## transpone x reordenando por los indices
    return x

print(reshape_and_transpose(
    x= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], 
    B=1, 
    C=2, 
    H=3, 
    W=4))
