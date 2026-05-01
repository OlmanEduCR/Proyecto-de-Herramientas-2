"""
Ejercicio #8
Einstein Summation (einsum) Basics
"""
import numpy as np
from typing import Dict, Union

def einsum_ops(A: np.ndarray, B: np.ndarray):
    Anp = np.array(A) ## Convierte el arreglo A en un arreglo de NumPy
    Bnp = np.array(B) ## Convierte el arreglo B en un arreglo de NumPy
    
    Diccionario_operaciones: Dict[str, Union[np.ndarray, float]] = {
        "transpose": np.einsum("ij->ji", Anp),
        "sum" : float(np.einsum("ij->", Anp)),
        "col_sum" : np.einsum("ij->j", Anp),
        "row_sum" : np.einsum("ij->i", Anp),
        "matmul" : np.einsum("ik,kj->ij", Anp, Bnp)
        }

    return Diccionario_operaciones


print(einsum_ops(A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]))
    
    