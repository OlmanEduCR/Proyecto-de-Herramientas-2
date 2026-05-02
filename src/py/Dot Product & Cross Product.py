import numpy as np

def vector_products(a, b):
    # Producto punto
    dot = np.sum(a * b, axis=1)
    
    # Producto cruz
    cross = np.stack([
        a[:,1]*b[:,2] - a[:,2]*b[:,1],
        a[:,2]*b[:,0] - a[:,0]*b[:,2],
        a[:,0]*b[:,1] - a[:,1]*b[:,0]
    ], axis=1)
    
    return {
        "dot": dot,
        "cross": cross
    }

#Ejemplo
a = np.array([[1, 0, 0]])
b = np.array([[0, 1, 0]])
vector_products(a, b)
