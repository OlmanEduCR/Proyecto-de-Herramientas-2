import numpy as np

def elementwise_ops(a, b):
    epsilon = 1e-8
    
    add = a + b
    mul = a * b
    div = a/(b + epsilon)
    
    return {
        "add" : add,
        "mul" : mul,
        "div" : div,
        }
    
#Ejemplo 1
a = np.array([1.0, 2.0])
b = np.array([0.0, 2.0])
print(elementwise_ops(a, b))
