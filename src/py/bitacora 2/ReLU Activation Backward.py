import numpy as np

def relu_backward(gradiente, x):
    boolean = (x > 0)
    dx = gradiente * boolean
    return dx

#Ejemplos
x =np.array([[-1, 2],
     [-3, 4]]) 

dout =np.array([[0.1, 0.2],
        [0.3, 0.4]])

print(relu_backward(dout, x))