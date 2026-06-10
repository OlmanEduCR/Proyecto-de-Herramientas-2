import numpy as np
def linear_forward(x,w,b):
    x = np.asarray(x)
    w = np.asarray(w)
    b = np.asarray(b)
    xw = x @ w
    Y = xw + b
    return Y

x = [[1, 2, 3],
     [4, 5, 6]]

w = [[0.1, 0.2],
     [0.3, 0.4],
     [0.5, 0.6]]

b = [1.0, 2.0]

resultado = linear_forward(x,w,b)
print(resultado)