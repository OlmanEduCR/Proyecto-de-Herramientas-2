
import numpy as np
def reduc_tensote(x, axis):
    dic = {
        "Suma" : np.sum(x, axis = axis),
        "Media": np.mean(x, axis = axis),
        "Maximp": np.max(x, axis = axis),
        "valor maximo": np.argmax(x, axis = axis)
    }
    return dic
x = np.array([[1,2,3],
              [4,5,6]])
axis = 1

calculo = reduc_tensote(x, axis)

for i, v in calculo.items():
    print(i, end=":")
    print(v)


