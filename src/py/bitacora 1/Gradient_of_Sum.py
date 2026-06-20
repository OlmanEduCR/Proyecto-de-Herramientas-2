
# 10) Gradient of Sum

import numpy as np

# Lo que se nos dice es que, al ser y la suma de los x, cualquier cambio en un xi produce el mismo cambio en y, es decir, la derivada es 1 para todos
# Aplicando la regla de la cadena, el gradiente respecto a xi es igual a grad_y
# Por lo tanto, cada elemento recibe el mismo gradiente
# Asi que el resultado final sera simplemente una matriz con el mismo tamaño de x, donde todos los elementos son iguales a grad_y

def grad_sum(grad_y , x_shape):
    return np.full(x_shape , grad_y) # crea y retorna una matriz en la que todos los elementos tienen el valor de grad_y



# Ejemplo de prueba 

x_shape=(3)
grad_y=0.5
print(grad_sum(grad_y, x_shape))





























