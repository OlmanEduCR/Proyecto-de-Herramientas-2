import numpy as np

def log_sum_exp(x):
    max_x = np.max(x, axis=1, keepdims=True)
    
    x_shifted = x - max_x
    
    exp_shifted = np.exp(x_shifted)
    
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    
    log_sum = np.log(sum_exp)
    
    result = max_x + log_sum
    
    return result.flatten()

#Ejemplo
x = [[2.0, 1.0, 0.1],    
     [0.5, 2.5, 0.3]]

print(log_sum_exp(x))
