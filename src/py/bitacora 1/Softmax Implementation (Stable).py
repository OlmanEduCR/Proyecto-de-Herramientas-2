import numpy as np

def softmax(x):
    max_x = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - max_x)
    sum_exp = np.sum(exp_x, axis=1, keepdims=True)
    s = exp_x / sum_exp
    return s


logits = np.array([
    [2.0, 1.0, 0.1],  
    [0.5, 2.5, 0.3]])   


probabilidades = softmax(logits)

print(probabilidades)