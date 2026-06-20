""" 
Ejercicio #19
Bitacora #2
Batch Normalization Backward
"""

import numpy as np

def batchnorm_backward(dout, cache):
    dout = np.array(dout)
    x = np.array(cache[0])
    x_norm = np.array(cache[1])
    mean = np.array(cache[2])
    var = np.array(cache[3])
    gamma = np.array(cache[4])
    eps = cache[5]
    
    ## DBeta
    dbeta = np.sum(dout, axis=0)
    
    ## DGamma
    dgamma = np.sum(dout * x_norm, axis=0)
    
    ## Dx
    dhat_x = dout * gamma
    dmu = np.mean(dhat_x, axis=0)
    dvar = dmu * (x - mean)
    std = np.sqrt(var+eps)
    dx = (1/std) * (dhat_x - dmu - x_norm * dvar)
    
    Diccionario = {
        "dx": dx,
        "dgamma": dgamma,
        "dbeta": dbeta
        }
    return Diccionario

print(batchnorm_backward([[0.1, 0.2],[0.3, 0.4],[0.5, 0.6]],([[1.0, 2.0],[3.0, 4.0],[5.0, 6.0]],[[-1.225, -1.225],[0, 0],[1.225, 1.225]],[3.0, 4.0], [2.67, 2.67], [1.0, 1.0], 1e-5)))
