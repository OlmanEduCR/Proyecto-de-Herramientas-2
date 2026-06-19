""" 
Ejercicio #15
Bitacora #2
Weight Initialization (Kaiming/He)
"""

import numpy as np

def kaiming_init(shape):
    fan_in = shape[0]
        
    if fan_in <= 0:
        weights = np.zeros(shape)
        return weights
    
    ## Desviación Estandar
    std = np.sqrt((2.0 / fan_in))
        
    ## Distribución normal
    weights = np.random.normal(
        loc = 0.0,
        scale = std, 
        size = shape
        )
    return weights
                
print(kaiming_init((100,50)))
