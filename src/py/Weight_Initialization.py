
# 14) Weight Initialization (Xavier/Glorot)

import numpy as np

def xavier_init(shape):
    fan_in,fan_out = shape 
    
    limit = np.sqrt( 6 / (fan_in + fan_out) )
    
    w= np.random.uniform(low=-limit,
                         high=limit,
                         size=shape) # matriz de muestra con distribucion uniforme de tamaño shape
    return w
    






























