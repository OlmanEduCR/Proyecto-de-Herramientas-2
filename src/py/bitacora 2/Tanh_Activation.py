
#6) Tanh Activation

import numpy as np

def tanh_ops(x,dout):
    
    out = np.tanh(x) # aplica tanh a x elemento a elemento
    dx = dout*(1-out**2) # gradiente 

    return{"out":out,
           "dx":dx}













































