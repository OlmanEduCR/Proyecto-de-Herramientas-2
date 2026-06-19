
#2) Linear Layer Backward

import numpy as np

def linear_backward(dout,x,w,b):
    dx = dout @ w.T  # gradiente respecto al input 
    dw = x.T @ dout # gradiente respecto a los pesos
    db = np.sum(dout, axis=0) # gradiente respecto al bias
    
    return {"dx": dx,
            "dw": dw,
            "db":db } 











































