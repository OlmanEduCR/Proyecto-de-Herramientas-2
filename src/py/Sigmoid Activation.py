import numpy as np
def sigmoid_ops(x, dout):
    x = np.asarray(x, dtype = float)
    dout = np. asarray(dout, dtype = float)
    pos_mask = (x >= 0)

    out = np.zeros_like(x)
    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))

    out[~pos_mask] = np.exp(x[~pos_mask]) / (1.0 + np.exp(x[~pos_mask]))
    dx = dout * out * (1 - out)
    
    return {"out": out,
            "dx": dx}

x = [-2.0, -1.0, 0.0, 1.0, 2.0]
dout = [0.1, 0.2, 0.3, 0.4, 0.5]

resultado = sigmoid_ops(x, dout)

print("Forward (out):", np.round(resultado["out"], 4))
print("Backward (dx):", np.round(resultado["dx"], 4))