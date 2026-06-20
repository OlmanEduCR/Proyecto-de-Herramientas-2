import numpy as np

def layernorm(x, gamma, beta, eps=1e-5):

    # Estadísticas por muestra
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)

    # Normalización
    x_norm = (x - mean) / np.sqrt(var + eps)

    # Escalamiento y desplazamiento
    out = gamma * x_norm + beta

    cache = {
        "x": x,
        "x_norm": x_norm,
        "mean": mean,
        "var": var,
        "gamma": gamma,
        "eps": eps
    }

    return out, cache


def layernorm_backward(dout, cache):
    x_norm = cache["x_norm"]
    var = cache["var"]
    gamma = cache["gamma"]
    eps = cache["eps"]

    N, D = dout.shape

    # Gradientes de beta y gamma
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)

    # Gradiente respecto a la normalización
    dx_norm = dout * gamma

    # Fórmula vectorizada de LayerNorm
    std_inv = 1.0 / np.sqrt(var + eps)

    dx = (
        (1.0 / D)
        * std_inv
        * (
            D * dx_norm
            - np.sum(dx_norm, axis=1, keepdims=True)
            - x_norm * np.sum(dx_norm * x_norm, axis=1, keepdims=True)
        )
    )

    return dx, dgamma, dbeta