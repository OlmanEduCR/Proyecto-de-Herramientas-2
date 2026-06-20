import numpy as np

def bce_loss(y_pred, y_true):
    eps = 1e-9

    N = y_true.shape[0]

    loss = -np.mean(
        y_true * np.log(y_pred + eps)
        + (1 - y_true) * np.log(1 - y_pred + eps)
    )

    dx = (y_pred - y_true) / N

    return {
        "loss": loss,
        "dx": dx
    }

#Ejemplos
y_pred = np.array([0.1, 0.9, 0.5])  # Probabilities (after sigmoid)
y_true = np.array([0, 1, 1])

print(bce_loss(y_pred, y_true))