""" 
Ejercicio #7
Bitacora #2
MSE Loss
"""
import numpy as np

def mse_loss(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    N_pred = y_pred.shape[0]
    D_pred = y_pred.shape[1]
    
    N_true = y_true.shape[0]
    D_true = y_true.shape[1]
    
    if N_pred == N_true and D_pred == D_true:
        N = N_pred
        D = D_pred
        
        if D >= 1 and N != 0:
            errors = y_pred - y_true
            squared_errors = errors ** 2
            loss = np.mean(squared_errors)
            loss = np.round(loss, 3)
            
            dx = (2 / (N * D)) * errors
            dx = np.round(dx, 3)
            lista_resultado = {
                "loss": loss,
                "dx": dx
                }
            return lista_resultado
    
print(mse_loss(y_pred = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], y_true = [[1.5, 2.0, 2.5], [3.5, 5.5, 6.5]]))
