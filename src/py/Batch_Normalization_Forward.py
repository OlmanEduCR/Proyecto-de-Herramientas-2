
# 18) Batch Normalization Forward (Train vs Eval)
import numpy as np

def batchnorm_forward(x, gamma, beta, eps=1e-5, momentum=0.9, running_mean=None, running_var=None, train=True):
    
    N,D = x.shape
    
    if running_mean is None:
        running_mean = np.zeros(D) # controlar inicializacion de running_mean None
        
    if running_var is None:
        running_var= np.zeros(D) # controlar inicializacion de running_var None

    
    if train: # cambiar variables si train=True
        mean = np.mean(x,axis=0)
        var= np.mean((x-mean)**2,axis=0)
        
        x_norm = (x-mean) / np.sqrt(var + eps)
        
        new_running_mean= momentum*running_mean+(1-momentum)* mean
        
        new_running_var= momentum*running_var+(1- momentum) * var
    else: # variables si Train=False, osea esta en modo Eval y se usa running statistics
        mean=running_mean
        var=running_var
        x_norm = (x-running_mean) / np.sqrt(running_var + eps)
        new_running_mean=running_mean
        new_running_var=running_var
        
    out= gamma * x_norm + beta # no cambia 

    cache={
        "x":x,
        "x_norm": x_norm,
        "mean": mean,
        "var": var,
        "gamma": gamma,
        "eps":eps}
    
    return (out,cache,new_running_mean,new_running_var)






















