import numpy as np

def am_step(w, dw, lr, beta1, beta2, eps, t, m=None, v=None):
    if m is None:
        m = np.zeros_like(w)
    if v is None:
        v = np.zeros_like(w)
    
    m_new = beta1 * m + (1 - beta1) * dw
    v_new = beta2 * v + (1 - beta2) * (dw**2)

    bias_correction1 = 1 - (beta1 ** (t+1))
    bias_correction2 = 1 - (beta2 ** (t+1))
    
    m_hat = m_new / bias_correction1
    v_hat = v_new / bias_correction2

    w_new = w - (lr / (np.sqrt(v_hat) + eps)) * m_hat
    return w_new, m_new, v_new



w = np.array([[1.0, 2.0],
              [3.0, 4.0]])

dw = np.array([[0.5, -0.3],
               [-0.2, 0.1]])

m = np.array([[0.1, -0.05],
              [-0.02, 0.01]])

v = np.array([[0.1, 0.05],
              [0.02, 0.01]])

lr = 0.001
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
t = 10

w_new, m_new, v_new = am_step(w, dw, lr, beta1, beta2, eps, t, m, v)

print("w_new:\n", np.round(w_new, 6))
print("\nm_new:\n", np.round(m_new, 3))
print("\nv_new:\n", np.round(v_new, 5))




