def grad_matmul(grad_C, A, B):
    grad_A = grad_C @ B.T
    grad_B = A.T @ grad_C
    
    return {
        "grad_A": grad_A,
        "grad_B": grad_B
    }

#Ejemplo 20:40
A = np.array([[1, 2],     
     [3, 4]])

B = np.array([[5, 6],     
     [7, 8]])

C = A @ B

grad_C = np.array([[0.1, 0.2],
                   [0.3, 0.4]])

print(grad_matmul(grad_C, A, B))
