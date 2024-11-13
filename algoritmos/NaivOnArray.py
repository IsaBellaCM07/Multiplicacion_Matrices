import numpy as np

def naive_multiplication(A, B):
    """
    Multiplica dos matrices A y B usando el método NaivOnArray.
    """
    n = len(A)  # Asume matrices cuadradas
    # Crear matriz de resultado con ceros
    C = np.zeros((n, n))

    # Multiplicación de matrices
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

    return C
