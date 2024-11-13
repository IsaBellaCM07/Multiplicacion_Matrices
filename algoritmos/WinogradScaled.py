import numpy as np

def winograd_scaled(A, B):
    """
    Multiplica dos matrices A y B usando el algoritmo de Winograd escalado para mejorar la precisión.
    """
    n = len(A)
    C = np.zeros((n, n))

    # Preparación de productos parciales para filas de A y columnas de B, escalados
    row_factors = np.zeros(n)
    col_factors = np.zeros(n)

    # Calcular los factores de las filas de A con un factor de escalado
    for i in range(n):
        for j in range(0, n // 2):
            row_factors[i] += A[i][2 * j] * A[i][2 * j + 1] * 0.5  # Escalado de 0.5

    # Calcular los factores de las columnas de B con un factor de escalado
    for j in range(n):
        for i in range(0, n // 2):
            col_factors[j] += B[2 * i][j] * B[2 * i + 1][j] * 0.5  # Escalado de 0.5

    # Calcular la matriz C usando los factores escalados
    for i in range(n):
        for j in range(n):
            C[i][j] = -row_factors[i] - col_factors[j]
            for k in range(0, n // 2):
                C[i][j] += (A[i][2 * k] + B[2 * k + 1][j]) * (A[i][2 * k + 1] + B[2 * k][j]) * 0.5  # Escalado de 0.5

    # Ajuste si el tamaño de la matriz es impar
    if n % 2 == 1:
        for i in range(n):
            for j in range(n):
                C[i][j] += A[i][n - 1] * B[n - 1][j]

    return C
