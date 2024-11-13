import numpy as np

def add_matrices(A, B):
    """ Suma dos matrices A y B. """
    return A + B

def subtract_matrices(A, B):
    """ Resta la matriz B de la matriz A. """
    return A - B

def strassen_naive(A, B):
    """
    Multiplica dos matrices A y B usando el algoritmo de Strassen.
    """
    n = len(A)
    if n == 1:
        return A * B  # Caso base: multiplicación de elementos únicos

    # Dividir matrices en submatrices de tamaño n/2 x n/2
    mid = n // 2
    A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
    B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]

    # Calcular los productos de Strassen
    M1 = strassen_naive(add_matrices(A11, A22), add_matrices(B11, B22))
    M2 = strassen_naive(add_matrices(A21, A22), B11)
    M3 = strassen_naive(A11, subtract_matrices(B12, B22))
    M4 = strassen_naive(A22, subtract_matrices(B21, B11))
    M5 = strassen_naive(add_matrices(A11, A12), B22)
    M6 = strassen_naive(subtract_matrices(A21, A11), add_matrices(B11, B12))
    M7 = strassen_naive(subtract_matrices(A12, A22), add_matrices(B21, B22))

    # Combinar submatrices en el resultado final
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Combinar C11, C12, C21, y C22 en la matriz de salida
    C = np.zeros((n, n))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22

    return C
