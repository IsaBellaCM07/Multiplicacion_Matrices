import numpy as np

def naive_loop_unrolling_two(A, B):
    """
    Multiplica dos matrices A y B usando Naiv con Loop Unrolling de 2.
    """
    n = len(A)
    C = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sum = 0
            k = 0
            # Desenrollado de bucle para procesar de dos en dos
            while k + 1 < n:
                sum += A[i][k] * B[k][j] + A[i][k + 1] * B[k + 1][j]
                k += 2
            # Si el tamaño es impar, procesa el último elemento
            if k < n:
                sum += A[i][k] * B[k][j]
            C[i][j] = sum

    return C
