import numpy as np

def naive_loop_unrolling_four(A, B):
    """
    Multiplica dos matrices A y B usando Naiv con Loop Unrolling de 4.
    """
    n = len(A)
    C = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sum = 0
            k = 0
            # Desenrollado de bucle para procesar de cuatro en cuatro
            while k + 3 < n:
                sum += (A[i][k] * B[k][j] +
                        A[i][k + 1] * B[k + 1][j] +
                        A[i][k + 2] * B[k + 2][j] +
                        A[i][k + 3] * B[k + 3][j])
                k += 4
            # Procesa los elementos restantes si el tamaño no es múltiplo de 4
            while k < n:
                sum += A[i][k] * B[k][j]
                k += 1
            C[i][j] = sum

    return C
