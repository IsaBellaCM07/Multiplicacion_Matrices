import numpy as np

def block_multiply_v3(A, B, C, block_size):
    """
    Realiza la multiplicación de matrices A y B en bloques secuenciales,
    almacenando el resultado en la matriz C.
    """
    n = len(A)
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                # Multiplicar sub-bloques de A y B
                for i_block in range(i, min(i + block_size, n)):
                    for j_block in range(j, min(j + block_size, n)):
                        for k_block in range(k, min(k + block_size, n)):
                            C[i_block][j_block] += A[i_block][k_block] * B[k_block][j_block]

def sequential_block_multiplication_v3(A, B, block_size):
    """
    Multiplica las matrices A y B utilizando el enfoque de bloques secuenciales V.3.
    """
    n = len(A)
    C = np.zeros((n, n))  # Inicializar la matriz de resultado en ceros

    # Llamada a la función que realiza la multiplicación por bloques
    block_multiply_v3(A, B, C, block_size)

    return C
