import numpy as np

def multiply_block(A, B, C, block_size):
    """
    Multiplica dos matrices A y B utilizando un enfoque de bloques secuenciales.
    El resultado se almacena en la matriz C.
    """
    n = len(A)  # Tamaño de la matriz
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                # Multiplicar los bloques de A y B
                for i_block in range(i, min(i + block_size, n)):
                    for j_block in range(j, min(j + block_size, n)):
                        for k_block in range(k, min(k + block_size, n)):
                            C[i_block][j_block] += A[i_block][k_block] * B[k_block][j_block]

def sequential_block_multiplication_II3(A, B, block_size):
    """
    Realiza la multiplicación de matrices A y B en bloques secuenciales.
    """
    n = len(A)
    C = np.zeros((n, n))

    # Llamar a la función para multiplicar en bloques
    multiply_block(A, B, C, block_size)

    return C
