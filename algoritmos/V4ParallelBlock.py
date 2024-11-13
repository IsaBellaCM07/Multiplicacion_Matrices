import numpy as np
import concurrent.futures


def block_multiply_parallel(A, B, C, block_size, start_row, end_row, start_col, end_col):
    """
    Multiplica bloques de A y B y almacena el resultado en C, usando paralelización para mejorar el rendimiento.
    """
    for i in range(start_row, end_row, block_size):
        for j in range(start_col, end_col, block_size):
            for k in range(0, len(A), block_size):
                # Multiplicar sub-bloques de A y B
                for i_block in range(i, min(i + block_size, len(A))):
                    for j_block in range(j, min(j + block_size, len(B))):
                        for k_block in range(k, min(k + block_size, len(A))):
                            C[i_block][j_block] += A[i_block][k_block] * B[k_block][j_block]


def parallel_block_multiplication(A, B, block_size, num_workers=4):
    """
    Realiza la multiplicación de matrices A y B usando bloques paralelizados.
    """
    n = len(A)
    C = np.zeros((n, n))  # Inicializar la matriz de resultado en ceros

    # Definir los parámetros para dividir la carga de trabajo entre los hilos
    block_rows = n // block_size
    block_cols = n // block_size

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        # Distribuir la multiplicación de bloques entre los hilos
        for i in range(block_rows):
            for j in range(block_cols):
                start_row = i * block_size
                end_row = (i + 1) * block_size
                start_col = j * block_size
                end_col = (j + 1) * block_size

                futures.append(
                    executor.submit(block_multiply_parallel, A, B, C, block_size, start_row, end_row, start_col,
                                    end_col))

        # Esperar a que todos los hilos terminen
        for future in futures:
            future.result()

    return C
