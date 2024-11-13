import os
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from algoritmos.NaivOnArray import naive_multiplication
from algoritmos.NaivLoopUnrollingTwo import naive_loop_unrolling_two
from algoritmos.NaivLoopUnrollingFour import naive_loop_unrolling_four
from algoritmos.WinogradOriginal import winograd_original
from algoritmos.WinogradScaled import winograd_scaled
from algoritmos.StrassenNaiv import strassen_naive
from algoritmos.V3SequentialBlock import sequential_block_multiplication_v3
from algoritmos.V4ParallelBlock import parallel_block_multiplication
from algoritmos.IV3SequentialBlock import sequential_block_multiplication_IV3
from algoritmos.III3SequentialBlock import sequential_block_multiplication_II3

# Crear las carpetas necesarias
os.makedirs('resultados/matrices', exist_ok=True)
os.makedirs('resultados/graficos', exist_ok=True)


# Funciones auxiliares
def generar_matriz(n, min_value=100000, max_value=999999):
    """Genera una matriz de tamaño n * n con valores aleatorios entre min_value y max_value."""
    return np.random.randint(min_value, max_value + 1, size=(n, n), dtype=np.int64)


def guardar_matriz(matriz, nombre_archivo):
    """Guarda una matriz en un archivo de texto."""
    np.savetxt(nombre_archivo, matriz, fmt="%d")


def ejecutar_algoritmo(algoritmo, A, B, block_size=None):
    """Ejecuta un algoritmo de multiplicación de matrices y mide el tiempo de ejecución."""
    start_time = time.time()
    if block_size is not None:
        C = algoritmo(A, B, block_size)  # Solo pasa block_size si es necesario
    else:
        C = algoritmo(A, B)
    end_time = time.time()
    tiempo_ejecucion = end_time - start_time
    return C, tiempo_ejecucion


def guardar_resultados(tiempos, archivo_tiempos="resultados/tiempos_ejecucion.csv"):
    """Guarda los tiempos de ejecución en un archivo CSV."""
    with open(archivo_tiempos, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Algoritmo", "Tamaño Matriz", "Caso 1", "Caso 2", "Caso 3", "Caso 4", "Caso 5", "Caso 6", "Caso 7",
             "Caso 8"])
        for alg, data in tiempos.items():
            writer.writerow([alg] + data)


def guardar_resultado_matriz(C, nombre_archivo):
    """Guarda el resultado de la multiplicación de matrices en un archivo."""
    np.savetxt(nombre_archivo, C, fmt="%d")


def generar_grafico(tiempos, archivo_grafico="resultados/graficos/tiempos_ejecucion_comparativa.png"):
    """Genera y guarda un gráfico de los tiempos de ejecución de los algoritmos."""
    algoritmos = list(tiempos.keys())
    tiempos_promedio = [np.mean(data) for data in tiempos.values()]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(algoritmos, tiempos_promedio, color='skyblue')

    # Agregar los tiempos de ejecución sobre las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    plt.xlabel('Algoritmos')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    plt.title('Comparación de Tiempos de Ejecución de Algoritmos de Multiplicación de Matrices')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(archivo_grafico)
    plt.show()


# Definir los tamaños de matrices a probar (n debe ser factor de 2^n)
casos_tamano = [2 ** i for i in range(1, 9)]  # Tamaños: 2^1, 2^2, ..., 2^8

# Diccionario para almacenar los tiempos de ejecución de cada algoritmo
tiempos = {}

# Definir los algoritmos de multiplicación de matrices
algoritmos = {
    "NaivOnArray": naive_multiplication,
    "NaivLoopUnrollingTwo": naive_loop_unrolling_two,
    "NaivLoopUnrollingFour": naive_loop_unrolling_four,
    "WinogradOriginal": winograd_original,
    "WinogradScaled": winograd_scaled,
    "StrassenNaiv": strassen_naive,
    "III.3 Sequential block": sequential_block_multiplication_II3,
    "IV.3 Sequential block": sequential_block_multiplication_IV3,
    "V.3 Sequential block": sequential_block_multiplication_v3,
    "V.4 Parallel Block": parallel_block_multiplication,
}

# Ejecutar pruebas con diferentes tamaños de matrices
for n in casos_tamano:
    A = generar_matriz(n)
    B = generar_matriz(n)

    # Guardar las matrices generadas en archivos
    guardar_matriz(A, f"resultados/matrices/matriz_A_{n}.txt")
    guardar_matriz(B, f"resultados/matrices/matriz_B_{n}.txt")

    # Ejecutar los algoritmos con estas matrices
    for nombre_algoritmo, algoritmo in algoritmos.items():
        # Asignar block_size si es necesario
        block_size = 64  # Ajusta el valor de block_size según tus necesidades
        if nombre_algoritmo in ["III.3 Sequential block", "IV.3 Sequential block", "V.3 Sequential block",
                                "V.4 Parallel Block"]:
            C, tiempo = ejecutar_algoritmo(algoritmo, A, B, block_size)
        else:
            C, tiempo = ejecutar_algoritmo(algoritmo, A, B)

        # Guardar los resultados de las matrices
        guardar_resultado_matriz(C, f"resultados/resultados_matrices/resultado_{nombre_algoritmo}_{n}.txt")

        # Registrar los tiempos
        if nombre_algoritmo not in tiempos:
            tiempos[nombre_algoritmo] = []
        tiempos[nombre_algoritmo].append(tiempo)

# Guardar los tiempos de ejecución en un archivo CSV
guardar_resultados(tiempos)

# Generar y guardar el gráfico de los tiempos de ejecución
generar_grafico(tiempos)
