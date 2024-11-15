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
    start_time = time.perf_counter()
    if block_size is not None:
        C = algoritmo(A, B, block_size)  # Solo pasa block_size si es necesario
    else:
        C = algoritmo(A, B)
    end_time = time.perf_counter()
    tiempo_ejecucion = end_time - start_time

    return C, tiempo_ejecucion


def guardar_resultados(tiempos, archivo_tiempos="resultados/tiempos_ejecucion.csv"):
    """Guarda los tiempos de ejecución en un archivo CSV."""
    with open(archivo_tiempos, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Algoritmo", "Tiempo Promedio (segundos)"])
        for alg, data in tiempos.items():
            writer.writerow([alg] + [np.mean(data)])


def guardar_resultado_matriz(C, nombre_archivo):
    """Guarda el resultado de la multiplicación de matrices en un archivo."""
    np.savetxt(nombre_archivo, C, fmt="%d")


def generar_grafico(tiempos, archivo_grafico="resultados/graficos/tiempos_ejecucion_comparativa.png"):
    """Genera y guarda un gráfico de los tiempos de ejecución de los algoritmos."""
    algoritmos = list(tiempos.keys())
    tiempos_promedio = [np.mean(data) for data in tiempos.values()]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(algoritmos, tiempos_promedio, color='darkorchid')

    # Agregar los tiempos de ejecución sobre las barras con 5 decimales
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.5f}', ha='center', va='bottom', fontsize=10)

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
    matriz_A_path = f"resultados/matrices/matriz_A_{n}.txt"
    matriz_B_path = f"resultados/matrices/matriz_B_{n}.txt"

    # Verificar si las matrices ya existen
    if not os.path.exists(matriz_A_path) or not os.path.exists(matriz_B_path):
        A = generar_matriz(n)
        B = generar_matriz(n)

        # Guardar las matrices generadas en archivos
        if not os.path.exists(matriz_A_path):
            guardar_matriz(A, matriz_A_path)
        if not os.path.exists(matriz_B_path):
            guardar_matriz(B, matriz_B_path)
    else:
        # Si las matrices ya existen, cargarlas
        A = np.loadtxt(matriz_A_path, dtype=np.int64)
        B = np.loadtxt(matriz_B_path, dtype=np.int64)

    # Inicializar un diccionario para los tiempos de este tamaño específico de matriz
    tiempos_por_tamano = {}

    # Ejecutar los algoritmos con estas matrices
    for nombre_algoritmo, algoritmo in algoritmos.items():
        # Asignar block_size si es necesario
        block_size = 64  # Ajusta el valor de block_size según tus necesidades
        if nombre_algoritmo in ["III.3 Sequential block", "IV.3 Sequential block", "V.3 Sequential block",
                                "V.4 Parallel Block"]:
            C, tiempo = ejecutar_algoritmo(algoritmo, A, B, block_size)
        else:
            C, tiempo = ejecutar_algoritmo(algoritmo, A, B)

        # Registrar los tiempos para este tamaño de matriz
        if nombre_algoritmo not in tiempos_por_tamano:
            tiempos_por_tamano[nombre_algoritmo] = []
        tiempos_por_tamano[nombre_algoritmo].append(tiempo)

    # Guardar los tiempos de ejecución por tamaño de matriz en su archivo CSV respectivo
    guardar_resultados(tiempos_por_tamano, archivo_tiempos=f"resultados/tiempos/tiempos_ejecucion_{n}.csv")

    # Acumular los tiempos en el diccionario global
    for alg, tiempo in tiempos_por_tamano.items():
        if alg not in tiempos:
            tiempos[alg] = []
        tiempos[alg].extend(tiempo)

    # Generar y guardar el gráfico de los tiempos de ejecución para este tamaño de matriz
    generar_grafico(tiempos_por_tamano, archivo_grafico=f"resultados/graficos/tiempos_ejecucion_{n}.png")

# Guardar los tiempos de ejecución acumulados en el archivo CSV de promedios
guardar_resultados(tiempos, archivo_tiempos="resultados/tiempos/tiempos_ejecucion_promedio.csv")

# Generar y guardar el gráfico de los tiempos de ejecución promedios
generar_grafico(tiempos, archivo_grafico="resultados/graficos/tiempos_ejecucion_promedio.png")
