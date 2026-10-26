"""Actividad 2.3. Búsqueda local: Hill Climbing y Simulated Annealing.

Caso didáctico: optimización de una ruta de reparto de artesanías.
Las coordenadas y distancias son simuladas; no representan mediciones oficiales.

Instalación:
    python -m pip install matplotlib networkx

Ejecución:
    python actividad_2_3_busqueda_local.py
"""
from __future__ import annotations

import itertools
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figuras"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Coordenadas didácticas en un plano expresado en kilómetros aproximados.
# La primera ubicación es el depósito y siempre será el inicio y el final.
UBICACIONES = {
    "Depósito/Taller": (11.0, 9.0),
    "Artesano A": (12.0, 4.0),
    "Artesano B": (16.0, 16.0),
    "Artesano C": (13.0, 3.0),
    "Artesano D": (10.0, 7.0),
    "Artesano E": (5.0, 19.0),
    "Punto de venta": (20.0, 1.0),
    "Centro de acopio": (17.0, 12.0),
    "Mercado local": (8.0, 6.0),
}
NOMBRES = list(UBICACIONES)
COORDENADAS = [UBICACIONES[nombre] for nombre in NOMBRES]
N = len(NOMBRES)
DEPOSITO = 0


def crear_matriz_distancias(coordenadas: list[tuple[float, float]]) -> list[list[float]]:
    """Calcula distancias euclidianas entre todos los puntos simulados."""
    matriz = []
    for x1, y1 in coordenadas:
        fila = []
        for x2, y2 in coordenadas:
            fila.append(math.hypot(x2 - x1, y2 - y1))
        matriz.append(fila)
    return matriz


DISTANCIAS = crear_matriz_distancias(COORDENADAS)


def costo_ruta(ruta: list[int]) -> float:
    """Suma los tramos desde el depósito, pasando por la ruta y regresando."""
    recorrido = [DEPOSITO] + ruta + [DEPOSITO]
    return sum(DISTANCIAS[recorrido[i]][recorrido[i + 1]]
               for i in range(len(recorrido) - 1))


def vecino_2opt(ruta: list[int], i: int, j: int) -> list[int]:
    """Genera un vecino invirtiendo el segmento de ruta comprendido entre i y j."""
    candidato = ruta[:]
    candidato[i:j + 1] = reversed(candidato[i:j + 1])
    return candidato


def hill_climbing(ruta_inicial: list[int], max_rondas: int = 100) -> dict:
    """Hill Climbing de mejor mejora: acepta solo el mejor vecino más barato."""
    actual = ruta_inicial[:]
    costo_actual = costo_ruta(actual)
    historial = [costo_actual]
    evaluaciones = 0
    inicio = time.perf_counter()

    for _ in range(max_rondas):
        mejor_vecino = actual
        mejor_costo = costo_actual
        # El vecindario 2-opt considera todas las inversiones de segmentos.
        for i in range(len(actual) - 1):
            for j in range(i + 1, len(actual)):
                candidato = vecino_2opt(actual, i, j)
                candidato_costo = costo_ruta(candidato)
                evaluaciones += 1
                if candidato_costo < mejor_costo - 1e-9:
                    mejor_vecino = candidato
                    mejor_costo = candidato_costo

        # Si ningún vecino mejora la función, se alcanzó un óptimo local.
        if mejor_costo >= costo_actual - 1e-9:
            historial.append(costo_actual)
            break
        actual = mejor_vecino[:]
        costo_actual = mejor_costo
        historial.append(costo_actual)

    segundos = time.perf_counter() - inicio
    return {
        "ruta": actual,
        "costo": costo_actual,
        "historial": historial,
        "evaluaciones": evaluaciones,
        "segundos": segundos,
        "rondas": len(historial) - 1,
    }


def simulated_annealing(
    ruta_inicial: list[int],
    temperatura_inicial: float = 12.0,
    eta: float = 0.997,
    iteraciones: int = 5000,
    semilla: int = 2026,
) -> dict:
    """Simulated Annealing: permite aceptar empeoramientos según temperatura."""
    generador = random.Random(semilla)
    actual = ruta_inicial[:]
    costo_actual = costo_ruta(actual)
    mejor_ruta = actual[:]
    mejor_costo = costo_actual
    historial = [mejor_costo]
    aceptados = 0
    inicio = time.perf_counter()

    for t in range(iteraciones):
        temperatura = temperatura_inicial * (eta ** t)
        i, j = sorted(generador.sample(range(len(actual)), 2))
        candidato = vecino_2opt(actual, i, j)
        costo_candidato = costo_ruta(candidato)
        diferencia = costo_candidato - costo_actual

        # Mejoras siempre aceptadas; empeoramientos se aceptan con probabilidad
        # exp(-diferencia / temperatura), lo que facilita escapar de óptimos locales.
        aceptar = diferencia < 0
        if not aceptar and temperatura > 1e-12:
            aceptar = generador.random() < math.exp(-diferencia / temperatura)
        if aceptar:
            actual = candidato
            costo_actual = costo_candidato
            aceptados += 1

        if costo_actual < mejor_costo - 1e-9:
            mejor_ruta = actual[:]
            mejor_costo = costo_actual
        historial.append(mejor_costo)

    segundos = time.perf_counter() - inicio
    return {
        "ruta": mejor_ruta,
        "costo": mejor_costo,
        "historial": historial,
        "evaluaciones": iteraciones,
        "segundos": segundos,
        "iteraciones": iteraciones,
        "aceptados": aceptados,
        "T0": temperatura_inicial,
        "eta": eta,
    }


def buscar_optimo_exacto() -> dict:
    """Enumera las (N-1)! rutas para obtener un punto de referencia exacto."""
    mejor_ruta = None
    mejor_costo = float("inf")
    evaluaciones = 0
    for permutacion in itertools.permutations(range(1, N)):
        costo = costo_ruta(list(permutacion))
        evaluaciones += 1
        if costo < mejor_costo:
            mejor_costo = costo
            mejor_ruta = list(permutacion)
    return {"ruta": mejor_ruta, "costo": mejor_costo, "evaluaciones": evaluaciones}


def nombres_ruta(ruta: list[int]) -> list[str]:
    """Convierte índices a nombres e incluye depósito al inicio y al final."""
    return [NOMBRES[i] for i in [DEPOSITO] + ruta + [DEPOSITO]]


def normalizar_historial(historial: list[float], puntos: int = 101) -> list[float]:
    """Muestrea un historial para comparar visualmente ejecuciones de distinta duración."""
    if len(historial) == 1:
        return historial * puntos
    resultado = []
    for indice in range(puntos):
        posicion = indice * (len(historial) - 1) / (puntos - 1)
        resultado.append(historial[round(posicion)])
    return resultado


def crear_grafica_convergencia(hc: dict, sa: dict, exacto: dict) -> Path:
    """Guarda la gráfica de convergencia en escala de progreso normalizado."""
    x = list(range(101))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(x, normalizar_historial(hc["historial"]), label="Hill Climbing", linewidth=2)
    ax.plot(x, normalizar_historial(sa["historial"]), label="Simulated Annealing", linewidth=1.8)
    ax.axhline(exacto["costo"], linestyle="--", linewidth=1.4,
               label=f"Óptimo exacto: {exacto['costo']:.2f} km")
    ax.set_title("Convergencia de la búsqueda local")
    ax.set_xlabel("Progreso normalizado de cada ejecución (%)")
    ax.set_ylabel("Distancia total de la ruta (km simulados)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    destino = FIG_DIR / "convergencia_busqueda_local.png"
    fig.savefig(destino, dpi=180)
    plt.close(fig)
    return destino


def crear_grafica_rutas(hc: dict, sa: dict) -> Path:
    """Dibuja las rutas finales de ambos algoritmos sobre las coordenadas simuladas."""
    grafo = nx.Graph()
    for i, nombre in enumerate(NOMBRES):
        grafo.add_node(i, label=nombre)
    posiciones = {i: COORDENADAS[i] for i in range(N)}

    fig, ejes = plt.subplots(1, 2, figsize=(12, 5.2))
    resultados = [(hc, "Hill Climbing"), (sa, "Simulated Annealing")]
    for ax, (resultado, titulo) in zip(ejes, resultados):
        grafo_ruta = nx.Graph()
        grafo_ruta.add_nodes_from(range(N))
        camino = [DEPOSITO] + resultado["ruta"] + [DEPOSITO]
        grafo_ruta.add_edges_from(zip(camino[:-1], camino[1:]))
        nx.draw_networkx_edges(grafo_ruta, posiciones, ax=ax, width=2, edge_color="#3178A8")
        nx.draw_networkx_nodes(grafo_ruta, posiciones, ax=ax, node_size=420,
                               node_color="#F4D35E", edgecolors="#333333")
        etiquetas = {i: NOMBRES[i] for i in range(N)}
        nx.draw_networkx_labels(grafo_ruta, posiciones, etiquetas, ax=ax, font_size=7)
        ax.set_title(f"{titulo}\nCosto: {resultado['costo']:.2f} km simulados")
        ax.set_axis_off()
    fig.suptitle("Rutas finales obtenidas por búsqueda local")
    fig.tight_layout()
    destino = FIG_DIR / "rutas_comparadas.png"
    fig.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destino


def main() -> None:
    """Ejecuta los métodos, genera figuras y guarda los resultados reproducibles."""
    generador = random.Random(1010)
    ruta_inicial = list(range(1, N))
    generador.shuffle(ruta_inicial)
    costo_inicial = costo_ruta(ruta_inicial)

    hc = hill_climbing(ruta_inicial)
    sa = simulated_annealing(ruta_inicial)
    exacto = buscar_optimo_exacto()
    crear_grafica_convergencia(hc, sa, exacto)
    crear_grafica_rutas(hc, sa)

    resultados = {
        "nota_datos": "Coordenadas y distancias simuladas para fines académicos.",
        "ubicaciones": UBICACIONES,
        "ruta_inicial": nombres_ruta(ruta_inicial),
        "costo_inicial_km": costo_inicial,
        "hill_climbing": {
            "ruta": nombres_ruta(hc["ruta"]),
            "costo_km": hc["costo"],
            "rondas": hc["rondas"],
            "evaluaciones": hc["evaluaciones"],
            "tiempo_segundos": hc["segundos"],
        },
        "simulated_annealing": {
            "ruta": nombres_ruta(sa["ruta"]),
            "costo_km": sa["costo"],
            "iteraciones": sa["iteraciones"],
            "evaluaciones": sa["evaluaciones"],
            "aceptados": sa["aceptados"],
            "temperatura_inicial": sa["T0"],
            "eta": sa["eta"],
            "tiempo_segundos": sa["segundos"],
        },
        "optimo_exacto": {
            "ruta": nombres_ruta(exacto["ruta"]),
            "costo_km": exacto["costo"],
            "rutas_evaluadas": exacto["evaluaciones"],
        },
    }
    salida_json = BASE_DIR / "resultados_2_3.json"
    salida_json.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    print("ACTIVIDAD 2.3 - RESULTADOS (distancias simuladas)")
    print(f"Ruta inicial: {resultados['ruta_inicial']}")
    print(f"Costo inicial: {costo_inicial:.2f} km")
    print(f"Hill Climbing: {hc['costo']:.2f} km; rondas={hc['rondas']}; "
          f"evaluaciones={hc['evaluaciones']}; tiempo={hc['segundos']:.6f} s")
    print(f"Simulated Annealing: {sa['costo']:.2f} km; iteraciones={sa['iteraciones']}; "
          f"evaluaciones={sa['evaluaciones']}; aceptados={sa['aceptados']}; "
          f"tiempo={sa['segundos']:.6f} s")
    print(f"Óptimo exacto: {exacto['costo']:.2f} km; rutas enumeradas={exacto['evaluaciones']}")
    print(f"Figuras: {FIG_DIR}")
    print(f"Resultados: {salida_json}")


if __name__ == "__main__":
    main()
