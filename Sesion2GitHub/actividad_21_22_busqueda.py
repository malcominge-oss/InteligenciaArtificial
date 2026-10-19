"""Actividad 2.1 y 2.2: problemas y espacios de búsqueda con NetworkX.

Los costos de las aristas son datos didácticos simulados. No representan
mediciones oficiales de tránsito ni distancias tomadas de un mapa real.

Requisitos:
    python -m pip install networkx matplotlib
Ejecución:
    python actividad_21_22_busqueda.py
"""

from collections import deque
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import networkx as nx

CARPETA_SALIDA = Path(__file__).resolve().parent / "diagramas"


def crear_problema_ambulancia() -> tuple[nx.DiGraph, dict, dict, str, str, str]:
    """Construye el espacio de rutas de una ambulancia.

    Cada nodo representa una ubicación/intersección y cada arista dirigida,
    un tramo transitable. El peso se interpreta como minutos de recorrido.
    """
    grafo = nx.DiGraph(nombre="Ruta de ambulancia")

    etiquetas = {
        "Base": "Base de\nambulancias",
        "A": "Cruce A",
        "B": "Cruce B",
        "C": "Cruce C",
        "D": "Cruce D",
        "E": "Cruce E",
        "Hospital": "Hospital",
    }
    grafo.add_nodes_from(etiquetas)

    # (origen, destino, minutos): cada conexión se puede recorrer en el
    # sentido de la flecha; no se presupone que la calle permita volver.
    tramos = [
        ("Base", "A", 4),
        ("Base", "B", 7),
        ("A", "C", 5),
        ("A", "D", 8),
        ("B", "D", 3),
        ("B", "E", 6),
        ("C", "Hospital", 6),
        ("D", "C", 2),
        ("D", "Hospital", 7),
        ("E", "D", 2),
        ("E", "Hospital", 8),
    ]
    grafo.add_weighted_edges_from(tramos, weight="weight")

    posiciones = {
        "Base": (-0.2, 1.0),
        "A": (1.0, 2.0),
        "B": (1.0, 0.0),
        "C": (2.8, 2.0),
        "D": (2.7, 0.6),
        "E": (2.5, -0.8),
        "Hospital": (4.5, 1.0),
    }
    return grafo, etiquetas, posiciones, "Base", "Hospital", "min"


def crear_problema_artesanias() -> tuple[nx.DiGraph, dict, dict, str, str, str]:
    """Construye el espacio de rutas de entrega de artesanías.

    Cada nodo es un punto de paso o entrega y el peso indica kilómetros
    estimados para el ejemplo académico.
    """
    grafo = nx.DiGraph(nombre="Distribución de artesanías")

    etiquetas = {
        "Taller": "Taller\nartesanal",
        "Norte": "Cruce\nNorte",
        "Sur": "Cruce\nSur",
        "Empaque": "Centro de\nempaque",
        "Mercado": "Mercado\nlocal",
        "Entrega": "Punto de\nentrega",
    }
    grafo.add_nodes_from(etiquetas)

    # (origen, destino, kilómetros). Los sentidos reflejan las conexiones
    # permitidas dentro de este pequeño escenario de ejemplo.
    tramos = [
        ("Taller", "Norte", 3),
        ("Taller", "Sur", 5),
        ("Norte", "Sur", 4),
        ("Norte", "Empaque", 5),
        ("Sur", "Empaque", 2),
        ("Sur", "Mercado", 7),
        ("Empaque", "Mercado", 3),
        ("Empaque", "Entrega", 7),
        ("Mercado", "Entrega", 1),
        ("Norte", "Entrega", 14),
    ]
    grafo.add_weighted_edges_from(tramos, weight="weight")

    posiciones = {
        "Taller": (-0.2, 1.0),
        "Norte": (1.2, 2.5),
        "Sur": (1.2, 0.0),
        "Empaque": (2.7, 1.0),
        "Mercado": (4.3, 0.2),
        "Entrega": (5.9, 1.0),
    }
    return grafo, etiquetas, posiciones, "Taller", "Entrega", "km"


def generar_estados(grafo: nx.DiGraph, estado_inicial: str) -> Iterator[str]:
    """Genera los estados alcanzables desde el estado inicial mediante BFS.

    Se recorren únicamente las aristas salientes, porque el grafo es dirigido.
    Cada estado se emite una sola vez para evitar ciclos infinitos.
    """
    if estado_inicial not in grafo:
        raise ValueError(f"El estado inicial {estado_inicial!r} no existe en el grafo.")

    frontera = deque([estado_inicial])
    visitados = {estado_inicial}

    while frontera:
        estado_actual = frontera.popleft()
        yield estado_actual

        for siguiente in grafo.successors(estado_actual):
            if siguiente not in visitados:
                visitados.add(siguiente)
                frontera.append(siguiente)


def validar_pesos(grafo: nx.DiGraph) -> None:
    """Revisa que todas las aristas tengan peso numérico no negativo."""
    for origen, destino, datos in grafo.edges(data=True):
        peso = datos.get("weight")
        if not isinstance(peso, (int, float)) or peso < 0:
            raise ValueError(
                f"Peso inválido en la arista {origen!r} -> {destino!r}: {peso!r}"
            )


def resolver_problema(
    grafo: nx.DiGraph, inicial: str, meta: str
) -> tuple[list[str], float, list[str]]:
    """Genera los estados y obtiene la ruta de menor costo con Dijkstra."""
    validar_pesos(grafo)
    estados = list(generar_estados(grafo, inicial))

    if meta not in estados:
        raise nx.NetworkXNoPath(f"No existe una ruta dirigida de {inicial} a {meta}.")

    # Dijkstra es apropiado porque todos los pesos del modelo son no negativos.
    ruta = nx.dijkstra_path(grafo, source=inicial, target=meta, weight="weight")
    costo = nx.dijkstra_path_length(
        grafo, source=inicial, target=meta, weight="weight"
    )

    # Verificación independiente: suma manual de los pesos de la ruta obtenida.
    costo_verificado = sum(
        grafo[u][v]["weight"] for u, v in zip(ruta, ruta[1:])
    )
    if costo != costo_verificado:
        raise AssertionError("El costo calculado no coincide con la suma de las aristas.")

    return ruta, costo, estados


def dibujar_grafo(
    grafo: nx.DiGraph,
    etiquetas: dict,
    posiciones: dict,
    ruta: list[str],
    unidad: str,
    archivo: Path,
    titulo: str,
) -> None:
    """Guarda un diagrama dirigido y ponderado; resalta la ruta óptima."""
    archivo.parent.mkdir(parents=True, exist_ok=True)
    aristas_ruta = set(zip(ruta, ruta[1:]))
    colores_aristas = [
        "#D1495B" if (u, v) in aristas_ruta else "#7B8794"
        for u, v in grafo.edges()
    ]
    anchos_aristas = [
        3.0 if (u, v) in aristas_ruta else 1.4 for u, v in grafo.edges()
    ]

    plt.figure(figsize=(11, 6.3), dpi=200)
    nx.draw_networkx_nodes(
        grafo,
        posiciones,
        node_size=2450,
        node_color="#E7F0FA",
        edgecolors="#315A7D",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(
        grafo,
        posiciones,
        labels=etiquetas,
        font_size=8.5,
        font_weight="bold",
        font_color="#172B4D",
    )
    nx.draw_networkx_edges(
        grafo,
        posiciones,
        edge_color=colores_aristas,
        width=anchos_aristas,
        arrows=True,
        arrowsize=19,
        node_size=2450,
        connectionstyle="arc3,rad=0.04",
        min_source_margin=12,
        min_target_margin=17,
    )
    etiquetas_aristas = {
        (u, v): f"{datos['weight']} {unidad}"
        for u, v, datos in grafo.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        grafo,
        posiciones,
        edge_labels=etiquetas_aristas,
        font_size=7.5,
        font_color="#253858",
        rotate=False,
        label_pos=0.52,
        bbox={"alpha": 0.88, "color": "white", "pad": 0.9, "linewidth": 0},
    )
    plt.title(titulo, fontsize=13, fontweight="bold", pad=16)
    plt.text(
        0.5,
        0.02,
        "Ruta óptima resaltada en rojo; los números indican el costo de cada tramo.",
        transform=plt.gcf().transFigure,
        ha="center",
        fontsize=8,
    )
    plt.axis("off")
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(archivo, bbox_inches="tight", facecolor="white")
    plt.close()


def ejecutar_caso(
    nombre: str,
    creador,
    nombre_archivo_grafo: str,
) -> None:
    """Ejecuta el flujo completo: crear, generar estados, buscar y graficar."""
    grafo, etiquetas, posiciones, inicial, meta, unidad = creador()
    ruta, costo, estados = resolver_problema(grafo, inicial, meta)

    print(f"\n=== {nombre} ===")
    print(f"Tipo de grafo: dirigido y ponderado (DiGraph)")
    print(f"Nodos (V): {grafo.number_of_nodes()}")
    print(f"Aristas (E): {grafo.number_of_edges()}")
    print(f"Estados alcanzables desde {inicial}: {', '.join(estados)}")
    print(f"Meta: {meta}")
    print(f"Ruta de menor costo: {' -> '.join(ruta)}")
    print(f"Costo total: {costo} {unidad}")

    archivo = CARPETA_SALIDA / nombre_archivo_grafo
    dibujar_grafo(
        grafo,
        etiquetas,
        posiciones,
        ruta,
        unidad,
        archivo,
        nombre,
    )
    print(f"Diagrama guardado en: {archivo}")


def main() -> None:
    """Resuelve ambos casos y crea los diagramas para el reporte."""
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    ejecutar_caso(
        "Caso 1. Ruta de ambulancia (costo: minutos)",
        crear_problema_ambulancia,
        "grafo_ambulancia.png",
    )
    ejecutar_caso(
        "Caso 2. Entrega de artesanías (costo: kilómetros)",
        crear_problema_artesanias,
        "grafo_artesanias.png",
    )
    print("\nEjecución finalizada. Ambos costos se verificaron sumando sus aristas.")


if __name__ == "__main__":
    main()
