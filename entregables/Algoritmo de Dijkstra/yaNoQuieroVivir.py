import heapq
import random
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from IPython.display import Video

def dijkstra(graph, start):
    priority_queue = [(0, start)]
    shortest_paths = {vertex: float('inf') for vertex in graph}
    shortest_paths[start] = 0
    predecessors = {vertex: None for vertex in graph}
    states = []

    while priority_queue:
        current_cost, current_vertex = heapq.heappop(priority_queue)
        states.append((current_vertex, shortest_paths.copy(), predecessors.copy()))

        for neighbor, weight in graph[current_vertex].items():
            cost = current_cost + weight
            if cost < shortest_paths[neighbor]:
                shortest_paths[neighbor] = cost
                predecessors[neighbor] = current_vertex
                heapq.heappush(priority_queue, (cost, neighbor))
                states.append((neighbor, shortest_paths.copy(), predecessors.copy()))

    return shortest_paths, predecessors, states

def create_random_graph(num_vertices, num_edges):
    graph = {i: {} for i in range(num_vertices)}
    for _ in range(num_edges):
        u = random.randint(0, num_vertices - 1)
        v = random.randint(0, num_vertices - 1)
        if u != v:
            weight = random.randint(1, 10)
            graph[u][v] = weight
            graph[v][u] = weight
    return graph

# Define the graph
graph = {
    'A': {'B': 8, 'C': 6, 'D': 2, 'E': 7},
    'B': {'A': 8, 'C': 1, 'F': 4, 'G': 9},
    'C': {'A': 6, 'B': 1, 'D': 3, 'H': 2},
    'D': {'A': 2, 'C': 3, 'E': 5, 'I': 4},
    'E': {'A': 7, 'D': 5, 'J': 8},
    'F': {'B': 4, 'G': 2, 'K': 6},
    'G': {'B': 9, 'F': 2, 'H': 7, 'L': 5},
    'H': {'C': 2, 'G': 7, 'I': 9, 'M': 3},
    'I': {'D': 4, 'H': 9, 'J': 2, 'N': 7},
    'J': {'E': 8, 'I': 2, 'O': 6},
    'K': {'F': 6, 'L': 3, 'P': 1},
    'L': {'G': 5, 'K': 3, 'M': 7, 'Q': 4},
    'M': {'H': 3, 'L': 7, 'N': 2, 'R': 9},
    'N': {'I': 7, 'M': 2, 'O': 1, 'S': 5},
    'O': {'J': 6, 'N': 1, 'T': 8},
    'P': {'K': 1, 'Q': 6, 'U': 7},
    'Q': {'L': 4, 'P': 6, 'R': 3, 'V': 5},
    'R': {'M': 9, 'Q': 3, 'S': 4, 'W': 2},
    'S': {'N': 5, 'R': 4, 'T': 1, 'X': 7},
    'T': {'O': 8, 'S': 1, 'Y': 6},
    'U': {'P': 7, 'V': 3, 'Z': 5},
    'V': {'Q': 5, 'U': 3, 'W': 8, 'A': 2},
    'W': {'R': 2, 'V': 8, 'X': 4, 'B': 9},
    'X': {'S': 7, 'W': 4, 'Y': 1, 'C': 6},
    'Y': {'T': 6, 'X': 1, 'Z': 3, 'D': 5},
    'Z': {'U': 5, 'Y': 3, 'E': 2}
}

shortest_paths, predecessors, states = dijkstra(graph, 'A')

G = nx.Graph()
for vertex in graph:
    for neighbor, weight in graph[vertex].items():
        G.add_edge(vertex, neighbor, weight=weight)

pos = nx.spring_layout(G)

fig, ax = plt.subplots()

def update(num):
    ax.clear()
    nx.draw(G, pos, with_labels=True, labels={node: node for node in G.nodes()}, ax=ax, node_color='skyblue', node_size=700, font_size=16)
    current_vertex, current_shortest_paths, current_predecessors = states[num]

    for vertex in current_shortest_paths:
        if current_predecessors[vertex] is not None:
            nx.draw_networkx_edges(G, pos, edgelist=[(vertex, current_predecessors[vertex])], width=2.5, alpha=0.6, edge_color='r', ax=ax)

    labels = {vertex: f"{vertex}\n{current_shortest_paths[vertex]}" for vertex in current_shortest_paths}
    nx.draw_networkx_labels(G, pos, labels, font_size=16, font_color='black', ax=ax)
    ax.set_title(f"Paso {num + 1}: Visitando nodo {current_vertex}")

ani = FuncAnimation(fig, update, frames=len(states), interval=1000, repeat=False)

# Guardar la animación como archivo de video utilizando ffmpeg
try:
    ani.save('dijkstra_animation.mp4', writer='ffmpeg', fps=1)
    print("Animación guardada como 'dijkstra_animation.mp4'.")
except Exception as e:
    print(f"Error al guardar la animación: {e}")

# Reproducir el video utilizando IPython
try:
    video = Video("dijkstra_animation.mp4", embed=True)
    display(video)
except Exception as e:
    print(f"Error al reproducir el video: {e}")

# Al finalizar la animación, resaltar el camino más corto
def highlight_shortest_path():
    ax.clear()
    nx.draw(G, pos, with_labels=True, labels={node: node for node in G.nodes()}, ax=ax, node_color='skyblue', node_size=700, font_size=16)

    # Resaltar los bordes del camino más corto
    shortest_path_edges = []
    for vertex in predecessors:
        if predecessors[vertex] is not None:
            shortest_path_edges.append((vertex, predecessors[vertex]))

    nx.draw_networkx_edges(G, pos, edgelist=shortest_path_edges, width=2.5, alpha=0.6, edge_color='g', ax=ax)

    # Dibujar las etiquetas de nuevo para evitar superposición
    final_labels = {vertex: vertex for vertex in shortest_paths}
    nx.draw_networkx_labels(G, pos, final_labels, font_size=16, font_color='black', ax=ax)
    ax.set_title("Camino más corto resaltado")

highlight_shortest_path()
plt.show()
