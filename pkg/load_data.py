import osmnx as ox
import os

def importCityGraph(cityName): 
    filename = f"data/{cityName}.graphml"
    if os.path.exists(filename):
        graph = ox.load_graphml(filename)
    else:
        graph = ox.graph_from_place(cityName, network_type='drive')
        ox.save_graphml(graph, filename)
    return graph