import inspect
import numpy as np
from collections import defaultdict

from heaps import *
from frontier import *
from algorithms import *
from load_data import *
from graph import *

@dataclass
class RunConfig:
    alg: str = "bmssp"
    heap: str = "binary"
    frontier: str = "block"

    graph: str = "random"
    seed: int = 5
    n: int = 100_000
    m: int = 1_000_000
    transform: bool = True
    transform_delta = 10

    niters: int = 5
    nsources: int = 1

def runSearch(cfg: RunConfig):
    # convert params to classes and functions
    conv_alg = { "bmssp":    BMSSP,
                 "dijkstra": Dijkstra,
                 "bf":       BellmanFord }

    conv_heap = { "binary":    BinaryHeap,
                  "fibonacci": FibonacciHeap }

    conv_frontier = { "heap":  HeapFrontier,
                      "block": BlockFrontier }

    # instantiate run parameters
    conv_graph = { "random": lambda : Graph.random_graph(n=cfg.n, m=cfg.m, seed=cfg.seed) }

    G = conv_graph[cfg.graph]()
    if cfg.transform: 
        print("Performing transformation...")
        G = G.constant_outdegree_transform(cfg.transform_delta)[0]

    alg = conv_alg[cfg.alg](
        frontier=conv_frontier[cfg.frontier]
    )

    # run simulations
    out = []
    for i in range(cfg.niters):
        print(f"Started iteration {i}...")
        src = G.random_vertices(seed=i, samples=cfg.nsources)
        result = alg.search(graph=G, src=src)
        out.append((result, alg.metrics))

    return out

    # graph = importCityGraph(cityName)

    # algs = filter(lambda x: not inspect.isabstract(x[1]) and x[0] != "ABC", 
    #               inspect.getmembers(algorithms, inspect.isclass))
    # hps = filter(lambda x: not inspect.isabstract(x[1]) and x[0] != "ABC",
    #              inspect.getmembers(heaps, inspect.isclass)) 
    # n = graph.number_of_nodes()
    # stats = defaultdict(lambda: {"comp": {}})

    # for alg_name, alg_cls in algs:
    #     # Dijkstra algorithms
    #     if alg_cls == algorithms.Dijkstra:
    #         # Iterate through heaps
    #         for heap_name, heap_cls in hps:
    #             run_id = alg_name + "_" + heap_name
    #             compMeans = np.zeros(m+1, dtype=float)
    #             compStds  = np.zeros(m+1, dtype=float)

    #             # Iterate through each param
    #             for i in range(m+1):
    #                 comps = np.zeros(niters, dtype=float)

    #                 # Iterate through iterations
    #                 for j in range(niters):
    #                     src = np.random.choice(list(graph.nodes()))
    #                     q = heap_cls()
    #                     comp = alg_cls(q).search(graph, src, m)
    #                     # comps[j] = comp + q.countComps
    #                     comps[j] =  comp

    #                 # Calculate stats per param
    #                 compMeans[i] = comps.mean()
    #                 compStds[i] = comps.std()
                
    #             stats[run_id]["comp"]["mean"] = compMeans / n
    #             stats[run_id]["comp"]["std"] = compStds / n
                # time
                # memory
                # print(f"finished {run_id} with {stats}")


# runSearch()

print(runSearch(RunConfig()))