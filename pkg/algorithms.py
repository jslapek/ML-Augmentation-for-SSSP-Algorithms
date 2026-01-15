from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import math

from frontier import *
from graph import *
from heaps import *

#############################
######### Utilities ######### 
#############################

@dataclass 
class AlgResults:
    distances: List[Float]
    predecessors: List[Optional[Vertex]]

@dataclass
class BaseMetrics:
    edges_relaxed: int = 0
    queue_pops: int = 0
    # max_frontier_size: int = 0
    # come up with more metrics?

@dataclass
class BMSSPMetrics(BaseMetrics):
    block_pulls: int = 0
    findpivot_rounds: int = 0

##################################
######### Abstract Class ######### 
##################################

class SPAlgorithm(ABC):
    metrics: BaseMetrics

    def init_metrics(self):
        return BaseMetrics()

    @abstractmethod
    def search(self, graph, src):
        raise NotImplementedError

#################################################
############# Dijkstra's Algorithms #############
#################################################

class Dijkstra(SPAlgorithm):
    def __init__(self, pq):
        self.pq = pq

    def search(self, graph, src):
        ## call init_metrics
        dist = { node: float('inf') for node in graph.nodes }
        dist[src] = 0
        keyNode = { 0 : [src] }
        pq = self.pq

        allKeys = []
        pq.insert(0)
        count = 0
        while not pq.isEmpty():
            current_distance = pq.extractMin()
            current_node = keyNode[current_distance].pop()
            if current_distance > dist[current_node]: continue

            for neighbour, attributes in graph[current_node].items():
                count += 1
                distance = attributes[0]["length"]
                new_distance = current_distance + distance

                if new_distance < dist[neighbour]:
                    dist[neighbour] = new_distance
                    if new_distance not in keyNode:
                        keyNode[new_distance] = []
                    
                    keyNode[new_distance].append(neighbour)
                    pq.insert(new_distance)
                    allKeys.append(new_distance)
        return count

###################################################
############# Bellman-Ford Algorithms #############
###################################################

class BellmanFord(SPAlgorithm):
    def __init__(self):
        super().__init__()

    def search(self, graph, src):
        pass

#################################
############# BMSSP #############
#################################

class BMSSP(SPAlgorithm):
    # config
    def __init__(self, frontier):
        self.frontier_cls = frontier

    def init_metrics(self):
        return BMSSPMetrics()

    def search(self, graph, src):
        ############ Initialisation ############ 

        self.metrics = self.init_metrics()
        self.sources = src
        self.G = graph

        # Distances and predecessors in (possibly transformed) space
        self.dhat: List[Float] = [math.inf] * self.G.n
        self.pred: List[Optional[Vertex]] = [None] * self.G.n
        self.complete: List[bool] = [False] * self.G.n
        self.root: List[int] = [-1] * self.G.n
        for s in self.sources:
            self.dhat[s] = 0.0
            self.complete[s] = True
            self.root[s] = s

        # Parameters (k, t, levels) with safety bounds
        n = max(2, self.G.n)
        # if self.cfg.k_t_auto:
        log2n = math.log2(n)
        k = max(1, int(round(log2n ** (1.0 / 3.0))))
        t = max(1, int(round(log2n ** (2.0 / 3.0))))
        k = max(1, min(k, t))
        # Cap to reasonable values to prevent runaway algorithms
        k = min(k, 100)
        t = min(t, 20)
        # else:
        #     k = max(1, min(self.cfg.k, 100))  # Safety cap
        #     t = max(1, min(self.cfg.t, 20))  # Safety cap
        self.k: int = k
        self.t: int = t
        self.L: int = max(1, min(math.ceil(math.log2(n) / t), 10))  # Cap levels

        # # Best-clone cache for each original vertex after solve()
        # self._best_clone_for_orig: Optional[List[int]] = None
        # # Compressed distances in original space (after solve())
        # self._distances_original: Optional[List[Float]] = None

        ############## Run Algorithm ############
        top_level = self.L
        B = math.inf
        S0 = set(self.sources)
        _Bprime, _U = self._bmssp(top_level, B, S0)
        return AlgResults(distances=self.dhat, predecessors=self.pred)

    ############### Utils ###############

    def _relax(self, u: Vertex, v: Vertex, w: Float) -> bool:
        """Relax edge ``(u, v)``.

        Args:
            u: Tail vertex.
            v: Head vertex.
            w: Edge weight.

        Returns:
            ``True`` if ``v`` improved or tied with its previous distance.
        """
        self.metrics.edges_relaxed += 1
        cand = self.dhat[u] + w
        if cand <= self.dhat[v]:
            if (
                cand < self.dhat[v]
                or self.pred[v] is None
                or (cand == self.dhat[v] and self.root[u] < self.root[v])
            ):
                self.dhat[v] = cand
                self.pred[v] = u
                self.root[v] = self.root[u]
            return True
        return False

    def _weight(self, u: Vertex, v: Vertex) -> Float:
        """Return ``w(u, v)`` by scanning ``u``'s adjacency list."""
        for vv, w in self.G.adj[u]:
            if vv == v:
                return w
        return 0.0

    ############### Base Case ###############

    ## !!!!!!! heap #########
    def _base_case(self, B: Float, S: Set[Vertex]) -> Tuple[Float, Set[Vertex]]:
        """Explore from a single pivot using a bounded Dijkstra search.

        Args:
            B: Upper bound on distances considered.
            S: Set containing exactly one pivot vertex ``x``.

        Returns:
            A tuple ``(B', U)`` where ``U`` are vertices completed with distance
            less than ``B'``.
        """
        # if len(S) != 1:
        #     raise AlgorithmError("BaseCase expects a singleton set.")
        # if self.dhat[x] == math.inf:
        #     raise AlgorithmError("BaseCase requires a finite pivot distance.")

        import heapq

        (x,) = tuple(S)
        U0: List[Vertex] = []
        seen: Set[Vertex] = set()
        heap: List[Tuple[Float, Vertex]] = [(self.dhat[x], x)]
        in_heap: Set[Vertex] = {x}

        # Safety limits to prevent infinite loops
        iterations = 0
        max_iterations = min(self.k * 1000, self.G.n * 10)

        while heap and len(U0) < self.k + 1 and iterations < max_iterations:
            self.metrics.queue_pops += 1
            iterations += 1

            du, u = heapq.heappop(heap)
            in_heap.discard(u)
            if du != self.dhat[u] or u in seen:
                continue
            seen.add(u)
            self.complete[u] = True
            U0.append(u)

            for v, w in self.G.adj[u]:
                if self._relax(u, v, w) and self.dhat[u] + w < B:
                    if v not in in_heap:
                        heapq.heappush(heap, (self.dhat[v], v))
                        in_heap.add(v)

        # if iterations >= max_iterations:
        #     self.counters["iterations_protected"] += 1

        if len(U0) <= self.k:
            return (B, set(U0))
        Bprime = max(self.dhat[v] for v in U0)
        U = {v for v in U0 if self.dhat[v] < Bprime}
        return (Bprime, U)

    ############# Find Pivots #############

    def _find_pivots(self, B: Float, S: Set[Vertex]) -> Tuple[Set[Vertex], Set[Vertex]]:
        """Run ``k`` relax rounds from ``S`` to collect candidate pivots.

        Args:
            B: Distance bound.
            S: Current set of vertices.

        Returns:
            A tuple ``(P, W)`` where ``P`` are chosen pivots and ``W`` is the
            set of vertices reached during the relax rounds.
        """
        W: Set[Vertex] = set(S)
        current: Set[Vertex] = set(S)

        # Safety limits for findpivots
        iterations = 0
        max_iterations = min(self.k * len(S) * 100, self.G.n * 10)

        for round_num in range(1, self.k + 1):
            if iterations >= max_iterations:
                # self.counters["iterations_protected"] += 1
                break

            self.metrics.findpivot_rounds += 1
            nxt: Set[Vertex] = set()

            for u in current:
                iterations += 1
                if iterations >= max_iterations:
                    break

                for v, w in self.G.adj[u]:
                    if self._relax(u, v, w) and (self.dhat[u] + w < B):
                        nxt.add(v)

            if not nxt:
                break
            W |= nxt

            # Early termination if W gets too large
            if len(W) > self.k * max(1, len(S)) * 5:  # More generous limit
                return set(S), W
            current = nxt

        # Build pivot tree with safety limits
        children: Dict[Vertex, List[Vertex]] = {u: [] for u in W}
        for v in W:
            p = self.pred[v]
            if p is not None and p in W and self.dhat[p] + self._weight(p, v) == self.dhat[v]:
                children[p].append(v)

        P: Set[Vertex] = set()
        for u in S:
            size = 0
            stack = [u]
            seen: Set[Vertex] = set()
            iterations = 0
            max_tree_iterations = min(self.k * 10, len(W))

            while stack and iterations < max_tree_iterations:
                iterations += 1
                a = stack.pop()
                if a in seen:
                    continue
                seen.add(a)
                size += 1
                stack.extend(children.get(a, ()))
                if size >= self.k:
                    P.add(u)
                    break
        return P, W

    
    ######## BMSSP ########

    def _make_frontier(self, level: int, B: Float) -> AbstractFrontier:
        # Cap the frontier size to prevent excessive memory usage
        M = max(1, min(2 ** ((level - 1) * self.t), 10000))
        return self.frontier_cls(M=M, B=B)

        # if self.cfg.frontier == "heap":
        #     return HeapFrontier(M=M, B=B)
        # if self.cfg.frontier == "block":
        #     return BlockFrontier(M=M, B=B)
        # raise ConfigError(f"unknown frontier '{self.cfg.frontier}'")

    def _bmssp(
        self, level: int, B: Float, S: Set[Vertex], depth: int = 0
    ) -> Tuple[Float, Set[Vertex]]:
        # Prevent excessive recursion
        if depth > 50 or level > 20:
            return self._base_case(B, S)

        if level == 0:
            return self._base_case(B, S)

        P, W = self._find_pivots(B, S)
        D = self._make_frontier(level, B)
        for x in P:
            D.insert(x, self.dhat[x])

        U_accum: Set[Vertex] = set()
        cap = min(self.k * max(1, 2 ** (level * self.t)), self.G.n)  # Cap to graph size
        pull_iterations = 0
        max_pull_iterations = min(cap * 10, 1000)  # Safety limit on pulls

        while len(U_accum) < cap and pull_iterations < max_pull_iterations:
            self.metrics.block_pulls += 1
            pull_iterations += 1

            S_i, B_i = D.pull()
            if not S_i:
                Bprime = B
                U_accum |= {x for x in W if self.dhat[x] < Bprime}
                for u in U_accum:
                    self.complete[u] = True
                return Bprime, U_accum

            B_i_prime, U_i = self._bmssp(level - 1, B_i, S_i, depth + 1)
            for u in U_i:
                self.complete[u] = True
            U_accum |= U_i

            K_pairs: List[Tuple[Vertex, Float]] = []
            for u in U_i:
                du = self.dhat[u]
                for v, w in self.G.adj[u]:
                    if self._relax(u, v, w):
                        val = du + w
                        if B_i <= val < B:
                            D.insert(v, val)
                        elif B_i_prime <= val < B_i:
                            K_pairs.append((v, val))

            extra_pairs = [(x, self.dhat[x]) for x in S_i if B_i_prime <= self.dhat[x] < B_i]
            if K_pairs or extra_pairs:
                D.batch_prepend(K_pairs + extra_pairs)

            if len(U_accum) >= cap:
                Bprime = B_i_prime
                U_accum |= {x for x in W if self.dhat[x] < Bprime}
                for u in U_accum:
                    self.complete[u] = True
                return Bprime, U_accum

        # if pull_iterations >= max_pull_iterations:
        #     self.counters["iterations_protected"] += 1

        return B, U_accum

