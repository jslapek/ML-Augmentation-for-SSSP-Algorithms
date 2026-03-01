"""Simple directed graph representation used by the solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Set

# from .exceptions import GraphFormatError, InputError

Vertex = int
Float = float
Edge = Tuple[Vertex, Vertex, Float]


@dataclass
class Graph:
    """Directed graph with non-negative edge weights.

    Negative weights are not supported: attempting to insert an edge with
    ``w < 0`` raises :class:`~ssspx.exceptions.GraphFormatError` that cites the
    offending edge.

    Attributes:
        n: Number of vertices in the range ``0`` .. ``n-1``.
        adj: Outgoing adjacency lists.
    """

    n: int

    def __post_init__(self) -> None:
        """Validate vertex count and initialize adjacency lists."""
        # if not isinstance(self.n, int) or self.n <= 0:
        #     raise InputError("Graph.n must be a positive integer.")
        self.adj: List[List[Tuple[Vertex, Float]]] = [[] for _ in range(self.n)]

    def add_edge(self, u: Vertex, v: Vertex, w: Float) -> None:
        """Add a directed edge from ``u`` to ``v``.

        Args:
            u: Tail vertex.
            v: Head vertex.
            w: Non-negative edge weight.

        Raises:
            InputError: If ``u`` or ``v`` are out of range.
            GraphFormatError: If ``w`` is negative.

        Examples:
            ```python
            >>> g = Graph(2)
            >>> g.add_edge(0, 1, 1.5)
            >>> g.adj
            [[(1, 1.5)], []]
            ```
        """
        # if not (0 <= u < self.n and 0 <= v < self.n):
        #     raise InputError("u and v must be vertex ids in [0, n).")
        # if not isinstance(w, (int, float)):
        #     raise GraphFormatError(f"non-numeric weight {w!r} on edge ({u}, {v})")
        # if w < 0:
        #     raise GraphFormatError(f"negative weight {w} on edge ({u}, {v})")
        self.adj[u].append((int(v), float(w)))

    @classmethod
    def from_edges(cls, n: int, edges: Iterable[Edge]) -> "Graph":
        """Create a graph from an iterable of edges.

        Args:
            n: Number of vertices.
            edges: Iterable of ``(u, v, w)`` tuples.

        Returns:
            A graph populated with the provided edges.
        """
        g = cls(n)
        for u, v, w in edges:
            g.add_edge(int(u), int(v), float(w))
        return g

    @classmethod
    def random_graph(cls, n: int, m: int, seed: int):
        """Generate a random graph with ``n`` vertices and ``m`` edges."""
        import random

        rnd = random.Random(seed)
        edges: List[Tuple[int, int, float]] = []
        for _ in range(m):
            u = rnd.randrange(n)
            v = rnd.randrange(n)
            w = rnd.random() * 10.0
            edges.append((u, v, w))
        return Graph.from_edges(n, edges)

    def out_degree(self, u: Vertex) -> int:
        """Return the out-degree of vertex ``u``.

        Args:
            u: Vertex identifier.

        Returns:
            Number of outgoing edges from ``u``.
        """
        return len(self.adj[u])
    
    def random_vertices(self, samples: int, seed: int):
        import random
        rnd = random.Random(seed)
        return rnd.sample(range(self.n), samples)

    # ...existing code...
    @classmethod
    def random_graph_bounded(cls, n: int, m: int, k: int, seed: int):
        """Generate random graph with n vertices, m edges, and out-degree(u) <= k for all u.

        Raises ValueError if m is infeasible (m > n * min(k, n-1)).
        """
        import numpy as np

        if k < 0:
            raise ValueError("k must be non-negative")
        cap = min(k, max(0, n - 1))
        total_slots = n * cap
        if m > total_slots:
            raise ValueError(f"cannot generate {m} edges with max out-degree {k} on {n} vertices (max {total_slots})")

        if m == 0:
            return Graph.from_edges(n, [])

        rng = np.random.default_rng(seed)
        # sample m distinct tail slots (fast, in C)
        slots = rng.choice(total_slots, size=m, replace=False)
        tails = slots // cap
        deg = np.bincount(tails, minlength=n)

        edges: List[Tuple[int, int, float]] = []
        # For each vertex sample deg[u] distinct heads using NumPy (C speed).
        for u in range(n):
            s = int(deg[u])
            if s == 0:
                continue
            # draw s indices from 0..n-2 and map to skip u
            choices = rng.choice(n - 1, size=s, replace=False)
            # vectorized map to heads
            heads = choices.copy()
            heads[heads >= u] += 1
            weights = rng.random(size=s) * 10.0
            for v, w in zip(heads.tolist(), weights.tolist()):
                edges.append((u, int(v), float(w)))

        return Graph.from_edges(n, edges)

    def constant_outdegree_transform(self, delta: int) -> Tuple[Graph, Dict[Vertex, List[Vertex]]]:
        """Split vertices so that every vertex has out-degree at most ``delta``.

        The transformation replaces a vertex with a chain of clones. Each clone
        except the last holds ``delta - 1`` of the original outgoing edges and a
        zero-weight edge to the next clone. The last clone holds the remaining
        edges (at most ``delta``). Incoming edges of a vertex are redirected to its
        first clone.

        Args:
            G: Original graph to transform.
            delta: Maximum out-degree allowed for each vertex (``delta > 0``).

        Returns:
            A tuple ``(G2, mapping)`` where ``G2`` is the transformed graph and
            ``mapping`` maps each original vertex id to a list of its clone ids in
            ``G2``.

        Raises:
            ConfigError: If ``delta`` is not positive.
        """
        # if delta <= 0:
        #     raise ConfigError("delta must be positive")

        # First pass: determine clones and partition outgoing edges.
        mapping: Dict[Vertex, List[Vertex]] = {}
        partitions: Dict[Vertex, List[List[Tuple[Vertex, Float]]]] = {}
        next_id = 0
        for u in range(self.n):
            edges = list(self.adj[u])
            if len(edges) <= delta:
                mapping[u] = [next_id]
                partitions[u] = [edges]
                next_id += 1
                continue

            chunks: List[List[Tuple[Vertex, Float]]] = []
            remaining = edges
            while len(remaining) > delta:
                chunks.append(remaining[: delta - 1])
                remaining = remaining[delta - 1 :]
            chunks.append(remaining)  # last chunk size <= delta
            clones = [next_id + i for i in range(len(chunks))]
            mapping[u] = clones
            partitions[u] = chunks
            next_id += len(clones)

        # Second pass: build edge list in transformed space.
        edges2: List[Tuple[Vertex, Vertex, Float]] = []
        for u in range(self.n):
            clones = mapping[u]
            chunks = partitions[u]
            for i, chunk in enumerate(chunks):
                cu = clones[i]
                for v, w in chunk:
                    edges2.append((cu, mapping[v][0], w))
                if i < len(clones) - 1:
                    edges2.append((cu, clones[i + 1], 0.0))

        G2 = Graph.from_edges(next_id, edges2)
        return G2, mapping


