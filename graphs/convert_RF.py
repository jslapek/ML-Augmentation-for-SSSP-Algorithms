from pathlib import Path
from torch_geometric.datasets import MalNetTiny
import torch

INPUT_ROOT = "./RF_raw"
OUTPUT_DIR = "./RF_5k_gr"
EDGE_WEIGHT = 1 

def write_dimacs_gr(data, out_path, weight=1):
    edge_index = data.edge_index

    if edge_index.numel() == 0:
        # fall back if there are isolated nodes only
        num_nodes = int(data.num_nodes) if data.num_nodes is not None else 0
        num_edges = 0
        edges = []
    else:
        # PyG is 0-based; DIMACS is 1-based
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        edges = [(u + 1, v + 1) for u, v in zip(src, dst)]

        if data.num_nodes is not None:
            num_nodes = int(data.num_nodes)
        else:
            num_nodes = int(edge_index.max().item()) + 1

        num_edges = len(edges)

    with open(out_path, "w", newline="\n") as f:
        f.write(f"p sp {num_nodes} {num_edges}\n")
        for u, v in edges:
            f.write(f"a {u} {v} {weight}\n")


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = MalNetTiny(root=INPUT_ROOT)
    print(f"Loaded {len(ds)} graphs")

    for i, data in enumerate(ds, start=1):
        out_path = out_dir / f"graph_{i}.gr"
        write_dimacs_gr(data, out_path, weight=EDGE_WEIGHT)

        if i % 100 == 0 or i == len(ds):
            print(f"Wrote {i}/{len(ds)}: {out_path}")

    print(f"Done. Files are in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()