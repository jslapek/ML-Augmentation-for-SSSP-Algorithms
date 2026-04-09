from pathlib import Path
import random
import csv
from lxml import html

ROOT_HINT = Path(r"./RD_raw")
OUTPUT_DIR = Path(r"./RD_5k_gr")
TOTAL_SAMPLES = 5000
MARKETS = ["AT", "DE", "FI", "GB", "NL", "NO", "SE", "US"]
RNG_SEED = 42
EDGE_WEIGHT = 1


def find_dataset_root(root_hint: Path) -> Path:
    root_hint = root_hint.resolve()
    if (root_hint / "train").exists() and (root_hint / "test").exists():
        return root_hint

    for p in root_hint.rglob("*"):
        if p.is_dir() and (p / "train").exists() and (p / "test").exists():
            return p

    raise FileNotFoundError(f"Could not find Klarna dataset root under {root_hint}")


def tag_name(el) -> str:
    t = getattr(el, "tag", None)
    if not isinstance(t, str):
        return ""
    if "}" in t:
        t = t.split("}", 1)[1]
    return t.lower()


def normal_element_children(el):
    return [child for child in el if isinstance(getattr(child, "tag", None), str)]


def find_html_root(doc):
    root = doc.getroottree().getroot()
    if root is None:
        raise ValueError("No root found")

    if tag_name(root) == "html":
        return root

    for el in root.iter():
        if tag_name(el) == "html":
            return el

    return root


def parse_source_html_to_dom_tree(source_html_path: Path):
    text = source_html_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("Empty source.html")

    parser = html.HTMLParser(recover=True, encoding="utf-8")
    doc = html.document_fromstring(text, parser=parser)
    root = find_html_root(doc)

    edges = []
    next_vid = 1
    stack = [(root, None)]

    while stack:
        node, parent_vid = stack.pop()
        my_vid = next_vid
        next_vid += 1

        if parent_vid is not None:
            edges.append((parent_vid, my_vid))

        children = normal_element_children(node)
        for child in reversed(children):
            stack.append((child, my_vid))

    num_nodes = next_vid - 1
    return num_nodes, edges


def write_dimacs_gr(out_path: Path, num_nodes: int, edges, weight: int = 1):
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"p sp {num_nodes} {len(edges)}\n")
        for u, v in edges:
            f.write(f"a {u} {v} {weight}\n")


def collect_page_dirs(dataset_root: Path):
    grouped = {market: [] for market in MARKETS}

    for split in ["train", "test"]:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue

        for market in MARKETS:
            market_dir = split_dir / market
            if not market_dir.exists():
                continue

            for source_html in market_dir.glob("**/source.html"):
                page_dir = source_html.parent
                rel = page_dir.relative_to(market_dir)

                grouped[market].append({
                    "split": split,
                    "market": market,
                    "page_dir": page_dir,
                    "source_html": source_html,
                    "relative_page_dir": str(rel),
                })

    return grouped


def allocate_counts_even_with_caps(grouped, total_samples):
    """
    Allocate as evenly as possible across markets, but never exceed availability
    and never overshoot total_samples.
    """
    available = {m: len(grouped[m]) for m in MARKETS}

    if sum(available.values()) < total_samples:
        raise RuntimeError(
            f"Not enough pages in total: have {sum(available.values())}, need {total_samples}"
        )

    allocation = {m: 0 for m in MARKETS}
    remaining = total_samples
    active = {m for m in MARKETS if available[m] > 0}

    while remaining > 0 and active:
        fair_share = remaining // len(active)
        if fair_share == 0:
            fair_share = 1

        progressed = False

        for m in sorted(active):
            if remaining == 0:
                break

            room = available[m] - allocation[m]
            if room <= 0:
                continue

            take = min(fair_share, room, remaining)
            if take > 0:
                allocation[m] += take
                remaining -= take
                progressed = True

        active = {m for m in active if allocation[m] < available[m]}

        if not progressed:
            break

    if remaining != 0:
        raise RuntimeError(
            f"Could not allocate {total_samples} samples; short by {remaining}."
        )

    return allocation


def main():
    rng = random.Random(RNG_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_root = find_dataset_root(ROOT_HINT)
    print(f"Dataset root: {dataset_root}")

    grouped = collect_page_dirs(dataset_root)

    for market in MARKETS:
        print(f"{market}: found {len(grouped[market])} candidate pages")
        rng.shuffle(grouped[market])

    allocation = allocate_counts_even_with_caps(grouped, TOTAL_SAMPLES)

    print("\nSampling plan:")
    for market in MARKETS:
        print(f"  {market}: {allocation[market]}")

    manifest_rows = []
    graph_idx = 1

    for market in MARKETS:
        target = allocation[market]
        written = 0
        failed = 0

        for item in grouped[market]:
            if written >= target:
                break

            try:
                num_nodes, edges = parse_source_html_to_dom_tree(item["source_html"])
                if num_nodes <= 0:
                    failed += 1
                    continue

                out_path = OUTPUT_DIR / f"graph_{graph_idx}.gr"
                write_dimacs_gr(out_path, num_nodes, edges, EDGE_WEIGHT)

                manifest_rows.append({
                    "graph_id": graph_idx,
                    "market": item["market"],
                    "split": item["split"],
                    "page_dir": str(item["page_dir"]),
                    "relative_page_dir": item["relative_page_dir"],
                    "source_html": str(item["source_html"]),
                    "num_nodes": num_nodes,
                    "num_edges": len(edges),
                })

                written += 1
                graph_idx += 1

                if written % 50 == 0 or written == target:
                    print(f"{market}: wrote {written}/{target}")

            except Exception as e:
                failed += 1
                print(f"Skipping {item['source_html']} | parse error: {e}")

        if written < target:
            raise RuntimeError(
                f"Could only write {written} graphs for {market}, needed {target}"
            )

        print(f"{market}: wrote {written}, failed/skipped {failed}")

    manifest_path = OUTPUT_DIR / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_id",
                "market",
                "split",
                "page_dir",
                "relative_page_dir",
                "source_html",
                "num_nodes",
                "num_edges",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. Wrote {len(manifest_rows)} graphs to {OUTPUT_DIR.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()