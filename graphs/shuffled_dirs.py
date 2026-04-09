from __future__ import annotations

import random
import shutil
from pathlib import Path


TARGET_SIZE = 5000
SEED = 0
ROOT = Path(__file__).resolve().parent 
OUTPUT_DIRS = {"mix_real_5k", "mix_gen_5k", "mix_all_5k"}


def list_gr_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == ".gr")


def compute_uniform_counts(total: int, sources: list[Path], rng: random.Random) -> dict[Path, int]:
    if not sources:
        raise RuntimeError("No source directories were found.")

    base = total // len(sources)
    remainder = total % len(sources)

    order = list(sources)
    rng.shuffle(order)

    counts = {src: base for src in sources}
    for src in order[:remainder]:
        counts[src] += 1
    return counts


def validate_capacity(counts: dict[Path, int]) -> None:
    for src, need in counts.items():
        have = len(list_gr_files(src))
        if have < need:
            raise RuntimeError(f"{src} has only {have} .gr files, but {need} are required.")


def recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_mix(output_name: str, sources: list[Path], rng_seed: int) -> None:
    rng = random.Random(rng_seed)
    output_dir = ROOT / output_name
    recreate_dir(output_dir)

    counts = compute_uniform_counts(TARGET_SIZE, sources, rng)
    validate_capacity(counts)

    manifest_rows: list[str] = ["new_name\tsource_dir\tsource_file"]
    out_idx = 1

    for src in sorted(sources, key=lambda p: p.name):
        files = list_gr_files(src)
        chosen = rng.sample(files, counts[src])
        rng.shuffle(chosen)

        for file_path in chosen:
            new_name = f"graph_{out_idx}.gr"
            shutil.copy2(file_path, output_dir / new_name)
            manifest_rows.append(f"{new_name}\t{src.name}\t{file_path.name}")
            out_idx += 1

    manifest_path = output_dir / "manifest.tsv"
    manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

    print(f"Created {output_dir} with {out_idx - 1} files")
    for src in sorted(sources, key=lambda p: p.name):
        print(f"  {src.name}: {counts[src]}")


def get_real_sources() -> list[Path]:
    sources = [ROOT / "RD_5k", ROOT / "RF_5k"]
    missing = [str(p) for p in sources if not p.is_dir()]
    if missing:
        raise RuntimeError(f"Missing required real-data directories: {', '.join(missing)}")
    return sources


def get_gen_sources() -> list[Path]:
    sources = [
        p
        for p in sorted(ROOT.iterdir())
        if p.is_dir()
        and p.name.startswith("random")
        and p.name.endswith("5k")
        and p.name not in OUTPUT_DIRS
    ]
    if not sources:
        raise RuntimeError("No random*_5k directories were found for mix_gen_5k.")
    return sources


def get_all_sources() -> list[Path]:
    sources = [
        p
        for p in sorted(ROOT.iterdir())
        if p.is_dir() and p.name.endswith("5k") and p.name not in OUTPUT_DIRS
    ]
    if not sources:
        raise RuntimeError("No *5k directories were found for mix_all_5k.")
    return sources


def main() -> None:
    if not ROOT.is_dir():
        raise RuntimeError(f"Graphs directory not found: {ROOT}")

    real_sources = get_real_sources()
    gen_sources = get_gen_sources()
    all_sources = get_all_sources()

    build_mix("mix_real_5k", real_sources, SEED + 1)
    build_mix("mix_gen_5k", gen_sources, SEED + 2)
    build_mix("mix_all_5k", all_sources, SEED + 3)


if __name__ == "__main__":
    main()
