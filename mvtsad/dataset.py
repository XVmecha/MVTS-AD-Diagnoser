"""Dataset assembly (design.md sections 4.3-4.5).

Stratified over the v1 difficulty grid: SNR x depth x regime dimension, with
fault class balanced within each cell by rotation (class is not a grid axis).
Wiring-id ranges are exclusive per split, so no graph shape ever appears in
two splits; each scene gets a fresh wiring. Scenes are stored as JSONL
records (answer key + frozen evidence bundle) plus a manifest; raw signals
are NOT stored, because generation is deterministic and they can be
regenerated from (wiring_id, cell, seed) at any time.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from mvtsad.check.checker import is_recoverable
from mvtsad.extract.extractor import ExtractorConfig, extract
from mvtsad.gen.scene import GEN_VERSION, DifficultyCell, generate_scene
from mvtsad.schemas import FaultClass

SNR_LEVELS = ("low", "med", "high")
DEPTHS = (0, 1, 2)
REGIME = ("none", "switch", "switch_near")

# Exclusive wiring-id ranges per split (design section 4.4). grpo is
# reserved for the staged GRPO prompt set.
WIRING_RANGES = {"sft": 10_000, "grpo": 20_000, "eval": 30_000, "dev": 40_000}


@dataclass(frozen=True)
class SplitSpec:
    per_cell: int  # faulted scenes per grid cell (27 cells)
    clean: int  # clean scenes, alternating regime switch presence


SPLITS = {
    "dev": SplitSpec(per_cell=6, clean=40),  # 162 + 40 = 202
    "sft": SplitSpec(per_cell=74, clean=500),  # 1998 + 500 = 2498 (20% clean)
    "eval": SplitSpec(per_cell=30, clean=190),  # 810 + 190 = 1000
}

_CLASSES = list(FaultClass)


def grid_cells() -> list[tuple[str, DifficultyCell]]:
    cells = []
    for snr in SNR_LEVELS:
        for depth in DEPTHS:
            for reg in REGIME:
                cells.append(
                    (
                        f"{snr}-d{depth}-{reg}",
                        DifficultyCell(
                            snr=snr,
                            depth=depth,
                            regime_switch=reg != "none",
                            onset_near_switch=reg == "switch_near",
                        ),
                    )
                )
    return cells


def iter_split(name: str, spec: SplitSpec | None = None):
    """Yield (scene_id, cell_id, wiring_id, seed, cell) for a split,
    deterministically."""
    spec = spec or SPLITS[name]
    base = WIRING_RANGES[name]
    idx = 0
    for ci, (cell_id, cell) in enumerate(grid_cells()):
        for k in range(spec.per_cell):
            fc = _CLASSES[(ci + k) % len(_CLASSES)]  # rotate class balance
            yield (
                f"{name}-{idx:05d}",
                cell_id,
                base + idx,
                base + idx,
                DifficultyCell(
                    snr=cell.snr,
                    depth=cell.depth,
                    regime_switch=cell.regime_switch,
                    onset_near_switch=cell.onset_near_switch,
                    fault_class=fc,
                ),
            )
            idx += 1
    for k in range(spec.clean):
        cell = DifficultyCell(clean=True, regime_switch=(k % 2 == 0))
        cell_id = "clean-switch" if k % 2 == 0 else "clean-none"
        yield f"{name}-{idx:05d}", cell_id, base + idx, base + idx, cell
        idx += 1


def build_split(
    name: str,
    out_dir: str | Path = "data",
    spec: SplitSpec | None = None,
    extractor_cfg: ExtractorConfig = ExtractorConfig(),
) -> dict:
    """Generate, extract, and freeze one split. Returns the manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_records = 0
    n_events = n_rec_events = 0
    n_roots = n_rec_roots = 0
    n_items = 0
    per_cell: dict[str, int] = {}

    with open(out / f"{name}.jsonl", "w") as f:
        for scene_id, cell_id, wiring_id, seed, cell in iter_split(name, spec):
            scene = generate_scene(scene_id, name, wiring_id, cell, seed)
            bundle = extract(scene.signals, scene_id, extractor_cfg)
            record = {
                "scene_id": scene_id,
                "cell_id": cell_id,
                "wiring_id": wiring_id,
                "seed": seed,
                "key": scene.key.model_dump(mode="json"),
                "evidence": bundle.model_dump(mode="json"),
            }
            f.write(json.dumps(record, sort_keys=True) + "\n")

            n_records += 1
            n_items += len(bundle.items)
            per_cell[cell_id] = per_cell.get(cell_id, 0) + 1
            for ev in scene.key.roots:
                n_roots += 1
                n_rec_roots += is_recoverable(ev, bundle)
            for ev in list(scene.key.roots) + list(scene.key.induced):
                n_events += 1
                n_rec_events += is_recoverable(ev, bundle)

    manifest = {
        "split": name,
        "gen_version": GEN_VERSION,
        "extractor_config": asdict(extractor_cfg),
        "wiring_range_start": WIRING_RANGES[name],
        "n_records": n_records,
        "per_cell": per_cell,
        "mean_items_per_scene": round(n_items / n_records, 2) if n_records else 0.0,
        "root_recoverable_frac": round(n_rec_roots / n_roots, 4) if n_roots else None,
        "extractor_ceiling": round(1 - n_rec_events / n_events, 4) if n_events else None,
    }
    with open(out / f"{name}.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def load_split(name: str, data_dir: str | Path = "data") -> list[dict]:
    with open(Path(data_dir) / f"{name}.jsonl") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    for split in sys.argv[1:] or ["dev"]:
        m = build_split(split)
        print(json.dumps(m, indent=2, sort_keys=True))
