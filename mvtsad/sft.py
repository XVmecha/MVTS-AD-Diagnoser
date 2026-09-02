"""SFT data export: MLX-LM chat format (design.md sections 7 and 10).

Each sample: the frozen baseline system prompt (identical to the frontier
baseline's, so the results table compares weights, not prompts), the
serialized evidence bundle as the user turn, and the template-rendered gold
completion as the assistant turn. Train/valid split by seeded shuffle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from mvtsad.baseline import system_prompt
from mvtsad.dataset import load_split
from mvtsad.render import render_gold, serialize_evidence
from mvtsad.schemas import AnswerKey, EvidenceBundle


def export_sft(
    data_dir: str | Path = "data",
    out_dir: str | Path = "data/sft_chat",
    valid_frac: float = 0.02,
    seed: int = 7,
    limit: int | None = None,
) -> dict:
    records = load_split("sft", data_dir)
    if limit:
        records = records[:limit]
    order = np.random.default_rng(seed).permutation(len(records))
    n_valid = max(1, int(len(records) * valid_frac))
    sys_prompt = system_prompt()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, idxs in (("valid", order[:n_valid]), ("train", order[n_valid:])):
        with open(out / f"{name}.jsonl", "w") as f:
            for i in idxs:
                r = records[int(i)]
                key = AnswerKey.model_validate(r["key"])
                bundle = EvidenceBundle.model_validate(r["evidence"])
                f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": serialize_evidence(bundle, key.n_timesteps)},
                                {"role": "assistant", "content": render_gold(key, bundle, seed=r["seed"])},
                            ]
                        }
                    )
                    + "\n"
                )
        counts[name] = len(idxs)
    return counts


if __name__ == "__main__":
    print(export_sft(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None))
