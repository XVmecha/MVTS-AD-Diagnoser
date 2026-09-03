"""Local model eval: run a split through an MLX model (base or + adapter).

Same protocol as the frontier baseline: frozen system prompt, serialized
evidence as the user turn, temperature 0, one re-prompt on invalid JSON.
Output rows are report.py-compatible, so every table row (frontier, base,
fine-tuned) flows through the identical scorer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mvtsad.baseline import system_prompt
from mvtsad.check.checker import parse_diagnosis
from mvtsad.dataset import load_split
from mvtsad.render import serialize_evidence
from mvtsad.schemas import EvidenceBundle

REPROMPT = "Your answer did not contain one valid JSON object in the required schema. Answer again, ending with exactly one valid JSON object."


def run_local(
    name: str,
    model_path: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    adapter_path: str | None = None,
    split: str = "eval",
    limit: int | None = None,
    data_dir: str = "data",
    max_tokens: int = 1500,
) -> Path:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    sampler = make_sampler(temp=0.0)
    sys_prompt = system_prompt()

    out = Path("runs") / f"{split}-{name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["scene_id"] for line in out.open()}
    records = [r for r in load_split(split, data_dir) if r["scene_id"] not in done]
    if limit:
        records = records[:limit]

    def gen(messages: list[dict]) -> str:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)

    with out.open("a") as f:
        for i, r in enumerate(records):
            bundle = EvidenceBundle.model_validate(r["evidence"])
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": serialize_evidence(bundle, r["key"]["n_timesteps"])},
            ]
            text = gen(messages)
            reprompted = False
            if parse_diagnosis(text) is None:
                text = gen(
                    messages
                    + [{"role": "assistant", "content": text}, {"role": "user", "content": REPROMPT}]
                )
                reprompted = True
            f.write(
                json.dumps(
                    {
                        "scene_id": r["scene_id"],
                        "cell_id": r["cell_id"],
                        "mode": name,
                        "model": f"{model_path}+{adapter_path}" if adapter_path else model_path,
                        "reprompted": reprompted,
                        "text": text,
                        "usage": {},
                    }
                )
                + "\n"
            )
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"{split}-{name}: {i + 1}/{len(records)}", flush=True)
    print(f"{split}-{name} done", flush=True)
    return out


if __name__ == "__main__":
    # usage: python -m mvtsad.local_eval <name> [adapter_path] [limit]
    name = sys.argv[1]
    adapter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    run_local(name, adapter_path=adapter, limit=limit)
