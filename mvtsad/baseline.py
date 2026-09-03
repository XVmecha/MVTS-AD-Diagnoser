"""Frontier baseline runner (design.md section 11).

Sends the frozen system prompt plus serialized evidence to Mistral Large on
La Plateforme, zero-shot or few-shot. Fairness protocol: identical evidence
and schema for every model, one re-prompt on invalid JSON (tracked, reported
separately), temperature 0, prompt frozen in prompts/baseline_system.md
before any results are produced.

Few-shot mode prepends worked examples as prior conversation turns: dev-split
scenes with gold-rendered answers (a shared prefix, so provider-side prompt
caching applies where offered). Dev scenes only; eval scenes are never used
as examples.

Runs are resumable: one JSONL line per scene in runs/, already-answered
scene_ids are skipped on restart.
"""

from __future__ import annotations

import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from mvtsad.check.checker import parse_diagnosis
from mvtsad.dataset import load_split
from mvtsad.render import render_gold, serialize_evidence
from mvtsad.schemas import EvidenceBundle

API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-large-latest"
PROMPT_PATH = Path("prompts/baseline_system.md")


def api_key() -> str:
    import os

    for name in ("MISTRAL_API_KEY", "MISTRAL_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    for line in Path(".env").read_text().splitlines():
        if "=" in line:
            name, _, value = line.partition("=")
            if "mistral" in name.strip().lower():
                return value.strip().strip("'\"")
    raise RuntimeError("no Mistral API key in environment or .env")


def system_prompt() -> str:
    text = PROMPT_PATH.read_text()
    return text.split("<!--PROMPT-START-->")[1].split("<!--PROMPT-END-->")[0].strip()


def _user_text(record: dict) -> str:
    bundle = EvidenceBundle.model_validate(record["evidence"])
    return serialize_evidence(bundle, record["key"]["n_timesteps"])


def few_shot_turns(data_dir: str = "data") -> list[dict]:
    """Worked examples from the dev split: the first scene with both a root
    and an induced event in the key, the first clean scene. Deterministic."""
    records = load_split("dev", data_dir)
    faulted = next(r for r in records if r["key"]["roots"] and r["key"]["induced"])
    clean = next(r for r in records if not r["key"]["roots"])
    turns = []
    for r in (faulted, clean):
        from mvtsad.schemas import AnswerKey

        gold = render_gold(
            AnswerKey.model_validate(r["key"]),
            EvidenceBundle.model_validate(r["evidence"]),
            seed=r["seed"],
        )
        turns.append({"role": "user", "content": _user_text(r)})
        turns.append({"role": "assistant", "content": gold})
    return turns


def build_messages(record: dict, mode: str, shots: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt()}]
    if mode == "few":
        messages += shots
    messages.append({"role": "user", "content": _user_text(record)})
    return messages


def chat(messages: list[dict], key: str, max_retries: int = 6) -> dict:
    body = json.dumps(
        {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 1500}
    ).encode()
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                retry_after = e.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else min(60.0, 2.0**attempt * 2))
                continue
            raise RuntimeError(f"API error {e.code}: {e.read().decode()[:500]}") from e
        except (OSError, http.client.HTTPException):
            # Covers URLError, timeouts, connection resets, bad status lines.
            if attempt < max_retries - 1:
                time.sleep(min(60.0, 2.0**attempt * 2))
                continue
            raise


def _one_scene(record: dict, mode: str, shots: list[dict], key: str) -> dict:
    messages = build_messages(record, mode, shots)
    resp = chat(messages, key)
    text = resp["choices"][0]["message"]["content"]
    reprompted = False
    if parse_diagnosis(text) is None:
        # Fairness protocol: exactly one re-prompt on invalid JSON.
        retry_messages = messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": "Your answer did not contain one valid JSON object in the required schema. Answer again, ending with exactly one valid JSON object.",
            },
        ]
        resp = chat(retry_messages, key)
        text = resp["choices"][0]["message"]["content"]
        reprompted = True
    return {
        "scene_id": record["scene_id"],
        "cell_id": record["cell_id"],
        "mode": mode,
        "model": resp.get("model", MODEL),
        "reprompted": reprompted,
        "text": text,
        "usage": resp.get("usage", {}),
    }


def run_split(
    split: str,
    mode: str,
    limit: int | None = None,
    data_dir: str = "data",
    workers: int = 3,
) -> Path:
    assert mode in ("zero", "few")
    import concurrent.futures
    import threading

    key = api_key()
    shots = few_shot_turns(data_dir) if mode == "few" else []
    out = Path("runs") / f"{split}-{mode}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["scene_id"] for line in out.open()}

    records = [r for r in load_split(split, data_dir) if r["scene_id"] not in done]
    # Deterministic shuffle of PROCESSING order only (results are per-scene and
    # order-independent): the split file is grid-ordered, so without this an
    # interim prefix would cover only the first cells; shuffled, any prefix is
    # approximately stratified and can be scored for an early saturation check.
    import numpy as np

    records = [records[i] for i in np.random.default_rng(0).permutation(len(records))]
    if limit:
        records = records[:limit]
    total_in = total_out = n_done = 0
    lock = threading.Lock()
    with out.open("a") as f, concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(_one_scene, r, mode, shots, key) for r in records]
        for fut in concurrent.futures.as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                # One scene must not kill the pool; resume re-runs it later.
                print(f"scene failed after retries: {e}", flush=True)
                continue
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
                total_in += row["usage"].get("prompt_tokens", 0)
                total_out += row["usage"].get("completion_tokens", 0)
                n_done += 1
                if n_done % 25 == 0:
                    print(f"{split}-{mode}: {n_done}/{len(records)}", flush=True)
    print(f"{split}-{mode} done: tokens in={total_in} out={total_out}", flush=True)
    return out


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "eval"
    modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["zero", "few"]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    for m in modes:
        run_split(split, m, limit)
