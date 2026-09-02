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
MIN_SECONDS_PER_REQUEST = 1.05  # free-tier limit is 1 request/second


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
        except (urllib.error.URLError, TimeoutError):
            if attempt < max_retries - 1:
                time.sleep(min(60.0, 2.0**attempt * 2))
                continue
            raise


def run_split(split: str, mode: str, limit: int | None = None, data_dir: str = "data") -> Path:
    assert mode in ("zero", "few")
    key = api_key()
    shots = few_shot_turns(data_dir) if mode == "few" else []
    out = Path("runs") / f"{split}-{mode}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["scene_id"] for line in out.open()}

    records = load_split(split, data_dir)
    if limit:
        records = records[:limit]
    total_in = total_out = 0
    with out.open("a") as f:
        for i, record in enumerate(records):
            if record["scene_id"] in done:
                continue
            t0 = time.time()
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
                time.sleep(MIN_SECONDS_PER_REQUEST)
                resp = chat(retry_messages, key)
                text = resp["choices"][0]["message"]["content"]
                reprompted = True
            usage = resp.get("usage", {})
            total_in += usage.get("prompt_tokens", 0)
            total_out += usage.get("completion_tokens", 0)
            f.write(
                json.dumps(
                    {
                        "scene_id": record["scene_id"],
                        "cell_id": record["cell_id"],
                        "mode": mode,
                        "model": resp.get("model", MODEL),
                        "reprompted": reprompted,
                        "text": text,
                        "usage": usage,
                    }
                )
                + "\n"
            )
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"{split}-{mode}: {i + 1}/{len(records)}", flush=True)
            time.sleep(max(0.0, MIN_SECONDS_PER_REQUEST - (time.time() - t0)))
    print(f"{split}-{mode} done: tokens in={total_in} out={total_out}", flush=True)
    return out


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "eval"
    modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["zero", "few"]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    for m in modes:
        run_split(split, m, limit)
