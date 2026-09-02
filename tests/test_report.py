"""Report scoring: a synthetic run made of gold completions must score
perfectly; a run of garbage must show parse failures."""

import json

from mvtsad.dataset import SplitSpec, build_split, load_split
from mvtsad.render import render_gold
from mvtsad.report import render_table, score_run
from mvtsad.schemas import AnswerKey, EvidenceBundle


def test_gold_run_scores_clean(tmp_path):
    build_split("dev", tmp_path, spec=SplitSpec(per_cell=1, clean=2))
    records = load_split("dev", tmp_path)
    run = tmp_path / "dev-zero.jsonl"
    with run.open("w") as f:
        for r in records:
            gold = render_gold(
                AnswerKey.model_validate(r["key"]),
                EvidenceBundle.model_validate(r["evidence"]),
                seed=r["seed"],
            )
            f.write(json.dumps({"scene_id": r["scene_id"], "reprompted": False, "text": gold}) + "\n")

    scored = score_run(run, split="dev", data_dir=tmp_path)
    totals = {k: sum(a[k] for a in scored["per_cell"].values()) for k in ("scenes", "claims", "strict", "halluc", "parse_fail", "false_alarm")}
    assert totals["scenes"] == 29
    assert totals["strict"] == totals["claims"] > 0
    assert totals["halluc"] == 0 and totals["parse_fail"] == 0 and totals["false_alarm"] == 0
    table = render_table(scored, "gold")
    assert "| ALL |" in table and "### By fault class" in table


def test_garbage_run_shows_parse_failures(tmp_path):
    build_split("dev", tmp_path, spec=SplitSpec(per_cell=0, clean=2))
    records = load_split("dev", tmp_path)
    run = tmp_path / "dev-zero.jsonl"
    with run.open("w") as f:
        for r in records:
            f.write(json.dumps({"scene_id": r["scene_id"], "reprompted": True, "text": "no json here"}) + "\n")
    scored = score_run(run, split="dev", data_dir=tmp_path)
    a = scored["per_cell"]
    assert sum(c["parse_fail"] for c in a.values()) == 2
