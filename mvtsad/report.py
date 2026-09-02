"""Gate report: score a baseline run against the answer keys, per cell.

Metrics per design.md section 13: strict binary claim accuracy, event recall
(vs recoverable), hallucinated and grounded-but-wrong claims per diagnosis,
grounding rate, attribution, false-alarm rate on clean scenes, parse-failure
and re-prompt rates. The per-cell table IS the failure curve the week-2 gate
reads.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from mvtsad.check.checker import check_diagnosis, parse_diagnosis
from mvtsad.dataset import load_split
from mvtsad.schemas import AnswerKey, EvidenceBundle


def score_run(run_path: str | Path, split: str = "eval", data_dir: str = "data") -> dict:
    records = {r["scene_id"]: r for r in load_split(split, data_dir)}
    per_cell: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    per_class: dict[str, dict] = defaultdict(lambda: defaultdict(float))

    def add(agg: dict, rec: dict, report, reprompted: bool) -> None:
        key = rec["key"]
        agg["scenes"] += 1
        agg["reprompted"] += reprompted
        agg["parse_fail"] += not report.parsed
        if not report.parsed:
            return
        agg["claims"] += len(report.verdicts)
        agg["strict"] += sum(v.strict_correct for v in report.verdicts)
        agg["halluc"] += report.n_hallucinated
        agg["gbw"] += report.n_grounded_but_wrong
        agg["grounding_sum"] += sum(v.grounding for v in report.verdicts)
        agg["recoverable"] += report.n_recoverable
        agg["rec_matched"] += report.n_recoverable_matched
        if key["induced"]:
            agg["attr_scenes"] += 1
            agg["attr_sum"] += report.attribution_score
        if not key["roots"]:
            agg["clean_scenes"] += 1
            agg["false_alarm"] += bool(report.verdicts)

    for line in open(run_path):
        row = json.loads(line)
        rec = records[row["scene_id"]]
        report = check_diagnosis(
            parse_diagnosis(row["text"]),
            AnswerKey.model_validate(rec["key"]),
            EvidenceBundle.model_validate(rec["evidence"]),
        )
        add(per_cell[rec["cell_id"]], rec, report, row["reprompted"])
        fc = rec["key"]["difficulty"].get("fault_class")
        if fc:
            add(per_class[fc], rec, report, row["reprompted"])

    return {"per_cell": dict(per_cell), "per_class": dict(per_class)}


def _row(name: str, a: dict) -> str:
    scenes = int(a["scenes"])
    claims = int(a["claims"])
    cols = [
        name,
        str(scenes),
        f"{a['strict'] / claims:.2f}" if claims else "-",
        f"{a['rec_matched'] / a['recoverable']:.2f}" if a["recoverable"] else "-",
        f"{a['halluc'] / scenes:.2f}",
        f"{a['gbw'] / scenes:.2f}",
        f"{a['grounding_sum'] / claims:.2f}" if claims else "-",
        f"{a['attr_sum'] / a['attr_scenes']:.2f}" if a["attr_scenes"] else "-",
        f"{a['false_alarm'] / a['clean_scenes']:.2f}" if a["clean_scenes"] else "-",
        f"{a['parse_fail'] / scenes:.2f}",
        f"{a['reprompted'] / scenes:.2f}",
    ]
    return "| " + " | ".join(cols) + " |"


def render_table(scored: dict, title: str) -> str:
    header = (
        "| cell | n | strict | recall | halluc/scene | gbw/scene | grounding | "
        "attrib | false-alarm | parse-fail | reprompt |"
    )
    sep = "|" + "---|" * 11
    lines = [f"## {title}", "", header, sep]

    def total(aggs: list[dict]) -> dict:
        t: dict = defaultdict(float)
        for a in aggs:
            for k, v in a.items():
                t[k] += v
        return t

    lines.append(_row("ALL", total(list(scored["per_cell"].values()))))
    for cell in sorted(scored["per_cell"]):
        lines.append(_row(cell, scored["per_cell"][cell]))
    lines += ["", "### By fault class", "", header.replace("cell", "class"), sep]
    for fc in sorted(scored["per_class"]):
        lines.append(_row(fc, scored["per_class"][fc]))
    return "\n".join(lines)


if __name__ == "__main__":
    for run_path in sys.argv[1:]:
        scored = score_run(run_path)
        table = render_table(scored, Path(run_path).stem)
        out = Path(run_path).with_suffix(".report.md")
        out.write_text(table + "\n")
        print(table, "\n")
        print(f"written: {out}")
