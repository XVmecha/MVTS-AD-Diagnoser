"""SFT export: format, determinism, and gold validity inside the samples."""

import json

from mvtsad.check.checker import parse_diagnosis
from mvtsad.sft import export_sft


def test_export_format_and_determinism(tmp_path):
    counts = export_sft(out_dir=tmp_path / "a", limit=20)
    assert counts == {"valid": 1, "train": 19}
    export_sft(out_dir=tmp_path / "b", limit=20)
    for name in ("train", "valid"):
        assert (tmp_path / "a" / f"{name}.jsonl").read_bytes() == (tmp_path / "b" / f"{name}.jsonl").read_bytes()
    rows = [json.loads(l) for l in open(tmp_path / "a" / "train.jsonl")]
    for row in rows:
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert parse_diagnosis(row["messages"][2]["content"]) is not None
        assert "Evidence flags:" in row["messages"][1]["content"]
