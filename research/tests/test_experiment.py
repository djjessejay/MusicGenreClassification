import json
from pathlib import Path

import experiment


def test_quick_end_to_end(tmp_path: Path):
    assert experiment.main(["--quick", "--out", str(tmp_path)]) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert set(res["summary"]) == set(experiment.CONFIGS)
    assert [h["id"] for h in res["hypotheses"]] == [h["id"] for h in experiment.HYPOTHESES]
    for h in res["hypotheses"]:
        assert "verdict" in h and "p_corrected" in h["test"]
    report = (tmp_path / "REPORT.md").read_text()
    assert "## Hypothesen" in report and "Konfusionsmatrix" in report


def test_hypotheses_reference_existing_configs():
    for h in experiment.HYPOTHESES:
        assert h["a"] in experiment.CONFIGS and h["b"] in experiment.CONFIGS and h["a"] != h["b"]
