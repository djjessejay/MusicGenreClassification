import json
from pathlib import Path

import experiment


def test_quick_end_to_end(tmp_path: Path):
    assert experiment.main(["--quick", "--out", str(tmp_path)]) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert set(res["summary"]) == set(experiment.CONFIGS)
    assert [h["id"] for h in res["hypotheses"]] == [h["id"] for h in experiment.HYPOTHESES]
    for h in res["hypotheses"]:
        assert "verdict" in h and "p_corrected" in h["test"] and "ci95_corrected" in h["test"]
        assert h["test"]["p_holm"] >= h["test"]["p_corrected"]
    assert sum(res["hyperparams_chosen"]["mlp_all"].values()) == 3
    assert res["meta"]["n_clips"] == 985 and len(res["meta"]["dropped_duplicates"]) == 13
    report = (tmp_path / "REPORT.md").read_text()
    assert "## Hypothesen" in report and "Konfusionsmatrix" in report


def test_quick_without_tuning(tmp_path: Path):
    assert experiment.main(["--quick", "--no-tune", "--out", str(tmp_path)]) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["meta"]["tune"] is False
    for name, cfg in experiment.CONFIGS.items():
        chosen = res["hyperparams_chosen"][name]
        assert sum(chosen.values()) == 3  # 1 x 3-fold
        assert chosen == {f"lr={cfg['lr']:g}, l2={cfg['l2']:g}": 3}


def test_tune_hyperparams_returns_grid_point():
    import numpy as np
    from mgc_jax import load_mgc
    ds = load_mgc().select(experiment.MFCC_FEATURES)
    idx = np.arange(0, len(ds.y), 4)
    grid = [(1e-3, 1e-4), (1e-2, 1e-3)]
    lr, l2 = experiment.tune_hyperparams(ds.X[idx], ds.y[idx], [], ds.n_classes, grid=grid,
                                         inner_folds=2, steps=20, batch_size=32, seed=0)
    assert (lr, l2) in grid


def test_hypotheses_reference_existing_configs():
    for h in experiment.HYPOTHESES:
        assert h["a"] in experiment.CONFIGS and h["b"] in experiment.CONFIGS and h["a"] != h["b"]
