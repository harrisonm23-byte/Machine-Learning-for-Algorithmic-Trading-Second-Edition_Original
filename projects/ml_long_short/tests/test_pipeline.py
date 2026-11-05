import json

from ml4t_strategy.pipeline import Config, run


def test_end_to_end_recovers_planted_signal(tmp_path):
    cfg = Config(
        source="synthetic", horizon=5, train_length=252, test_length=63, n_splits=4,
        n_long=5, n_short=5, loader_kwargs={"n_assets": 30, "n_days": 700, "signal_strength": 0.2},
        lgb_params={"n_estimators": 100},
    )
    result = run(cfg, output_dir=tmp_path)
    summary = result["summary"]
    assert summary["n_folds"] == 4
    assert summary["ic_mean"] > 0.03
    for name in ["summary.json", "predictions.parquet", "daily_returns.csv", "report.png"]:
        assert (tmp_path / name).exists()
    assert json.loads((tmp_path / "summary.json").read_text())["n_folds"] == 4
