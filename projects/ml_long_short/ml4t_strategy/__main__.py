"""Command line entry point: ``python -m ml4t_strategy --help``."""
from __future__ import annotations

import argparse
import json
import logging

from .pipeline import Config, run


def parse_args(argv=None) -> tuple[Config, str]:
    p = argparse.ArgumentParser(description="ML long-short equity strategy: train, predict, backtest.")
    d = Config()
    p.add_argument("--source", default=d.source, choices=["synthetic", "csv", "fmp", "alpaca", "yfinance"])
    p.add_argument("--tickers", nargs="+", help="symbols (default: 50 US large caps)")
    p.add_argument("--start", default=d.start)
    p.add_argument("--end", default=None, help="default: yesterday")
    p.add_argument("--csv-path")
    p.add_argument("--cache-dir", default=d.cache_dir)
    p.add_argument("--horizon", type=int, default=d.horizon, help="forward-return days = rebalance period")
    p.add_argument("--train-length", type=int, default=d.train_length, help="training window in days")
    p.add_argument("--test-length", type=int, default=d.test_length, help="test window in days")
    p.add_argument("--n-splits", type=int, default=None, help="default: as many as the data allows")
    p.add_argument("--n-long", type=int, default=d.n_long)
    p.add_argument("--n-short", type=int, default=d.n_short)
    p.add_argument("--cost-bps", type=float, default=d.cost_bps)
    p.add_argument("--execution-lag", type=int, default=d.execution_lag)
    p.add_argument("--alpaca-feed", default="sip", choices=["sip", "iex"])
    p.add_argument("--n-assets", type=int, default=50, help="synthetic only")
    p.add_argument("--n-days", type=int, default=1500, help="synthetic only")
    p.add_argument("--signal-strength", type=float, default=0.15, help="synthetic only; 0 = pure noise")
    p.add_argument("--output-dir", default="output")
    a = p.parse_args(argv)

    loader_kwargs = {}
    if a.source == "synthetic":
        loader_kwargs = {"n_assets": a.n_assets, "n_days": a.n_days, "signal_strength": a.signal_strength}
    elif a.source == "alpaca":
        loader_kwargs = {"feed": a.alpaca_feed}

    cfg = Config(
        source=a.source, tickers=a.tickers, start=a.start, end=a.end, csv_path=a.csv_path,
        cache_dir=a.cache_dir, horizon=a.horizon, train_length=a.train_length,
        test_length=a.test_length, n_splits=a.n_splits, n_long=a.n_long, n_short=a.n_short,
        cost_bps=a.cost_bps, execution_lag=a.execution_lag, loader_kwargs=loader_kwargs,
    )
    return cfg, a.output_dir


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, output_dir = parse_args(argv)
    summary = run(cfg, output_dir=output_dir)["summary"]
    keys = ["universe_size", "n_folds", "oos_start", "oos_end", "ic_mean", "ic_ir",
            "avg_turnover_per_rebalance", "strategy_net", "benchmark_equal_weight"]
    print(json.dumps({k: summary[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
