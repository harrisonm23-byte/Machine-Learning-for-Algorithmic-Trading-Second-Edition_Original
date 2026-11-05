"""End-to-end run: data -> features -> walk-forward model -> backtest -> report."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .backtest import information_coefficient, long_short_backtest, performance_summary
from .cv import PurgedWalkForwardCV
from .data import load_prices
from .features import build_dataset
from .model import walk_forward_predict

log = logging.getLogger(__name__)


@dataclass
class Config:
    # data
    source: str = "synthetic"
    tickers: list[str] | None = None
    start: str = "2018-01-01"
    end: str | None = None
    csv_path: str | None = None
    cache_dir: str = "data_cache"
    # target / validation
    horizon: int = 5
    train_length: int = 504
    test_length: int = 63
    n_splits: int | None = None
    # portfolio
    n_long: int = 10
    n_short: int = 10
    cost_bps: float = 5.0
    execution_lag: int = 1
    # model
    lgb_params: dict = field(default_factory=dict)
    # extra kwargs for the data loader (e.g. synthetic n_assets, alpaca feed)
    loader_kwargs: dict = field(default_factory=dict)


def run(config: Config, output_dir: str | Path | None = "output") -> dict:
    log.info("Loading prices from %s", config.source)
    prices = load_prices(
        source=config.source, tickers=config.tickers, start=config.start, end=config.end,
        csv_path=config.csv_path, cache_dir=config.cache_dir, **config.loader_kwargs,
    )
    n_symbols = prices.index.get_level_values("symbol").nunique()
    log.info("%d rows, %d symbols", len(prices), n_symbols)

    X, y = build_dataset(prices, horizon=config.horizon)
    cv = PurgedWalkForwardCV(
        train_length=config.train_length, test_length=config.test_length,
        lookahead=config.horizon, n_splits=config.n_splits,
    )
    log.info("Training %d walk-forward folds on %d features", cv.get_n_splits(X), X.shape[1])
    wf = walk_forward_predict(X, y, cv, params=config.lgb_params)

    ic = information_coefficient(wf.predictions)
    bt = long_short_backtest(
        wf.predictions["y_pred"], prices,
        n_long=config.n_long, n_short=config.n_short,
        holding_period=config.horizon, cost_bps=config.cost_bps,
        execution_lag=config.execution_lag,
    )

    summary = {
        "config": asdict(config),
        "universe_size": int(n_symbols),
        "n_folds": int(wf.predictions["fold"].nunique()),
        "oos_start": str(wf.predictions.index.get_level_values("date").min().date()),
        "oos_end": str(wf.predictions.index.get_level_values("date").max().date()),
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "ic_ir": float(ic.mean() / ic.std()) if ic.std() > 0 else None,
        "ic_positive_share": float((ic > 0).mean()),
        "avg_turnover_per_rebalance": float(bt.turnover.mean()),
        "strategy_net": performance_summary(bt.returns["net"]),
        "strategy_gross": performance_summary(bt.returns["gross"]),
        "benchmark_equal_weight": performance_summary(bt.returns["benchmark"]),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        wf.predictions.to_parquet(out / "predictions.parquet")
        bt.returns.to_csv(out / "daily_returns.csv")
        ic.rename("ic").to_csv(out / "daily_ic.csv")
        importance = wf.feature_importance.mean(axis=1).sort_values(ascending=False)
        importance.rename("gain").to_csv(out / "feature_importance.csv")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        _plot(bt.returns, ic, importance, out / "report.png")
        log.info("Wrote results to %s", out.resolve())

    return {"summary": summary, "predictions": wf.predictions, "ic": ic, "backtest": bt,
            "feature_importance": wf.feature_importance}


def _plot(returns: pd.DataFrame, ic: pd.Series, importance: pd.Series, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), gridspec_kw={"height_ratios": [3, 2, 3]})

    ax = axes[0]
    equity = (1 + returns[["net", "gross", "benchmark"]]).cumprod() - 1
    equity.rename(columns={"net": "Long-short (net of costs)", "gross": "Long-short (gross)",
                           "benchmark": "Equal-weight universe"}).plot(ax=ax, lw=1.4,
                                                                       color=["#2a6fdb", "#9bb8ec", "#888888"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Cumulative return (out of sample)")
    ax.set_xlabel("")

    ax = axes[1]
    ic.plot(ax=ax, lw=0.5, alpha=0.4, color="#888888", label="Daily IC")
    ic.rolling(63).mean().plot(ax=ax, lw=1.6, color="#2a6fdb", label="63-day mean")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Information coefficient (mean {ic.mean():.3f})")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlabel("")

    ax = axes[2]
    top = importance.head(15)[::-1]
    ax.barh(top.index, top.values / top.values.sum(), color="#2a6fdb")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Feature importance (share of gain, top 15)")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
