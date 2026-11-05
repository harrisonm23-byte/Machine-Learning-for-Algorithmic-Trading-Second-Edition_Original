"""Vectorised long-short backtest of model predictions (cf. chapter 8)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.DataFrame   # columns: gross, cost, net, benchmark (daily)
    weights: pd.DataFrame   # dates x symbols, weights *held* each day
    turnover: pd.Series     # sum |delta w| on each trade day


def target_weights(scores: pd.DataFrame, n_long: int, n_short: int) -> pd.DataFrame:
    """Equal-weight the top ``n_long`` (+1/n_long each) and bottom ``n_short``
    (-1/n_short each) names per row. Long and short legs each have 100% gross
    exposure, so the book is dollar neutral with 200% gross."""
    valid = scores.notna().sum(axis=1)
    rank_desc = scores.rank(axis=1, ascending=False, method="first")
    rank_asc = scores.rank(axis=1, ascending=True, method="first")
    enough = valid >= n_long + n_short  # never hold a name in both legs
    longs = rank_desc.le(n_long) & enough.values[:, None]
    shorts = rank_asc.le(n_short) & enough.values[:, None]
    return longs.astype(float) / n_long - shorts.astype(float) / n_short


def long_short_backtest(
    predictions: pd.Series,
    prices: pd.DataFrame,
    n_long: int = 10,
    n_short: int = 10,
    holding_period: int = 5,
    cost_bps: float = 5.0,
    execution_lag: int = 1,
) -> BacktestResult:
    """Simulate a periodically rebalanced long-short portfolio.

    Parameters
    ----------
    predictions : Series indexed by (symbol, date): the score known at the close of ``date``.
    prices : the OHLCV panel; daily close-to-close returns are used.
    holding_period : rebalance every this many trading days.
    cost_bps : one-way cost per unit of traded notional, in basis points.
    execution_lag : days between the signal and the trade. ``1`` (default)
        means the signal from day t's close is traded at day t+1's close,
        which avoids assuming you can trade at the very close you observe.
    """
    scores = predictions.unstack("symbol").sort_index()
    daily_ret = prices["close"].unstack("symbol").sort_index().pct_change()
    daily_ret = daily_ret.reindex(columns=scores.columns)

    rebalance_dates = scores.index[::holding_period]
    targets = target_weights(scores.loc[rebalance_dates], n_long, n_short)

    # positions are entered `execution_lag` trading days after the signal
    calendar = daily_ret.index
    pos = calendar.get_indexer(rebalance_dates) + execution_lag
    keep = pos < len(calendar)
    targets = targets[keep]
    targets.index = calendar[pos[keep]]

    # hold from each trade date until the next one; stop after the last test date
    last = calendar[min(calendar.get_loc(scores.index[-1]) + execution_lag, len(calendar) - 1)]
    trade_window = calendar[(calendar >= targets.index[0]) & (calendar <= last)]
    weights = targets.reindex(trade_window).ffill().fillna(0.0)

    # weights set at the close of day d earn the return of day d+1
    held = weights.shift(1).fillna(0.0)
    period_ret = daily_ret.reindex(trade_window).fillna(0.0)
    gross = (held * period_ret).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = turnover.shift(1).fillna(0.0) * cost_bps / 1e4

    benchmark = daily_ret.reindex(trade_window).mean(axis=1)  # equal-weight universe, long only
    returns = pd.DataFrame(
        {"gross": gross, "cost": cost, "net": gross - cost, "benchmark": benchmark}
    ).iloc[1:]  # first row only opens positions
    return BacktestResult(returns=returns, weights=held.iloc[1:], turnover=turnover[turnover > 0])


def information_coefficient(predictions: pd.DataFrame, pred="y_pred", true="y_true") -> pd.Series:
    """Daily Spearman rank correlation between predictions and realised targets."""
    ranks = predictions[[pred, true]].groupby(level="date").rank()
    return ranks.groupby(level="date").apply(lambda g: g[pred].corr(g[true])).dropna()


def performance_summary(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Standard performance statistics for a daily return series."""
    r = returns.dropna()
    if r.empty:
        return {}
    equity = (1 + r).cumprod()
    years = len(r) / periods_per_year
    ann_ret = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = r.std() * np.sqrt(periods_per_year)
    downside = r[r < 0].std() * np.sqrt(periods_per_year)
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "annual_return": float(ann_ret),
        "annual_volatility": float(ann_vol),
        "sharpe": float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else np.nan,
        "sortino": float(r.mean() * periods_per_year / downside) if downside > 0 else np.nan,
        "max_drawdown": float(max_dd),
        "calmar": float(ann_ret / -max_dd) if max_dd < 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "n_days": int(len(r)),
    }
