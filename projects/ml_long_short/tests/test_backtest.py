import numpy as np
import pandas as pd

from ml4t_strategy.backtest import (
    information_coefficient,
    long_short_backtest,
    performance_summary,
    target_weights,
)
from ml4t_strategy.data import load_synthetic
from ml4t_strategy.features import forward_returns


def noise_prices():
    return load_synthetic(n_assets=30, n_days=600, seed=11, signal_strength=0.0)


def test_target_weights_are_dollar_neutral():
    scores = pd.DataFrame(np.random.default_rng(0).standard_normal((5, 12)))
    w = target_weights(scores, n_long=3, n_short=3)
    assert np.allclose(w.sum(axis=1), 0)
    assert np.allclose(w.abs().sum(axis=1), 2)
    assert ((w > 0).sum(axis=1) == 3).all() and ((w < 0).sum(axis=1) == 3).all()


def test_target_weights_flat_when_too_few_names():
    scores = pd.DataFrame([[1.0, 2.0, np.nan, np.nan]])
    assert target_weights(scores, 2, 2).abs().sum().sum() == 0


def test_perfect_foresight_wins_and_lag_removes_it():
    prices = noise_prices()
    fwd1 = forward_returns(prices, 1).dropna()
    same_close = long_short_backtest(fwd1, prices, holding_period=1, cost_bps=0, execution_lag=0)
    next_close = long_short_backtest(fwd1, prices, holding_period=1, cost_bps=0, execution_lag=1)
    assert performance_summary(same_close.returns["net"])["sharpe"] > 20
    # on pure-noise prices yesterday's 1-day-ahead return carries no information
    assert abs(performance_summary(next_close.returns["net"])["sharpe"]) < 1.5


def test_random_signal_has_no_edge_on_average():
    prices = noise_prices()
    sharpes = []
    for seed in range(10):
        pred = pd.Series(np.random.default_rng(seed).standard_normal(len(prices)), index=prices.index)
        bt = long_short_backtest(pred, prices, cost_bps=0)
        sharpes.append(performance_summary(bt.returns["net"])["sharpe"])
    assert abs(np.mean(sharpes)) < 0.6


def test_costs_reduce_returns_by_turnover():
    prices = noise_prices()
    pred = pd.Series(np.random.default_rng(1).standard_normal(len(prices)), index=prices.index)
    free = long_short_backtest(pred, prices, cost_bps=0)
    costly = long_short_backtest(pred, prices, cost_bps=10)
    pd.testing.assert_series_equal(free.returns["gross"], costly.returns["gross"])
    assert costly.returns["cost"].sum() > 0
    assert np.isclose(costly.returns["cost"].sum(), costly.turnover.sum() * 10 / 1e4)


def test_information_coefficient_perfect_and_inverse():
    idx = pd.MultiIndex.from_product([["A", "B", "C", "D"], pd.bdate_range("2021", periods=3)],
                                     names=["symbol", "date"])
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    df = pd.DataFrame({"y_true": y, "y_pred": y})
    assert np.allclose(information_coefficient(df), 1)
    df["y_pred"] = -y
    assert np.allclose(information_coefficient(df), -1)


def test_performance_summary_known_values():
    r = pd.Series([0.01, -0.02, 0.01, 0.0])
    s = performance_summary(r)
    assert np.isclose(s["total_return"], 1.01 * 0.98 * 1.01 - 1)
    assert np.isclose(s["max_drawdown"], -0.02)
    assert np.isclose(s["hit_rate"], 0.5)
