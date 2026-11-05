import numpy as np
import pandas as pd

from ml4t_strategy.data import load_synthetic
from ml4t_strategy.features import build_dataset, compute_features, forward_returns, rsi


def test_features_use_no_future_data():
    """Features at date t must not change when data after t is removed."""
    prices = load_synthetic(n_assets=8, n_days=300, seed=1)
    cutoff = prices.index.get_level_values("date").unique()[200]
    full = compute_features(prices)
    truncated = compute_features(prices[prices.index.get_level_values("date") <= cutoff])
    full = full[full.index.get_level_values("date") <= cutoff]
    pd.testing.assert_frame_equal(full.sort_index(), truncated.sort_index(), check_exact=False, rtol=1e-9)


def test_forward_returns_alignment():
    prices = load_synthetic(n_assets=3, n_days=50, seed=2)
    fwd = forward_returns(prices, 5)
    close = prices["close"].xs("SYN000")
    expected = close.shift(-5) / close - 1
    pd.testing.assert_series_equal(fwd.xs("SYN000"), expected, check_names=False)


def test_rsi_bounds_and_trend():
    up = pd.Series(np.linspace(100, 200, 60))
    assert rsi(up).dropna().gt(99).all()
    noisy = pd.Series(100 + np.random.default_rng(0).standard_normal(500).cumsum())
    values = rsi(noisy).dropna()
    assert values.between(0, 100).all()


def test_build_dataset_has_no_missing_values_and_demeaned_target():
    prices = load_synthetic(n_assets=10, n_days=200, seed=3)
    X, y = build_dataset(prices, horizon=5)
    assert not X.isna().any().any() and not y.isna().any()
    assert y.groupby(level="date").mean().abs().max() < 1e-12
    assert X.index.equals(y.index)
