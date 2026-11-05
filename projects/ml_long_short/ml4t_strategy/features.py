"""Alpha factors and forward-return targets.

Adapted from chapters 4 and 12 of the book (``04_preparing_the_model_data``),
with two deliberate changes:

* indicators are implemented in pandas, so TA-Lib is not required;
* every feature uses **only data available at the close of day t**. The book's
  MACD/ATR normalisation subtracts the full-sample mean and std, which leaks
  future information; here oscillators are scaled by price or by trailing
  statistics instead. ``tests/test_features.py`` checks this property.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACKS = (1, 5, 10, 21, 42, 63)


def _by_symbol(s: pd.Series):
    return s.groupby(level="symbol", group_keys=False)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (0-100) for a single price series."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def bollinger_z(close: pd.Series, window: int = 20) -> pd.Series:
    """Distance from the moving average in units of rolling std."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mid) / std.replace(0, np.nan)


def ppo(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Percentage price oscillator: (EMA_fast - EMA_slow) / EMA_slow."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    return (ema_fast - ema_slow) / ema_slow


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram scaled by price, so it is comparable across stocks."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return (macd - sig) / close


def natr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Normalised average true range (ATR / close) for one symbol's OHLC."""
    prev_close = df["close"].shift()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return atr / df["close"]


def compute_features(prices: pd.DataFrame, lookbacks=LOOKBACKS) -> pd.DataFrame:
    """Compute the factor panel for ``(symbol, date)``-indexed OHLCV prices."""
    prices = prices.sort_index()
    close = prices["close"]
    by_sym = _by_symbol(close)
    feats = pd.DataFrame(index=prices.index)

    # momentum / reversal at several horizons, raw and cross-sectionally ranked
    for t in lookbacks:
        feats[f"r{t:02d}"] = by_sym.pct_change(t)
    for t in lookbacks:
        feats[f"r{t:02d}_rank"] = feats[f"r{t:02d}"].groupby(level="date").rank(pct=True)
    # 12-1 style momentum: 63-day return skipping the most recent week
    feats["mom_63_5"] = (1 + feats["r63"]) / (1 + feats["r05"]) - 1

    # technical oscillators
    feats["rsi"] = by_sym.transform(rsi)
    feats["bb_z"] = by_sym.transform(bollinger_z)
    feats["ppo"] = by_sym.transform(ppo)
    feats["macd_hist"] = by_sym.transform(macd_hist)
    feats["natr"] = prices.groupby(level="symbol", group_keys=False)[["high", "low", "close"]].apply(natr)

    # risk and liquidity
    feats["vol_21"] = _by_symbol(feats["r01"]).transform(lambda r: r.rolling(21).std())
    feats["vol_63"] = _by_symbol(feats["r01"]).transform(lambda r: r.rolling(63).std())
    dollar_vol = (close * prices["volume"])
    dollar_vol = _by_symbol(dollar_vol).transform(lambda x: x.rolling(21).mean())
    feats["dollar_vol_rank"] = dollar_vol.groupby(level="date").rank(pct=True)

    # calendar
    dates = feats.index.get_level_values("date")
    feats["month"] = dates.month
    feats["weekday"] = dates.weekday

    return feats.replace([np.inf, -np.inf], np.nan)


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.Series:
    """Return from the close of day t to the close of day t + horizon."""
    close = prices["close"].sort_index()
    fwd = _by_symbol(close).pct_change(horizon)
    return _by_symbol(fwd).shift(-horizon).rename(f"fwd_{horizon:02d}")


def build_dataset(
    prices: pd.DataFrame,
    horizon: int = 5,
    demean_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Features ``X`` and target ``y`` with incomplete rows removed.

    With ``demean_target`` the target is the forward return minus that day's
    cross-sectional mean, i.e. the model learns *relative* performance, which
    is what a market-neutral long-short book trades.
    """
    X = compute_features(prices)
    y = forward_returns(prices, horizon)
    if demean_target:
        y = y - y.groupby(level="date").transform("mean")
    data = X.join(y).dropna()
    # work in (symbol, date) order sorted by date for the CV splitter
    data = data.sort_index(level=["date", "symbol"])
    return data.drop(columns=y.name), data[y.name]
