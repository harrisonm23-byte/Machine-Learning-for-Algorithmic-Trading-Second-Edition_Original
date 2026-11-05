"""Price data loaders.

Every loader returns the same layout: a DataFrame indexed by
``(symbol, date)`` with float columns ``open, high, low, close, volume``,
where prices are split- and dividend-adjusted.

Sources
-------
- ``fmp``       Financial Modeling Prep (env ``FMP_API_KEY``)
- ``alpaca``    Alpaca Market Data v2 (env ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``)
- ``yfinance``  Yahoo Finance via the ``yfinance`` package (no key)
- ``csv``       a local file with columns symbol,date,open,high,low,close,volume
- ``synthetic`` simulated prices with a known, persistent alpha (offline tests)

Downloads are cached as Parquet under ``cache_dir`` so repeated runs are fast.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]

# Liquid large caps across sectors; a reasonable default research universe.
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE",
    "CRM", "INTC", "AMD", "QCOM", "TXN", "IBM", "JPM", "BAC", "WFC", "GS",
    "MS", "C", "AXP", "BLK", "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY",
    "TMO", "ABT", "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE",
    "SBUX", "XOM", "CVX", "COP", "CAT", "BA", "GE", "HON", "UPS", "DIS",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise index/columns, sort, drop empty rows."""
    df = df.copy()
    df.index = df.index.set_names(["symbol", "date"])
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["symbol"] = df["symbol"].astype(str)
    df = df.set_index(["symbol", "date"])[OHLCV].astype(float)
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["close"]).sort_index()
    if df.empty:
        raise ValueError("No price data was returned.")
    return df


def _cache_path(cache_dir: Path, source: str, tickers: Sequence[str], start, end) -> Path:
    key = f"{source}|{','.join(sorted(tickers))}|{start}|{end}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return cache_dir / f"{source}_{digest}.parquet"


def _get_json(session, url: str, params: dict, retries: int = 4):
    """GET with exponential backoff on 429/5xx."""
    for attempt in range(retries):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 2 ** (attempt + 1)
            log.warning("HTTP %s from %s; retrying in %ss", resp.status_code, url, wait)
            time.sleep(wait)
            continue
        if resp.status_code in (401, 403):
            raise PermissionError(
                f"HTTP {resp.status_code} from {url}: check the API key and plan. "
                f"Response: {resp.text[:200]}"
            )
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# remote sources
# --------------------------------------------------------------------------- #
FMP_URL = "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"


def load_fmp(tickers: Iterable[str], start: str, end: str, api_key: str | None = None) -> pd.DataFrame:
    """Dividend- and split-adjusted daily bars from Financial Modeling Prep."""
    import requests

    api_key = api_key or os.environ.get("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("Set FMP_API_KEY to use the 'fmp' source.")
    session = requests.Session()
    session.headers["apikey"] = api_key  # FMP accepts the key as a header

    frames = []
    for symbol in tickers:
        rows = _get_json(session, FMP_URL, {"symbol": symbol, "from": start, "to": end})
        if isinstance(rows, dict):  # older payload shape / error message
            rows = rows.get("historical", [])
        if not rows:
            log.warning("FMP returned no rows for %s", symbol)
            continue
        df = pd.DataFrame(rows).rename(
            columns={"adjOpen": "open", "adjHigh": "high", "adjLow": "low", "adjClose": "close"}
        )
        df["symbol"] = symbol
        frames.append(df.set_index(["symbol", "date"]))
    if not frames:
        raise ValueError("FMP returned no data for any ticker.")
    return _finalize(pd.concat(frames))


ALPACA_URL = "https://data.alpaca.markets/v2/stocks/bars"


def load_alpaca(
    tickers: Iterable[str],
    start: str,
    end: str,
    key_id: str | None = None,
    secret: str | None = None,
    feed: str = "sip",
) -> pd.DataFrame:
    """Adjusted daily bars from Alpaca Market Data v2.

    ``feed='sip'`` (consolidated tape) works on free accounts for history older
    than 15 minutes; use ``feed='iex'`` if your plan rejects it.
    """
    import requests

    key_id = key_id or os.environ.get("APCA_API_KEY_ID")
    secret = secret or os.environ.get("APCA_API_SECRET_KEY")
    if not (key_id and secret):
        raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY to use the 'alpaca' source.")
    session = requests.Session()
    session.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret})

    tickers = list(tickers)
    records = []
    for i in range(0, len(tickers), 50):  # keep the symbols query string short
        params = {
            "symbols": ",".join(tickers[i : i + 50]),
            "timeframe": "1Day",
            "start": start,
            "end": end,
            "adjustment": "all",
            "feed": feed,
            "limit": 10000,
        }
        while True:
            payload = _get_json(session, ALPACA_URL, params)
            for symbol, bars in (payload.get("bars") or {}).items():
                for b in bars:
                    records.append((symbol, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]))
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
    if not records:
        raise ValueError("Alpaca returned no data.")
    df = pd.DataFrame(records, columns=["symbol", "date", *OHLCV]).set_index(["symbol", "date"])
    return _finalize(df)


def load_yfinance(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column", threads=True)
    if raw.empty:
        raise ValueError("yfinance returned no data (network blocked or bad tickers?).")
    raw.columns = raw.columns.set_names(["field", "symbol"])
    df = raw.stack(level="symbol").rename(columns=str.lower)
    df = df.swaplevel().sort_index()
    return _finalize(df)


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    missing = {"symbol", "date", *OHLCV} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    return _finalize(df.set_index(["symbol", "date"]))


# --------------------------------------------------------------------------- #
# synthetic source
# --------------------------------------------------------------------------- #
def load_synthetic(
    n_assets: int = 50,
    n_days: int = 1500,
    start: str = "2015-01-02",
    seed: int = 42,
    signal_strength: float = 0.15,
    persistence: float = 0.98,
) -> pd.DataFrame:
    """Simulate prices whose returns contain a slow-moving latent alpha.

    Daily return = beta * market + idio_vol * (signal_strength * alpha + noise),
    with alpha an AR(1) process (``persistence``). Past returns therefore carry
    information about future returns, so a working pipeline should find a
    positive information coefficient. ``signal_strength=0`` gives pure noise,
    which is useful for checking that the backtest does *not* invent alpha.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    symbols = [f"SYN{i:03d}" for i in range(n_assets)]

    market = rng.normal(0.0003, 0.01, n_days)
    beta = rng.uniform(0.6, 1.4, n_assets)
    idio_vol = rng.uniform(0.01, 0.025, n_assets)

    shocks = rng.standard_normal((n_days, n_assets))
    alpha = np.empty_like(shocks)
    alpha[0] = shocks[0]
    scale = np.sqrt(1 - persistence**2)
    for t in range(1, n_days):
        alpha[t] = persistence * alpha[t - 1] + scale * shocks[t]

    # alpha known at t-1 drives the return on day t
    lagged_alpha = np.vstack([np.zeros(n_assets), alpha[:-1]])
    noise = rng.standard_normal((n_days, n_assets))
    rets = market[:, None] * beta + idio_vol * (signal_strength * lagged_alpha + noise)

    close = 100 * np.exp(np.cumsum(np.log1p(rets), axis=0))
    prev_close = np.vstack([close[0], close[:-1]])
    open_ = prev_close * (1 + rng.normal(0, 0.002, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, close.shape)))
    volume = rng.lognormal(mean=14, sigma=0.5, size=close.shape)

    panel = {
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }
    frames = {k: pd.DataFrame(v, index=dates, columns=symbols).stack() for k, v in panel.items()}
    df = pd.DataFrame(frames)
    df.index = df.index.set_names(["date", "symbol"])
    return _finalize(df.swaplevel())


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
LOADERS = {"fmp": load_fmp, "alpaca": load_alpaca, "yfinance": load_yfinance}


def load_prices(
    source: str = "synthetic",
    tickers: Sequence[str] | None = None,
    start: str = "2018-01-01",
    end: str | None = None,
    csv_path: str | Path | None = None,
    cache_dir: str | Path | None = "data_cache",
    refresh: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Load adjusted OHLCV prices from ``source`` (see module docstring)."""
    if source == "synthetic":
        return load_synthetic(**kwargs)
    if source == "csv":
        if csv_path is None:
            raise ValueError("csv source requires csv_path")
        return load_csv(csv_path)
    if source not in LOADERS:
        raise ValueError(f"Unknown source {source!r}; choose from synthetic, csv, {', '.join(LOADERS)}")

    tickers = list(tickers or DEFAULT_TICKERS)
    end = end or (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = _cache_path(cache_dir, source, tickers, start, end)
        if cache_file.exists() and not refresh:
            log.info("Loading cached prices from %s", cache_file)
            return pd.read_parquet(cache_file)

    prices = LOADERS[source](tickers, start, end, **kwargs)
    if cache_file is not None:
        prices.to_parquet(cache_file)
    return prices
