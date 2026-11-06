"""Loader tests. Remote APIs are mocked, so these run offline."""
import pandas as pd
import pytest

from ml4t_strategy import data
from ml4t_strategy.data import OHLCV, load_csv, load_prices, load_synthetic


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, responder):
        self.headers, self.calls, self._responder = {}, [], responder

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        return self._responder(url, params)


def install_session(monkeypatch, responder):
    import requests

    session = FakeSession(responder)
    monkeypatch.setattr(requests, "Session", lambda: session)
    return session


def check_layout(df):
    assert list(df.index.names) == ["symbol", "date"]
    assert list(df.columns) == OHLCV
    assert df.index.is_monotonic_increasing


def test_synthetic_layout():
    df = load_synthetic(n_assets=4, n_days=30)
    check_layout(df)
    assert len(df) == 120
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()


def test_csv_roundtrip(tmp_path):
    src = load_synthetic(n_assets=2, n_days=10)
    path = tmp_path / "prices.csv"
    src.reset_index().rename(columns=str.upper).to_csv(path, index=False)
    pd.testing.assert_frame_equal(load_csv(path), src)


def test_fmp_parsing(monkeypatch):
    def responder(url, params):
        sym = params["symbol"]
        return FakeResponse([
            {"symbol": sym, "date": "2024-01-03", "adjOpen": 2, "adjHigh": 3, "adjLow": 1, "adjClose": 2.5, "volume": 10},
            {"symbol": sym, "date": "2024-01-02", "adjOpen": 1, "adjHigh": 2, "adjLow": 1, "adjClose": 1.5, "volume": 20},
        ])

    session = install_session(monkeypatch, responder)
    df = data.load_fmp(["AAA", "BBB"], "2024-01-01", "2024-01-05", api_key="k")
    check_layout(df)
    assert session.headers["apikey"] == "k"
    assert len(df) == 4 and df.loc[("AAA", pd.Timestamp("2024-01-03")), "close"] == 2.5


def test_alpaca_pagination(monkeypatch):
    pages = [
        {"bars": {"AAA": [{"t": "2024-01-02T05:00:00Z", "o": 1, "h": 2, "l": 1, "c": 1.5, "v": 9}]},
         "next_page_token": "p2"},
        {"bars": {"AAA": [{"t": "2024-01-03T05:00:00Z", "o": 2, "h": 3, "l": 2, "c": 2.5, "v": 8}],
                  "BBB": [{"t": "2024-01-03T05:00:00Z", "o": 5, "h": 6, "l": 4, "c": 5.5, "v": 7}]},
         "next_page_token": None},
    ]
    session = install_session(monkeypatch, lambda url, params: FakeResponse(pages.pop(0)))
    df = data.load_alpaca(["AAA", "BBB"], "2024-01-01", "2024-01-05", key_id="id", secret="s")
    check_layout(df)
    assert len(df) == 3
    assert session.calls[1][1]["page_token"] == "p2"
    assert session.calls[0][1]["adjustment"] == "all"
    assert session.headers["APCA-API-KEY-ID"] == "id"


def test_fmp_skips_symbols_outside_plan(monkeypatch, caplog):
    def responder(url, params):
        if params["symbol"] == "PREMIUM":
            return FakeResponse("Premium Query Parameter", 402)
        return FakeResponse([{"symbol": params["symbol"], "date": "2024-01-02", "adjOpen": 1,
                              "adjHigh": 2, "adjLow": 1, "adjClose": 1.5, "volume": 5}])

    install_session(monkeypatch, responder)
    df = data.load_fmp(["AAA", "PREMIUM"], "2024-01-01", "2024-01-05", api_key="k")
    assert df.index.get_level_values("symbol").unique().tolist() == ["AAA"]
    assert "PREMIUM" in caplog.text


def test_auth_error_is_explicit(monkeypatch):
    install_session(monkeypatch, lambda url, params: FakeResponse({"error": "bad key"}, 401))
    with pytest.raises(PermissionError):
        data.load_fmp(["AAA"], "2024-01-01", "2024-01-05", api_key="bad")


def test_missing_key_message(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FMP_API_KEY"):
        data.load_fmp(["AAA"], "2024-01-01", "2024-01-05")


def test_load_prices_caches(monkeypatch, tmp_path):
    calls = []

    def fake_loader(tickers, start, end):
        calls.append(1)
        return load_synthetic(n_assets=2, n_days=5)

    monkeypatch.setitem(data.LOADERS, "fmp", fake_loader)
    a = load_prices("fmp", ["X"], "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    b = load_prices("fmp", ["X"], "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    pd.testing.assert_frame_equal(a, b, check_freq=False)
    assert len(calls) == 1
