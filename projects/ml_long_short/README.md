# ML long-short equity strategy

A self-contained, tested version of the book's end-to-end ML4T workflow:

```
prices ──► alpha factors ──► LightGBM (purged walk-forward) ──► long-short backtest ──► report
 data.py     features.py        cv.py + model.py                  backtest.py           pipeline.py
```

It condenses chapters 4 (alpha factors), 7/12 (model data, gradient boosting) and 8
(vectorised backtest) into a small package. It runs on modern Python with plain pandas and does
not depend on the book's data store (`assets.h5`), Zipline or TA-Lib.

## Quick start

```bash
cd projects/ml_long_short
pip install -r requirements.txt
python -m ml4t_strategy --source synthetic      # offline, ~15 s
python -m pytest -q                              # 23 tests
```

Results land in `output/`: `summary.json`, `report.png` (equity curve, rolling IC, feature
importance), `predictions.parquet`, `daily_returns.csv`, `daily_ic.csv` and `feature_importance.csv`.

## Data sources

| `--source`  | Credentials (environment variables)          | Notes |
|-------------|----------------------------------------------|-------|
| `fmp`       | `FMP_API_KEY`                                | dividend- and split-adjusted EOD bars, one request per ticker |
| `alpaca`    | `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`     | `adjustment=all`; `--alpaca-feed iex` if your plan rejects `sip` |
| `yfinance`  | none                                         | unofficial Yahoo API |
| `csv`       | none                                         | `--csv-path`, columns `symbol,date,open,high,low,close,volume` |
| `synthetic` | none                                         | prices with a planted persistent alpha (`--signal-strength`, 0 = pure noise) |

Downloads are cached as Parquet in `data_cache/`, keyed by source, tickers and dates.
The default universe is 50 US large caps (`data.DEFAULT_TICKERS`); pass `--tickers` to override.

```bash
export FMP_API_KEY=...
python -m ml4t_strategy --source fmp --start 2016-01-01 --horizon 5 --n-long 10 --n-short 10
```

## What the pipeline does

1. **Features** (`features.py`): returns over 1-63 days, raw and cross-sectionally ranked,
   63-1-week momentum, RSI, Bollinger z-score, PPO, MACD histogram, NATR, 21/63-day volatility,
   dollar-volume rank and calendar dummies.
2. **Target:** the `horizon`-day forward return minus that day's cross-sectional mean, so the
   model ranks stocks against each other instead of timing the market.
3. **Validation** (`cv.py`): rolling walk-forward with a 504-day training window and a 63-day
   test window by default. The last training date is `horizon` trading days before the first
   test date, so its label ends exactly where the test period begins. That is also the data you
   would actually have on that day. The model is refit every fold; all reported results are out of sample.
4. **Portfolio** (`backtest.py`): every `horizon` days, go long the top `n_long` and short the
   bottom `n_short` predictions with equal weights (each leg 100% gross, net 0). The signal from day
   t's close is traded at day t+1's close (`--execution-lag 1`), and costs are charged on turnover
   (`--cost-bps`, default 5 bps one way).
5. **Evaluation:** daily Spearman IC, Sharpe, Sortino, max drawdown, turnover, and an
   equal-weight long-only benchmark.

### Guards against look-ahead bias

These checks are tested in `tests/`:

- Features at date t are identical whether or not data after t exists (`test_features.py`).
  The book's ch. 12 notebook normalises MACD/ATR with full-sample mean and std, which leaks
  future data; this version scales by price instead.
- CV folds are purged by the label horizon (`test_cv.py`).
- A perfect-foresight signal makes money only if it is traded at the same close it was
  computed from; with the default one-day lag, its edge on noise prices disappears. Random
  signals average a Sharpe of about 0 (`test_backtest.py`).

## Caveats before reading too much into a backtest

- **Survivorship bias:** today's large caps were, by construction, winners. Use a
  point-in-time universe for serious research.
- **Multiple testing:** each parameter sweep you run raises the bar for significance (book ch. 8,
  `01_multiple_testing`; deflated Sharpe ratio).
- Shorting costs, borrow availability and market impact are not modelled beyond a flat cost
  per unit of turnover.

## Next steps

- Add fundamental factors from FMP (`/stable/ratios`, `/stable/key-metrics`) with point-in-time lags.
- Tune LightGBM hyperparameters inside the walk-forward loop (book ch. 12, `05_trading_signals_with_lightgbm_and_catboost`).
- Add a `predict_latest` step plus an Alpaca paper-trading order router.
