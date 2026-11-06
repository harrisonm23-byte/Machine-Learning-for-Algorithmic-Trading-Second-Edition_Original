# Notes for Claude

## User background and focus

- The user thinks mostly in terms of **technical indicators (book chapter 24, `24_alpha_factor_library`)**
  and **candlestick dynamics and patterns**. Frame ideas and explanations in those terms first.
  Chapter 24 does not cover candlestick patterns, so those need to be added separately.
- Secondary interests: chapter 2 (market and fundamental data), chapter 7 (linear factor models)
  and chapter 9 (time series, volatility forecasting, pairs trading).
- The user has their own **volatility strategy**, explained in the artifact
  https://claude.ai/artifact/VS1UEX42XB2jgZ3z1NADh3 ("Buy High, Sell Low"). The main version is
  **short SPX options, delta-hedged with ES futures**: 30-day options at 15 delta, sold as 2 puts
  and 1 call ("Short (put + strangle)"). Variants on the page: short strangle, iron condor (with
  5-delta wings), long strangle, long put, long call. Its historical odds start from days when the
  VIX is between 12 and 18. The page explains the Greeks (delta, gamma, vega, volga, vanna, gaps).
  It does not define exit rules, re-hedge frequency or position sizing.
- Data check: FMP serves daily `^VIX` and `^GSPC` from 2010 on this plan. `^VIX3M` and `^VIX9D`
  are not available. FMP has no historical option chains.
- They trade on Robinhood (charts) and have FMP and Alpaca API access.
- Work so far: `projects/ml_long_short/` reproduces the book's chapter 4/8/12 LightGBM long-short
  workflow. On FMP data (30 large caps, 2018-2026) it showed no edge after costs. See its README.

## Reading the user's Robinhood chart screenshots

The user often shares Robinhood mobile chart screenshots to show market conditions or
discuss trade ideas. Their chart setup (black background) uses these colors:

### Price pane
| Element | Color | Notes |
|---|---|---|
| Candles | green = up, red/orange-red = down | |
| EMA (9) | blue line | value shown in blue in the header |
| SMA (9) | orange/salmon line | value shown in orange |
| Bollinger Bands (20) | upper and lower bands: blue lines; middle band: pale/white-cyan line; dark navy fill between the bands | header shows Lower / Mid / Upper |
| VWAP | yellow/gold line (the thickest line) | value shown in yellow |
| Volume | green (up) / dark red-brown (down) bars at the bottom of the price pane | |
| Last price | green pill on the right axis with a dotted green horizontal line | |
| Crosshair readout | grey price pill on the right axis, orange pill on the left | appears when the user is touching the chart |

The header row `O H L C V` (orange text) and the indicator values refer to the **latest
candle**, or to the candle under the crosshair if one is shown.

### RSI pane: RSI (14)
- RSI line: pale cyan/light blue.
- Dashed blue lines at 70 (top) and 30 (bottom), with a dark navy fill between them.

### MACD pane: MACD (12, 26, 9)
- Header values in order: MACD line, signal line, histogram.
- MACD line: pale cyan/light blue. Signal line: orange/yellow-orange.
- Histogram bars: green above zero, orange-red below zero.

### Timeframe
- The range buttons (1D, 1W, 1M, 3M, YTD, 1Y, 5Y) set the visible window. The green pill at the
  bottom right sets the candle interval (e.g. `4h`). Check both before reading signals.

### Matching the screenshots with code
The indicator settings (RSI 14, Bollinger 20, MACD 12/26/9) match the defaults in
`projects/ml_long_short/ml4t_strategy/features.py`, so values seen in a screenshot can be
reproduced from FMP data. FMP access works in this cloud environment through a saved network
secret; set `FMP_API_KEY=proxy-injected` when running the loader.
