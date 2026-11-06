# Notes for Claude

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
