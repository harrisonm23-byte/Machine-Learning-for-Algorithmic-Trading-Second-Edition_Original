"""ML-driven long-short equity strategy built on the ML4T (2nd ed.) workflow."""
from .backtest import information_coefficient, long_short_backtest, performance_summary
from .cv import PurgedWalkForwardCV
from .data import load_prices
from .features import build_dataset, compute_features, forward_returns
from .model import walk_forward_predict
from .pipeline import Config, run

__all__ = [
    "Config", "PurgedWalkForwardCV", "build_dataset", "compute_features", "forward_returns",
    "information_coefficient", "load_prices", "long_short_backtest", "performance_summary",
    "run", "walk_forward_predict",
]
