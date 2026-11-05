"""Purged walk-forward cross-validation for panel data.

A cleaned-up version of ``MultipleTimeSeriesCV`` from the repo's ``utils.py``.
Folds are yielded oldest first. Each training window ends early enough that
no training label (a return over the next ``lookahead`` days) overlaps the
test period, which is also exactly the data you would have in live trading
on the first test day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedWalkForwardCV:
    def __init__(
        self,
        train_length: int = 504,
        test_length: int = 63,
        lookahead: int = 5,
        n_splits: int | None = None,
        date_level: str = "date",
    ):
        if min(train_length, test_length, lookahead) < 1:
            raise ValueError("train_length, test_length and lookahead must be >= 1")
        self.train_length = train_length
        self.test_length = test_length
        self.lookahead = lookahead
        self.n_splits = n_splits
        self.date_level = date_level

    def _max_splits(self, n_dates: int) -> int:
        return max(0, (n_dates - self.train_length - self.lookahead + 1) // self.test_length)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        n_max = self._max_splits(X.index.get_level_values(self.date_level).nunique())
        return n_max if self.n_splits is None else min(self.n_splits, n_max)

    def split(self, X: pd.DataFrame, y=None, groups=None):
        dates = X.index.get_level_values(self.date_level)
        unique = np.sort(dates.unique())
        n = len(unique)
        n_splits = self.get_n_splits(X)
        if n_splits == 0:
            raise ValueError(
                f"Only {n} dates: need at least train_length + lookahead + test_length - 1 "
                f"= {self.train_length + self.lookahead + self.test_length - 1}."
            )
        for i in reversed(range(n_splits)):
            test_end = n - i * self.test_length          # exclusive
            test_start = test_end - self.test_length
            train_end = test_start - self.lookahead + 1  # exclusive: purge overlap
            train_start = train_end - self.train_length
            train_mask = (dates >= unique[train_start]) & (dates < unique[train_end])
            test_mask = (dates >= unique[test_start]) & (dates <= unique[test_end - 1])
            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)
