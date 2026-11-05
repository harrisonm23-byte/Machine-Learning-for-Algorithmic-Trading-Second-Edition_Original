import numpy as np
import pandas as pd
import pytest

from ml4t_strategy.cv import PurgedWalkForwardCV


def panel(n_dates=300, n_symbols=4):
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([[f"S{i}" for i in range(n_symbols)], dates], names=["symbol", "date"])
    return pd.DataFrame({"x": np.arange(len(idx))}, index=idx)


@pytest.mark.parametrize("lookahead", [1, 5, 21])
def test_folds_are_purged_chronological_and_contiguous(lookahead):
    X = panel()
    cv = PurgedWalkForwardCV(train_length=100, test_length=20, lookahead=lookahead)
    dates = X.index.get_level_values("date")
    unique = np.sort(dates.unique())
    folds = list(cv.split(X))
    assert len(folds) == cv.get_n_splits(X) > 0

    prev_test_end = None
    for train_idx, test_idx in folds:
        train_dates = np.sort(dates[train_idx].unique())
        test_dates = np.sort(dates[test_idx].unique())
        assert len(train_dates) == 100 and len(test_dates) == 20
        # last training label (t, t+lookahead] must end on or before the first test date
        last_train_pos = np.searchsorted(unique, train_dates[-1])
        first_test_pos = np.searchsorted(unique, test_dates[0])
        assert last_train_pos + lookahead == first_test_pos
        if prev_test_end is not None:
            assert np.searchsorted(unique, test_dates[0]) == prev_test_end + 1
        prev_test_end = np.searchsorted(unique, test_dates[-1])
    # the last fold ends on the last date
    assert prev_test_end == len(unique) - 1


def test_n_splits_cap_and_too_little_data():
    X = panel(n_dates=300)
    assert PurgedWalkForwardCV(100, 20, 5, n_splits=3).get_n_splits(X) == 3
    with pytest.raises(ValueError):
        list(PurgedWalkForwardCV(400, 20, 5).split(X))
