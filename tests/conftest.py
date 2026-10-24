import pandas as pd
import pytest

from quantlab.data import generate_synthetic_prices

TICKERS = list("ABCDEFGH")


@pytest.fixture(scope="session")
def rw_prices():
    return generate_synthetic_prices(TICKERS, n_days=252 * 15, seed=7, scenario="random_walk")


@pytest.fixture
def small_prices():
    return generate_synthetic_prices(TICKERS, n_days=900, seed=3, scenario="regime")


def make_prices(values: dict, start="2020-01-01"):
    n = len(next(iter(values.values())))
    return pd.DataFrame(values, index=pd.bdate_range(start, periods=n), dtype=float)
