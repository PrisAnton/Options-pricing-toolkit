import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from options_pricing_toolkit import (
    MarketInputs,
    black_scholes_price,
    implied_spot,
    implied_volatility,
)


BASE_MARKET = MarketInputs(
    spot=100.0,
    strike=100.0,
    volatility=0.20,
    rate=0.05,
    maturity=1.0,
    dividend_yield=0.02,
)


def test_implied_volatility_recovers_call_volatility():
    market_price = black_scholes_price("call", BASE_MARKET)
    solved_volatility = implied_volatility("call", market_price, BASE_MARKET)

    assert math.isclose(solved_volatility, BASE_MARKET.volatility, rel_tol=1e-8, abs_tol=1e-8)


def test_implied_volatility_recovers_put_volatility():
    market_price = black_scholes_price("put", BASE_MARKET)
    solved_volatility = implied_volatility("put", market_price, BASE_MARKET)

    assert math.isclose(solved_volatility, BASE_MARKET.volatility, rel_tol=1e-8, abs_tol=1e-8)


def test_implied_spot_recovers_call_spot():
    market_price = black_scholes_price("call", BASE_MARKET)
    solved_spot = implied_spot("call", market_price, BASE_MARKET)

    assert math.isclose(solved_spot, BASE_MARKET.spot, rel_tol=1e-8, abs_tol=1e-8)


def test_implied_volatility_rejects_non_positive_target_price():
    with pytest.raises(ValueError):
        implied_volatility("call", 0.0, BASE_MARKET)
