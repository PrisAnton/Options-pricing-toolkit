# use "python -m pytest tests" to test.
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from options_pricing_toolkit import MarketInputs, black_scholes_greeks

def test_black_scholes_at_the_money_call():
    market = MarketInputs(100, 100, 0.20, 0.05, 1, 0.02)
    result = black_scholes_greeks("call", market)

    assert math.isclose(result.price, 9.2270, rel_tol=1e-4)
    assert math.isclose(result.delta, 0.5869, rel_tol=1e-4)

def test_black_scholes_at_the_money_put():
    market = MarketInputs(100, 100, 0.20, 0.05, 1, 0.02)
    result = black_scholes_greeks("put", market)

    assert math.isclose(result.price, 6.3301, rel_tol=1e-4)
    assert math.isclose(result.delta, -0.3933, rel_tol=1e-4)
