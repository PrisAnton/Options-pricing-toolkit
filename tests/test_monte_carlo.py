import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from options_pricing_toolkit import (
    MarketInputs,
    black_scholes_price,
    monte_carlo_convergence,
    monte_carlo_option,
)


BASE_MARKET = MarketInputs(
    spot=100.0,
    strike=100.0,
    volatility=0.20,
    rate=0.05,
    maturity=1.0,
    dividend_yield=0.02,
)


def test_monte_carlo_call_price_is_close_to_black_scholes_benchmark():
    result = monte_carlo_option("call", BASE_MARKET, paths=50_000, seed=42, antithetic=True)
    benchmark = black_scholes_price("call", BASE_MARKET)

    assert result.paths == 50_000
    assert result.antithetic is True
    assert result.standard_error > 0
    assert abs(result.price - benchmark) < 3.0 * result.standard_error


def test_monte_carlo_put_price_is_close_to_black_scholes_benchmark():
    result = monte_carlo_option("put", BASE_MARKET, paths=50_000, seed=42, antithetic=True)
    benchmark = black_scholes_price("put", BASE_MARKET)

    assert result.paths == 50_000
    assert result.standard_error > 0
    assert abs(result.price - benchmark) < 3.0 * result.standard_error


def test_monte_carlo_convergence_report_has_expected_shape():
    report = monte_carlo_convergence(
        "call",
        BASE_MARKET,
        path_counts=(1_000, 5_000, 20_000),
        seed=42,
        antithetic=True,
    )

    assert report["option_type"] == "call"
    assert math.isclose(report["black_scholes_price"], black_scholes_price("call", BASE_MARKET), rel_tol=1e-12)
    assert [row["paths"] for row in report["runs"]] == [1_000, 5_000, 20_000]
    assert report["runs"][0]["standard_error"] > report["runs"][-1]["standard_error"]


def test_monte_carlo_degenerate_case_returns_closed_form_value():
    market = MarketInputs(
        spot=120.0,
        strike=100.0,
        volatility=0.0,
        rate=0.05,
        maturity=1.0,
        dividend_yield=0.02,
    )
    result = monte_carlo_option("call", market, paths=1_000, seed=42, antithetic=True)
    benchmark = black_scholes_price("call", market)

    assert math.isclose(result.price, benchmark, rel_tol=1e-12)
    assert result.standard_error == 0.0
    assert result.absolute_error == 0.0
