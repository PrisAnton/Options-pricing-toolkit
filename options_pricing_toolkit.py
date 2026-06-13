#!/usr/bin/env python3
"""
Options Pricing Toolkit
-----------------------
A small, dependency-free option pricing library. Some useful pricing routines were turned 
into regular Python functions and a lightweight CLI.

Functionalities included:
    - Black-Scholes prices and Greeks
    - implied volatility and implied spot solvers
    - Cox-Ross-Rubinstein / forward / lognormal binomial trees
    - Bermudan-style vesting support in the tree engine
    - Monte Carlo pricing with Black-Scholes convergence benchmarking
    - perpetual American options, Merton jump diffusion, compound options

All rates, dividend yields and volatilities are annualized decimals.
Example: 5% is passed as 0.05, not 5.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

EPS = 1e-12
SQRT_2PI = math.sqrt(2.0 * math.pi)


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class TreeModel(str, Enum):
    # Matches the three binomial choices in the source workbook.
    FORWARD = "forward"
    CRR = "crr"
    LOGNORMAL = "lognormal"


@dataclass(frozen=True)
class MarketInputs:
    spot: float
    strike: float
    volatility: float
    rate: float
    maturity: float
    dividend_yield: float = 0.0


@dataclass(frozen=True)
class BlackScholesResult:
    price: float
    delta: float
    gamma: float
    vega: float      # price change for a 1 vol-point move
    rho: float       # price change for a 1 rate-point move
    theta: float     # daily theta
    psi: float       # sensitivity to dividend yield, per 1 dividend-point move
    elasticity: float


@dataclass(frozen=True)
class BinomialResult:
    price: float
    delta: float
    gamma: float
    theta: float     # daily theta
    risk_neutral_up_probability: float
    up_factor: float
    down_factor: float
    steps: int
    model: str
    exercise_style: str


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_interval_95: Tuple[float, float]
    paths: int
    antithetic: bool
    seed: int
    black_scholes_price: float
    absolute_error: float
    relative_error: float


@dataclass(frozen=True)
class CompoundResult:
    price: float
    critical_spot: float


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def poisson_pmf(k: int, lam: float) -> float:
    if k < 0 or lam < 0:
        raise ValueError("Poisson inputs must be non-negative")
    if lam == 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1.0))


def validate_market_inputs(m: MarketInputs) -> None:
    if m.spot <= 0:
        raise ValueError("spot must be positive")
    if m.strike <= 0:
        raise ValueError("strike must be positive")
    if m.volatility < 0:
        raise ValueError("volatility cannot be negative")
    if m.maturity < 0:
        raise ValueError("maturity cannot be negative")


def _d1(m: MarketInputs) -> float:
    if m.volatility <= 0 or m.maturity <= 0:
        raise ValueError("d1 is undefined when volatility or maturity is zero")
    carry = m.rate - m.dividend_yield
    return (math.log(m.spot / m.strike) + (carry + 0.5 * m.volatility**2) * m.maturity) / (
        m.volatility * math.sqrt(m.maturity)
    )


def _d2(m: MarketInputs) -> float:
    return _d1(m) - m.volatility * math.sqrt(m.maturity)


def intrinsic_value(option_type: OptionType | str, spot: float, strike: float) -> float:
    option_type = OptionType(option_type)
    if option_type == OptionType.CALL:
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def black_scholes_price(option_type: OptionType | str, m: MarketInputs) -> float:
    """European option price with continuous dividend yield."""
    validate_market_inputs(m)
    option_type = OptionType(option_type)

    if m.maturity <= EPS or m.volatility * math.sqrt(max(m.maturity, 0.0)) <= EPS:
        forward_intrinsic = intrinsic_value(
            option_type,
            m.spot * math.exp((m.rate - m.dividend_yield) * m.maturity),
            m.strike,
        )
        return math.exp(-m.rate * m.maturity) * forward_intrinsic

    d1 = _d1(m)
    d2 = d1 - m.volatility * math.sqrt(m.maturity)
    df_r = math.exp(-m.rate * m.maturity)
    df_q = math.exp(-m.dividend_yield * m.maturity)

    if option_type == OptionType.CALL:
        return m.spot * df_q * norm_cdf(d1) - m.strike * df_r * norm_cdf(d2)
    return m.strike * df_r * norm_cdf(-d2) - m.spot * df_q * norm_cdf(-d1)



def _sample_standard_error(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def monte_carlo_option(
    option_type: OptionType | str,
    m: MarketInputs,
    paths: int = 100_000,
    seed: int = 42,
    antithetic: bool = True,
) -> MonteCarloResult:
    """European option price by risk-neutral Monte Carlo simulation."""
    validate_market_inputs(m)
    option_type = OptionType(option_type)
    if paths <= 0:
        raise ValueError("paths must be positive")

    benchmark = black_scholes_price(option_type, m)
    if m.maturity <= EPS or m.volatility <= EPS:
        return MonteCarloResult(
            price=benchmark,
            standard_error=0.0,
            confidence_interval_95=(benchmark, benchmark),
            paths=paths,
            antithetic=antithetic,
            seed=seed,
            black_scholes_price=benchmark,
            absolute_error=0.0,
            relative_error=0.0,
        )

    rng = random.Random(seed)
    drift = (m.rate - m.dividend_yield - 0.5 * m.volatility**2) * m.maturity
    vol_step = m.volatility * math.sqrt(m.maturity)
    discount = math.exp(-m.rate * m.maturity)
    direction = 1.0 if option_type == OptionType.CALL else -1.0

    observations: List[float] = []
    if antithetic:
        pair_count = (paths + 1) // 2
        for _ in range(pair_count):
            z = rng.gauss(0.0, 1.0)
            st_up = m.spot * math.exp(drift + vol_step * z)
            st_down = m.spot * math.exp(drift - vol_step * z)
            payoff_up = max(direction * (st_up - m.strike), 0.0)
            payoff_down = max(direction * (st_down - m.strike), 0.0)
            observations.append(0.5 * (payoff_up + payoff_down))
        actual_paths = pair_count * 2
    else:
        for _ in range(paths):
            z = rng.gauss(0.0, 1.0)
            st = m.spot * math.exp(drift + vol_step * z)
            observations.append(max(direction * (st - m.strike), 0.0))
        actual_paths = paths

    mean_payoff = sum(observations) / len(observations)
    price = discount * mean_payoff
    standard_error = discount * _sample_standard_error(observations)
    ci_low = price - 1.96 * standard_error
    ci_high = price + 1.96 * standard_error
    absolute_error = abs(price - benchmark)
    relative_error = absolute_error / benchmark if benchmark > EPS else 0.0

    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_interval_95=(ci_low, ci_high),
        paths=actual_paths,
        antithetic=antithetic,
        seed=seed,
        black_scholes_price=benchmark,
        absolute_error=absolute_error,
        relative_error=relative_error,
    )


def monte_carlo_convergence(
    option_type: OptionType | str,
    m: MarketInputs,
    path_counts: Sequence[int] = (1_000, 5_000, 10_000, 50_000, 100_000),
    seed: int = 42,
    antithetic: bool = True,
) -> Dict[str, Any]:
    """Compare Monte Carlo estimates with the Black-Scholes closed-form price."""
    if not path_counts:
        raise ValueError("path_counts cannot be empty")
    if any(count <= 0 for count in path_counts):
        raise ValueError("all path counts must be positive")

    benchmark = black_scholes_price(option_type, m)
    rows: List[Dict[str, Any]] = []
    for i, count in enumerate(path_counts):
        result = monte_carlo_option(option_type, m, paths=count, seed=seed + i, antithetic=antithetic)
        rows.append(
            {
                "paths": result.paths,
                "estimate": result.price,
                "standard_error": result.standard_error,
                "confidence_interval_95": result.confidence_interval_95,
                "absolute_error": result.absolute_error,
                "relative_error": result.relative_error,
            }
        )

    return {
        "option_type": OptionType(option_type).value,
        "black_scholes_price": benchmark,
        "antithetic": antithetic,
        "seed": seed,
        "runs": rows,
    }


def black_scholes_greeks(option_type: OptionType | str, m: MarketInputs) -> BlackScholesResult:
    """Return price and the Greeks used most often on an options desk."""
    validate_market_inputs(m)
    option_type = OptionType(option_type)
    price = black_scholes_price(option_type, m)

    if m.maturity <= EPS or m.volatility <= EPS:
        if option_type == OptionType.CALL:
            delta = 1.0 if m.spot > m.strike else 0.0
        else:
            delta = -1.0 if m.spot < m.strike else 0.0
        return BlackScholesResult(
            price=price,
            delta=delta,
            gamma=0.0,
            vega=0.0,
            rho=0.0,
            theta=0.0,
            psi=0.0,
            elasticity=(delta * m.spot / price) if price > EPS else 0.0,
        )

    d1 = _d1(m)
    d2 = _d2(m)
    sqrt_t = math.sqrt(m.maturity)
    df_r = math.exp(-m.rate * m.maturity)
    df_q = math.exp(-m.dividend_yield * m.maturity)

    gamma = df_q * norm_pdf(d1) / (m.spot * m.volatility * sqrt_t)
    vega = df_q * m.spot * sqrt_t * norm_pdf(d1) / 100.0

    if option_type == OptionType.CALL:
        delta = df_q * norm_cdf(d1)
        rho = m.maturity * m.strike * df_r * norm_cdf(d2) / 100.0
        theta = (
            m.dividend_yield * df_q * m.spot * norm_cdf(d1)
            - m.rate * df_r * m.strike * norm_cdf(d2)
            - m.strike * df_r * m.volatility * norm_pdf(d2) / (2.0 * sqrt_t)
        ) / 365.0
        psi = -m.maturity * m.spot * df_q * norm_cdf(d1) / 100.0
    else:
        delta = df_q * (norm_cdf(d1) - 1.0)
        rho = -m.maturity * m.strike * df_r * norm_cdf(-d2) / 100.0
        theta = (
            -m.dividend_yield * df_q * m.spot * norm_cdf(-d1)
            + m.rate * df_r * m.strike * norm_cdf(-d2)
            - m.strike * df_r * m.volatility * norm_pdf(d2) / (2.0 * sqrt_t)
        ) / 365.0
        psi = m.maturity * m.spot * df_q * norm_cdf(-d1) / 100.0

    elasticity = delta * m.spot / price if price > EPS else 0.0
    return BlackScholesResult(price, delta, gamma, vega, rho, theta, psi, elasticity)


def implied_volatility(
    option_type: OptionType | str,
    target_price: float,
    m: MarketInputs,
    initial_guess: float = 0.20,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Solve implied volatility with Newton first, then a bracketed fallback."""
    option_type = OptionType(option_type)
    if target_price <= 0:
        raise ValueError("target_price must be positive")

    vol = max(initial_guess, m.volatility, 0.05)
    for _ in range(max_iterations):
        trial = MarketInputs(m.spot, m.strike, vol, m.rate, m.maturity, m.dividend_yield)
        price = black_scholes_price(option_type, trial)
        diff = price - target_price
        if abs(diff) < tolerance:
            return vol
        vega = black_scholes_greeks(option_type, trial).vega * 100.0
        if abs(vega) < EPS:
            break
        next_vol = vol - diff / vega
        if not math.isfinite(next_vol) or next_vol <= 0:
            break
        vol = next_vol

    low, high = 1e-8, max(1.0, vol * 2.0)
    while black_scholes_price(option_type, MarketInputs(m.spot, m.strike, high, m.rate, m.maturity, m.dividend_yield)) < target_price:
        high *= 2.0
        if high > 10.0:
            raise RuntimeError("could not bracket implied volatility")

    for _ in range(max_iterations * 2):
        mid = 0.5 * (low + high)
        trial = MarketInputs(m.spot, m.strike, mid, m.rate, m.maturity, m.dividend_yield)
        price = black_scholes_price(option_type, trial)
        if abs(price - target_price) < tolerance:
            return mid
        if price < target_price:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def implied_spot(
    option_type: OptionType | str,
    target_price: float,
    m: MarketInputs,
    tolerance: float = 1e-8,
    max_iterations: int = 120,
) -> float:
    """Back out the spot level that matches a quoted Black-Scholes price."""
    option_type = OptionType(option_type)
    if target_price <= 0:
        raise ValueError("target_price must be positive")

    low = EPS
    high = max(m.spot, m.strike, 1.0)

    def priced_at(spot: float) -> float:
        trial = MarketInputs(spot, m.strike, m.volatility, m.rate, m.maturity, m.dividend_yield)
        return black_scholes_price(option_type, trial)

    if option_type == OptionType.CALL:
        while priced_at(high) < target_price:
            high *= 2.0
            if high > 1e9:
                raise RuntimeError("could not bracket implied spot")
    else:
        high = max(high, m.strike * 4.0)
        # A put price falls as spot rises. The lower bracket carries the high price.
        if priced_at(low) < target_price:
            raise RuntimeError("target put price is above the model's maximum value")

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        price = priced_at(mid)
        if abs(price - target_price) < tolerance:
            return mid
        if option_type == OptionType.CALL:
            if price < target_price:
                low = mid
            else:
                high = mid
        else:
            if price > target_price:
                low = mid
            else:
                high = mid
    return 0.5 * (low + high)


def _tree_factors(m: MarketInputs, steps: int, model: TreeModel) -> Tuple[float, float, float, float, float]:
    h = m.maturity / steps
    rhat = math.exp(m.rate * h)
    dhat = math.exp(m.dividend_yield * h)
    ahat = rhat / dhat

    if model == TreeModel.FORWARD:
        up = math.exp((m.rate - m.dividend_yield) * h + m.volatility * math.sqrt(h))
        down = math.exp((m.rate - m.dividend_yield) * h - m.volatility * math.sqrt(h))
    elif model == TreeModel.CRR:
        up = math.exp(m.volatility * math.sqrt(h))
        down = 1.0 / up
    elif model == TreeModel.LOGNORMAL:
        drift = m.rate - m.dividend_yield - 0.5 * m.volatility**2
        up = math.exp(drift * h + m.volatility * math.sqrt(h))
        down = math.exp(drift * h - m.volatility * math.sqrt(h))
    else:
        raise ValueError(f"unknown tree model: {model}")

    if abs(up - down) < EPS:
        raise ValueError("tree has collapsed; increase volatility or maturity")
    if not (down <= ahat <= up):
        raise ValueError("inputs violate the binomial no-arbitrage condition")

    p_up = (ahat - down) / (up - down)
    return up, down, p_up, rhat, dhat


def binomial_option(
    option_type: OptionType | str,
    m: MarketInputs,
    steps: int = 100,
    exercise_style: ExerciseStyle | str = ExerciseStyle.AMERICAN,
    model: TreeModel | str = TreeModel.CRR,
    vesting_time: float = 0.0,
) -> BinomialResult:
    """Price an option on a recombining binomial tree."""
    validate_market_inputs(m)
    option_type = OptionType(option_type)
    exercise_style = ExerciseStyle(exercise_style)
    model = TreeModel(model)

    if steps < 3:
        raise ValueError("steps must be at least 3 so delta/gamma/theta can be estimated")
    if m.maturity <= 0:
        value = intrinsic_value(option_type, m.spot, m.strike)
        return BinomialResult(value, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, steps, model.value, exercise_style.value)

    h = m.maturity / steps
    vesting_time = min(max(vesting_time, 0.0), max(m.maturity - 1e-8, 0.0))
    up, down, p_up, rhat, dhat = _tree_factors(m, steps, model)
    p_down = 1.0 - p_up
    direction = 1.0 if option_type == OptionType.CALL else -1.0

    stock = [m.spot * (up**i) * (down ** (steps - i)) for i in range(steps + 1)]
    values = [max(0.0, direction * (s - m.strike)) for s in stock]
    w1: Optional[List[float]] = None
    w2: Optional[List[float]] = None

    for period in range(steps - 1, -1, -1):
        new_stock: List[float] = []
        new_values: List[float] = []
        for i in range(period + 1):
            s_node = stock[i + 1] / up
            continuation = (p_up * values[i + 1] + p_down * values[i]) / rhat
            value = continuation
            if exercise_style == ExerciseStyle.AMERICAN and vesting_time <= period * h:
                value = max(value, direction * (s_node - m.strike), 0.0)
            new_stock.append(s_node)
            new_values.append(value)
        stock = new_stock
        values = new_values
        if period == 2:
            w2 = values[:]
        elif period == 1:
            w1 = values[:]

    if w1 is None or w2 is None:
        # Guard kept for readability; the steps >= 3 check should make this unreachable.
        raise RuntimeError("tree did not store enough levels for Greek estimates")

    price = values[0]
    delta = (w1[1] - w1[0]) / (m.spot * (up - down)) / dhat
    del1 = (w2[2] - w2[1]) / (m.spot * (up**2 - up * down)) / dhat
    del2 = (w2[1] - w2[0]) / (m.spot * (up * down - down**2)) / dhat
    gamma = (del1 - del2) / (m.spot * (up - down))

    # The forward-style tree does not always center exactly at S after two steps.
    # This adjustment keeps theta comparable across the three tree variants.
    epsilon = m.spot * (up * down - 1.0)
    price_at_center = w2[1] - (epsilon * delta + 0.5 * epsilon * epsilon * gamma)
    theta = (price_at_center - price) / (2.0 * h * 365.0)

    return BinomialResult(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta,
        risk_neutral_up_probability=p_up,
        up_factor=up,
        down_factor=down,
        steps=steps,
        model=model.value,
        exercise_style=exercise_style.value,
    )


def binomial_lattice(
    option_type: OptionType | str,
    m: MarketInputs,
    steps: int = 5,
    exercise_style: ExerciseStyle | str = ExerciseStyle.AMERICAN,
    model: TreeModel | str = TreeModel.CRR,
) -> List[List[Dict[str, float]]]:
    """Return a small stock/option lattice for inspection or reporting."""
    if steps < 1 or steps > 25:
        raise ValueError("lattice display is intended for 1..25 steps")
    option_type = OptionType(option_type)
    exercise_style = ExerciseStyle(exercise_style)
    model = TreeModel(model)
    direction = 1.0 if option_type == OptionType.CALL else -1.0
    up, down, p_up, rhat, _ = _tree_factors(m, steps, model)
    p_down = 1.0 - p_up

    stock_levels: List[List[float]] = []
    for period in range(steps + 1):
        stock_levels.append([m.spot * (up**i) * (down ** (period - i)) for i in range(period + 1)])

    value_levels: List[List[float]] = [[0.0 for _ in range(period + 1)] for period in range(steps + 1)]
    value_levels[steps] = [max(0.0, direction * (s - m.strike)) for s in stock_levels[steps]]

    for period in range(steps - 1, -1, -1):
        for i in range(period + 1):
            continuation = (p_up * value_levels[period + 1][i + 1] + p_down * value_levels[period + 1][i]) / rhat
            value = continuation
            if exercise_style == ExerciseStyle.AMERICAN:
                value = max(value, direction * (stock_levels[period][i] - m.strike), 0.0)
            value_levels[period][i] = value

    lattice: List[List[Dict[str, float]]] = []
    for period in range(steps + 1):
        lattice.append(
            [
                {"stock": stock_levels[period][i], "option": value_levels[period][i]}
                for i in range(period + 1)
            ]
        )
    return lattice


def perpetual_american(option_type: OptionType | str, spot: float, strike: float, volatility: float, rate: float, dividend_yield: float) -> Tuple[float, float]:
    """Perpetual American option value and exercise boundary."""
    option_type = OptionType(option_type)
    if spot <= 0 or strike <= 0 or volatility <= 0:
        raise ValueError("spot, strike and volatility must be positive")

    root_term = math.sqrt(((rate - dividend_yield) / volatility**2 - 0.5) ** 2 + 2.0 * rate / volatility**2)
    if option_type == OptionType.CALL:
        beta = 0.5 - (rate - dividend_yield) / volatility**2 + root_term
        boundary = strike * beta / (beta - 1.0)
        value = spot - strike if spot > boundary else (boundary - strike) * (spot / boundary) ** beta
    else:
        beta = 0.5 - (rate - dividend_yield) / volatility**2 - root_term
        boundary = strike * beta / (beta - 1.0)
        value = strike - spot if spot < boundary else (strike - boundary) * (spot / boundary) ** beta
    return value, boundary


def merton_jump_diffusion(
    m: MarketInputs,
    jump_intensity: float,
    mean_jump_log: float,
    jump_volatility: float,
    tolerance: float = 1e-8,
    max_terms: int = 250,
) -> Dict[str, float]:
    """Merton jump-diffusion call and put prices via a Poisson mixture."""
    if jump_intensity < 0 or jump_volatility < 0:
        raise ValueError("jump inputs must be non-negative")
    validate_market_inputs(m)

    adjusted_intensity = jump_intensity * math.exp(mean_jump_log)
    kappa = math.exp(mean_jump_log) - 1.0
    call = 0.0

    for i in range(max_terms):
        prob = poisson_pmf(i, adjusted_intensity * m.maturity)
        vol_i = math.sqrt(m.volatility**2 + (i * jump_volatility**2 / m.maturity if m.maturity > EPS else 0.0))
        rate_i = m.rate - jump_intensity * kappa + (i * mean_jump_log / m.maturity if m.maturity > EPS else 0.0)
        trial = MarketInputs(m.spot, m.strike, vol_i, rate_i, m.maturity, m.dividend_yield)
        call += prob * black_scholes_price(OptionType.CALL, trial)
        if i > 0 and prob < tolerance:
            break
    else:
        raise RuntimeError("Merton series did not converge")

    put = call + m.strike * math.exp(-m.rate * m.maturity) - m.spot * math.exp(-m.dividend_yield * m.maturity)
    return {"call": call, "put": put, "terms_used": i + 1}


def biv_norm_cdf(x: float, y: float, rho: float, terms: int = 35) -> float:
    """Bivariate standard-normal CDF by the Taylor expansion used in the workbook."""
    if not -1.0 < rho < 1.0:
        if rho >= 1.0:
            return norm_cdf(min(x, y))
        return max(norm_cdf(x) - norm_cdf(-y), 0.0)

    # Coefficients for derivatives of the standard normal density.
    coeffs = [[0.0 for _ in range(terms + 1)] for _ in range(terms + 1)]
    coeffs[0][0] = 1.0
    for i in range(1, terms):
        coeffs[i][0] = coeffs[i - 1][1]
        for j in range(i, 0, -2):
            coeffs[i][j] = (j + 1) * coeffs[i - 1][j + 1] - coeffs[i - 1][j - 1]
    coeffs[terms][terms] = -coeffs[terms - 1][terms - 1]
    coeffs[terms][0] = coeffs[terms - 1][1]
    for j in range(terms - 2, 0, -2):
        coeffs[terms][j] = (j + 1) * coeffs[terms - 1][j + 1] - coeffs[terms - 1][j - 1]

    x_powers = [1.0]
    y_powers = [1.0]
    for _ in range(terms):
        x_powers.append(x_powers[-1] * x)
        y_powers.append(y_powers[-1] * y)

    density_xy = norm_pdf(x) * norm_pdf(y)
    total = norm_cdf(x) * norm_cdf(y)
    factorial = 1.0
    rho_power = 1.0

    for i in range(terms + 1):
        dx = 0.0
        dy = 0.0
        for j in range(i, -1, -2):
            dx += coeffs[i][j] * x_powers[j]
            dy += coeffs[i][j] * y_powers[j]
        factorial *= i + 1.0
        rho_power *= rho
        total += density_xy * rho_power * dx * dy / factorial
    return min(max(total, 0.0), 1.0)


def compound_option(
    kind: str,
    m: MarketInputs,
    compound_strike: float,
    compound_maturity: float,
    underlying_maturity: float,
) -> CompoundResult:
    """Four classic compound option cases: call_on_call, put_on_call, call_on_put, put_on_put."""
    kind = kind.lower().replace("-", "_")
    if not 0 < compound_maturity < underlying_maturity:
        raise ValueError("compound_maturity must be between 0 and underlying_maturity")

    remaining = underlying_maturity - compound_maturity
    sub_market = MarketInputs(m.spot, m.strike, m.volatility, m.rate, remaining, m.dividend_yield)
    if kind in {"call_on_call", "put_on_call"}:
        critical = implied_spot(OptionType.CALL, compound_strike, sub_market)
    elif kind in {"call_on_put", "put_on_put"}:
        critical = implied_spot(OptionType.PUT, compound_strike, sub_market)
    else:
        raise ValueError("kind must be call_on_call, put_on_call, call_on_put or put_on_put")

    a1_market = MarketInputs(m.spot, critical, m.volatility, m.rate, compound_maturity, m.dividend_yield)
    a1 = _d1(a1_market)
    a2 = a1 - m.volatility * math.sqrt(compound_maturity)
    full_market = MarketInputs(m.spot, m.strike, m.volatility, m.rate, underlying_maturity, m.dividend_yield)
    d1 = _d1(full_market)
    d2 = d1 - m.volatility * math.sqrt(underlying_maturity)
    rho = math.sqrt(compound_maturity / underlying_maturity)
    df_q = math.exp(-m.dividend_yield * underlying_maturity)
    df_r_full = math.exp(-m.rate * underlying_maturity)
    df_r_compound = math.exp(-m.rate * compound_maturity)

    if kind == "call_on_call":
        price = (
            m.spot * df_q * biv_norm_cdf(a1, d1, rho)
            - m.strike * df_r_full * biv_norm_cdf(a2, d2, rho)
            - compound_strike * df_r_compound * norm_cdf(a2)
        )
    elif kind == "put_on_call":
        price = (
            -m.spot * df_q * biv_norm_cdf(-a1, d1, -rho)
            + m.strike * df_r_full * biv_norm_cdf(-a2, d2, -rho)
            + compound_strike * df_r_compound * norm_cdf(-a2)
        )
    elif kind == "call_on_put":
        price = (
            -m.spot * df_q * biv_norm_cdf(-a1, -d1, rho)
            + m.strike * df_r_full * biv_norm_cdf(-a2, -d2, rho)
            - compound_strike * df_r_compound * norm_cdf(-a2)
        )
    else:
        price = (
            m.spot * df_q * biv_norm_cdf(a1, -d1, -rho)
            - m.strike * df_r_full * biv_norm_cdf(a2, -d2, -rho)
            + compound_strike * df_r_compound * norm_cdf(a2)
        )
    return CompoundResult(price=price, critical_spot=critical)


def matrix_from_descriptor(option_type: OptionType | str, m: MarketInputs, descriptor: str) -> List[List[float]]:
    """Small replacement for the workbook's descriptor-string Black-Scholes array."""
    option_type = OptionType(option_type)
    result = black_scholes_greeks(option_type, m)
    lookup = {
        "P": result.price,
        "D": result.delta,
        "G": result.gamma,
        "V": result.vega,
        "R": result.rho,
        "T": result.theta,
        "S": result.psi,
        "E": result.elasticity,
    }
    rows: List[List[float]] = []
    for raw_row in descriptor.upper().split("/"):
        row: List[float] = []
        for ch in raw_row.replace("+", ""):
            if ch.strip():
                row.append(lookup[ch])
        if row:
            rows.append(row)
    return rows


def _market_from_args(args: argparse.Namespace) -> MarketInputs:
    return MarketInputs(
        spot=args.spot,
        strike=args.strike,
        volatility=args.volatility,
        rate=args.rate,
        maturity=args.maturity,
        dividend_yield=args.dividend_yield,
    )


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_demo() -> None:
    m = MarketInputs(spot=100.0, strike=100.0, volatility=0.20, rate=0.05, maturity=1.0, dividend_yield=0.02)
    demo_payload = {
        "black_scholes_call": asdict(black_scholes_greeks(OptionType.CALL, m)),
        "black_scholes_put": asdict(black_scholes_greeks(OptionType.PUT, m)),
        "implied_vol_from_call_price_10": implied_volatility(OptionType.CALL, 10.0, m),
        "american_put_crr_tree": asdict(
            binomial_option(OptionType.PUT, m, steps=200, exercise_style=ExerciseStyle.AMERICAN, model=TreeModel.CRR)
        ),
        "merton_jump_diffusion": merton_jump_diffusion(m, jump_intensity=0.30, mean_jump_log=-0.08, jump_volatility=0.25),
        "monte_carlo_call": asdict(monte_carlo_option(OptionType.CALL, m, paths=50_000, seed=42, antithetic=True)),
        "monte_carlo_convergence": monte_carlo_convergence(OptionType.CALL, m, path_counts=(1_000, 5_000, 10_000, 50_000), seed=42),
        "perpetual_american_put": {
            "price": perpetual_american(OptionType.PUT, 100.0, 100.0, 0.20, 0.05, 0.02)[0],
            "boundary": perpetual_american(OptionType.PUT, 100.0, 100.0, 0.20, 0.05, 0.02)[1],
        },
    }
    _print_json(demo_payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Options pricing toolkit converted from an Excel/VBA workbook.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_market_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--spot", type=float, default=100.0)
        p.add_argument("--strike", type=float, default=100.0)
        p.add_argument("--volatility", type=float, default=0.20)
        p.add_argument("--rate", type=float, default=0.05)
        p.add_argument("--maturity", type=float, default=1.0)
        p.add_argument("--dividend-yield", type=float, default=0.0)

    bs = sub.add_parser("bs", help="Black-Scholes price and Greeks")
    add_market_args(bs)
    bs.add_argument("--type", choices=[x.value for x in OptionType], default="call")
    bs.add_argument("--descriptor", default=None, help="Optional mini-matrix, e.g. PDGV/RT")

    iv = sub.add_parser("iv", help="Implied volatility from a quoted option price")
    add_market_args(iv)
    iv.add_argument("--type", choices=[x.value for x in OptionType], default="call")
    iv.add_argument("--price", type=float, required=True)

    tree = sub.add_parser("binomial", help="Binomial option price and tree Greeks")
    add_market_args(tree)
    tree.add_argument("--type", choices=[x.value for x in OptionType], default="put")
    tree.add_argument("--style", choices=[x.value for x in ExerciseStyle], default="american")
    tree.add_argument("--model", choices=[x.value for x in TreeModel], default="crr")
    tree.add_argument("--steps", type=int, default=200)
    tree.add_argument("--vesting-time", type=float, default=0.0)
    tree.add_argument("--show-lattice", action="store_true")

    jump = sub.add_parser("jump", help="Merton jump-diffusion call and put")
    add_market_args(jump)
    jump.add_argument("--jump-intensity", type=float, default=0.30)
    jump.add_argument("--mean-jump-log", type=float, default=-0.08)
    jump.add_argument("--jump-volatility", type=float, default=0.25)

    mc = sub.add_parser("mc", help="Monte Carlo European option price")
    add_market_args(mc)
    mc.add_argument("--type", choices=[x.value for x in OptionType], default="call")
    mc.add_argument("--paths", type=int, default=100_000)
    mc.add_argument("--seed", type=int, default=42)
    mc.add_argument("--no-antithetic", action="store_true")

    conv = sub.add_parser("mc-convergence", help="Benchmark Monte Carlo estimates against Black-Scholes")
    add_market_args(conv)
    conv.add_argument("--type", choices=[x.value for x in OptionType], default="call")
    conv.add_argument("--path-counts", default="1000,5000,10000,50000,100000")
    conv.add_argument("--seed", type=int, default=42)
    conv.add_argument("--no-antithetic", action="store_true")

    perp = sub.add_parser("perpetual", help="Perpetual American option")
    perp.add_argument("--type", choices=[x.value for x in OptionType], default="put")
    perp.add_argument("--spot", type=float, default=100.0)
    perp.add_argument("--strike", type=float, default=100.0)
    perp.add_argument("--volatility", type=float, default=0.20)
    perp.add_argument("--rate", type=float, default=0.05)
    perp.add_argument("--dividend-yield", type=float, default=0.02)

    comp = sub.add_parser("compound", help="Compound option price")
    add_market_args(comp)
    comp.add_argument("--kind", choices=["call_on_call", "put_on_call", "call_on_put", "put_on_put"], default="call_on_call")
    comp.add_argument("--compound-strike", type=float, default=8.0)
    comp.add_argument("--compound-maturity", type=float, default=0.5)
    comp.add_argument("--underlying-maturity", type=float, default=1.0)

    sub.add_parser("demo", help="Run a sample portfolio of calculations")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        run_demo()
        return 0

    if args.command == "bs":
        m = _market_from_args(args)
        result = black_scholes_greeks(args.type, m)
        if args.descriptor:
            _print_json({"descriptor_matrix": matrix_from_descriptor(args.type, m, args.descriptor)})
        else:
            _print_json(asdict(result))
        return 0

    if args.command == "iv":
        m = _market_from_args(args)
        _print_json({"implied_volatility": implied_volatility(args.type, args.price, m)})
        return 0

    if args.command == "binomial":
        m = _market_from_args(args)
        result = binomial_option(args.type, m, args.steps, args.style, args.model, args.vesting_time)
        payload: Dict[str, Any] = asdict(result)
        if args.show_lattice:
            payload["lattice"] = binomial_lattice(args.type, m, min(args.steps, 8), args.style, args.model)
        _print_json(payload)
        return 0

    if args.command == "jump":
        m = _market_from_args(args)
        _print_json(merton_jump_diffusion(m, args.jump_intensity, args.mean_jump_log, args.jump_volatility))
        return 0

    if args.command == "mc":
        m = _market_from_args(args)
        result = monte_carlo_option(args.type, m, args.paths, args.seed, not args.no_antithetic)
        _print_json(asdict(result))
        return 0

    if args.command == "mc-convergence":
        m = _market_from_args(args)
        path_counts = tuple(int(x.strip()) for x in args.path_counts.split(",") if x.strip())
        _print_json(monte_carlo_convergence(args.type, m, path_counts, args.seed, not args.no_antithetic))
        return 0

    if args.command == "perpetual":
        price, boundary = perpetual_american(args.type, args.spot, args.strike, args.volatility, args.rate, args.dividend_yield)
        _print_json({"price": price, "exercise_boundary": boundary})
        return 0

    if args.command == "compound":
        m = _market_from_args(args)
        result = compound_option(args.kind, m, args.compound_strike, args.compound_maturity, args.underlying_maturity)
        _print_json(asdict(result))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
