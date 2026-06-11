# Options Pricing Toolkit

A compact Python toolkit for pricing equity derivatives and short-rate products. The project started from a legacy Excel/VBA options workbook and was refactored into a standalone Python module with a small command-line interface.

The goal is not to reproduce the original spreadsheet screen by screen. The useful pricing logic was kept, cleaned up, and exposed as regular Python functions that can be used from the terminal or imported into another script.

## Features

- Black-Scholes pricing for European calls and puts
- Greeks: delta, gamma, vega, rho, theta, dividend-yield sensitivity, and elasticity
- Implied volatility and implied spot solvers
- Binomial option pricing with forward, Cox-Ross-Rubinstein, and lognormal trees
- European, American, and Bermudan-style vesting support in the binomial engine
- Merton jump-diffusion pricing
- Exchange option pricing
- Perpetual American option pricing
- CIR and Vasicek zero-coupon bond pricing
- Compound option pricing
- JSON output from the command-line interface

The Asian, Power, Binary, Barrier, and Rainbow spreadsheet sections were intentionally left out to keep this version focused and easier to maintain.

## Requirements

Python 3.10 or newer.

No third-party packages are required. The toolkit only uses the Python standard library.

## Project structure

```text
options-pricing-toolkit/
├── options_pricing_toolkit.py
└── README.md
```

## Quick start

Run the built-in demo:

```bash
python options_pricing_toolkit.py demo
```

On Windows, this may be:

```bash
py options_pricing_toolkit.py demo
```

The output is printed as JSON, which makes it easy to inspect manually or pipe into another tool.

## Command-line examples

Black-Scholes price and Greeks:

```bash
python options_pricing_toolkit.py bs \
  --type call \
  --spot 100 \
  --strike 100 \
  --volatility 0.20 \
  --rate 0.05 \
  --maturity 1 \
  --dividend-yield 0.02
```

Implied volatility from a quoted option price:

```bash
python options_pricing_toolkit.py iv \
  --type call \
  --price 10 \
  --spot 100 \
  --strike 100 \
  --rate 0.05 \
  --maturity 1 \
  --dividend-yield 0.02
```

American put using a CRR binomial tree:

```bash
python options_pricing_toolkit.py binomial \
  --type put \
  --style american \
  --model crr \
  --steps 200 \
  --spot 100 \
  --strike 100 \
  --volatility 0.20 \
  --rate 0.05 \
  --maturity 1 \
  --dividend-yield 0.02
```

Merton jump-diffusion model:

```bash
python options_pricing_toolkit.py jump \
  --spot 100 \
  --strike 100 \
  --volatility 0.20 \
  --rate 0.05 \
  --maturity 1 \
  --dividend-yield 0.02 \
  --jump-intensity 0.30 \
  --mean-jump-log -0.08 \
  --jump-volatility 0.25
```

CIR zero-coupon bond pricing:

```bash
python options_pricing_toolkit.py rates \
  --model cir \
  --a 0.60 \
  --b 0.05 \
  --sigma 0.12 \
  --short-rate 0.04 \
  --maturity 5
```

## Using it as a Python module

```python
from options_pricing_toolkit import MarketInputs, black_scholes_greeks, binomial_option

market = MarketInputs(
    spot=100,
    strike=100,
    volatility=0.20,
    rate=0.05,
    maturity=1,
    dividend_yield=0.02,
)

bs_result = black_scholes_greeks("call", market)
print(bs_result.price)
print(bs_result.delta)

binomial_result = binomial_option(
    option_type="put",
    m=market,
    steps=200,
    exercise_style="american",
    model="crr",
)
print(binomial_result.price)
```

## Input conventions

All market inputs are annualized decimals:

- `0.05` means 5% interest rate
- `0.20` means 20% volatility
- `1` means one year to maturity
- `0.02` means 2% continuous dividend yield

The command-line interface reports results in JSON. In the Black-Scholes output, vega, rho, and dividend-yield sensitivity are scaled to a one percentage-point move. Theta is reported as daily theta.

## Implementation notes

The module is intentionally kept dependency-free. Normal and bivariate normal calculations are implemented directly with the standard library, which keeps the project easy to run in a clean Python environment.

The binomial engine supports three tree specifications:

- `forward`: a forward-price tree adapted from the original workbook structure
- `crr`: Cox-Ross-Rubinstein tree
- `lognormal`: a lognormal tree specification

For numerical inversion, the implied-volatility and implied-spot routines use bracket expansion followed by bisection. This is slower than Newton-Raphson in some cases, but it is stable and avoids requiring an analytical derivative.

## Limitations

This is a study and portfolio project, not a production trading library. It does not include market-data ingestion, calibration, volatility-surface construction, portfolio risk aggregation, or numerical validation against a professional pricing system.

