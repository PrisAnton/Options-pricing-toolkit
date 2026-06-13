import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from options_pricing_toolkit import black_scholes_greeks, MarketInputs

def test_bs_at_the_money_call():
    m = MarketInputs(100, 100, 0.20, 0.05, 1, 0.02)
    r = black_scholes_greeks("call", m)
    assert 7.5 < r.price < 9.5   # ATM call大概这个区间
    assert 0.5 < r.delta < 0.7
    print("PASS")

if __name__ == "__main__":
    test_bs_at_the_money_call()