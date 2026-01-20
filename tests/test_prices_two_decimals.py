from decimal import Decimal

from utils import price_pair, quantize_price


def test_prices_two_decimals():
    p_up = quantize_price(0.515)
    assert round(p_up, 2) == p_up
    p_up2, p_down = price_pair(0.51)
    assert round(p_up2, 2) == p_up2
    assert round(p_down, 2) == p_down
    assert abs((p_up2 + p_down) - 1.0) < 1e-9
    assert Decimal("0.00") <= Decimal(str(p_up2)) <= Decimal("1.00")
