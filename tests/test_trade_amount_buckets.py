from config import Config
from env_updown import UpDownEnv


def test_trade_amount_buckets():
    cfg = Config()
    env = UpDownEnv(cfg)
    amounts = env.amounts
    assert len(amounts) == 17
    assert amounts[0] == 4.0
    assert amounts[-1] == 20.0
    for amt in amounts:
        assert round(amt, 2) == amt
        assert 4.0 <= amt <= 20.0
