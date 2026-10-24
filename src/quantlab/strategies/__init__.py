from .base import (
    LookaheadError,
    Strategy,
    check_no_lookahead,
    hold_between_rebalances,
    rebalance_mask,
)
from .momentum import CrossSectionalMomentum
from .trend import BuyAndHold, TrendFollowing

REGISTRY = {
    "trend_sma": TrendFollowing,
    "xs_momentum": CrossSectionalMomentum,
    "equal_weight_bh": BuyAndHold,
}

__all__ = [
    "REGISTRY",
    "BuyAndHold",
    "CrossSectionalMomentum",
    "LookaheadError",
    "Strategy",
    "TrendFollowing",
    "check_no_lookahead",
    "hold_between_rebalances",
    "rebalance_mask",
]
