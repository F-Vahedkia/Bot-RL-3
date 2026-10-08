# f04_features/_engine/adaptive_structure_engine.py

from collections import deque
from typing import Any, Dict, Optional
from online_structure_engine import OnlineStructureEngine, Candle

"""
نحوه استفاده در RL Pipeline

engine = AdaptiveStructureEngine()
for candle in data:
    engine.update(candle)
    state = engine.get_state()
    observation = feature_builder(state)
    agent.step(observation)

✅ causal
✅ online
✅ volatility adaptive
✅ RL compatible
✅ deterministic replay
✅ training/live parity
"""

class AdaptiveStructureEngine(OnlineStructureEngine):
    """
    Online Structure Engine with adaptive threshold using ATR and volatility regimes
    """

    def __init__(
        self,
        atr_period: int = 14,
        low_vol_mult: float = 1.2,
        normal_vol_mult: float = 2.0,
        high_vol_mult: float = 3.0,
    ):

        super().__init__(
            reversal_ratio=0.01,
            min_bars_between_pivots=1,
            use_close_for_confirmation=False
        )

        self.atr_period = atr_period
        self.tr_values = deque(maxlen=atr_period)

        self.prev_close: Optional[float] = None
        self.atr: Optional[float] = None

        self.low_vol_mult = low_vol_mult
        self.normal_vol_mult = normal_vol_mult
        self.high_vol_mult = high_vol_mult

    # -------------------------------------

    def compute_tr(self, candle: Candle):

        if self.prev_close is None:
            tr = candle.high - candle.low
        else:
            tr = max(
                candle.high - candle.low,
                abs(candle.high - self.prev_close),
                abs(candle.low - self.prev_close),
            )

        self.prev_close = candle.close
        return tr

    # -------------------------------------

    def update_atr(self, candle: Candle):

        tr = self.compute_tr(candle)
        self.tr_values.append(tr)

        if len(self.tr_values) < self.atr_period:
            return None

        if self.atr is None:
            self.atr = sum(self.tr_values) / len(self.tr_values)
        else:
            alpha = 2 / (self.atr_period + 1)
            self.atr = alpha * tr + (1 - alpha) * self.atr

        return self.atr

    # -------------------------------------

    def detect_regime(self):

        if self.atr is None:
            return "normal"

        mean_tr = sum(self.tr_values) / len(self.tr_values)

        if self.atr < 0.8 * mean_tr:
            return "low"

        if self.atr > 1.3 * mean_tr:
            return "high"

        return "normal"

    # -------------------------------------

    def get_threshold(self):

        if self.atr is None:
            return None

        regime = self.detect_regime()

        if regime == "low":
            mult = self.low_vol_mult

        elif regime == "high":
            mult = self.high_vol_mult

        else:
            mult = self.normal_vol_mult

        return self.atr * mult

    # -------------------------------------

    def check_pivot_confirmation(self, candle: Candle) -> bool:
        """
        override pivot confirmation using ATR threshold
        """
        if not self.provisional_leg.end_price:
            return False

        threshold = self.get_threshold()

        if threshold is None:
            return False

        direction = self.current_leg.direction

        if direction == "up":
            return candle.low <= self.provisional_leg.end_price - threshold
        else:
            return candle.high >= self.provisional_leg.end_price + threshold

    # -------------------------------------

    def update(self, candle: Candle) -> Dict[str, Any]:

        self.update_atr(candle)

        return super().update(candle)
