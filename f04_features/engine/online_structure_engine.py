# f04_features/_engine/online_structure_engine.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, List, Dict, Any


# ============================================================
# Data Models
# ============================================================

@dataclass(frozen=True)
class Candle:
    ts: Any
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class PivotType(str, Enum):
    HIGH = "high"
    LOW = "low"


class EngineState(str, Enum):
    INIT = "init"
    SEEK_HIGH = "seek_high"
    SEEK_LOW = "seek_low"


@dataclass(frozen=True)
class Pivot:
    ts: Any
    index: int
    price: float
    kind: PivotType


@dataclass(frozen=True)
class Leg:
    start_ts: Any
    end_ts: Any
    start_index: int
    end_index: int
    start_price: float
    end_price: float
    direction: str   # "up" | "down"
    length_price: float
    length_pct: float

    @staticmethod
    def from_pivots(p1: Pivot, p2: Pivot) -> "Leg":
        direction = "up" if p2.price >= p1.price else "down"
        length_price = p2.price - p1.price
        base = p1.price if p1.price != 0 else 1e-12
        length_pct = length_price / base
        return Leg(
            start_ts=p1.ts,
            end_ts=p2.ts,
            start_index=p1.index,
            end_index=p2.index,
            start_price=p1.price,
            end_price=p2.price,
            direction=direction,
            length_price=length_price,
            length_pct=length_pct,
        )


@dataclass(frozen=True)
class StructureEvent:
    name: str
    ts: Any
    index: int
    payload: Dict[str, Any]


# ============================================================
# Final Engine
# ============================================================

class OnlineStructureEngine:
    """
    Final causal/stateful/replayable market structure engine.

    Core properties:
      - no lookahead
      - deterministic
      - incremental O(1)
      - same logic usable in train and live
      - confirmed pivots only after sufficient reversal
    """

    def __init__(
        self,
        reversal_ratio: float = 0.01,
        min_bars_between_pivots: int = 1,
        use_close_for_confirmation: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        reversal_ratio:
            Minimum reversal needed to confirm an extreme as pivot.
            Example: 0.01 => 1%

        min_bars_between_pivots:
            Minimal bar distance between consecutive confirmed pivots.

        use_close_for_confirmation:
            If True, reversal confirmation uses candle.close
            If False, uses candle.low / candle.high intrabar extremes.
        """
        if reversal_ratio <= 0:
            raise ValueError("reversal_ratio must be > 0")
        if min_bars_between_pivots < 1:
            raise ValueError("min_bars_between_pivots must be >= 1")

        self.reversal_ratio = float(reversal_ratio)
        self.min_bars_between_pivots = int(min_bars_between_pivots)
        self.use_close_for_confirmation = bool(use_close_for_confirmation)

        self.reset()

    # ========================================================
    # Lifecycle
    # ========================================================

    def reset(self) -> None:
        self.state: EngineState = EngineState.INIT
        self.index: int = -1

        self.confirmed_pivots: List[Pivot] = []
        self.confirmed_legs: List[Leg] = []

        self.last_event: Optional[StructureEvent] = None

        # seed / last confirmed pivot
        self.last_confirmed_pivot: Optional[Pivot] = None

        # current tracked extreme in active search direction
        self.candidate_extreme_price: Optional[float] = None
        self.candidate_extreme_ts: Optional[Any] = None
        self.candidate_extreme_index: Optional[int] = None

        # bootstrap first candle memory
        self.first_candle: Optional[Candle] = None

    # ========================================================
    # Public API
    # ========================================================

    def update(self, candle: Candle) -> Dict[str, Any]:
        """
        Consume one candle and update the structure state.

        Returns a snapshot dict after processing this candle.
        """
        self.index += 1
        self.last_event = None

        if self.state == EngineState.INIT:
            self._handle_init(candle)
        elif self.state == EngineState.SEEK_HIGH:
            self._handle_seek_high(candle)
        elif self.state == EngineState.SEEK_LOW:
            self._handle_seek_low(candle)
        else:
            raise RuntimeError(f"Unknown state: {self.state}")

        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "state": self.state.value,
            "last_confirmed_pivot": asdict(self.last_confirmed_pivot) if self.last_confirmed_pivot else None,
            "candidate_extreme": self._candidate_extreme_snapshot(),
            "confirmed_pivots": [asdict(p) for p in self.confirmed_pivots],
            "confirmed_legs": [asdict(l) for l in self.confirmed_legs],
            "active_leg": self._active_leg_snapshot(),
            "current_bias": self._current_bias(),
            "last_event": asdict(self.last_event) if self.last_event else None,
        }

    # ========================================================
    # State Handlers
    # ========================================================

    def _handle_init(self, candle: Candle) -> None:
        """
        Bootstrap logic:
        - first candle فقط ذخیره می‌شود
        - second candle determines initial direction
        - one initial pivot is seeded causally
        """
        if self.first_candle is None:
            self.first_candle = candle
            return

        prev = self.first_candle

        # اگر حرکت اولیه صعودی باشد، low اولیه را pivot seed می‌کنیم
        if candle.high > prev.high or candle.close > prev.close:
            seed = Pivot(
                ts=prev.ts,
                index=0,
                price=prev.low,
                kind=PivotType.LOW,
            )
            self._register_initial_pivot(seed)

            self.state = EngineState.SEEK_HIGH
            self.candidate_extreme_price = max(prev.high, candle.high)
            self.candidate_extreme_ts = candle.ts if candle.high >= prev.high else prev.ts
            self.candidate_extreme_index = self.index if candle.high >= prev.high else 0
            return

        # اگر حرکت اولیه نزولی باشد، high اولیه را pivot seed می‌کنیم
        if candle.low < prev.low or candle.close < prev.close:
            seed = Pivot(
                ts=prev.ts,
                index=0,
                price=prev.high,
                kind=PivotType.HIGH,
            )
            self._register_initial_pivot(seed)

            self.state = EngineState.SEEK_LOW
            self.candidate_extreme_price = min(prev.low, candle.low)
            self.candidate_extreme_ts = candle.ts if candle.low <= prev.low else prev.ts
            self.candidate_extreme_index = self.index if candle.low <= prev.low else 0
            return

        # هنوز جهت نگرفته
        self.first_candle = prev

    def _handle_seek_high(self, candle: Candle) -> None:
        """
        We have a confirmed LOW pivot.
        We seek an extreme HIGH.
        Confirm HIGH only after sufficient downward reversal.
        """
        assert self.last_confirmed_pivot is not None
        assert self.last_confirmed_pivot.kind == PivotType.LOW

        # 1) extend candidate high
        if self.candidate_extreme_price is None or candle.high >= self.candidate_extreme_price:
            self.candidate_extreme_price = candle.high
            self.candidate_extreme_ts = candle.ts
            self.candidate_extreme_index = self.index

        # 2) check reversal from candidate high
        candidate_high = self.candidate_extreme_price
        assert candidate_high is not None

        reversal_probe = candle.close if self.use_close_for_confirmation else candle.low
        reversal_ratio = (candidate_high - reversal_probe) / max(abs(candidate_high), 1e-12)

        enough_reversal = reversal_ratio >= self.reversal_ratio
        enough_spacing = (
            self.candidate_extreme_index is not None
            and self.last_confirmed_pivot is not None
            and (self.candidate_extreme_index - self.last_confirmed_pivot.index) >= self.min_bars_between_pivots
        )

        if enough_reversal and enough_spacing:
            new_pivot = Pivot(
                ts=self.candidate_extreme_ts,
                index=self.candidate_extreme_index,
                price=self.candidate_extreme_price,
                kind=PivotType.HIGH,
            )
            self._confirm_new_pivot(new_pivot)

            # switch state
            self.state = EngineState.SEEK_LOW

            # start tracking next low from current candle
            self.candidate_extreme_price = candle.low
            self.candidate_extreme_ts = candle.ts
            self.candidate_extreme_index = self.index

    def _handle_seek_low(self, candle: Candle) -> None:
        """
        We have a confirmed HIGH pivot.
        We seek an extreme LOW.
        Confirm LOW only after sufficient upward reversal.
        """
        assert self.last_confirmed_pivot is not None
        assert self.last_confirmed_pivot.kind == PivotType.HIGH

        # 1) extend candidate low
        if self.candidate_extreme_price is None or candle.low <= self.candidate_extreme_price:
            self.candidate_extreme_price = candle.low
            self.candidate_extreme_ts = candle.ts
            self.candidate_extreme_index = self.index

        # 2) check reversal from candidate low
        candidate_low = self.candidate_extreme_price
        assert candidate_low is not None

        reversal_probe = candle.close if self.use_close_for_confirmation else candle.high
        reversal_ratio = (reversal_probe - candidate_low) / max(abs(candidate_low), 1e-12)

        enough_reversal = reversal_ratio >= self.reversal_ratio
        enough_spacing = (
            self.candidate_extreme_index is not None
            and self.last_confirmed_pivot is not None
            and (self.candidate_extreme_index - self.last_confirmed_pivot.index) >= self.min_bars_between_pivots
        )

        if enough_reversal and enough_spacing:
            new_pivot = Pivot(
                ts=self.candidate_extreme_ts,
                index=self.candidate_extreme_index,
                price=self.candidate_extreme_price,
                kind=PivotType.LOW,
            )
            self._confirm_new_pivot(new_pivot)

            # switch state
            self.state = EngineState.SEEK_HIGH

            # start tracking next high from current candle
            self.candidate_extreme_price = candle.high
            self.candidate_extreme_ts = candle.ts
            self.candidate_extreme_index = self.index

    # ========================================================
    # Pivot / Leg registration
    # ========================================================

    def _register_initial_pivot(self, pivot: Pivot) -> None:
        self.confirmed_pivots.append(pivot)
        self.last_confirmed_pivot = pivot
        self.last_event = StructureEvent(
            name="initial_pivot_seeded",
            ts=pivot.ts,
            index=pivot.index,
            payload={
                "kind": pivot.kind.value,
                "price": pivot.price,
            },
        )

    def _confirm_new_pivot(self, pivot: Pivot) -> None:
        prev = self.last_confirmed_pivot
        if prev is None:
            self._register_initial_pivot(pivot)
            return

        # جلوگیری از pivot هم‌نوع پشت‌سرهم
        if prev.kind == pivot.kind:
            # theoretically should not happen under current state machine
            # but keep hard safety
            if pivot.kind == PivotType.HIGH and pivot.price > prev.price:
                self.confirmed_pivots[-1] = pivot
                self.last_confirmed_pivot = pivot
            elif pivot.kind == PivotType.LOW and pivot.price < prev.price:
                self.confirmed_pivots[-1] = pivot
                self.last_confirmed_pivot = pivot
            return

        self.confirmed_pivots.append(pivot)
        self.confirmed_legs.append(Leg.from_pivots(prev, pivot))
        self.last_confirmed_pivot = pivot

        self.last_event = StructureEvent(
            name="pivot_confirmed",
            ts=pivot.ts,
            index=pivot.index,
            payload={
                "kind": pivot.kind.value,
                "price": pivot.price,
                "previous_kind": prev.kind.value,
                "previous_price": prev.price,
                "leg_direction": self.confirmed_legs[-1].direction,
                "leg_length_price": self.confirmed_legs[-1].length_price,
                "leg_length_pct": self.confirmed_legs[-1].length_pct,
            },
        )

    # ========================================================
    # Derived Views
    # ========================================================

    def _candidate_extreme_snapshot(self) -> Optional[Dict[str, Any]]:
        if self.candidate_extreme_price is None:
            return None
        return {
            "price": self.candidate_extreme_price,
            "ts": self.candidate_extreme_ts,
            "index": self.candidate_extreme_index,
        }

    def _active_leg_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Mutable current leg:
        from last confirmed pivot to current candidate extreme.
        """
        if self.last_confirmed_pivot is None or self.candidate_extreme_price is None:
            return None

        start = self.last_confirmed_pivot.price
        end = self.candidate_extreme_price
        direction = "up" if end >= start else "down"
        length_price = end - start
        length_pct = length_price / max(abs(start), 1e-12)

        return {
            "start_ts": self.last_confirmed_pivot.ts,
            "start_index": self.last_confirmed_pivot.index,
            "start_price": start,
            "end_ts": self.candidate_extreme_ts,
            "end_index": self.candidate_extreme_index,
            "end_price": end,
            "direction": direction,
            "length_price": length_price,
            "length_pct": length_pct,
            "confirmed": False,
        }

    def _current_bias(self) -> Optional[str]:
        """
        Structural bias based on state:
          SEEK_HIGH => currently building upward leg
          SEEK_LOW  => currently building downward leg
        """
        if self.state == EngineState.SEEK_HIGH:
            return "up"
        if self.state == EngineState.SEEK_LOW:
            return "down"
        return None

