"""
Wrapper for AdaptiveStructureEngine with snapshot access and state management.

استفاده:
wrapper = AdaptiveStructureWrapper(atr_period=14)
snapshot = wrapper.update(candle)
print(f"ATR: {snapshot.atr}, Regime: {snapshot.regime}")
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from adaptive_structure_engine import AdaptiveStructureEngine
from online_structure_engine import Candle, Pivot, Leg, StructureEvent


@dataclass
class StructureSnapshot:
    """Snapshot of structure engine state at a point in time."""
    index: int
    state: str
    last_confirmed_pivot: Optional[Pivot]
    candidate_extreme: Optional[Dict[str, Any]]
    confirmed_pivots: List[Pivot]
    confirmed_legs: List[Leg]
    active_leg: Optional[Leg]
    current_bias: str
    last_event: Optional[StructureEvent]
    atr: Optional[float]
    regime: str
    threshold: Optional[float]


class AdaptiveStructureWrapper:
    """
    Wrapper around AdaptiveStructureEngine providing:
    - Easy snapshot access
    - State inspection methods
    - Typed return values
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        low_vol_mult: float = 1.2,
        normal_vol_mult: float = 2.0,
        high_vol_mult: float = 3.0,
    ):
        """
        Initialize wrapper with adaptive structure engine.
        
        Args:
            atr_period: Period for ATR calculation
            low_vol_mult: Threshold multiplier for low volatility regime
            normal_vol_mult: Threshold multiplier for normal volatility regime
            high_vol_mult: Threshold multiplier for high volatility regime
        """
        self.engine = AdaptiveStructureEngine(
            atr_period=atr_period,
            low_vol_mult=low_vol_mult,
            normal_vol_mult=normal_vol_mult,
            high_vol_mult=high_vol_mult,
        )
        self._last_snapshot: Optional[StructureSnapshot] = None
    
    def update(self, candle: Candle) -> StructureSnapshot:
        """
        Update engine with new candle and return typed snapshot.
        
        Args:
            candle: New candle data
            
        Returns:
            StructureSnapshot with current state
        """
        result = self.engine.update(candle)
        
        # Extract adaptive-specific fields
        atr = self.engine.atr
        regime = self.engine.detect_regime()
        threshold = self.engine.get_threshold()
        
        # Build typed snapshot
        snapshot = StructureSnapshot(
            index=result['index'],
            state=result['state'],
            last_confirmed_pivot=result['last_confirmed_pivot'],
            candidate_extreme=result['candidate_extreme'],
            confirmed_pivots=result['confirmed_pivots'],
            confirmed_legs=result['confirmed_legs'],
            active_leg=result['active_leg'],
            current_bias=result['current_bias'],
            last_event=result['last_event'],
            atr=atr,
            regime=regime,
            threshold=threshold,
        )
        
        self._last_snapshot = snapshot
        return snapshot
    
    def get_snapshot(self) -> Optional[StructureSnapshot]:
        """
        Get current snapshot without updating.
        
        Returns:
            Last snapshot or None if no update has occurred
        """
        if self._last_snapshot is None:
            return None
        return self._last_snapshot
    
    def get_current_state(self) -> Optional[str]:
        """Get current engine state."""
        return self._last_snapshot.state if self._last_snapshot else None
    
    def get_atr(self) -> Optional[float]:
        """Get current ATR value."""
        return self.engine.atr
    
    def get_regime(self) -> str:
        """Get current volatility regime (low/normal/high)."""
        return self.engine.detect_regime()
    
    def get_threshold(self) -> Optional[float]:
        """Get current adaptive threshold."""
        return self.engine.get_threshold()
    
    def get_confirmed_pivots(self) -> List[Pivot]:
        """Get all confirmed pivots."""
        return self._last_snapshot.confirmed_pivots if self._last_snapshot else []
    
    def get_confirmed_legs(self) -> List[Leg]:
        """Get all confirmed legs."""
        return self._last_snapshot.confirmed_legs if self._last_snapshot else []
    
    def get_active_leg(self) -> Optional[Leg]:
        """Get current active leg."""
        return self._last_snapshot.active_leg if self._last_snapshot else None
    
    def get_last_event(self) -> Optional[StructureEvent]:
        """Get last structure event."""
        return self._last_snapshot.last_event if self._last_snapshot else None
    
    def get_current_bias(self) -> Optional[str]:
        """Get current market bias."""
        return self._last_snapshot.current_bias if self._last_snapshot else None
    
    def is_ready(self) -> bool:
        """Check if engine has enough data (ATR calculated)."""
        return self.engine.atr is not None
    
    def reset(self):
        """Reset wrapper and create new engine instance."""
        self.engine = AdaptiveStructureEngine(
            atr_period=self.engine.atr_period,
            low_vol_mult=self.engine.low_vol_mult,
            normal_vol_mult=self.engine.normal_vol_mult,
            high_vol_mult=self.engine.high_vol_mult,
        )
        self._last_snapshot = None
