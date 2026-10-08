

from dataclasses import dataclass, field
from typing import Optional

# ============================================================================
# Price-Action States
# ============================================================================
# ------------------------------------- 24 = PA-1
@dataclass
class SwingState:
    """State for swing/pivot detection."""
    swing_highs: list = field(default_factory=list)
    swing_lows: list = field(default_factory=list)
    last_swing_high: Optional[tuple] = None
    last_swing_low: Optional[tuple] = None
    current_high_candidate: Optional[tuple] = None
    current_low_candidate: Optional[tuple] = None
    lookback_left: int = 5
    lookback_right: int = 5
    buffer_high: list = field(default_factory=list)
    buffer_low: list = field(default_factory=list)
    current_idx: int = 0

    def update(self, high: float, low: float, left: int = 5, right: int = 5) -> dict:
        """Update swing state incrementally."""
        self.lookback_left = left
        self.lookback_right = right
        
        self.buffer_high.append((self.current_idx, high))
        self.buffer_low.append((self.current_idx, low))
        
        window_size = left + right + 1
        if len(self.buffer_high) > window_size:
            self.buffer_high.pop(0)
            self.buffer_low.pop(0)
        
        result = {'swing_high': None, 'swing_low': None}
        
        if len(self.buffer_high) >= window_size:
            # Check for swing high
            mid_idx = left
            mid_high = self.buffer_high[mid_idx][1]
            is_swing_high = all(mid_high >= self.buffer_high[i][1] for i in range(window_size) if i != mid_idx)
            
            if is_swing_high:
                swing_high = self.buffer_high[mid_idx]
                self.last_swing_high = swing_high
                self.swing_highs.append(swing_high)
                result['swing_high'] = swing_high
            
            # Check for swing low
            mid_low = self.buffer_low[mid_idx][1]
            is_swing_low = all(mid_low <= self.buffer_low[i][1] for i in range(window_size) if i != mid_idx)
            
            if is_swing_low:
                swing_low = self.buffer_low[mid_idx]
                self.last_swing_low = swing_low
                self.swing_lows.append(swing_low)
                result['swing_low'] = swing_low
        
        self.current_idx += 1
        return result

# ------------------------------------- 25 = PA-2
@dataclass
class StructureState:
    """State for market structure (HH/HL/LH/LL)."""
    structure_type: str = "ranging"
    last_hh: Optional[tuple] = None
    last_hl: Optional[tuple] = None
    last_lh: Optional[tuple] = None
    last_ll: Optional[tuple] = None
    structure_breaks: list = field(default_factory=list)
    swing_state: Optional['SwingState'] = None

    def update(self, swing_result: dict) -> dict:
        """Update structure state based on swing detection."""
        result = {'structure_change': None, 'bos': None}
        
        if swing_result['swing_high'] is not None:
            idx, price = swing_result['swing_high']
            
            if self.last_hh is None:
                self.last_hh = (idx, price)
            elif price > self.last_hh[1]:
                # Higher High
                self.last_hh = (idx, price)
                if self.structure_type != "uptrend":
                    self.structure_type = "uptrend"
                    result['structure_change'] = "uptrend"
            elif price < self.last_hh[1]:
                # Lower High
                self.last_lh = (idx, price)
                if self.structure_type != "downtrend":
                    self.structure_type = "downtrend"
                    result['structure_change'] = "downtrend"
                    result['bos'] = {'type': 'bearish', 'idx': idx, 'price': price}
                    self.structure_breaks.append(result['bos'])
        
        if swing_result['swing_low'] is not None:
            idx, price = swing_result['swing_low']
            
            if self.last_ll is None:
                self.last_ll = (idx, price)
            elif price < self.last_ll[1]:
                # Lower Low
                self.last_ll = (idx, price)
                if self.structure_type != "downtrend":
                    self.structure_type = "downtrend"
                    result['structure_change'] = "downtrend"
            elif price > self.last_ll[1]:
                # Higher Low
                self.last_hl = (idx, price)
                if self.structure_type != "uptrend":
                    self.structure_type = "uptrend"
                    result['structure_change'] = "uptrend"
                    result['bos'] = {'type': 'bullish', 'idx': idx, 'price': price}
                    self.structure_breaks.append(result['bos'])
        
        return result

# ------------------------------------- 26 = PA-3
@dataclass
class BOSCHOCHState:
    """State for Break of Structure (BOS) and Change of Character (CHOCH)."""
    last_bos: Optional[dict] = None
    last_choch: Optional[dict] = None
    bos_events: list = field(default_factory=list)
    choch_events: list = field(default_factory=list)
    current_trend: str = "neutral"
    structure_state: Optional['StructureState'] = None

    def update(self, structure_result: dict, close: float) -> dict:
        """Update BOS/CHOCH state based on structure changes."""
        result = {'bos': None, 'choch': None}
        
        if structure_result.get('bos') is not None:
            bos = structure_result['bos']
            
            # Check if this is a CHOCH (reversal) or BOS (continuation)
            if self.current_trend == "neutral":
                self.last_bos = bos
                self.bos_events.append(bos)
                result['bos'] = bos
                self.current_trend = "bullish" if bos['type'] == 'bullish' else "bearish"
            
            elif (self.current_trend == "bullish" and bos['type'] == 'bearish') or \
                 (self.current_trend == "bearish" and bos['type'] == 'bullish'):
                # Change of Character
                choch = {**bos, 'event_type': 'choch'}
                self.last_choch = choch
                self.choch_events.append(choch)
                result['choch'] = choch
                self.current_trend = "bullish" if bos['type'] == 'bullish' else "bearish"
            
            else:
                # Break of Structure (continuation)
                self.last_bos = bos
                self.bos_events.append(bos)
                result['bos'] = bos
        
        return result

# ------------------------------------- 27 = PA-4
@dataclass
class FVGState:
    """State for Fair Value Gap (imbalance) tracking."""
    active_fvgs: list = field(default_factory=list)
    bullish_fvgs: list = field(default_factory=list)
    bearish_fvgs: list = field(default_factory=list)
    last_fvg: Optional[dict] = None
    candle_buffer: list = field(default_factory=list)
    current_idx: int = 0

    def update(self, high: float, low: float, close: float) -> dict:
        """Update FVG state incrementally."""
        self.candle_buffer.append({'idx': self.current_idx, 'high': high, 'low': low, 'close': close})
        
        if len(self.candle_buffer) > 3:
            self.candle_buffer.pop(0)
        
        result = {'new_fvg': None, 'filled_fvgs': []}
        
        if len(self.candle_buffer) == 3:
            c1, c2, c3 = self.candle_buffer
            
            # Bullish FVG: c3.low > c1.high
            if c3['low'] > c1['high']:
                fvg = {
                    'type': 'bullish',
                    'start_idx': c1['idx'],
                    'end_idx': c3['idx'],
                    'top': c3['low'],
                    'bottom': c1['high'],
                    'filled': False
                }
                self.active_fvgs.append(fvg)
                self.bullish_fvgs.append(fvg)
                self.last_fvg = fvg
                result['new_fvg'] = fvg
            
            # Bearish FVG: c3.high < c1.low
            elif c3['high'] < c1['low']:
                fvg = {
                    'type': 'bearish',
                    'start_idx': c1['idx'],
                    'end_idx': c3['idx'],
                    'top': c1['low'],
                    'bottom': c3['high'],
                    'filled': False
                }
                self.active_fvgs.append(fvg)
                self.bearish_fvgs.append(fvg)
                self.last_fvg = fvg
                result['new_fvg'] = fvg
        
        # Check if any active FVGs are filled
        for fvg in self.active_fvgs:
            if not fvg['filled']:
                if fvg['type'] == 'bullish' and low <= fvg['bottom']:
                    fvg['filled'] = True
                    result['filled_fvgs'].append(fvg)
                elif fvg['type'] == 'bearish' and high >= fvg['top']:
                    fvg['filled'] = True
                    result['filled_fvgs'].append(fvg)
        
        self.current_idx += 1
        return result

# ------------------------------------- 28 = PA-5
@dataclass
class LiquidityPoolState:
    """State for liquidity pool/level tracking."""
    liquidity_levels: list = field(default_factory=list)
    equal_highs: list = field(default_factory=list)
    equal_lows: list = field(default_factory=list)
    swept_levels: list = field(default_factory=list)
    swing_state: Optional['SwingState'] = None

    def update(self, swing_result: dict, high: float, low: float, tolerance: float = 0.001) -> dict:
        """Update liquidity pool state based on swing points."""
        result = {'new_liquidity': None, 'swept': []}
        
        # Detect equal highs
        if swing_result.get('swing_high') is not None:
            idx, price = swing_result['swing_high']
            
            # Check for equal highs
            for existing in self.equal_highs:
                if abs(price - existing['price']) / existing['price'] <= tolerance:
                    existing['touches'] += 1
                    existing['indices'].append(idx)
                    result['new_liquidity'] = existing
                    break
            else:
                level = {
                    'type': 'equal_high',
                    'price': price,
                    'touches': 1,
                    'indices': [idx],
                    'swept': False
                }
                self.equal_highs.append(level)
                self.liquidity_levels.append(level)
                result['new_liquidity'] = level
        
        # Detect equal lows
        if swing_result.get('swing_low') is not None:
            idx, price = swing_result['swing_low']
            
            for existing in self.equal_lows:
                if abs(price - existing['price']) / existing['price'] <= tolerance:
                    existing['touches'] += 1
                    existing['indices'].append(idx)
                    result['new_liquidity'] = existing
                    break
            else:
                level = {
                    'type': 'equal_low',
                    'price': price,
                    'touches': 1,
                    'indices': [idx],
                    'swept': False
                }
                self.equal_lows.append(level)
                self.liquidity_levels.append(level)
                result['new_liquidity'] = level
        
        # Check for liquidity sweeps
        for level in self.liquidity_levels:
            if not level['swept']:
                if level['type'] == 'equal_high' and high > level['price']:
                    level['swept'] = True
                    self.swept_levels.append(level)
                    result['swept'].append(level)
                elif level['type'] == 'equal_low' and low < level['price']:
                    level['swept'] = True
                    self.swept_levels.append(level)
                    result['swept'].append(level)
        
        return result

# ------------------------------------- 29 = PA-6
@dataclass
class ZoneLifecycleState:
    """State for supply/demand zone lifecycle tracking."""
    active_zones: list = field(default_factory=list)
    supply_zones: list = field(default_factory=list)
    demand_zones: list = field(default_factory=list)
    tested_zones: list = field(default_factory=list)
    broken_zones: list = field(default_factory=list)

    def update(self, zone_type: str, top: float, bottom: float, origin_idx: int, high: float, low: float, close: float) -> dict:
        """Update zone lifecycle state."""
        result = {'new_zone': None, 'tested': [], 'broken': []}
        
        # Add new zone
        zone = {
            'type': zone_type,
            'top': top,
            'bottom': bottom,
            'origin_idx': origin_idx,
            'strength': 1.0,
            'touches': 0,
            'status': 'active'
        }
        self.active_zones.append(zone)
        
        if zone_type == 'supply':
            self.supply_zones.append(zone)
        else:
            self.demand_zones.append(zone)
        
        result['new_zone'] = zone
        
        # Check existing zones
        for z in self.active_zones:
            if z['status'] == 'active':
                # Check if zone is tested
                if z['type'] == 'supply' and high >= z['bottom'] and low <= z['top']:
                    z['touches'] += 1
                    z['status'] = 'tested'
                    self.tested_zones.append(z)
                    result['tested'].append(z)
                    
                    # Check if broken
                    if close > z['top']:
                        z['status'] = 'broken'
                        self.broken_zones.append(z)
                        result['broken'].append(z)
                
                elif z['type'] == 'demand' and low <= z['top'] and high >= z['bottom']:
                    z['touches'] += 1
                    z['status'] = 'tested'
                    self.tested_zones.append(z)
                    result['tested'].append(z)
                    
                    # Check if broken
                    if close < z['bottom']:
                        z['status'] = 'broken'
                        self.broken_zones.append(z)
                        result['broken'].append(z)
        
        return result

# ------------------------------------- 30 = PA-7
@dataclass
class BreakoutState:
    """State for breakout detection and confirmation."""
    breakout_events: list = field(default_factory=list)
    pending_breakouts: list = field(default_factory=list)
    confirmed_breakouts: list = field(default_factory=list)
    failed_breakouts: list = field(default_factory=list)
    last_breakout: Optional[dict] = None

    def update(self, level: float, level_type: str, close: float, volume: float, avg_volume: float, idx: int) -> dict:
        """Update breakout state."""
        result = {'breakout': None, 'confirmed': [], 'failed': []}
        
        # Detect breakout
        breakout_occurred = False
        if level_type == 'resistance' and close > level:
            breakout_occurred = True
            direction = 'bullish'
        elif level_type == 'support' and close < level:
            breakout_occurred = True
            direction = 'bearish'
        
        if breakout_occurred:
            volume_confirmed = volume > avg_volume * 1.5
            breakout = {
                'idx': idx,
                'type': direction,
                'level': level,
                'level_type': level_type,
                'volume_confirmed': volume_confirmed,
                'status': 'confirmed' if volume_confirmed else 'pending'
            }

            self.breakout_events.append(breakout)
            self.last_breakout = breakout

            if volume_confirmed:
                self.confirmed_breakouts.append(breakout)
                result['confirmed'].append(breakout)
            else:
                self.pending_breakouts.append(breakout)

            result['breakout'] = breakout

        return result

# ------------------------------------- 31 = PA-8
@dataclass
class ConfluenceState:
    """State for multi-indicator/multi-timeframe confluence."""
    confluence_zones: list = field(default_factory=list)
    active_signals: dict = field(default_factory=dict)
    confluence_score: float = 0.0
    mtf_alignment: dict = field(default_factory=dict)

    def update(
        self,
        signal_type: str,
        signal_value,
        timeframe: str = "current",
        weight: float = 1.0
    ) -> dict:
        """Update confluence state."""
        if signal_type not in self.active_signals:
            self.active_signals[signal_type] = []

        signal = {
            'timeframe': timeframe,
            'value': signal_value,
            'weight': weight
        }

        self.active_signals[signal_type].append(signal)

        # Calculate total confluence score
        total_weight = 0.0
        weighted_score = 0.0

        for signals in self.active_signals.values():
            for s in signals[-5:]:  # recent signals only
                total_weight += s['weight']

                if isinstance(s['value'], (int, float)):
                    weighted_score += s['value'] * s['weight']
                else:
                    weighted_score += s['weight']

        self.confluence_score = (
            weighted_score / total_weight
            if total_weight > 0
            else 0.0
        )

        return {
            'signal_added': signal,
            'confluence_score': self.confluence_score
        }

# ------------------------------------- 32 = PA-9
@dataclass
class RegimeState:
    """State for market regime detection."""
    current_regime: str = "ranging"
    regime_strength: float = 0.0
    regime_duration: int = 0
    regime_changes: list = field(default_factory=list)
    atr_percentile: float = 0.5
    adx_value: float = 0.0

    def update(
        self,
        adx: float,
        atr_ratio: float,
        trend_strength: float = 0.0
    ) -> dict:
        """Update market regime state."""
        previous_regime = self.current_regime

        self.adx_value = adx
        self.atr_percentile = atr_ratio

        # Detect regime
        if adx >= 25:
            if trend_strength >= 0:
                self.current_regime = "trending_up"
            else:
                self.current_regime = "trending_down"

            self.regime_strength = min(adx / 50.0, 1.0)

        elif atr_ratio >= 0.8:
            self.current_regime = "volatile"
            self.regime_strength = atr_ratio

        else:
            self.current_regime = "ranging"
            self.regime_strength = 1.0 - (adx / 25.0)

        # Track regime changes
        regime_changed = previous_regime != self.current_regime

        if regime_changed:
            change = {
                'from_regime': previous_regime,
                'to_regime': self.current_regime,
                'strength': self.regime_strength
            }

            self.regime_changes.append(change)
            self.regime_duration = 0

        else:
            self.regime_duration += 1

        return {
            'current_regime': self.current_regime,
            'regime_strength': self.regime_strength,
            'regime_changed': regime_changed
        }

