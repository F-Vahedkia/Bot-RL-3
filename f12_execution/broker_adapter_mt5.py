# -*- coding: utf-8 -*-
# f12_execution/broker_adapter_mt5.py   (آداپتر بروکر MT5 - اسکلت)
# Status in (Bot-RL-3): Completed

"""
آداپتر بروکر:
- MT5Broker: قابل‌استفاده در صورت فراهم‌بودن اتصال واقعی (این‌جا فقط اسکلت لاگ‌محور).
- NoOpBroker: برای اجراهای غیر زنده (dry/semi-live) که صرفاً لاگ می‌دهد.

نکته: مطابق قوانین پروژه، از حدس دربارهٔ API های خارجی خودداری می‌کنیم؛
در صورت نیاز به اتصال واقعی، همین اسکلت باید با کُد MT5 موجود شما جایگزین/تکمیل شود.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple
# [QC:TICKTIME:IMPORTS] — unique anchor
from datetime import datetime, timezone

from f02_utils.config_loader import load_config
from f03_data.mt5_connector import MT5Connector  # استفاده از کانکتور واقعی موجود

try:
    import MetaTrader5 as mt5  # کتابخانهٔ رسمی؛ قبلاً در پروژه استفاده شده
except Exception as _ex:  # pragma: no cover
    mt5 = None
    logging.getLogger("broker").warning("MetaTrader5 module not available: %s", _ex)


LOGGER = logging.getLogger("broker")


class NoOpBroker:
    """آداپتر No-Op برای اجراهای بدون بروکر (صرفاً لاگ)."""
    def get_prices(self, symbol: str) -> Tuple[float | None, float | None]:
        return None, None
    
    # [QC:TICKTIME:NOOP] — unique anchor
    def get_last_tick_time(self, symbol: str):
        return None

    def get_account_equity(self) -> float | None:
        return None
    def get_account_leverage(self):
        return None
    def get_pip_value_per_lot(self, symbol: str) -> float | None:
        return None

    def buy(self, symbol: str, lot: float, sl: float | None = None, tp: float | None = None) -> Tuple[bool, Optional[int]]:
        LOGGER.info("[NOOP] BUY %s lot=%.3f sl=%s tp=%s", symbol, lot, sl, tp)
        return True, None

    def sell(self, symbol: str, lot: float, sl: float | None = None, tp: float | None = None) -> Tuple[bool, Optional[int]]:
        LOGGER.info("[NOOP] SELL %s lot=%.3f sl=%s tp=%s", symbol, lot, sl, tp)
        return True, None

    def close_long(self, symbol: str) -> Tuple[bool, Optional[int]]:
        LOGGER.info("[NOOP] CLOSE LONG %s", symbol)
        return True, None

    def close_short(self, symbol: str) -> Tuple[bool, Optional[int]]:
        LOGGER.info("[NOOP] CLOSE SHORT %s", symbol)
        return True, None

    def close_partial(self, symbol: str, side: str, volume: float) -> Tuple[bool, Optional[int]]:
        LOGGER.info("[NOOP] PARTIAL CLOSE %s side=%s vol=%.3f", symbol, side, volume)
        return True, None

    def modify_position_sl_tp(self, position_id: int, sl: float | None, tp: float | None) -> bool:
        LOGGER.info("[NOOP] MODIFY position=%s sl=%s tp=%s", position_id, sl, tp)
        return True


class MT5Broker:
    """
    آداپتر MT5 با استفاده از MT5Connector موجود پروژه:
    - login/health طبق Connector
    - سفارش‌های Market با mt5.order_send + ریترای هوشمند
    - SL/TP واقعی (در زمان ارسال یا اصلاح)
    - گارد مارجین و نرمال‌سازی حجم
    - Partial-close و خواندن قیمت جاری
    """
    def __init__(self) -> None:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 module not available.")
        self._cfg = load_config("f01_config/config.yaml", enable_env_override=True)
        self._conn = MT5Connector(config=self._cfg)
        ok = self._conn.initialize()
        if not ok:
            raise RuntimeError("Failed to initialize MT5Connector (login/connect).")
        LOGGER.info("MT5Broker initialized and connected.")

    # ---------- Prices
    def get_prices(self, symbol: str) -> Tuple[float, float]:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick({symbol}) returned None.")
        bid = float(getattr(tick, "bid", 0.0))
        ask = float(getattr(tick, "ask", 0.0))
        if bid <= 0 or ask <= 0:
            raise RuntimeError(f"Invalid tick prices for {symbol}: bid={bid} ask={ask}")
        return bid, ask

    # [QC:TICKTIME:MT5] — unique anchor
    # این متد زمان آخرین تیک را بر میگرداند
    def get_last_tick_time(self, symbol: str):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        ts = getattr(tick, "time", None)
        if ts is None:
            return None
        try:
            return datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except Exception:
            return None

    # ---------- Equity & PipValue
    def get_account_equity(self) -> float | None:
        try:
            acct = mt5.account_info()
            return float(getattr(acct, "equity", None)) if acct is not None else None
        except Exception:
            LOGGER.exception("get_account_equity() failed.")
            return None


    def get_account_leverage(self):
        acct = mt5.account_info()
        lv = getattr(acct, "leverage", None) if acct is not None else None
        return float(lv) if lv is not None else None


    def get_pip_value_per_lot(self, symbol: str) -> float | None:
        try:
            si = mt5.symbol_info(symbol)
            if si is None: return None
            ttv = getattr(si, "trade_tick_value", None)
            tts = getattr(si, "trade_tick_size", None)
            if ttv is None or tts in (None, 0): return None
            return float(ttv) / float(tts)
        except Exception:
            LOGGER.exception("get_pip_value_per_lot(%s) failed.", symbol)
            return None

    # ---------- Lot normalization & margin guard
    def _normalize_lot(self, symbol: str, lot: float) -> float:
        si = mt5.symbol_info(symbol)
        if si is None:
            return max(0.0, float(lot))
        vmin = float(getattr(si, "volume_min", 0.01) or 0.01)
        vmax = float(getattr(si, "volume_max", 100.0) or 100.0)
        vstep = float(getattr(si, "volume_step", 0.01) or 0.01)
        lot = max(vmin, min(vmax, float(lot)))
        if vstep > 0:
            k = round(lot / vstep)
            lot = k * vstep

        # [LOT:NORMALIZE_LOG] — unique anchor
        # این لاگ فقط وقتی فعال می‌شود که رُندینگ به‌علت volume_step/volume_min اتفاق افتاده باشد.
        # پیام «due to margin» همچنان فقط در مسیر «affordable» باقی می‌ماند.
        try:
            desired = float(lot)
            normed = float(f"{lot:.4f}")
            if abs(normed - desired) > 1e-9:
                LOGGER.warning("Lot normalized (step/min): desired=%.4f -> normalized=%.4f (vmin=%.2f, vstep=%.2f)",
                               desired, normed, vmin, vstep)
        except Exception:
            pass
        return float(f"{lot:.4f}")


    def _max_affordable_lot(self, symbol: str, desired_lot: float, order_type: int) -> float:
        acct = mt5.account_info()
        if acct is None:
            return 0.0
        free_margin = float(getattr(acct, "margin_free", 0.0) or 0.0)
        if free_margin <= 0:
            return 0.0
        si = mt5.symbol_info(symbol)
        if si is None:
            return 0.0

        lo = 0.0
        hi = self._normalize_lot(symbol, desired_lot)
        if hi <= 0:
            return 0.0


        # quick check hi
        try:
            bid, ask = self.get_prices(symbol)
            px = ask if order_type == mt5.ORDER_TYPE_BUY else bid
            margin_req = mt5.order_calc_margin(order_type, symbol, hi, px)
            if margin_req is not None and margin_req <= free_margin:
                return hi
        except Exception:
            pass

        # binary search
        for _ in range(24):
            mid = self._normalize_lot(symbol, (lo + hi) / 2.0)
            if mid <= 0 or abs(hi - lo) < 1e-6:
                break
            try:
                bid, ask = self.get_prices(symbol)
                px = ask if order_type == mt5.ORDER_TYPE_BUY else bid
                margin_req = mt5.order_calc_margin(order_type, symbol, mid, px)
                if margin_req is not None and margin_req <= free_margin:
                    lo = mid
                else:
                    hi = mid
            except Exception:
                hi = mid
        return self._normalize_lot(symbol, lo)

    def _prepare_affordable_lot(self, symbol: str, desired_lot: float, order_type: int) -> float:
        desired_lot = self._normalize_lot(symbol, desired_lot)
        affordable = self._max_affordable_lot(symbol, desired_lot, order_type)
        return self._normalize_lot(symbol, affordable)

    # ---------- Order send (deal) with retry + optional SL/TP
    def _send_deal(self, *, symbol: str, lot: float, order_type: int, price: float, position_id: Optional[int] = None, sl: float | None = None, tp: float | None = None) -> Tuple[bool, Optional[int]]:
        deviation = int(((self._cfg.get("executor") or {}).get("slippage_cap_pips") or 10))
        allowed_retry = {getattr(mt5, "TRADE_RETCODE_REQUOTE", 10004), getattr(mt5, "TRADE_RETCODE_PRICE_CHANGED", 10032), getattr(mt5, "TRADE_RETCODE_TRADE_CONTEXT_BUSY", 10017)}
        attempt = 0
        while True:
            attempt += 1

            # --- StopsLevel guard (adjust SL/TP to satisfy broker min distance) ----- start
            try:
                si = mt5.symbol_info(symbol)
                if si and getattr(si, "point", None) is not None:
                    point = float(si.point or 0.0)
                    stops = float(getattr(si, "stops_level", 0) or 0) * point
                    if stops > 0.0:
                        if order_type == mt5.ORDER_TYPE_BUY:
                            if sl is not None and (price - float(sl)) < stops:
                                sl = float(price - stops)
                            if tp is not None and (float(tp) - price) < stops:
                                tp = float(price + stops)
                        else:  # SELL
                            if sl is not None and (float(sl) - price) < stops:
                                sl = float(price + stops)
                            if tp is not None and (price - float(tp)) < stops:
                                tp = float(price - stops)
            except Exception:
                # گارد دفاعی: در صورت خطای MT5 این بخش را نادیده بگیر
                pass
            # --- StopsLevel guard (adjust SL/TP to satisfy broker min distance) ----- end

            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lot),
                "type": order_type,
                "price": float(price),
                "deviation": deviation,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            if position_id is not None:
                req["position"] = int(position_id)
            if sl is not None:
                req["sl"] = float(sl)
            if tp is not None:
                req["tp"] = float(tp)

            # برای هر تلاش (و تلاش‌های retry)، تاخیرِ ارسال، تایید و اسلیپیج نسبت به قیمت درخواستی در لاگ استاندارد دیده می‌شود؛
            # قبل از order_send:
            import time  # اگر در بالای فایل نیست، اضافه‌اش کنید
            t0 = time.perf_counter()
            res = mt5.order_send(req)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            # بلافاصله بعد از res:
            fill_price = getattr(res, "price", None)
            try:
                slip = (abs((fill_price or req["price"]) - float(req["price"])) if fill_price is not None else None)
            except Exception:
                slip = None
            LOGGER.info("ExecMetrics: latency_ms=%.2f req_price=%s fill_price=%s slippage=%s deviation=%s",
                        lat_ms, req.get("price"), fill_price, slip, deviation)
            #  پایان بلوک تاخیر ارسال، تایید و اسلیپیج


            try:
                retcode = int(getattr(res, "retcode", -1))
                ticket = getattr(res, "order", None)
                if retcode == mt5.TRADE_RETCODE_DONE:
                    LOGGER.info("Order OK: sym=%s type=%s lot=%.3f ticket=%s sl=%s tp=%s", symbol, order_type, lot, ticket, sl, tp)
                    return True, int(ticket) if ticket is not None else None
                elif retcode in allowed_retry and attempt <= 3:
                    LOGGER.warning("Order retry (%d): retcode=%s comment=%s", attempt, retcode, getattr(res, "comment", None))
                    # refresh price then retry
                    bid, ask = self.get_prices(symbol)
                    price = ask if order_type == mt5.ORDER_TYPE_BUY else bid
                    continue
                else:
                    LOGGER.error("Order FAIL: sym=%s type=%s lot=%.3f retcode=%s comment=%s", symbol, order_type, lot, retcode, getattr(res, "comment", None))
                    return False, None
            except Exception as ex:
                LOGGER.exception("order_send parse failed: %s", ex)
                return False, None

    # ---------- Public trading API
    def buy(self, symbol: str, lot: float, sl: float | None = None, tp: float | None = None) -> Tuple[bool, Optional[int]]:
        lv, eq = self.get_account_leverage(), self.get_account_equity()
        if (lv is None or lv <= 0) or (eq is None or eq <= 0):
            LOGGER.error("Safety gate: invalid leverage/equity. Refusing to trade."); return False, None

        bid, ask = self.get_prices(symbol)
        lot2 = self._prepare_affordable_lot(symbol, lot, mt5.ORDER_TYPE_BUY)
        if lot2 <= 0:
            LOGGER.error("Order CANCELLED (no affordable lot): symbol=%s desired=%.4f", symbol, lot)
            return False, None
        if abs(lot2 - lot) > 1e-6:
            LOGGER.warning("Lot adjusted due to margin: desired=%.4f -> affordable=%.4f", lot, lot2)
        return self._send_deal(symbol=symbol, lot=lot2, order_type=mt5.ORDER_TYPE_BUY, price=ask, sl=sl, tp=tp)

    def sell(self, symbol: str, lot: float, sl: float | None = None, tp: float | None = None) -> Tuple[bool, Optional[int]]:
        lv, eq = self.get_account_leverage(), self.get_account_equity()
        if (lv is None or lv <= 0) or (eq is None or eq <= 0):
            LOGGER.error("Safety gate: invalid leverage/equity. Refusing to trade."); return False, None
        bid, ask = self.get_prices(symbol)
        lot2 = self._prepare_affordable_lot(symbol, lot, mt5.ORDER_TYPE_SELL)
        if lot2 <= 0:
            LOGGER.error("Order CANCELLED (no affordable lot): symbol=%s desired=%.4f", symbol, lot)
            return False, None
        if abs(lot2 - lot) > 1e-6:
            LOGGER.warning("Lot adjusted due to margin: desired=%.4f -> affordable=%.4f", lot, lot2)
        return self._send_deal(symbol=symbol, lot=lot2, order_type=mt5.ORDER_TYPE_SELL, price=bid, sl=sl, tp=tp)

    def close_long(self, symbol: str) -> Tuple[bool, Optional[int]]:
        pos_list = mt5.positions_get(symbol=symbol)
        if not pos_list:
            return True, None
        ok_any, last_ticket = True, None
        bid, ask = self.get_prices(symbol)
        for p in pos_list:
            if int(getattr(p, "type", -1)) == mt5.POSITION_TYPE_BUY:
                vol = float(getattr(p, "volume", 0.0))
                pid = int(getattr(p, "ticket", 0))
                ok, ticket = self._send_deal(symbol=symbol, lot=vol, order_type=mt5.ORDER_TYPE_SELL, price=bid, position_id=pid)
                ok_any &= ok
                last_ticket = ticket or last_ticket
        return ok_any, last_ticket

    def close_short(self, symbol: str) -> Tuple[bool, Optional[int]]:
        pos_list = mt5.positions_get(symbol=symbol)
        if not pos_list:
            return True, None
        ok_any, last_ticket = True, None
        bid, ask = self.get_prices(symbol)
        for p in pos_list:
            if int(getattr(p, "type", -1)) == mt5.POSITION_TYPE_SELL:
                vol = float(getattr(p, "volume", 0.0))
                pid = int(getattr(p, "ticket", 0))
                ok, ticket = self._send_deal(symbol=symbol, lot=vol, order_type=mt5.ORDER_TYPE_BUY, price=ask, position_id=pid)
                ok_any &= ok
                last_ticket = ticket or last_ticket
        return ok_any, last_ticket

    def close_partial(self, symbol: str, side: str, volume: float) -> Tuple[bool, Optional[int]]:
        """side in {'long','short'}: بستن بخشی از پوزیشن‌ها به اندازهٔ volume"""
        if volume <= 0:
            return False, None
        pos_list = mt5.positions_get(symbol=symbol)
        if not pos_list:
            return False, None
        bid, ask = self.get_prices(symbol)
        remaining = float(volume)
        ok_any, last_ticket = True, None
        for p in pos_list:
            ptype = int(getattr(p, "type", -1))
            if (side == "long" and ptype == mt5.POSITION_TYPE_BUY) or (side == "short" and ptype == mt5.POSITION_TYPE_SELL):
                vol = float(getattr(p, "volume", 0.0))
                pid = int(getattr(p, "ticket", 0))
                use = min(vol, remaining)
                if use <= 0:
                    continue
                if side == "long":
                    ok, ticket = self._send_deal(symbol=symbol, lot=use, order_type=mt5.ORDER_TYPE_SELL, price=bid, position_id=pid)
                else:
                    ok, ticket = self._send_deal(symbol=symbol, lot=use, order_type=mt5.ORDER_TYPE_BUY, price=ask, position_id=pid)
                ok_any &= ok
                last_ticket = ticket or last_ticket
                remaining -= use
                if remaining <= 1e-9:
                    break
        return ok_any, last_ticket

    # ---------- Modify SL/TP for a position
    def modify_position_sl_tp(self, position_id: int, sl: float | None, tp: float | None) -> bool:
        try:
            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": int(position_id),
            }
            if sl is not None: req["sl"] = float(sl)
            if tp is not None: req["tp"] = float(tp)
            res = mt5.order_send(req)
            ok = (res is not None and int(getattr(res, "retcode", -1)) == mt5.TRADE_RETCODE_DONE)
            if not ok:
                LOGGER.error("SLTP MODIFY FAIL: pos=%s ret=%s comment=%s", position_id, getattr(res, "retcode", None), getattr(res, "comment", None))
            else:
                LOGGER.info("SLTP MODIFY OK: pos=%s sl=%s tp=%s", position_id, sl, tp)
            return ok
        except Exception:
            LOGGER.exception("modify_position_sl_tp() failed.")
            return False

    def __del__(self):
        try:
            if hasattr(self, "_conn") and self._conn is not None:
                self._conn.shutdown()
        except Exception:
            pass


