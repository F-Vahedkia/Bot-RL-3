# -*- coding: utf-8 -*-
# f12_execution/executor.py   (اجرای زنده/نیمه‌زنده پالیسی روی Env)
# Status in (Bot-RL-3): Completed

"""
اجرای زنده/نیمه‌زنده پالیسی روی Env
- فقط از APIهای موجود پروژه استفاده می‌کند (بدون حدس).
- از cfg «در حافظه» استفاده می‌شود؛ هیچ نوشتن دیسکی وسط اجرا انجام نمی‌شود.
- اگر --live فعال باشد و آداپتر بروکر در دسترس نباشد، اجرا متوقف می‌شود.

پیام‌های runtime انگلیسی هستند.
توضیحات فارسی صرفاً برای توسعه‌دهندگان داخل پروژه است.
"""

from __future__ import annotations

import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Optional
import json, csv, os, time

from f02_utils.config_loader import load_config
from f06_env.OLD_files.OLD_utils import paths_from_cfg, resolve_spread_selection
from f06_env.OLD_files.OLD_trading_env import TradingEnv, EnvConfig  # موجود در پروژه
from f10_evaluation.backtest import transform_obs

try:
    from f09_training.agent_sb3 import load_sb3 as _load_sb3
except Exception:
    _load_sb3 = None

from f06_env.OLD_files.OLD_risk import RiskManager

# [OMS:IMPORTS] — unique anchor
from hashlib import blake2s
# [NEWS_GATE:IMPORTS]
# وارد کردن make_news_gate و زمان UTC
from f06_news.integration import make_news_gate
from datetime import datetime, timezone


LOGGER = logging.getLogger("executor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live/Semi-live executor")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("-c", "--config", default="f01_config/config.yaml")
    p.add_argument("--base-tf", default=None, help="override base timeframe, e.g. M5")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--reward", default="pnl")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "live"])
    # [MODE:CLI]
    # ------------------------------------------- start
    p.add_argument("--mode", choices=["dry", "semi", "live"], default=None,
                   help="execution mode: dry (no broker), semi (canary), live (full)")
    p.add_argument("--canary-volume", type=float, default=None,
                   help="0..1; scales effective lot in semi/live (defaults from config/executor.canary_volume or 1.0)")
    p.add_argument("--live", action="store_true", help="(legacy) enable broker execution (requires adapter)")
    p.add_argument("--model-path", default=None, help="optional absolute model path")
    p.add_argument("--sb3-alg", default="auto", choices=["auto","PPO","SAC","TD3","A2C","DQN","DDPG"])
    # ------------------------------------------- end
    return p.parse_args()



def _resolve_model_path(cfg: dict, symbol: str, base_tf: str, explicit: Optional[str]) -> Optional[Path]:
    """
    تعیین مسیر مدل SB3 (.zip) برای اجرا:
      - اگر explicit داده شده و وجود داشته باشد → همان
      - در غیر این صورت: ابتدا در prod سپس staging جدیدترین .zip با الگوی
        "<ALG>_<SYMBOL>_<TF>_*.zip" یا "<ALG>_<SYMBOL>_<TF>.zip"
    """
    paths = paths_from_cfg(cfg)
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    prod_dir = paths["models_dir"] / "prod"
    stg_dir  = paths["models_dir"] / "staging"
    patterns = [f"*_{symbol}_{base_tf}_*.zip", f"*_{symbol}_{base_tf}.zip"]

    for d in (prod_dir, stg_dir):
        cands = []
        for pat in patterns:
            cands.extend(d.glob(pat))
        if cands:
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0]
    return None



def _load_broker_adapter(live: bool):
    """
    تلاش برای لود آداپتر بروکر.
    - اگر --live False باشد: No-OpAdapter برمی‌گرداند (صرفاً برای سازگاری).
    - اگر --live True و ماژول در دسترس نباشد: خطا می‌دهیم و اجرا را قطع می‌کنیم.
    """
    if not live:
        from f12_execution.broker_adapter_mt5 import NoOpBroker as BA  # موجود در همین پوشه
        LOGGER.info("Broker execution disabled; using No-Op broker adapter.")
        return BA()
    try:
        from f12_execution.broker_adapter_mt5 import MT5Broker as BA  # اگر MT5 آماده باشد
        b = BA()
        LOGGER.info("Broker adapter initialized.")
        return b
    except Exception as e:
        LOGGER.error("Live broker adapter is required but not available: %s", e)
        raise


def _env_from_cfg(cfg: dict, symbol: str, base_tf: Optional[str], window: int, reward: str, normalize: bool) -> TradingEnv:
    """ساخت Env از روی cfg در حافظه (بدون خواندن/نوشتن دیسک)."""
    base_tf_cfg = (cfg.get("features") or {}).get("base_timeframe", "M5")
    base_tf_final = (base_tf or base_tf_cfg).upper()
    env_cfg = EnvConfig(
        symbol=symbol,
        base_tf=base_tf_final,
        window_size=window,
        reward_mode=reward,
        normalize=bool(normalize),
    )
    return TradingEnv(cfg=cfg, env_cfg=env_cfg)


def _act_and_execute(broker, prev_pos: int, action: int, symbol: str, lot: float, sl: float | None = None, tp: float | None = None):
    """
    خروجی: (new_pos, {"opened": (success, ticket) or None, "closed": (success, ticket) or None, "entry_price": price or None, "position_id": id or None})
    """
    result = {"opened": None, "closed": None, "entry_price": None, "position_id": None}
    if action not in (-1, 0, 1):
        return prev_pos, result
    if action == prev_pos:
        return prev_pos, result

    # close previous
    if prev_pos == 1:
        result["closed"] = broker.close_long(symbol)
    elif prev_pos == -1:
        result["closed"] = broker.close_short(symbol)

    # open new
    if action == 1:
        ok, ticket = broker.buy(symbol, lot, sl=sl, tp=tp)
        result["opened"] = (ok, ticket)
        # ذخیرهٔ قیمت ورود (برای R) اگر بتوانیم
        if hasattr(broker, "get_prices"):
            _, ask = broker.get_prices(symbol)
            result["entry_price"] = float(ask) if ask else None
    elif action == -1:
        ok, ticket = broker.sell(symbol, lot, sl=sl, tp=tp)
        result["opened"] = (ok, ticket)
        if hasattr(broker, "get_prices"):
            bid, _ = broker.get_prices(symbol)
            result["entry_price"] = float(bid) if bid else None

    # position_id را اگر بتوانیم از positions_get برداریم (ساده‌سازی: آخرین پوزیشن سمت مربوط)
    try:
        import MetaTrader5 as mt5  # امن است چون در مسیر live است
        poss = mt5.positions_get(symbol=symbol)
        if poss:
            if action == 1:
                cand = [p for p in poss if int(getattr(p, "type", -1)) == mt5.POSITION_TYPE_BUY]
            else:
                cand = [p for p in poss if int(getattr(p, "type", -1)) == mt5.POSITION_TYPE_SELL]
            if cand:
                result["position_id"] = int(getattr(cand[-1], "ticket", 0))
    except Exception:
        pass

    return action, result




def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config, enable_env_override=True)

    # کم‌حرف کردن لاگ‌ها در اجرا
    logging.getLogger("hparam_search").setLevel(logging.WARNING)
    logging.getLogger("self_optimizer").setLevel(logging.WARNING)
    logging.getLogger("backtest_api").setLevel(logging.WARNING)

    # Overrideهای CLI روی cfg در حافظه
    (cfg.setdefault("env", {}))["normalize"] = bool(args.normalize)
    if args.base_tf:
        (cfg.setdefault("features", {}))["base_timeframe"] = str(args.base_tf).upper()

    # --- Pre-flight: load broker & validate live inputs BEFORE building Env ------------ start
    # [MODE:INIT] — unique anchor
    exec_mode = (args.mode or ("live" if args.live else "dry")).lower()
    if exec_mode not in ("dry", "semi", "live"):
        exec_mode = "dry"

    # [MODE:INIT][CANARY:CFG] — unique anchor
    cd = ((cfg.get("executor") or {}).get("canary_deployment") or {})
    canary_enabled = bool(cd.get("enabled", False))
    canary_volume = float(args.canary_volume if (args.canary_volume is not None) else (cd.get("volume_multiplier") or 1.0))
    canary_volume = max(0.0, min(1.0, canary_volume))


    is_live = (exec_mode in ("semi", "live"))

    bt = ((cfg.get("evaluation") or {}).get("backtest") or {})
    sel = resolve_spread_selection(cfg, has_broker_series=bool(is_live or bt.get("reuse_historical_spread")))
    LOGGER.info("[COSTS] spread source=%s fixed_pts=%.4f", sel["source"], float(sel["value"]))

    # هم‌خوانی کامل هزینه‌ها با کانفیگ (بدون override) --- start
    tr = cfg.setdefault("trading", {}); costs = tr.setdefault("costs", {})
    bt = ((cfg.get("evaluation") or {}).get("backtest") or {})

    # فقط در صورت نبود مقدار، دیفالت‌ها را از evaluation.backtest ست می‌کنیم
    costs.setdefault("commission_per_lot", float(bt.get("commissions_per_lot_usd") or 0.0))
    costs.setdefault("slippage_pts",       float(bt.get("slippage_pips_mean") or 0.0))
    # نکتهٔ کلیدی: point_value هرگز در trading.costs ذخیره نشود؛ منبع فقط evaluation.backtest است
    _pv_bt = float(bt.get("point_value") or 0.0)

    LOGGER.info(
        "Costs fallback set: commission=%.4f slippage=%.4f point_value(evaluation.backtest)=%.4f",
        float(costs["commission_per_lot"]), float(costs["slippage_pts"]), _pv_bt
    )
    # --- end

    broker = _load_broker_adapter(is_live)
    LOGGER.info("ExecMode=%s canary_volume=%.3f live=%s", exec_mode, canary_volume, is_live)
    # --- Pre-flight: load broker & validate live inputs BEFORE building Env ------------ end

    eq0 = broker.get_account_equity() if hasattr(broker, "get_account_equity") else None
    lv0 = broker.get_account_leverage() if hasattr(broker, "get_account_leverage") else None
    pv0 = broker.get_pip_value_per_lot(args.symbol) if hasattr(broker, "get_pip_value_per_lot") else None
    if (args.live or args.split=="live") and (not eq0 or not lv0 or not pv0):
        LOGGER.error("Live aborted: missing/invalid pre-flight inputs (equity/leverage/pip_value).")
        return 2

    env = _env_from_cfg(cfg, args.symbol, args.base_tf, args.window, args.reward, args.normalize)


    base_tf_cfg = (cfg.get("features") or {}).get("base_timeframe", "M5")
    base_tf_final = (args.base_tf or base_tf_cfg).upper()
    model_path = _resolve_model_path(cfg, args.symbol, base_tf_final, args.model_path)

    policy = None
    if model_path is None:
        LOGGER.warning("No model found (prod/staging). Will act randomly.")
    else:
        try:
            policy = (_load_sb3(args.sb3_alg, model_path) if (_load_sb3 and model_path) else None)
            LOGGER.info("Model loaded: %s", model_path)
        except Exception as e:
            LOGGER.error("Failed to load SB3 model: %s", e)
            policy = None


    broker = broker if ("broker" in locals() and broker) else _load_broker_adapter(args.live)



    # Journal setup
    jdir = os.path.join("f13_models", "logs")
    os.makedirs(jdir, exist_ok=True)
    jcsv = os.path.join(jdir, f"journal_{args.symbol}.csv")
    jjson = os.path.join(jdir, f"journal_{args.symbol}.jsonl")
    if not os.path.exists(jcsv):
        with open(jcsv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts","event","symbol","action","lot","price","ticket","ok","ret","pnl"])
    
    # [OBS:STATE] — unique anchor
    obs_metrics = {"attempts": 0, "opens": 0, "fails": 0, "closes": 0}
    # [QC:STATE] — unique anchor
    qc_prev_mid = None

    latency_sla_ms = int(((cfg.get("executor") or {}).get("latency_sla_ms") or 0) or 0)

    # [OMS:STATE] — unique anchor
    sent_intents = set()
    last_intent_key = None

    # --- Risk wiring ---
    rm = RiskManager.from_config(cfg)
    # [NEWS_GATE:BUILD]
    # ------------------------------------------- start
    # ساخت NewsGate بر اساس config و کش خبر؛ اگر غیرفعال/خالی باشد، بی‌اثر و فقط لاگ.
    news_gate = None
    try:
        news_gate = make_news_gate(cfg, symbol=args.symbol)
        if news_gate is None:
            LOGGER.info("NewsGate: disabled or empty cache.")
        else:
            LOGGER.info("NewsGate: initialized for symbol=%s", args.symbol)
    except Exception as _ex:
        LOGGER.warning("NewsGate init failed: %s", _ex)
    # ------------------------------------------- end

    lot_base = float(((cfg.get("executor") or {}).get("lot") or 0.10))  # fallback when dynamic inputs unavailable

    # وضعیت ورودی‌های ریسک پویا را یک‌بار اعلام می‌کنیم (برای شفافیت)
    try:
        eq0 = broker.get_account_equity() if hasattr(broker, "get_account_equity") else None
        lv0 = broker.get_account_leverage() if hasattr(broker, "get_account_leverage") else None
        pv0 = broker.get_pip_value_per_lot(args.symbol) if hasattr(broker, "get_pip_value_per_lot") else None
        if (args.live or args.split == "live") and (eq0 is None or float(eq0) <= 0):
            LOGGER.error("Equity unavailable (None/<=0). Aborting execution by safety rule.")
            return 2
        if (args.live or args.split == "live") and (lv0 is None or float(lv0) <= 0):
            LOGGER.error("Leverage unavailable (None/<=0). Aborting execution by safety rule.")
            return 2

        LOGGER.info("Risk inputs snapshot: equity=%s leverage=%s pip_value_per_lot=%s lot_base=%.5f", eq0, lv0, pv0, lot_base)
        
    except Exception:
        if args.live or args.split == "live":
            LOGGER.error("Leverage probe failed. Aborting execution by safety rule.")
            return 2
        LOGGER.info("Risk inputs snapshot: unavailable (falling back to lot_base=%.5f)", lot_base)



    obs, info = env.reset(split=args.split)
    if (args.live or args.split == "live") and ((eq0 is None or (isinstance(eq0,(int,float)) and eq0<=0)) or (lv0 is None or (isinstance(lv0,(int,float)) and lv0<=0)) or (pv0 is None or (isinstance(pv0,(int,float)) and pv0<=0))):
        try:
            with open(jcsv, "a", newline="") as f:
                csv.writer(f).writerow([int(time.time()), "abort", args.symbol, None, None, None, None, False, "ABORT_PREFLIGHT", None])
        except Exception:
            pass

        LOGGER.error("Live aborted: missing/invalid pre-flight inputs (equity/leverage/pip_value).")
        return 2

    try:
        if isinstance(info, dict):
            info["leverage"] = lv0 if ('lv0' in locals()) else (broker.get_account_leverage() if hasattr(broker, "get_account_leverage") else None)
    except Exception:
        pass

    total_r, steps, pos = 0.0, 0, 0
    # Idempotency: avoid duplicate opens within this session
    recent_intents = set()

    eq_start_day = eq0
    eq_peak = eq0

    # مدیریت پرایس‌اکشن/ریسک: وضعیت محلی
    pa_state = {
        "entry": None,           # قیمت ورود آخرین پوزیشن
        "stop_dist": None,       # فاصله‌ی SL اولیه (برحسب قیمت)
        "be_done": False,        # آیا BE اعمال شده؟
        "last_sl": None,         # آخرین SL اعمال‌شده برای جلوگیری از آپدیت‌های تکراری
        "last_mng_step": -9999,  # آخرین گامی که مدیریت پوزیشن اجرا شد (برای interval)
    }
    # فاصله‌ی بین اجرای مدیریت پوزیشن (گام‌ها)؛ اگر در cfg نبود، 5 پیش‌فرض
    ps_manage_interval = int(((cfg.get("risk") or {}).get("position_sizing") or {}).get("manage_interval", 5))
    if ps_manage_interval < 1:
        ps_manage_interval = 5


    while steps < args.steps:
        x = transform_obs(obs, mode="mean", last_k=16).astype(np.float32)
        if policy is None:
            a = int(np.random.randint(0, 3))  # {0,1,2}
        else:
            a, _ = policy.predict(x, deterministic=True)
            a = int(a)


        step = env.step(a - 1)
        obs = step.observation
        r = float(step.reward)
        term = bool(step.terminated)
        trunc = bool(step.truncated)
        
        info = step.info
        if args.live and hasattr(broker,"get_account_equity"):
            eq_now = broker.get_account_equity()
            if (eq_now is not None) and (rm.daily_loss_breached(eq_start_day, eq_now) or rm.total_drawdown_breached(eq_peak, eq_now)):
                LOGGER.error("Risk gate breached (daily/total). Aborting live execution.")
                return 2
            if (eq_now is not None) and (eq_now > (eq_peak or 0)): eq_peak = eq_now



        total_r += r
        steps += 1

        # --- Dynamic position sizing (if inputs are available) ---
        lot_eff = lot_base   # Effective Lot Size
        equity = None
        pip_val = None
        atr_price = None
        try:
            if hasattr(broker, "get_account_equity"):
                equity = broker.get_account_equity()
            if hasattr(broker, "get_pip_value_per_lot"):
                pip_val = broker.get_pip_value_per_lot(args.symbol)
            if isinstance(info, dict):
                atr_price = info.get("atr_price") or info.get("atr")
            if (args.live or args.split == "live") and (not equity or not pip_val or not atr_price):
                LOGGER.error("Live aborted: dynamic risk inputs unavailable (equity/pip_value/atr).")
                return 2

            if equity and pip_val and atr_price:
                stop_dist = rm.stop_distance_from_atr(float(atr_price))
                if stop_dist > 0.0:
                    lot_dyn = rm.size_fixed_fractional(
                        equity_usd=float(equity),
                        stop_distance_price=float(stop_dist),
                        pip_value_per_lot=float(pip_val),
                        price_point=(cfg.get("executor") or {}).get("price_point")
                    )
                    lot_eff = rm.cap_lot(lot_dyn)    # Effective Lot Size
                else:
                    lot_eff = rm.cap_lot(lot_base)
            else:
                lot_eff = rm.cap_lot(lot_base)
        except Exception:
            lot_eff = rm.cap_lot(lot_base)

        desired = (a - 1)
        # فقط وقتی اکشن تغییر کند سفارش/لاگ می‌دهیم
        if desired != pos:
            intent = (desired, round(float(atr_price or 0.0),3))
            if intent in recent_intents:
                LOGGER.info("Idempotency: duplicate intent skipped.")
                continue
            recent_intents.add(intent)

            # --- SL/TP محاسبه از روی ATR و تنظیمات ریسک
            sl_price = None
            tp_price = None
            try:
                # atr_price از info (در همین گام یا گام قبلی) آمده است
                if (equity is not None) and (pip_val is not None) and (atr_price is not None):
                    ps = rm.rcfg.position_sizing
                    stop_dist = rm.stop_distance_from_atr(float(atr_price)) if ps.use_atr_for_sl else 0.0
                    take_dist = float(ps.tp_atr_mult * float(atr_price)) if ps.trailing_enabled or ps.tp_atr_mult else float(ps.tp_atr_mult * float(atr_price))
                    bid, ask = broker.get_prices(args.symbol) if hasattr(broker, "get_prices") else (None, None)
                    if desired == 1 and ask:
                        if stop_dist > 0: sl_price = float(ask - stop_dist)
                        if take_dist > 0: tp_price = float(ask + take_dist)
                    elif desired == -1 and bid:
                        if stop_dist > 0: sl_price = float(bid + stop_dist)
                        if take_dist > 0: tp_price = float(bid - take_dist)
            except Exception:
                sl_price = None
                tp_price = None

            # --- Pre-trade risk gate (prevent order if already breached)
            if args.live and hasattr(broker, "get_account_equity"):
                eq_now = broker.get_account_equity()
                if (eq_now is not None) and (rm.daily_loss_breached(eq_start_day, eq_now) or rm.total_drawdown_breached(eq_peak, eq_now)):
                    LOGGER.error("Risk gate breached pre-trade. Aborting live execution.")
                    return 2

            # --- Pre-trade budget vs. entry-cost gate
            risk_cfg = (cfg.get("risk") or {})
            max_d = float(risk_cfg.get("max_daily_loss_pct") or 0)
            max_dd = float(risk_cfg.get("max_total_drawdown_pct") or 0)
            eq_now = broker.get_account_equity() if hasattr(broker, "get_account_equity") else None
            pp = ( (cfg.get("executor") or cfg.get("execution") or {}).get("price_point") )
            sp_thr_pips = ( (cfg.get("executor") or cfg.get("execution") or {}).get("slippage_cap_pips") )
            if args.live and eq_now and pp and pip_val and lot_eff:
                ba = broker.get_prices(args.symbol) if hasattr(broker,"get_prices") else None
                if ba and ba[0] and ba[1]:
                    spread_pips = (float(ba[1]) - float(ba[0])) / float(pp)
                    entry_cost = spread_pips * float(pip_val) * float(lot_eff)
                    # remaining budgets (USD)
                    daily_rem = max(0.0, (max_d * float(eq_start_day)) - max(0.0, float(eq_start_day) - float(eq_now)))
                    total_rem = max(0.0, (max_dd * float(eq_peak)) - max(0.0, float(eq_peak) - float(eq_now)))
                    budget_left = min(daily_rem, total_rem)
                    if budget_left <= entry_cost:
                        LOGGER.error("Pre-trade budget insufficient (entry_cost>=remaining_budget). Aborting live execution.")
                        return 2

            # [CANARY:APPLY]
            # ----------------------------------- start
            # [CANARY:APPLY][ENABLED] — unique anchor
            if canary_enabled and exec_mode in ("semi", "live") and canary_volume < 1.0:
                prev_lot = float(lot_eff)
                lot_eff = float(max(0.0, prev_lot * canary_volume))
                try:
                    with open(jcsv, "a", newline="") as f:
                        csv.writer(f).writerow([int(time.time()*1000), "canary_apply", args.symbol,
                                                desired, lot_eff, None, None, True,
                                                f"CANARY:prev={prev_lot:.4f};vol={canary_volume:.3f}", None])
                except Exception:
                    pass
            # ----------------------------------- end

            # [OMS:INTENT] — unique anchor
            key = f"{args.symbol}|{desired}|{round(lot_eff or 0.0,4)}|{round(sl_price or 0.0,5)}|{round(tp_price or 0.0,5)}"
            intent_id = blake2s(key.encode("utf-8"), digest_size=8).hexdigest()

            # [OBS:INTENT_TS] — unique anchor
            obs_metrics["attempts"] += 1
            last_intent_ts = int(time.time()*1000)

            if (last_intent_key == key) or (intent_id in sent_intents):
                try:
                    with open(jcsv,"a",newline="") as f:
                        csv.writer(f).writerow([int(time.time()*1000),"idempotent_skip",args.symbol,desired,lot_eff,None,None,True,"SKIP_IDEMPOTENT",None])
                except Exception: pass
                continue
            sent_intents.add(intent_id); last_intent_key = key
            try:
                with open(jjson,"a") as f: f.write(json.dumps({"ts":int(time.time()*1000),"event":"intent","symbol":args.symbol,"intent_id":intent_id,"key":key})+"\n")
            except Exception: pass

            LOGGER.info("Decision: action=%s lot_eff=%.5f sl=%s tp=%s", desired, lot_eff, sl_price, tp_price)
            # [NEWS_GATE:CHECK]
            # ----------------------------------- start
            # قبل از Spread Guard، وضعیت خبر بررسی می‌شود:
            # freeze: ترید لغو و در ژورنال ثبت می‌شود.
            # reduce_risk: lot_eff با reduce_scale (از کانفیگ) کاهش یافته و ثبت می‌شود.

            if news_gate is not None:
                _st = news_gate.status(datetime.now(timezone.utc))
                if _st.get("freeze"):
                    # Freeze: skip trade (log to journal)
                    try:
                        with open(jcsv, "a", newline="") as f:
                            csv.writer(f).writerow([int(time.time()*1000), "news_freeze", args.symbol, desired, None, None, None, True, "SKIP_NEWS_FREEZE", None])
                    except Exception:
                        pass
                    continue
                elif _st.get("reduce_risk"):
                    # Reduce: scale down effective lot
                    reduce_scale = float(_st.get("reduce_scale") or 0.5)
                    lot_eff = max(0.0, float(lot_eff) * reduce_scale)
                    try:
                        with open(jcsv, "a", newline="") as f:
                            csv.writer(f).writerow([int(time.time()*1000), "news_reduce", args.symbol, desired, lot_eff, None, None, True, f"NEWS_REDUCE@{reduce_scale}", None])
                    except Exception:
                        pass
            # ----------------------------------- end
            

            # [HOURS:GATE] — unique anchor
            # از کلیدهای موجود در config.yaml زیر sessions: (همان‌هایی که در f03_data هم استفاده می‌شوند) بهره می‌بریم.
            sessions = ((cfg.get("env") or {}).get("sessions") or {})
            if sessions:
                now_utc = datetime.now(timezone.utc).time()
                def _in(s):
                    try:
                        h1,m1 = [int(x) for x in str(s.get("start_utc","00:00")).split(":")[:2]]
                        h2,m2 = [int(x) for x in str(s.get("end_utc","00:00")).split(":")[:2]]
                        t1 = (h1, m1); t2 = (h2, m2); t0 = (now_utc.hour, now_utc.minute)
                        return (t1 <= t0 <= t2) if t1 <= t2 else (t0 >= t1 or t0 <= t2)  # پشتیبانی از بازه‌های شب‌رو
                    except Exception: return True
                open_now = any(_in(v) for v in sessions.values())
                if not open_now:
                    try:
                        with open(jcsv, "a", newline="") as f:
                            csv.writer(f).writerow([int(time.time()*1000),"trading_hours_block",args.symbol,desired,lot_eff,None,None,True,"SKIP_TRADING_HOURS",None])
                    except Exception:
                        pass
                    LOGGER.info("Trading-hours gate: market closed by configured sessions. Skipping.")
                    continue


            # [QC:GATE] — unique anchor
            # ------------------------- start
            qc = ((cfg.get("executor") or {}).get("qc") or {})
            pp = ((cfg.get("executor") or cfg.get("execution") or {}).get("price_point"))
            # stale: اگر ثانیه آستانه تنظیم باشد و آخرین تیک قدیمی‌تر از آن باشد → skip
            try:
                stale_s = int(qc.get("stale_seconds") or 0)
                # اگر آخرین تیک قدیمی تر از چند ثانیه باشد، رد میشود
            except Exception:
                stale_s = 0
            if stale_s and hasattr(broker, "get_last_tick_time"):
                ts = broker.get_last_tick_time(args.symbol)
                if ts and (datetime.now(timezone.utc) - ts).total_seconds() > stale_s:
                    try:
                        with open(jcsv,"a",newline="") as f:
                            csv.writer(f).writerow([int(time.time()*1000),"qc_stale",args.symbol,desired,lot_eff,None,None,True,f"STALE>{stale_s}s",None])
                    except Exception: pass
                    
                    # [QC:JSONL] stale --- start
                    try:
                        with open(jjson, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "ts": int(datetime.now(timezone.utc).timestamp()),
                                "event": "qc_skip", "symbol": args.symbol, "reason": "stale",
                                "value_sec": float((datetime.now(timezone.utc) - ts).total_seconds()),
                                "thr_sec": int(stale_s)
                            }) + "\n")
                    except Exception: pass
                    # --- end

                    LOGGER.warning("QC: Stale tick (> %ss). Skipping trade.", stale_s)
                    continue
            # spike: جهش شدید نسبت به گام قبلی (بر حسب پیپ) → skip
            # اگر جهش نسبت به گام قبلی بیشتر از چند پیپ باشد، رد میشود.
            spike_pips = qc.get("spike_pips")
            if pp and spike_pips and hasattr(broker, "get_prices"):
                try:
                    bid, ask = broker.get_prices(args.symbol)
                    if bid and ask:
                        mid = (float(bid) + float(ask)) / 2.0
                        if qc_prev_mid is not None:
                            dpips = abs(mid - qc_prev_mid) / float(pp)
                            if dpips > float(spike_pips):
                                try:
                                    with open(jcsv,"a",newline="") as f:
                                        csv.writer(f).writerow([int(time.time()*1000),"qc_spike",args.symbol,desired,lot_eff,None,None,True,f"SPIKE>{dpips:.2f}p",None])
                                except Exception: pass
                                
                                # [QC:JSONL] spike --- start
                                try:
                                    with open(jjson, "a", encoding="utf-8") as f:
                                        f.write(json.dumps({
                                            "ts": int(datetime.now(timezone.utc).timestamp()),
                                            "event": "qc_skip", "symbol": args.symbol, "reason": "spike",
                                            "value_pips": float(dpips), "thr_pips": float(spike_pips)
                                        }) + "\n")
                                except Exception: pass
                                # --- end
                                LOGGER.warning("QC: Spike detected (%.2fp). Skipping trade.", dpips)
                                continue
                        qc_prev_mid = mid
                except Exception:
                    pass
            # ------------------------- end

            # --- Spread Guard (if configured) ---
        
            pp = ((cfg.get("executor") or cfg.get("execution") or {}).get("price_point"))
            thr_pips = ((cfg.get("executor") or cfg.get("execution") or {}).get("slippage_cap_pips"))
            sp_thr = (float(thr_pips) * float(pp)) if (thr_pips and pp) else None
            if sp_thr and hasattr(broker,"get_prices"):
                bid_ask = broker.get_prices(args.symbol); 
                if bid_ask and all(bid_ask):
                    spread = float(bid_ask[1] - bid_ask[0])
                    LOGGER.info("SpreadGuard: pp=%s thr_pips=%s thr_price=%s bid=%s ask=%s spread=%s",
                            pp, thr_pips, sp_thr, bid_ask[0], bid_ask[1], spread)

                    if spread > float(sp_thr):
                        try:
                            with open(jcsv,"a",newline="") as f:
                                csv.writer(f).writerow([int(time.time()),"skip",args.symbol,desired,None,None,None,True,"SKIP_SPREAD_GUARD",None])
                        except Exception:
                            pass
                        continue



            pos, ord_info = _act_and_execute(broker, pos, desired, args.symbol, lot_eff, sl_price, tp_price)
            opened = ord_info.get("opened")
            closed = ord_info.get("closed")
            if opened is not None:
                ok, ticket = opened
                LOGGER.info("Order-OPEN result: ok=%s ticket=%s", ok, ticket)

                # [OBS:OPEN_METRICS] — unique anchor
                now_ms = int(time.time()*1000); latency_ms = (now_ms - last_intent_ts) if "last_intent_ts" in locals() else None
                if ok: obs_metrics["opens"] += 1
                try:
                    ba = broker.get_prices(args.symbol) if hasattr(broker,"get_prices") else None
                    spread = (float(ba[1]) - float(ba[0])) if (ba and ba[0] and ba[1]) else None
                    with open(jjson,"a") as f: f.write(json.dumps({"ts":now_ms,"event":"open_metrics","symbol":args.symbol,"intent_id":intent_id,"latency_ms":latency_ms,"spread":spread,"ok":bool(ok),"ticket":ticket})+"\n")
                    if latency_sla_ms and latency_ms and latency_ms>latency_sla_ms: f.write(json.dumps({"ts":now_ms,"event":"alert_latency_sla","symbol":args.symbol,"latency_ms":latency_ms,"sla_ms":latency_sla_ms})+"\n")
                except Exception: pass

                # [OMS:RECONCILE] — unique anchor
                if is_live:
                    try:
                        import MetaTrader5 as mt5
                        poss = mt5.positions_get(symbol=args.symbol)
                        found = len(poss or [])
                        with open(jjson,"a") as f:
                            f.write(json.dumps({"ts":int(time.time()*1000),"event":"reconcile_check","symbol":args.symbol,"intent_id":intent_id,"positions_found":found})+"\n")
                    except Exception:
                        pass


                # Journal: execution metrics at open (no new headers)
                try:
                    # [OBS:DEDUP_IMPORTS:OPEN] — unique anchor
                    pass

                    ba = broker.get_prices(args.symbol) if hasattr(broker, "get_prices") else None
                    spread = (float(ba[1]) - float(ba[0])) if (ba and ba[0] and ba[1]) else None
                    dev = ((cfg.get("executor") or cfg.get("execution") or {}).get("slippage_cap_pips"))
                    rp = (float(ba[1]) if desired == 1 else float(ba[0])) if ba else None  # requested price (ask for BUY, bid for SELL)
                    with open(jcsv, "a", newline="") as f:
                        csv.writer(f).writerow([int(time.time()), "metrics", args.symbol, desired, lot_eff, rp, ticket, ok, f"OPEN_METRICS:spread={spread};deviation={dev}", None])
                except Exception:
                    pass

                # [OMS:RECONCILE_VERIFY] — unique anchor
                if is_live:
                    try:
                        import MetaTrader5 as mt5
                        poss = mt5.positions_get(symbol=args.symbol)
                        expected_type = mt5.POSITION_TYPE_BUY if desired == 1 else mt5.POSITION_TYPE_SELL
                        if ok and not any(int(getattr(p,"type",-1))==expected_type for p in (poss or [])):
                            LOGGER.error("Reconcile failed: expected position not found. Aborting.")
                            return 2
                    except Exception:
                        LOGGER.error("Reconcile check failed unexpectedly."); return 2



                # اگر پوزیشن باز شد، state را تنظیم کن
                try:
                    if ok and isinstance(ord_info, dict):
                        if ord_info.get("entry_price") is not None:
                            pa_state["entry"] = float(ord_info["entry_price"])
                            # محاسبه‌ی stop_dist از ATR در صورت فعال بودن
                            ps = rm.rcfg.position_sizing
                            if (ps.use_atr_for_sl and (atr_price is not None)):
                                sd = rm.stop_distance_from_atr(float(atr_price))
                            else:
                                sd = None
                            pa_state["stop_dist"] = sd if (sd and sd > 0.0) else None
                            pa_state["be_done"] = False
                            pa_state["last_sl"] = None
                            pa_state["last_mng_step"] = steps
                except Exception:
                    pass
                try:
                    with open(jcsv,"a",newline="") as f: csv.writer(f).writerow([int(time.time()),"open",args.symbol,desired,lot_eff,(ask if desired==1 else bid if desired==-1 else None),ticket,ok,None,None])
                    with open(jjson,"a") as f: f.write(json.dumps({"ts":int(time.time()),"ev":"open","sym":args.symbol,"a":desired,"lot":lot_eff,"price":(ask if desired==1 else bid if desired==-1 else None),"ticket":ticket,"ok":ok})+"\n")
                except Exception: pass

            # [OBS:OPEN_FAIL] — unique anchor
            if opened is None or (opened and not opened[0]):
                obs_metrics["fails"] += 1

            if closed is not None:
                # [OBS:CLOSE_METRICS] — unique anchor
                obs_metrics["closes"] += 1
                try:
                    with open(jjson,"a") as f: f.write(json.dumps({"ts":int(time.time()*1000),"event":"close_metrics","symbol":args.symbol,"intent_id":intent_id,"closed_ok":bool(closed[0]),"ticket":closed[1] if len(closed)>1 else None})+"\n")
                except Exception: pass


                ok, ticket = closed
                LOGGER.info("Order-CLOSE result: ok=%s ticket=%s", ok, ticket)

                # Journal: execution metrics at close (no new headers)
                try:
                    # [OBS:DEDUP_IMPORTS:CLOSE] — unique anchor
                    pass

                    ba = broker.get_prices(args.symbol) if hasattr(broker, "get_prices") else None
                    spread = (float(ba[1]) - float(ba[0])) if (ba and ba[0] and ba[1]) else None
                    with open(jcsv, "a", newline="") as f:
                        csv.writer(f).writerow([int(time.time()), "metrics", args.symbol, 0, None, None, ticket, ok, f"CLOSE_METRICS:spread={spread}", None])
                except Exception:
                    pass


                try:
                    # قیمت لحظه‌ای برای ثبت در ژورنال
                    px_bid, px_ask = (broker.get_prices(args.symbol) if hasattr(broker, "get_prices") else (None, None))
                    with open(jcsv,"a",newline="") as f:
                        csv.writer(f).writerow([
                            int(time.time()), "close", args.symbol, desired, None,  # lot=None چون حجم کل/جزئی را از بروکر نگرفتیم
                            (px_bid if desired == -1 else px_ask), ticket, ok, None, None
                        ])
                    with open(jjson,"a") as f:
                        f.write(json.dumps({
                            "ts": int(time.time()), "ev": "close", "sym": args.symbol, "a": desired,
                            "lot": None, "price": (px_bid if desired == -1 else px_ask),
                            "ticket": ticket, "ok": ok
                        }) + "\n")
                except Exception:
                    pass


        else:
            # Heartbeat کم‌حجم هر 100 گام
            if (steps % 100) == 0:
                LOGGER.info("Heartbeat: pos=%s steps=%d", pos, steps)



        # --- مدیریت پوزیشن باز: Breakeven / Trailing / Scale-out ---
        try:
            ps = rm.rcfg.position_sizing

            # --- Epsilon و SL فعلی سرور ---
            sl_eps = 1e-3
            current_sl = None
            pid = None
            try:
                import MetaTrader5 as mt5
                # تعیین epsilon بر اساس point نماد
                si = mt5.symbol_info(args.symbol)
                if si and getattr(si, "point", None):
                    sl_eps = max(float(si.point) * 2.0, 1e-3)
                # گرفتن position_id و SL فعلی سمت جاری
                poss = mt5.positions_get(symbol=args.symbol)
                if poss:
                    sel = [p for p in poss if int(getattr(p, "type", -1)) == \
                           (mt5.POSITION_TYPE_BUY if pos == 1 else mt5.POSITION_TYPE_SELL)]
                    if sel:
                        last = sel[-1]
                        pid = int(getattr(last, "ticket", 0))
                        current_sl = getattr(last, "sl", None)
            except Exception:
                pass




            # اجرای مدیریت فقط هر ps_manage_interval گام
            if (steps - pa_state["last_mng_step"]) < ps_manage_interval:
                raise RuntimeError("skip-mng")  # خروج سریع از بلوک (نه خطا)

            # فقط اگر پوزیشن باز است و state معتبر داریم
            if pos in (1, -1) and pa_state["entry"] is not None and pa_state["stop_dist"]:
                if hasattr(broker, "get_prices"):
                    bid, ask = broker.get_prices(args.symbol)
                    last_price = float(ask) if pos == 1 else float(bid)

                    r_mult = (last_price - pa_state["entry"]) / \
                        pa_state["stop_dist"] if pos == 1 else (pa_state["entry"] - last_price) / pa_state["stop_dist"]

                    # --- Breakeven
                    if ps.breakeven_enabled and not pa_state["be_done"] and r_mult >= ps.breakeven_r_multiple:
                        # همیشه «position ticket» سمت سرور را ترجیح بده؛ اگر نبود، از ord_info استفاده کن
                        pid_local = pid if (pid is not None) else (ord_info.get("position_id") if ('ord_info' in locals() and ord_info) else None)

                        if pid and hasattr(broker, "modify_position_sl_tp"):
                            # فقط اگر SL فعلی متفاوت از entry است
                            new_sl = pa_state["entry"]
                            # اگر SL سرور عملاً برابر entry است، اصلاح لازم نیست
                            if current_sl is not None and abs(float(new_sl) - float(current_sl)) <= sl_eps:
                                new_sl = None

                            if (new_sl is not None) and ((pa_state["last_sl"] is None) or (abs(float(new_sl) - float(pa_state["last_sl"])) > sl_eps)):

                                ok_be = broker.modify_position_sl_tp(pid_local, sl=new_sl, tp=None)
                                LOGGER.info("Breakeven SL applied: ok=%s pos_id=%s sl=%s", ok_be, pid, new_sl)
                                if ok_be:
                                    pa_state["be_done"] = True
                                    pa_state["last_sl"] = new_sl
                                    pa_state["last_mng_step"] = steps  # فقط وقتی تغییر واقعی اعمال شد، interval ریست شود

                    # --- Trailing
                    if ps.trailing_enabled and (atr_price is not None):
                        trail_off = rm.trailing_offset_from_atr(float(atr_price))
                        if trail_off and trail_off > 0:
                            new_sl = (last_price - trail_off) if pos == 1 else (last_price + trail_off)
                            # از BE عقب‌تر نرویم
                            if pa_state["be_done"]:
                                new_sl = max(new_sl, pa_state["entry"]) if pos == 1 else min(new_sl, pa_state["entry"])
                            # اگر SL سرور با new_sl در یک محدوده است، اصلاح لازم نیست
                            if current_sl is not None and abs(float(new_sl) - float(current_sl)) <= sl_eps:
                                new_sl = None

                            # فقط اگر SL واقعاً تغییر معنادار کرده باشد، اصلاح کن
                            if (new_sl is not None) and \
                               ((pa_state["last_sl"] is None) or (abs(float(new_sl) - float(pa_state["last_sl"])) > sl_eps)):

                                # همیشه «position ticket» سمت سرور را ترجیح بده؛ اگر نبود، از ord_info استفاده کن
                                pid_local = pid if (pid is not None) else (ord_info.get("position_id") if ('ord_info' in locals() and ord_info) else None)

                                if pid and hasattr(broker, "modify_position_sl_tp"):
                                    ok_tr = broker.modify_position_sl_tp(pid, sl=float(new_sl), tp=None)
                                    LOGGER.info("Trailing SL updated: ok=%s pos_id=%s sl=%s", ok_tr, pid, new_sl)
                                    if ok_tr:
                                        pa_state["last_sl"] = float(new_sl)
                                        pa_state["last_mng_step"] = steps  # فقط وقتی تغییر واقعی اعمال شد، interval ریست شود


                    # --- Scale-out (اختیاری؛ اگر در cfg تعریف شده)
                    scale_cfg = ((getattr(rm.rcfg, "position_limits", {}) or {}).get("scale_out") if hasattr(rm.rcfg, "position_limits") else None)
                    if scale_cfg and isinstance(scale_cfg, list):
                        import MetaTrader5 as mt5
                        poss = mt5.positions_get(symbol=args.symbol)
                        if poss:
                            vol_now = sum(float(getattr(p, "volume", 0.0)) for p in poss if int(getattr(p, "type", -1)) == (mt5.POSITION_TYPE_BUY if pos == 1 else mt5.POSITION_TYPE_SELL))
                            for rule in scale_cfg:
                                try:
                                    thr = float(rule.get("at_r")); frac = float(rule.get("close_frac"))
                                except Exception:
                                    continue
                                if r_mult >= thr and 0 < frac < 1.0 and vol_now > 0:
                                    to_close = max(0.0, vol_now * frac)
                                    if to_close > 0 and hasattr(broker, "close_partial"):
                                        ok_sc, tkt = broker.close_partial(args.symbol, "long" if pos == 1 else "short", to_close)
                                        LOGGER.info("Scale-out: ok=%s ticket=%s vol=%.3f at_r=%.2f", ok_sc, tkt, to_close, r_mult)
                                        # (اختیاری) vol_now را کم کن تا از چندبار بستن اضافی جلوگیری شود
                                        if ok_sc:
                                            vol_now = max(0.0, vol_now - to_close)
            # موفقیت اجرای مدیریت این گام
            pa_state["last_mng_step"] = steps

        except RuntimeError as _skip:
            # اگر پیام 'skip-mng' بود یعنی فقط به‌خاطر interval این گام را رد کردیم
            if str(_skip) != "skip-mng":
                LOGGER.exception("PA management runtime error: %s", _skip)
            pass
        except Exception:
            LOGGER.exception("PA management (BE/Trailing/Scale-out) failed.")



        if term or trunc:
            LOGGER.info("Episode terminated: steps=%d total_reward=%.6f", steps, total_r)
            break

    LOGGER.info("Execution finished: steps=%d total_reward=%f avg_per_step=%f", steps, total_r, total_r / max(1, steps))

    # [OBS:SUMMARY] — unique anchor
    try:
        fr = (obs_metrics["opens"]/obs_metrics["attempts"]) if obs_metrics.get("attempts") else None
        payload = {"ts": int(time.time()*1000), "event": "obs_summary", "symbol": args.symbol,
                   "attempts": obs_metrics.get("attempts",0), "opens": obs_metrics.get("opens",0),
                   "fails": obs_metrics.get("fails",0), "closes": obs_metrics.get("closes",0), "fill_rate": fr}
        with open(jjson, "a") as f: f.write(json.dumps(payload) + "\n")
        LOGGER.info("OBS summary: attempts=%s opens=%s fails=%s closes=%s fill_rate=%s",
                    payload["attempts"], payload["opens"], payload["fails"], payload["closes"], fr)
    except Exception: pass


    if pos == 1:
        ok, ticket = broker.close_long(args.symbol)
        LOGGER.info("Order-CLOSE result: ok=%s ticket=%s", ok, ticket)
        try:
            px_bid, px_ask = (broker.get_prices(args.symbol) if hasattr(broker, "get_prices") else (None, None))
            with open(jcsv,"a",newline="") as f:
                csv.writer(f).writerow([
                    int(time.time()), "close", args.symbol, 0, None, px_bid, ticket, ok, None, None
                ])
            with open(jjson,"a") as f:
                f.write(json.dumps({
                    "ts": int(time.time()), "ev": "close", "sym": args.symbol, "a": 0,
                    "lot": None, "price": px_bid, "ticket": ticket, "ok": ok
                }) + "\n")
        except Exception:
            pass

    elif pos == -1:
        ok, ticket = broker.close_short(args.symbol)
        LOGGER.info("Order-CLOSE result: ok=%s ticket=%s", ok, ticket)
        try:
            px_bid, px_ask = (broker.get_prices(args.symbol) if hasattr(broker, "get_prices") else (None, None))
            with open(jcsv,"a",newline="") as f:
                csv.writer(f).writerow([
                    int(time.time()), "close", args.symbol, 0, None, px_ask, ticket, ok, None, None
                ])
            with open(jjson,"a") as f:
                f.write(json.dumps({
                    "ts": int(time.time()), "ev": "close", "sym": args.symbol, "a": 0,
                    "lot": None, "price": px_ask, "ticket": ticket, "ok": ok
                }) + "\n")
        except Exception:
            pass



if __name__ == "__main__":
    raise SystemExit(main())
