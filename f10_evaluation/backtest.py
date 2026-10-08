# -*- coding: utf-8 -*-
# f10_evaluation/backtest.py   (لود مدل و بک‌تست + گزارش)
# Status in (Bot-RL-3): Completed

"""
بک‌تست سریع یک مدل ذخیره‌شده روی TradingEnv
- رفتار: greedy (اکشن با بیشترین احتمال)
- گزارش: مجموع پاداش، میانگین پاداش، قدم‌ها و بازه‌ی زمانی
"""
# فرمان اجرای برنامه
# python -m f10_evaluation.backtest --symbol XAUUSD -c .\f01_config\config.yaml --base-tf M1 --window 128 --steps 5000 --reward atr_norm --normalize --split test

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np

from f02_utils.config_loader import load_config
from f06_env.OLD_files.OLD_trading_env import TradingEnv, EnvConfig
from f06_env.OLD_files.OLD_utils import paths_from_cfg, resolve_spread_selection
try:
    from f09_training.agent_sb3 import load_sb3 as _load_sb3
except Exception:
    _load_sb3 = None

#logger = logging.getLogger(__name__)
logger = logging.getLogger("backtest")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S")


def transform_obs(obs, mode: str = "mean", last_k: int = 16):
    import numpy as np
    a = np.asarray(obs, dtype=np.float32)
    if mode == "last":
        return a[-1].ravel()
    if mode in ("mean", "lastk_mean"):
        k = int(last_k)
        return a[-k:].mean(axis=0).ravel() if a.ndim == 2 else a.ravel()
    return a.ravel()  # flat fallback


def _parse_args():
    p = argparse.ArgumentParser(description="Backtest a saved SB3 policy")
    p.add_argument("--symbol", required=True)
    p.add_argument("-c","--config", default="f01_config/config.yaml")
    p.add_argument("--base-tf", default=None)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--reward", default="pnl", choices=["pnl","logret","atr_norm"])
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--model-path", default=None)
    p.add_argument("--obs-agg", default="mean", choices=["mean","last","lastk_mean","flat"])
    p.add_argument("--last-k", type=int, default=16)
    p.add_argument("--split", default="test", choices=["train","val","test"])
    p.add_argument("--sb3-alg", default="auto", choices=["auto","PPO","SAC","TD3","A2C","DQN","DDPG"])

    # [COSTS:CLI] — unique anchor
    p.add_argument("--cost-spread-pts", type=float, default=None, help="spread cost in price points (e.g., 2.0)")
    p.add_argument("--cost-commission-per-lot", type=float, default=None, help="commission per lot in account currency")
    p.add_argument("--cost-slippage-pts", type=float, default=None, help="slippage cost in price points")
    p.add_argument("--point-value", type=float, default=None, help="point value for symbol (price→pip)")
    return p.parse_args()


def main() -> int:
    """CLI wrapper: load cfg from disk, then delegate to run_backtest()."""
    args = _parse_args()
    cfg = load_config(args.config, enable_env_override=True)

    # اِعمال گزینه‌های CLI روی cfg (بدون دست‌کاری فایل روی دیسک)
    (cfg.setdefault("env", {}))["normalize"] = bool(args.normalize)
    if args.base_tf:
        (cfg.setdefault("features", {}))["base_timeframe"] = str(args.base_tf).upper()

    # [COSTS:CFG] — unique anchor
    # --------------------------------- start
    tr = cfg.setdefault("trading", {})
    costs = tr.setdefault("costs", {})

    # [COSTS:FALLBACK] — revised for 4-step spread
    bt = ((cfg.get("evaluation") or {}).get("backtest") or {})
    sel = resolve_spread_selection(cfg, has_broker_series=bool(bt.get("reuse_historical_spread")))
    spread_effective = 0.0 if sel["use_series"] else float(sel["value"])
    costs.setdefault("commission_per_lot", float(bt.get("commissions_per_lot_usd") or 0.0))
    costs.setdefault("slippage_pts", float(bt.get("slippage_pips_mean") or 0.0))

    # point_value comes only from evaluation.backtest (no executor.price_point fallback)
    # (do not store it under trading.costs anymore)
    _pv_bt = bt.get("point_value", None)

    logger.info(f"[COSTS] spread source={sel['source']} value={spread_effective}")

    #if args.cost_spread_pts is not None:
    #    costs["spread_pts"] = float(args.cost_spread_pts)
    # CLI override for fixed spread is disabled by design (no config key for constant spread)
    if args.cost_commission_per_lot is not None:
        costs["commission_per_lot"] = float(args.cost_commission_per_lot)
    if args.cost_slippage_pts is not None:
        costs["slippage_pts"] = float(args.cost_slippage_pts)
    if args.point_value is not None:
        #costs["point_value"] = float(args.point_value)
        (cfg.setdefault("evaluation",{}).setdefault("backtest",{}))["point_value"] = float(args.point_value)
    # --------------------------------- end

    # [STEPS:PASS] — unique anchor
    rep = run_backtest(
        symbol=args.symbol,
        cfg=cfg,
        tag="cli",
        model_path=(args.model_path or None),
        sb3_alg=args.sb3_alg,
        max_steps=int(args.steps),
    )

    print(
        f"Backtest finished: steps={rep.get('steps',0)} "
        f"total_reward={rep.get('total_reward',0.0):.6f} "
        f"avg_per_step={rep.get('avg_per_step',0.0):.6f}"
    )

    # USD overlay summary (if available) --- start
    if "total_reward_usd" in rep:
        print(
            f"USD overlay: total_reward_usd={rep.get('total_reward_usd',0.0):.6f} "
            f"avg_per_step_usd={rep.get('avg_per_step_usd',0.0):.6f} "
            f"sharpe_usd={rep.get('sharpe_usd',0.0):.6f} "
            f"profit_factor_usd={rep.get('profit_factor_usd',0.0):.6f} "
            f"winrate_usd={rep.get('winrate_usd',0.0):.6f}"
        )
    # --- end
    
    # [COSTS:REPORT] — unique anchor
    # چاپ خلاصهٔ کانفیگ هزینه‌ها در خروجی
    _c  = (cfg.get("trading") or {}).get("costs") or {}
    _bt = (cfg.get("evaluation") or {}).get("backtest") or {}
    #print(f"Costs: spread_pts={_c.get('spread_pts',0)} slippage_pts={_c.get('slippage_pts',0)} "
    #      f"commission_per_lot={_c.get('commission_per_lot',0)} point_value={_bt.get('point_value','MISSING')}")    
    print(f"Costs: spread_pts={spread_effective} slippage_pts={_c.get('slippage_pts',0)} "
          f"commission_per_lot={_c.get('commission_per_lot',0)} point_value={_bt.get('point_value','MISSING')}")
    
    print(f"Spread source: {sel['source']}")
    return 0


# --- In-process API for PA hparam search (no CLI, no disk writes) ---
def run_backtest(*,
                 symbol: str,
                 cfg: dict,
                 tag: str | None = None,
                 model_path: str | None = None,
                 forward_days: int | None = None,
                 cost_spread_pts: float | None = None,
                 cost_commission_per_lot: float | None = None,
                 cost_slippage_pts: float | None = None,
                 point_value: float | None = None,
                 sb3_alg: str = "auto",
                 max_steps: int | None = None) -> dict:

    """
    Run backtest in-process using the provided in-memory cfg.
    Returns a dict with at least: {"total_reward": float, "steps": int, "avg_per_step": float}
    Notes:
      - Uses the same logic as main(): TradingEnv + (SB3 policy if available else random).
      - Does NOT read config from disk; uses `cfg` passed by caller (self-optimizer).
      - Does NOT write anything to disk.
    """
    # [COSTS:INPROC] — unique anchor
    # --------------------------------- start
    tr = cfg.setdefault("trading", {})
    costs = tr.setdefault("costs", {})
    if cost_spread_pts is not None:
        costs["spread_pts"] = float(cost_spread_pts)
    if cost_commission_per_lot is not None:
        costs["commission_per_lot"] = float(cost_commission_per_lot)
    if cost_slippage_pts is not None:
        costs["slippage_pts"] = float(cost_slippage_pts)
    if point_value is not None:
        #costs["point_value"] = float(point_value)
        (cfg.setdefault("evaluation",{}).setdefault("backtest",{}))["point_value"] = float(point_value)
    # --------------------------------- end

    import numpy as np
    logger = logging.getLogger("backtest_api")

    # 1) derive base timeframe from cfg (exactly like main())
    # base_tf_cfg = (cfg.get("features") or {}).get("base_timeframe", "M5")
    # base_tf = str(base_tf_cfg).upper()

    base_tf_cfg = (cfg.get("env") or {}).get("base_tf", "M5")
    base_tf = str(base_tf_cfg).upper()

    # 2) build env from in-memory cfg
    env_cfg = EnvConfig(
        symbol=symbol,
        base_tf=base_tf,
        window_size=128,          # مطابق پیش‌فرض‌های مسیر main؛ در صورت نیاز از cfg بخوان
        reward_mode="pnl",        # همان پیش‌فرضی که در main استفاده می‌کنی
        normalize=bool((cfg.get("env") or {}).get("normalize", False)),
    )
    env = TradingEnv(cfg=cfg, env_cfg=env_cfg)

    # [SPREAD:SOURCE] — detect from env.df_full
    try:
        _has = hasattr(env, "df_full") and ("spread" in env.df_full.columns) and bool(env.df_full["spread"].notna().any())
    except Exception:
        _has = False
    _sel = resolve_spread_selection(cfg, has_broker_series=_has)
    spread_source = _sel["source"]

    # 3) فقط SB3: جدیدترین فایل .zip در staging برای symbol & TF
    if model_path is None:
        paths = paths_from_cfg(cfg)
        stg_dir = paths["models_dir"] / "staging"
        cands = sorted(
            stg_dir.glob(f"*_{symbol}_{base_tf}_*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        model_path = (cands[0] if cands else None)


    try:
        policy = (_load_sb3(sb3_alg, model_path) if (_load_sb3 and model_path and model_path.suffix == ".zip") else None)
        # اگر logger/LOGGER داری، همین‌جا یک info بزن (اختیاری):
        # logger.info("Model loaded: %s", model_path)  # یا LOGGER.info(...)
    except Exception:
        policy = None  # robust fallback: random acts


    # 4) backtest loop (same stepping pattern as main(), StepResult-safe)
    obs, info = env.reset(split="test")
    total_r, steps = 0.0, 0
    
    # --- [METRICS:INIT] محاسبهٔ متریک‌ها ---
    _rewards = []           # پاداشِ هر استپ
    _trade_pnls = []        # سود/زیان هر معامله (پوزیشنِ غیر صفر)
    _current_trade_pnl = 0.0
    pos_prev = int((info or {}).get("pos", 0))
    sign = (0 if pos_prev == 0 else (1 if pos_prev > 0 else -1))

    # برآورد طول هر استپ برحسب دقیقه (برای گردش روزانه)
    _tf2min = {"M1":1,"M5":5,"M15":15,"M30":30,"H1":60,"H4":240,"D1":1440,"W1":10080}
    tf_minutes = _tf2min.get(base_tf.upper(), 1)

    # --- [USD:INIT] متریک‌های دلاری ---
    # تعریف متغیرهای USD قبل از حلقه
    _rewards_usd = []              # reward دلاری هر استپ (درصورت وجود در info)
    _trade_pnls_usd = []           # PnL دلاری هر معامله
    _current_trade_pnl_usd = 0.0   # انباشت PnL دلاری برای معاملهٔ جاری
    _total_reward_usd = 0.0        # مجموع reward_usd (برای میانگین)

    # [STEPS:LIMIT] — unique anchor
    limit = int(max_steps) if (max_steps is not None) else 5000
    while steps < limit:

        x = transform_obs(obs, mode="mean", last_k=16).astype(np.float32)
        if policy is None:
            a = np.random.randint(0, 3)
        else:
            a, _ = policy.predict(x, deterministic=True); a = int(a)



        sr = env.step(a - 1)
        obs = sr.observation
        r = sr.reward
        term = sr.terminated
        trunc = sr.truncated
        total_r += float(r)
        steps += 1

        # --- [METRICS:ACCUM] انباشت ---
        _rewards.append(float(r))

        # --- [USD:ACCUM] اگر env در info مقدار USD داده باشد، انباشت کن --- start
        _usd = (sr.info or {}).get("reward_usd", None)
        if _usd is not None:
            try:
                usd = float(_usd)
                _rewards_usd.append(usd)
                _total_reward_usd += usd
                if sign != 0:
                    _current_trade_pnl_usd += usd
            except Exception:
                pass  # robust: اگر نادرست بود، نادیده بگیر
        # --- end

        new_pos = int((sr.info or {}).get("pos", pos_prev))
        new_sign = (0 if new_pos == 0 else (1 if new_pos > 0 else -1))

        # پاداشِ هر استپ را به معاملهٔ باز (اگر پوزیشن قبلی غیر صفر بوده) اضافه می‌کنیم
        if sign != 0:
            _current_trade_pnl += float(r)

        # بستن/تعویض معامله: اگر به صفر رسیدیم یا علامت عوض شد
        if sign != 0 and (new_sign == 0 or new_sign != sign):
            _trade_pnls.append(_current_trade_pnl)
            _current_trade_pnl = 0.0

        # اگر از صفر رفتیم به غیرصفر (شروع معاملهٔ جدید)، accumulator را صفر نگه می‌داریم
        sign = new_sign
        pos_prev = new_pos



        if term or trunc:
            break

    '''
    avg = total_r / max(1, steps)
    # حداقل سه کلید موردنیاز برای hparam_search
    return {"total_reward": total_r, "steps": steps, "avg_per_step": avg}
    '''

    # --- [METRICS:FINAL] محاسبهٔ نهایی ---
    import numpy as _np

    # اگر معامله‌ای باز مانده، ببندیم
    if sign != 0:
        _trade_pnls.append(_current_trade_pnl)

    steps = int(steps)
    avg = float(total_r) / max(1, steps)

    rewards = _np.asarray(_rewards, dtype=_np.float64)
    if rewards.size >= 2:
        sharpe = float((rewards.mean() / (rewards.std(ddof=1) + 1e-12)) * _np.sqrt(max(1, rewards.size)))
    else:
        sharpe = 0.0

    tp = _np.array([p for p in _trade_pnls if p > 0.0], dtype=_np.float64)
    tn = _np.array([p for p in _trade_pnls if p < 0.0], dtype=_np.float64)
    profit_factor = float((tp.sum() / (abs(tn.sum()) + 1e-12)) if tp.size > 0 else 0.0)
    winrate = float((tp.size / max(1, len(_trade_pnls))))

    # حداکثر دراودان روی جمع تجمعی پاداش‌ها
    if rewards.size > 0:
        equity = rewards.cumsum()
        peak = _np.maximum.accumulate(equity)
        dd_abs = (peak - equity).max() if equity.size else 0.0
        denom = max(1.0, abs(peak.max()))  # نرمال‌سازی نسبی ساده
        max_drawdown = float(dd_abs / denom)
    else:
        max_drawdown = 0.0

    # گردش روزانه ≈ شمار معاملات / تعداد روزهای مؤثر
    days = (steps * tf_minutes) / 1440.0
    turnover_per_day = float(len(_trade_pnls) / max(days, 1e-9))

    # --- [USD:FINAL] محاسبات انتهایی دلاری --- start
    _rewards_usd_np = _np.asarray(_rewards_usd, dtype=_np.float64) if len(_rewards_usd) else _np.array([], dtype=_np.float64)
    if sign != 0 and _current_trade_pnl_usd != 0.0:
        _trade_pnls_usd.append(_current_trade_pnl_usd)

    total_reward_usd = float(_np.nansum(_rewards_usd_np)) if _rewards_usd_np.size else 0.0
    avg_per_step_usd = float(total_reward_usd / max(1, steps))
    if _rewards_usd_np.size >= 2:
        sharpe_usd = float((_rewards_usd_np.mean() / (_rewards_usd_np.std(ddof=1) + 1e-12)) * _np.sqrt(max(1, _rewards_usd_np.size)))
    else:
        sharpe_usd = 0.0
    tp_usd = _np.array([p for p in _trade_pnls_usd if p > 0.0], dtype=_np.float64)
    tn_usd = _np.array([p for p in _trade_pnls_usd if p < 0.0], dtype=_np.float64)
    profit_factor_usd = float((tp_usd.sum() / (abs(tn_usd.sum()) + 1e-12)) if tp_usd.size > 0 else 0.0)
    winrate_usd = float((tp_usd.size / max(1, len(_trade_pnls_usd)))) if _trade_pnls_usd else 0.0
    # --- end

    return {
        "total_reward": float(total_r),
        "steps": steps,
        "avg_per_step": float(avg),
        "trades": int(len(_trade_pnls)),
        "winrate": float(winrate),
        "profit_factor": float(profit_factor),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "turnover_per_day": float(turnover_per_day),
        "spread_source": str(spread_source),
        # افزودن costs_used به خروجی بک‌تست (برای ردگیری کامل هزینه‌ها)
        "costs_used": {
            "spread_pts": float((costs or {}).get("spread_pts", 0.0)),
            "slippage_pts": float((costs or {}).get("slippage_pts", 0.0)),
            "commission_per_lot": float((costs or {}).get("commission_per_lot", 0.0)),
            "point_value": float(((cfg.get("evaluation") or {}).get("backtest") or {}).get("point_value", 0.0))},
        "total_reward_usd": float(total_reward_usd),
        "avg_per_step_usd": float(avg_per_step_usd),
        "sharpe_usd": float(sharpe_usd),
        "profit_factor_usd": float(profit_factor_usd),
        "winrate_usd": float(winrate_usd),

    }


if __name__ == "__main__":
    raise SystemExit(main())
