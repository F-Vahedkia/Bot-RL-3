# -*- coding: utf-8 -*-
# پروموشن مدل SB3 از staging به versions با خواندن گیت‌ها از cfg و خروجی گزارش JSON کنار مدل
# توضیحات فارسی (غیرقابل‌اجرا)؛ پیام‌های runtime انگلیسی هستند.
# Run: python -m f13_models.promote_from_staging --symbol XAUUSD --model f13_models/staging/best_model.zip --sb3-alg PPO --max-steps 5000

from __future__ import annotations
import argparse, logging, shutil, json, time
import hashlib, sys, platform

from pathlib import Path
from typing import Dict, Any, Tuple

from f02_utils.config_loader import load_config, save_config_versioned
from f06_env.OLD_files.OLD_utils import paths_from_cfg, split_summary_path
from f10_evaluation.backtest import run_backtest

logger = logging.getLogger("promote")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)

def _parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a staged SB3 model and promote to versions if gates pass"
    )
    p.add_argument("--symbol", required=True)
    p.add_argument("-c","--config", default="f01_config/config.yaml")
    p.add_argument("--model", required=True, help="path to .zip in staging")
    p.add_argument("--sb3-alg", default="auto", choices=["auto","PPO","SAC","TD3","A2C","DQN","DDPG"])
    p.add_argument("--max-steps", type=int, default=5000)
    # آستانه‌های CLI فقط اگر در cfg نبودند استفاده می‌شوند
    p.add_argument("--min-avg-per-step", type=float, default=None)
    p.add_argument("--min-steps", type=int, default=None)
    return p.parse_args()

'''
# جایگزین تابع خواندن گیت‌ها:
def _read_gates_from_cfg(cfg: Dict[str, Any]) -> int:
    acc = ((cfg.get("evaluation") or {}).get("acceptance_gates") or {})
    return int(acc.get("min_trades_total", 1000))  # ← فقط بر اساس کانفیگ
'''

def _write_report(model_path: Path, payload: Dict[str, Any]) -> Path:
    """
    گزارش بک‌تست/گیت‌ها را در کنار فایل مدل ذخیره می‌کند: <model>.report.json
    """
    out = model_path.with_suffix(model_path.suffix + ".report.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def _read_min_steps_from_cfg(cfg: Dict[str, Any]) -> int:
    """خواندن تنها گیت موجود از کانفیگ: evaluation.acceptance_gates.min_trades_total"""
    acc = ((cfg.get("evaluation") or {}).get("acceptance_gates") or {})
    return int(acc.get("min_trades_total", 1000))


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    paths = paths_from_cfg(cfg)

    model_path = Path(args.model)
    assert model_path.suffix == ".zip", "Model must be a .zip saved by SB3"
    assert model_path.exists(), f"Model not found: {model_path}"

    min_steps = _read_min_steps_from_cfg(cfg)
    logger.info("Gate: min_trades_total=%d", min_steps)

    # Backtest (split=test)
    rep = run_backtest(
        symbol=args.symbol.upper(),
        cfg=cfg,
        tag="promote",
        model_path=model_path,
        sb3_alg=args.sb3_alg,
        max_steps=int(args.max_steps),
    )
    steps = int(rep.get("steps", 0))
    logger.info("Backtest: steps=%d", steps)

    # passed = (steps >= min_steps)
    # logger.info("Backtest: steps=%d -> passed=%s", steps, passed)

    # Gates from config: evaluation.acceptance_gates
    gates = ((cfg.get("evaluation") or {}).get("acceptance_gates") or {})
    sharpe_min            = gates.get("sharpe_min")
    profit_factor_min     = gates.get("profit_factor_min")
    winrate_min           = gates.get("winrate_min")
    max_drawdown_max      = gates.get("max_drawdown_max")
    turnover_max_per_day  = gates.get("turnover_max_per_day")

    # Metrics from backtest
    sharpe           = float(rep.get("sharpe", 0.0))
    profit_factor    = float(rep.get("profit_factor", 0.0))
    winrate          = float(rep.get("winrate", 0.0))
    max_drawdown     = float(rep.get("max_drawdown", 0.0))
    turnover_per_day = float(rep.get("turnover_per_day", 0.0))

    conds = [steps >= min_steps]
    if sharpe_min is not None:            conds.append(sharpe >= float(sharpe_min))
    if profit_factor_min is not None:     conds.append(profit_factor >= float(profit_factor_min))
    if winrate_min is not None:           conds.append(winrate >= float(winrate_min))
    if max_drawdown_max is not None:      conds.append(max_drawdown <= float(max_drawdown_max))
    if turnover_max_per_day is not None:  conds.append(turnover_per_day <= float(turnover_max_per_day))

    passed = all(conds)
    logger.info(
        "Gates -> steps[%d >= %d], sharpe=%.4f, pf=%.4f, wr=%.4f, mdd=%.4f, turn=%.2f => passed=%s",
        steps, min_steps, sharpe, profit_factor, winrate, max_drawdown, turnover_per_day, passed
    )
    
    # تشخیص دقیق اینکه کدام گیت‌ها رد شده‌اند
    failed = []
    if steps < min_steps:                        failed.append(f"steps<{min_steps}")
    if (gates.get("sharpe_min") is not None) and (sharpe < float(gates["sharpe_min"])):                     failed.append(f"sharpe<{gates['sharpe_min']}")
    if (gates.get("profit_factor_min") is not None) and (profit_factor < float(gates["profit_factor_min"])): failed.append(f"pf<{gates['profit_factor_min']}")
    if (gates.get("winrate_min") is not None) and (winrate < float(gates["winrate_min"])):                  failed.append(f"wr<{gates['winrate_min']}")
    if (gates.get("max_drawdown_max") is not None) and (max_drawdown > float(gates["max_drawdown_max"])):   failed.append(f"mdd>{gates['max_drawdown_max']}")
    if (gates.get("turnover_max_per_day") is not None) and (turnover_per_day > float(gates["turnover_max_per_day"])): failed.append(f"turn>{gates['turnover_max_per_day']}")

    logger.warning("Gate failures: %s", ", ".join(failed) if failed else "NONE")

    # [GATES:DETAILS] — actionable deltas
    fd = []
    if steps < min_steps: fd.append({"gate":"steps","value":steps,"thr":int(min_steps),"delta":int(steps-min_steps)})
    if gates.get("sharpe_min") is not None and sharpe < float(gates["sharpe_min"]): fd.append({"gate":"sharpe","value":sharpe,"thr":float(gates["sharpe_min"]),"delta":sharpe-float(gates["sharpe_min"])})
    if gates.get("profit_factor_min") is not None and profit_factor < float(gates["profit_factor_min"]): fd.append({"gate":"profit_factor","value":profit_factor,"thr":float(gates["profit_factor_min"]),"delta":profit_factor-float(gates["profit_factor_min"])})
    if gates.get("winrate_min") is not None and winrate < float(gates["winrate_min"]): fd.append({"gate":"winrate","value":winrate,"thr":float(gates["winrate_min"]),"delta":winrate-float(gates["winrate_min"])})
    if gates.get("max_drawdown_max") is not None and max_drawdown > float(gates["max_drawdown_max"]): fd.append({"gate":"max_drawdown","value":max_drawdown,"thr":float(gates["max_drawdown_max"]),"delta":max_drawdown-float(gates["max_drawdown_max"])})
    if gates.get("turnover_max_per_day") is not None and turnover_per_day > float(gates["turnover_max_per_day"]): fd.append({"gate":"turnover_per_day","value":turnover_per_day,"thr":float(gates["turnover_max_per_day"]),"delta":turnover_per_day-float(gates["turnover_max_per_day"])})

    # [GATES:PRINT] — console summary for action --- start
    if fd:
        for d in fd:
            logger.warning("Gate '%s' failed: value=%.4f thr=%.4f delta=%.4f",
                        str(d.get("gate")), float(d.get("value") or 0.0),
                        float(d.get("thr") or 0.0), float(d.get("delta") or 0.0))
        # --- end

    # [COSTS:SNAPSHOT] — define once (point_value from evaluation.backtest)
    _costs = ((cfg.get("trading") or {}).get("costs") or {})
    _bt    = ((cfg.get("evaluation") or {}).get("backtest") or {})
    costs_used = {
        "spread_pts": float(_costs.get("spread_pts") or 0.0),
        "slippage_pts": float(_costs.get("slippage_pts") or 0.0),
        "commission_per_lot": float(_costs.get("commission_per_lot") or 0.0),
        "point_value": float(_bt.get("point_value") or 0.0),
    }
    base_tf = str(((cfg.get("features") or {}).get("base_timeframe") or "M5")).upper()


    # Report JSON beside model
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "symbol": args.symbol.upper(),
        "model": str(model_path),
        "sb3_alg": args.sb3_alg,
        "max_steps": int(args.max_steps),
        "metrics": rep,
        # فلت‌کردن ستون‌های USD در گزارش Promote --- start
        "total_reward_usd": float(rep.get("total_reward_usd", 0.0)),
        "avg_per_step_usd": float(rep.get("avg_per_step_usd", 0.0)),
        "sharpe_usd": float(rep.get("sharpe_usd", 0.0)),
        "profit_factor_usd": float(rep.get("profit_factor_usd", 0.0)),
        "winrate_usd": float(rep.get("winrate_usd", 0.0)),
        # --- end
        "gates": {
                "min_trades_total": int(min_steps),
                "sharpe_min": gates.get("sharpe_min"),
                "profit_factor_min": gates.get("profit_factor_min"),
                "winrate_min": gates.get("winrate_min"),
                "max_drawdown_max": gates.get("max_drawdown_max"),
                "turnover_max_per_day": gates.get("turnover_max_per_day"),
                },
        "failed_gates": failed,
        "passed": bool(passed),
        "failed_details": fd,
        # غنی‌سازی فایل گزارش پروموشن با منبع اسپرد و اسنپ‌شات هزینه‌ها
        "spread_source": rep.get("spread_source"),
        "costs_used": costs_used,
        # تصمیم پروموشن (accepted/rejected)
        "decision": "accepted" if passed else "rejected",
        "decision_reason": ("passed_all" if passed else ",".join([d.get("gate","") for d in (fd or [])])),
        # افزودن بازهٔ زمانی پایه (base_timeframe)
        "base_timeframe": base_tf,

    }
    report_path = _write_report(model_path, report)
    logger.info("Saved report: %s", report_path)

    if not passed:
        logger.warning("Gates NOT passed. Keep model in staging.")
        return 2

    # Promote to versions
    versions = paths["models_dir"] / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    target = versions / model_path.name
    shutil.copy2(model_path, target)
    logger.info("Promoted to versions: %s", target)

    # [MANIFEST:APPEND] — minimal promotion record
    try:
        manifest = versions / "manifest.jsonl"
        base_tf = str(((cfg.get("features") or {}).get("base_timeframe") or "M5")).upper()
        split_json = str(split_summary_path(args.symbol.upper(), base_tf, cfg))
        cfg_ver = str(save_config_versioned(cfg, prefix="promote_"))
        cfg_sha1 = hashlib.sha1(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        scaler_tag = f"{args.symbol.upper()}_{base_tf}_v1"
        """
        # costs snapshot for reproducibility
        _costs = ((cfg.get("trading") or {}).get("costs") or {})
        costs_used = {
            "spread_pts": float(_costs.get("spread_pts") or 0.0),
            "slippage_pts": float(_costs.get("slippage_pts") or 0.0),
            "commission_per_lot": float(_costs.get("commission_per_lot") or 0.0),
            "point_value": float(_costs.get("point_value") or 0.01),
        }
        """
        # costs_used already defined at [COSTS:SNAPSHOT] (point_value from evaluation.backtest)

        # [MANIFEST:HASHES] --- افزودن هش فایل مدل و Split به مانیفست --- start
        def _sha1(path):
            try:
                h=hashlib.sha1()
                with open(path,"rb") as f:
                    for b in iter(lambda:f.read(65536), b""): h.update(b)
                return h.hexdigest()
            except Exception: return None

        model_sha1 = _sha1(str(target))
        split_sha1 = _sha1(split_json) if split_json else None
        # --- end
        
        # ثبت متادیتای اجرا (نسخهٔ پایتون/پلتفرم) در مانیفست
        runtime = {"python": sys.version.split()[0], "platform": platform.platform()}

        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "symbol": args.symbol.upper(),
            "model": str(target), "sb3_alg": args.sb3_alg, "metrics": rep,
            # همان ستون‌های USD را علاوه بر metrics، در رکورد manifest.jsonl نیز top-level بنویسیم --- start
            "total_reward_usd": float(rep.get("total_reward_usd", 0.0)),
            "avg_per_step_usd": float(rep.get("avg_per_step_usd", 0.0)),
            "sharpe_usd": float(rep.get("sharpe_usd", 0.0)),
            "profit_factor_usd": float(rep.get("profit_factor_usd", 0.0)),
            "winrate_usd": float(rep.get("winrate_usd", 0.0)),       
            # --- end
            "gates": report.get("gates"), "failed": report.get("failed_gates"), "passed": bool(passed),
            "split_json": split_json, "scaler_tag": scaler_tag, "cfg_version": cfg_ver, "cfg_sha1": cfg_sha1,
            "failed_details": report.get("failed_details"),
            "costs_used": {**costs_used, "spread_source": rep.get("spread_source")},
            "model_sha1": model_sha1,
            "split_sha1": split_sha1,
            # تصمیم پروموشن (accepted/rejected)
            "decision": ("accepted" if passed else "rejected"),
            "decision_reason": ("passed_all" if passed else ",".join([d.get("gate","") for d in (fd or [])])),
            # افزودن بازهٔ زمانی پایه (base_timeframe)
            "base_timeframe": base_tf,
            # ثبت متادیتای اجرا (نسخهٔ پایتون/پلتفرم) در مانیفست
            "runtime": runtime,

        }
        # چاپ خلاصهٔ Costs + منبع اسپرد هنگام پروموشن (کنسول) --- start
        logger.info("Promotion costs snapshot: source=%s spread=%.4f slippage=%.4f commission=%.4f point_value=%.4f",
                    str(rep.get("spread_source")), float(costs_used["spread_pts"]), float(costs_used["slippage_pts"]),
                    float(costs_used["commission_per_lot"]), float(costs_used["point_value"]))
        # --- end
        with manifest.open("a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Manifest appended: %s", manifest)
    except Exception:
        logger.exception("Manifest append failed")

    return 0



if __name__ == "__main__":
    raise SystemExit(main())
