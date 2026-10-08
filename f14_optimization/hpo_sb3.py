# -*- coding: utf-8 -*-
# f14_optimization/hpo_sb3.py
# Status in (Bot-RL-3): New (Final)
#
# توضیح (فارسی):
# این ماژول جست‌وجوی هایپرپارامتر برای الگوریتم‌های SB3 را اجرا می‌کند:
# - فضای جست‌وجو را از یک فایل JSON/YAML خارجی می‌خواند (بدون افزودن کلید به config.yaml)
# - در هر تریل: نمونه‌گیری → اعمال روی cfg در حافظه → ذخیرهٔ موقت cfg → اجرای
#   f09_training.agent_sb3_train → کپی مدل بهترین به نام یکتا → backtest درون‌پردازه‌ای
# - امتیاز هدف از گزارش بک‌تست استخراج می‌شود (Sharpe / PF / total_reward / avg_per_step).
# - در پایان: خلاصهٔ نتایج + بهترین هایپرها در f13_models/staging ذخیره می‌شود.
#
# نکات انطباق با قوانین پروژه:
# - هیچ کلید جدیدی به config.yaml اضافه نمی‌شود؛ فقط در cfg کاریِ موقت ست می‌کنیم.
# - پیام‌های runtime انگلیسی‌اند؛ کامنت‌ها فارسی‌اند.
# - مسیرها از utilهای پروژه استفاده می‌شود؛ فولدر جدید ساخته نمی‌شود (فقط فایل در staging).
# - از APIهای موجود پروژه استفاده شده است: agent_sb3_train, backtest.run_backtest.
#
from __future__ import annotations

import argparse, json, logging, math, os, random, shutil, sys, time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --- Project imports (مطابق ساختار موجود) ---
from f02_utils.config_loader import load_config
from f06_env.OLD_files.OLD_utils import paths_from_cfg
from f10_evaluation.backtest import run_backtest  # in-process API  ← الزامی است
from f14_optimization.hparam_search import _sample_space, _extract_objective  # استفاده از نمونه‌گیر موجود

try:
    import yaml  # برای خواندن YAML اگر کاربر YAML داد
except Exception:
    yaml = None

logger = logging.getLogger("hpo_sb3")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------
# ابزارها
# ---------------------------------------------------------------------
def _read_space_file(path: str) -> Dict[str, Any]:
    """خواندن فضای جست‌وجو از JSON/YAML بدون حدس."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"space file not found: {p}")
    txt = p.read_text(encoding="utf-8")
    # اگر YAML در دسترس بود و پسوند .yml/.yaml بود، YAML؛ در غیر این صورت JSON
    if p.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not available but a YAML file was provided.")
        obj = yaml.safe_load(txt)
    else:
        obj = json.loads(txt)
    if not isinstance(obj, dict) or not obj:
        raise ValueError("invalid/empty search space file.")
    return obj


def _apply_sb3_hparams_to_cfg(cfg: Dict[str, Any], alg_name: str, params: Dict[str, Any]) -> None:
    """
    اعمال مستقیم هایپرپارامترها به cfg کاری (در حافظه).
    هایپرهای SB3 در config زیر مسیر rl.hyperparams[alg] استفاده می‌شوند.
    """
    rl = cfg.setdefault("rl", {})
    hp_all = rl.setdefault("hyperparams", {})
    alg_key = str(alg_name).lower()
    base = dict(hp_all.get(alg_key) or {})
    base.update(params or {})
    hp_all[alg_key] = base


def _run_training_cli(symbol: str, cfg_path: str, alg: str,
                      total_timesteps: int, eval_freq: int, eval_steps: int,
                      obs_agg: str, last_k: int) -> int:
    """
    اجرای ماژول آموزش SB3 از طریق CLI رسمی پروژه.
    (از نوشتن کد تکراری جلوگیری می‌کند؛ همان مسیر استاندارد استفاده می‌شود.)
    """
    import shlex, subprocess
    cmd = [
        sys.executable, "-m", "f09_training.agent_sb3_train",
        "--symbol", symbol, "-c", cfg_path, "--alg", alg,
        "--total-timesteps", str(int(total_timesteps)),
        "--eval-freq", str(int(eval_freq)),
        "--eval-steps", str(int(eval_steps)),
        "--obs-agg", obs_agg, "--last-k", str(int(last_k)),
    ]
    logger.info("Run SB3 train: %s", " ".join(shlex.quote(x) for x in cmd))
    res = subprocess.run(cmd, check=False)
    return int(res.returncode)


def _copy_best_model(staging_dir: Path, trial_tag: str) -> Optional[Path]:
    """
    کپی best_model.zip (که توسط EvalCallback ذخیره می‌شود) به نام یکتا.
    اگر best_model.zip نبود، تلاش می‌کنیم آخرین .zip تولیدشده را بیابیم.
    """
    best = staging_dir / "best_model.zip"
    if best.exists():
        dst = staging_dir / f"sb3_{trial_tag}.zip"
        shutil.copy2(best, dst)
        return dst

    # fallback: آخرین فایل .zip در استیجینگ
    zips = sorted(staging_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if zips:
        dst = staging_dir / f"sb3_{trial_tag}.zip"
        shutil.copy2(zips[0], dst)
        return dst
    return None


# ---------------------------------------------------------------------
# هستهٔ HPO
# ---------------------------------------------------------------------
def sb3_hparam_search(*,
                      cfg: Dict[str, Any],
                      space: Dict[str, Any],
                      max_trials: int,
                      symbol: str,
                      alg: str,
                      total_timesteps: int,
                      eval_freq: int,
                      eval_steps: int,
                      obs_agg: str = "mean",
                      last_k: int = 16) -> Dict[str, Any]:
    """
    جست‌وجوی هایپرپارامتر SB3 با استفاده از آموزش کوتاه + بک‌تست درون‌پردازه‌ای.

    خروجی:
      {
        "best_hparams": { ... },
        "best_score": float,
        "trials": [
          {"trial": i, "score": float, "params": {...}, "model_path": "..."},
          ...
        ]
      }
    """
    # حفاظت از ورودی‌ها
    if not isinstance(cfg, dict): raise TypeError("cfg must be dict.")
    if not isinstance(space, dict) or not space: raise ValueError("space is empty.")
    if not isinstance(max_trials, int) or max_trials <= 0: raise ValueError("max_trials must be >0.")
    if not isinstance(symbol, str) or not symbol: raise ValueError("symbol must be non-empty.")
    if alg not in {"PPO","SAC","TD3","A2C","DQN","DDPG"}:
        raise ValueError("Unsupported alg (must be one of PPO/SAC/TD3/A2C/DQN/DDPG).")

    # مسیرهای پروژه (برای دسترسی به staging فعلی پروژه)
    paths = paths_from_cfg(cfg)
    staging = paths["models_dir"] / "staging"
    staging.mkdir(parents=True, exist_ok=True)  # پوشهٔ موجود است؛ ساختن در صورت نبود مجاز است

    best_score: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None
    best_model_path: Optional[str] = None
    trial_summaries = []

    # base_tf برای تگ‌گذاری فایل مدل
    base_tf = str(((cfg.get("env") or {}).get("base_tf") or "M5")).upper()

    for t in range(1, max_trials + 1):
        # 1) نمونه‌گیری
        params = _sample_space(space)

        # 2) cfg کاری و اعمال هایپرها
        cfg_working = deepcopy(cfg)
        _apply_sb3_hparams_to_cfg(cfg_working, alg, params)

        # 3) cfg موقت روی دیسک (بدون تغییر فایل اصلی)
        import tempfile
        with tempfile.NamedTemporaryFile(prefix="cfg_sb3_", suffix=".yaml", delete=False, mode="w", encoding="utf-8") as tf:
            tmp_cfg_path = tf.name
            # YAML اگر موجود، YAML؛ وگرنه JSON
            if yaml is not None:
                yaml.safe_dump(cfg_working, tf, allow_unicode=True, sort_keys=False)
            else:
                tf.write(json.dumps(cfg_working, ensure_ascii=False, indent=2))

        # 4) آموزش کوتاه
        ret = _run_training_cli(
            symbol=symbol, cfg_path=tmp_cfg_path, alg=alg,
            total_timesteps=total_timesteps, eval_freq=eval_freq,
            eval_steps=eval_steps, obs_agg=obs_agg, last_k=last_k
        )
        # پاک‌سازی cfg موقت
        try: os.remove(tmp_cfg_path)
        except Exception: pass

        if ret != 0:
            logger.warning("Trial %d failed during training (exit=%d).", t, ret)
            continue

        # 5) کپی مدل best به نام یکتا و ارزیابی
        tag = f"hpo_{alg}_{symbol}_{base_tf}_{t:03d}_{int(time.time())}"
        model_p = _copy_best_model(staging, tag)
        if model_p is None:
            logger.warning("Trial %d: no model found in staging (best_model.zip missing).", t)
            continue

        # 6) بک‌تست درون‌پردازه‌ای با مدل SB3
        report = run_backtest(
            symbol=symbol, cfg=cfg_working, tag="hpo_sb3_trial",
            model_path=str(model_p), sb3_alg=alg, max_steps=int(eval_steps)
        )
        score = _extract_objective(report)  # Sharpe/PF/total_reward/avg_per_step

        trial_summaries.append({
            "trial": t, "score": float(score),
            "params": params, "model_path": str(model_p)
        })
        logger.info("Trial %d/%d — objective=%.6f — params=%s", t, max_trials, score, params)

        # 7) به‌روزکردن بهترین
        if (best_score is None) or (score > best_score):
            best_score = float(score)
            best_params = dict(params)
            best_model_path = str(model_p)

    if best_score is None or best_params is None:
        raise RuntimeError("HPO finished with no successful trials; cannot choose best_hparams.")

    # 8) ذخیرهٔ خلاصهٔ نهایی در staging (بدون ساخت فولدر جدید)
    summary = {
        "symbol": symbol,
        "alg": alg,
        "base_tf": base_tf,
        "best_score": best_score,
        "best_hparams": best_params,
        "best_model_path": best_model_path,
        "trials": trial_summaries,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = staging / f"hpo_sb3_{symbol}_{alg}_{int(time.time())}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("HPO finished. Best score=%.6f | saved summary: %s", best_score, out)
    return summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SB3 Hyperparameter Search (uses training+backtest pipeline)")
    p.add_argument("--symbol", required=True)
    p.add_argument("-c", "--config", default="f01_config/config.yaml")
    p.add_argument("--alg", required=True, choices=["PPO","SAC","TD3","A2C","DQN","DDPG"])
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--space-file", required=True, help="JSON/YAML file with SB3 search space")
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--eval-steps", type=int, default=3_000)
    p.add_argument("--obs-agg", default="mean", choices=["mean","last","lastk_mean","flat"])
    p.add_argument("--last-k", type=int, default=16)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)

    space = _read_space_file(args.space_file)
    # اجرای هستهٔ HPO
    _ = sb3_hparam_search(
        cfg=cfg, space=space, max_trials=int(args.trials),
        symbol=args.symbol.upper(), alg=str(args.alg),
        total_timesteps=int(args.total_timesteps),
        eval_freq=int(args.eval_freq), eval_steps=int(args.eval_steps),
        obs_agg=str(args.obs_agg), last_k=int(args.last_k),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
نکته‌های اتصال (خیلی کوتاه):

agent_sb3_train.py هایپرها را از rl.hyperparams[alg] می‌خواند
 و مدل را در استیجینگ ذخیره می‌کند؛ ما دقیقاً همان مسیر را تغذیه می‌کنیم.

بک‌تست درون‌پردازه‌ای با پارامتر sb3_alg و model_path در دسترس است
 و برای هدف HPO استفاده می‌شود.

Run:
python -m f14_optimization.hpo_sb3 `
  --symbol XAUUSD -c f01_config/config.yaml --alg PPO `
  --trials 8 --space-file f14_optimization/spaces/ppo_space.json `
  --total-timesteps 200000 --eval-freq 25000 --eval-steps 3000

"""