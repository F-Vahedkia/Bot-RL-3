# -*- coding: utf-8 -*-
# f14_optimization/hparam_search.py
# Status in (Bot-RL-3): Completed

"""
Hyperparameter Search (Final)
============================
این ماژول «موتور جست‌وجوی هایپرپارامتر» را برای پرایس‌اکشن (و سایر بخش‌ها)
ارائه می‌دهد تا توسط ارکستریتور `self_optimizer.py` صدا زده شود.

نکات کلیدی مطابق قوانین پروژه:
- بدون حدس روی ساختار پروژه: از پل رسمی اعمال هایپرپارامترها استفاده می‌کنیم
  (apply_pa_hparams_to_config) و ارزیابی را با backtest رسمی پروژه انجام می‌دهیم.
- پیاده‌سازی *read→apply→evaluate* کاملاً in-memory است؛ چیزی روی دیسک
  در طول جست‌وجو نوشته نمی‌شود.
- اگر backtest نتیجهٔ عددیِ قابل استفاده بازنگرداند، با پیام شفاف متوقف می‌شویم
  تا از انتخاب تصادفی/غلط جلوگیری شود (no silent guessing).
- پیام‌های runtime انگلیسی هستند؛ توضیحات فارسی در کامنت‌ها.
- علاوه بر شمای استاندارد، شمای فشرده مانند
  {min,max,step} / {choices} نیز به‌صورت خودکار پشتیبانی می‌شود.

امضای عمومی ماژول:
    hyperparam_search(cfg: dict, space: dict, max_trials: int, symbol: str) -> dict
خروجی:
    {"best_hparams": <dict sample with pa.* keys>, "best_score": <float>}

انتظار فضای جست‌وجو (نمونه‌های متداول):
    {
      "pa.market_structure.lookback": {"type": "int",   "low": 2, "high": 6, "step": 1},
      "pa.breakouts.range_window":    {"type": "int",   "low": 6, "high": 14, "step": 2},
      "pa.confluence.weights":        {"type": "float_vec", "size": 5, "low": 0.0, "high": 0.4, "normalize": true},
      "pa.microchannels.min_len":     {"type": "choice","values": [3,4,5]},
      ...
    }

توجه: اگر در `space` تایپ ناشناخته‌ای ببینیم یا کلیدها اصلاً وجود نداشته باشند،
      با پیام شفاف و توقف روبه‌رو می‌شویم تا از حدس جلوگیری شود.
"""

from __future__ import annotations

import math
import random
import logging
import subprocess, re, sys
import shlex
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List

from f14_optimization.hparam_bridge import apply_pa_hparams_to_config
from f04_features.price_action.config_wiring import build_pa_features_from_config  # جهت اطمینان از وایرینگ
# ارزیابی درون‌پردازه‌ای، اگر در دسترس باشد:
try:
    from f10_evaluation.backtest import run_backtest  # type: ignore
except Exception:  # pragma: no cover
    run_backtest = None
if run_backtest is None:
    raise ImportError("run_backtest is required for in-process PA hparam search (CLI fallback is disabled).")

logger = logging.getLogger("hparam_search")


def _normalize_spec(key: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    نرمال‌سازی شمای فشردهٔ کانفیگ به شمای استاندارد داخلی.
    پشتیبانی: {min,max,step} و {choices}
    - اگر همهٔ min/max/step عدد صحیح باشند → int
    - اگر min/max/step عدد اعشاری باشند → float
    - اگر choices موجود باشد → choice (همان values)
    """
    # اگر spec خودش لیست باشد → مانند choices
    if isinstance(spec, list):
        return {"type": "choice", "values": list(spec)}

    if not isinstance(spec, dict):
        raise ValueError(f"Invalid spec for key {key!r}: expected dict.")
    # حالت choices
    if "choices" in spec:
        vals = list(spec["choices"])
        if not vals:
            raise ValueError(f"Empty choices for key {key!r}.")
        return {"type": "choice", "values": vals}

    # حالت min/max/step
    has_range = {"min", "max"}.issubset(spec.keys())
    if has_range:
        lo, hi = spec["min"], spec["max"]
        step = spec.get("step", None)
        # تشخیص نوع: اگر هرکدام float باشد → float
        is_float = any(isinstance(x, float) for x in (lo, hi, step) if x is not None)
        if is_float:
            out = {"type": "float", "low": float(lo), "high": float(hi)}
            if step is not None:
                out["step"] = float(step)
            return out
        else:
            out = {"type": "int", "low": int(lo), "high": int(hi)}
            if step is not None:
                out["step"] = int(step)
            return out

    # اگر قبلاً شمای استاندارد ما را داشتید، همان را برگردانید
    if "type" in spec:
        return spec  # already normalized

    raise ValueError(f"Unsupported spec format for key {key!r}.")

# ---------------------------------------------------------------------
# ابزار اجرا از طریق CLI ماژول‌های پروژه (fallback امن)
# ---------------------------------------------------------------------
'''
def _run_module_cli(module: str, args: List[str]) -> int:
    """
    اجرای ماژول پروژه با python -m <module> ... و بازگرداندن کد خروج.
    این تابع فقط fallback است وقتی run_backtest درون‌پردازه‌ای در دسترس نیست.
    """
    cmd = [sys.executable, "-m", module] + list(args)
    logger.info("Running module: %s", " ".join(shlex.quote(x) for x in cmd))
    try:
        res = subprocess.run(cmd, check=False)
        return int(res.returncode)
    except Exception as e:  # pragma: no cover
        logger.exception("CLI run failed for %s: %s", module, e)
        return -1
'''

# ---------------------------------------------------------------------
# نمونه‌گیری از فضای جست‌وجو (بدون حدس؛ فقط تایپ‌های صریح پشتیبانی می‌شود)
# ---------------------------------------------------------------------
def _sample_one(spec: Dict[str, Any]) -> Any:
    """
    نمونه‌گیری یک پارامتر طبق spec.
    تایپ‌های پشتیبانی‌شده:
      - int:   {"type":"int","low":a,"high":b,"step":k?}
      - float: {"type":"float","low":a,"high":b,"step":k?}
      - choice:{"type":"choice","values":[...]}
      - float_vec: {"type":"float_vec","size":N,"low":a,"high":b,"normalize":bool?}
    """
    t = (spec.get("type") or "").lower()
    if t == "int":
        low, high = int(spec["low"]), int(spec["high"])
        step = int(spec.get("step", 1))
        # اطمینان از درستی بازه
        if step <= 0 or low > high:
            raise ValueError("Invalid int spec.")
        grid = list(range(low, high + 1, step))
        return random.choice(grid)
    elif t == "float":
        low, high = float(spec["low"]), float(spec["high"])
        step = spec.get("step", None)
        if step is None:
            return random.uniform(low, high)
        step = float(step)
        if step <= 0 or low > high:
            raise ValueError("Invalid float spec.")
        # شبکه‌ی یکنواخت
        n = max(1, int(math.floor((high - low) / step)))
        grid = [low + i * step for i in range(n + 1)]
        return random.choice(grid)
    elif t == "choice":
        vals = list(spec["values"])
        if not vals:
            raise ValueError("Empty choice list.")
        return random.choice(vals)
    elif t == "float_vec":
        size = int(spec.get("size", 4))
        low, high = float(spec["low"]), float(spec["high"])
        normalize = bool(spec.get("normalize", True))
        vec = [random.uniform(low, high) for _ in range(size)]
        if normalize:
            s = sum(vec)
            if s > 0:
                vec = [x / s for x in vec]
        return vec
    else:
        raise ValueError(f"Unsupported space type: {t!r}")


def _sample_space(space: Dict[str, Any]) -> Dict[str, Any]:
    """
    ساخت نمونهٔ کامل از فضای جست‌وجو.
    پشتیبانی از:
      - شِمای فشرده ({min,max,step} / {choices} / لیست خام)
      - rule وابسته به پارامترهای هم‌گروه (مثلاً breakouts.*)
    """
    if not isinstance(space, dict) or not space:
        raise ValueError("Empty or invalid search space.")

    # مرحله آماده‌سازی: تفکیک specهای rule از بقیه
    norm_map: Dict[str, Dict[str, Any]] = {}
    rule_map: Dict[str, Dict[str, Any]] = {}

    for key, raw in space.items():
        if isinstance(raw, dict) and "rule" in raw:
            rule_map[key] = raw                      # ← فعلاً نرمال‌سازی نشود
        else:
            if isinstance(raw, (dict, list)):
                norm_map[key] = _normalize_spec(key, raw)
            else:
                # مقدار خام مثل  [2,3,4,5]  → به‌مثابه choices
                norm_map[key] = _normalize_spec(key, {"choices": raw})

    # مرحله ۱: نمونه‌گیری کلیدهای بدون rule
    sample: Dict[str, Any] = {}
    for key, spec in norm_map.items():
        sample[key] = _sample_one(spec)

    # مرحله ۲: محاسبهٔ کلیدهای rule با دسترسی به خواهر/برادرها (هم‌گروه‌ها)
    import math
    allowed_funcs = {"max": max, "min": min, "int": int, "round": round, "floor": math.floor}
    for key, raw in rule_map.items():
        rule_expr = str(raw["rule"])
        # prefix مشترک گروه (مثلاً "pa.breakouts.")
        parts = key.split(".")
        prefix = ".".join(parts[:-1]) + "."

        # ساخت locals از هم‌گروه‌ها با نام کوتاه (مثلاً range_window)
        sibs = {k[len(prefix):]: v for k, v in sample.items() if k.startswith(prefix)}

        try:
            rhs = rule_expr.split("=", 1)[-1].strip()  # فقط سمت راست «=»
            val = eval(rhs, {}, {**allowed_funcs, **sibs})
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate rule for {key!r}: {e}")

        sample[key] = val

    return sample


# ---------------------------------------------------------------------
# استخراج عدد هدف از خروجی بک‌تست (بدون حدس خاموش)
# ---------------------------------------------------------------------
def _extract_objective(report: Any) -> float:
    """
    تلاش شفاف برای استخراج معیار هدف از خروجی backtest درون‌پردازه‌ای.
    فقط کلیدهای صریح و رایج بررسی می‌شوند. اگر پیدا نشود، با خطا متوقف می‌شویم.
    """
    if report is None:
        raise RuntimeError("Backtest returned None; cannot evaluate objective.")
    # تنها دیکشنری‌های عددی معتبرند
    if isinstance(report, dict):
        # اولویت کلیدها (بدون حدس آزاد؛ فقط این‌های صریح)
        for k in ("sharpe", "pf", "profit_factor", "total_reward", "avg_per_step"):
            v = report.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    raise RuntimeError("Unsupported backtest report format; cannot find a numeric objective.")


# ---------------------------------------------------------------------
# ارزیابی یک نمونه (run_backtest یا CLI fallback)
# ---------------------------------------------------------------------
def _evaluate_sample_old1(cfg_base: Dict[str, Any], sample: Dict[str, Any], symbol: str, cfg_path: str) -> Tuple[float, Dict[str, Any]]:
    """
    اعمال sample روی cfg کاری و اجرای بک‌تست. خروجی: (objective, cfg_applied)
    - اگر run_backtest درون‌پردازه‌ای در دسترس باشد → استفاده می‌کنیم.
    - در غیر این صورت، از CLI استفاده می‌کنیم و صرفاً موفقیت اجرا را بررسی می‌کنیم.
      (در حالت CLI اگر راه دریافتِ متریک عددی نداشته باشیم، با خطای شفاف متوقف می‌شویم.)
    """
    # کپی cfg و اعمال نمونه (فقط in-memory)
    cfg_working = deepcopy(cfg_base)
    apply_pa_hparams_to_config(cfg_working, sample)

    # (اختیاری) اطمینان از سالم‌بودن وایرینگ PA — اگر نیاز بود می‌توانیم یک بار build را صدا بزنیم
    try:
        # فقط بررسی لایت‌وزن؛ اگر کار نکرد، مشکلی برای ادامه ایجاد نکند
        _ = build_pa_features_from_config
    except Exception:
        pass

    # مسیر ۱: ارزیابی درون‌پردازه‌ای
    if run_backtest is not None:
        report = run_backtest(symbol=symbol, cfg=cfg_working, tag="hs_trial", model_path=None)
        score = _extract_objective(report)
        return score, cfg_working

    # مسیر ۲: fallback CLI (برای پروژه‌ای که run_backtest را فقط به صورت CLI دارد)
    # در این مسیر، چون دسترسی برنامه‌ای به متریک نداریم، از حدس جلوگیری می‌کنیم
    # و با پیام شفاف متوقف می‌شویم تا مسیر in-process فراهم شود.
    # متن بالا مربوط به قطعه کد قبلی بوده است.
    cmd = [sys.executable, "-m", "f10_evaluation.backtest", "--symbol", symbol, "-c", cfg_path]

    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Backtest CLI failed with exit code {res.returncode}.")
    out = (res.stdout or "") + "\n" + (res.stderr or "")
    m = re.search(r"total_reward=([+-]?\d+(?:\.\d+)?)", out)
    if not m:
        raise RuntimeError("Backtest objective not found in CLI output.")
    score = float(m.group(1))
    return score, cfg_working

def _evaluate_sample(cfg_base: Dict[str, Any], sample: Dict[str, Any], symbol: str, cfg_path: str) -> Tuple[float, Dict[str, Any]]:
    # ساخت cfg کاری در حافظه و اعمال نمونه
    cfg_working = deepcopy(cfg_base)
    apply_pa_hparams_to_config(cfg_working, sample)

    # ارزیابی درون‌پردازه‌ای با همان cfg کاری
    report = run_backtest(symbol=symbol, cfg=cfg_working, tag="hs_trial", model_path=None)
    score = _extract_objective(report)
    return score, cfg_working

# ---------------------------------------------------------------------
# نقطهٔ ورود عمومی: جست‌وجوی هایپرپارامتر
# ---------------------------------------------------------------------
def hyperparam_search(*, cfg: Dict[str, Any], space: Dict[str, Any], max_trials: int, symbol: str) -> Dict[str, Any]:
    """
    نقطهٔ ورود استاندارد برای ارکستریتور:
      - cfg:       دیکشنری کانفیگ
      - space:     فضای جست‌وجو از config.yaml (کلیدهای pa.* با specهایی که در بالا گفتیم)
      - max_trials: تعداد نمونه‌های تصادفی برای ارزیابی
      - symbol:    سمبل معاملاتی

    خروجی:
      {"best_hparams": <dict pa.* → value>, "best_score": <float>}
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict.")
    if not isinstance(space, dict) or not space:
        raise ValueError("search space is empty or invalid.")
    if not isinstance(max_trials, int) or max_trials <= 0:
        raise ValueError("max_trials must be a positive integer.")
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("symbol must be a non-empty string.")

    logger.info("Hyperparameter search started (trials=%d)", max_trials)

    best_score: Optional[float] = None
    best_sample: Optional[Dict[str, Any]] = None

    # برای reproducibility در صورت نیاز می‌توان seed گرفت؛ فعلاً از randomness پیش‌فرض استفاده می‌کنیم.
    for t in range(1, max_trials + 1):
        sample = _sample_space(space)
        try:
            score, _cfg_applied = _evaluate_sample(cfg, sample, symbol, cfg_path=cfg.get("__cfg_path__", "f01_config/config.yaml"))
        except Exception as e:
            logger.warning("Trial %d failed: %s", t, e)
            continue

        logger.debug("Trial %d/%d — objective=%.6f — sample=%s", t, max_trials, score, sample)

        if (best_score is None) or (score > best_score):
            best_score = score
            best_sample = sample

    if best_sample is None or best_score is None:
        # عدم انتخاب نتیجهٔ برتر به‌صورت ساکت مجاز نیست؛ شفاف متوقف می‌شویم
        raise RuntimeError("Hyperparameter search finished with no successful trials; cannot choose best_hparams.")

    logger.info("Hparam search done. Best score=%.6f, Best params=%s", best_score, best_sample)
    return {"best_hparams": best_sample, "best_score": float(best_score)}

