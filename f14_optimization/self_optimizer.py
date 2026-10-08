# -*- coding: utf-8 -*-
# f14_optimization/self_optimizer.py
# Status in (Bot-RL-3): Completed

"""
Self-Optimize Orchestrator (Final)
==================================
این فایل «هماهنگ‌کنندهٔ نهایی» فرایند خودبهینه‌سازی است و دقیقاً مطابق
ساختار پروژه‌ی فعلی پیاده‌سازی شده است:

- فقط روی **cfg در حافظه** کار می‌کند و در طول حلقهٔ تیون چیزی روی دیسک
  نمی‌نویسد. Persist نهایی فقط بعد از عبور از گیت‌ها انجام می‌شود.
- از پل هایپرپارامترها در f13 (`hparam_bridge`) برای تزریق pa.* به cfg استفاده می‌کند.
- از وایرینگ پرایس‌اکشن در f04 (`config_wiring`) پشتیبانی می‌کند (برای ساخت فیچرها، اگر جایی لازم باشد).
- به سایر ماژول‌های f0X فقط «هوشمندانه» فراخوان می‌دهد: با *بازتاب امضا* (inspect)
  تنها پارامترهایی را پاس می‌کند که آن توابع واقعاً پشتیبانی می‌کنند؛ بنابراین از حدس
  و SignatureError جلوگیری می‌شود.
- پیام‌های runtime انگلیسی هستند؛ توضیحات فارسی در کامنت‌ها نوشته شده‌اند.

توجه مهم (مطابق سیاست شما):
- در هر trial (آزمایش) فقط cfg کاری (in-memory) تغییر می‌کند.
- تنها پس از انتخاب کاندید و عبور از گیت‌ها، پیکربندی نسخه‌گذاری و اتمیک روی دیسک ذخیره می‌شود.

"""

from __future__ import annotations

# استانداردها
import os
import io
import sys
import time
import copy
import json
import uuid
import types
import shutil
import logging
import subprocess, shlex
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Callable

# --- وابستگی‌های داخلی پروژه (همگی در f01..f14 هستند) ---
# بارگذاری/ذخیرهٔ پیکربندی
try:
    from f02_utils.config_loader import load_config, save_config_versioned  # type: ignore
except Exception:  # pragma: no cover
    load_config = None
    save_config_versioned = None

# Logging setup
try:
    from f02_utils.logging_cfg import setup_logging  # type: ignore
except Exception:  # pragma: no cover
    setup_logging = None

# تازه‌سازی دیتاست‌ها
try:
    from f03_data.OLD.data_handler import refresh_datasets  # type: ignore
except Exception:  # pragma: no cover
    refresh_datasets = None

# آموزش/جستجوی هایپرپارامتر
from f14_optimization.hparam_search import hyperparam_search
try:
    from f09_training.train import finetune_model  # type: ignore
except Exception:  # pragma: no cover
    finetune_model = None

# بک‌تست و متریک
try:
    from f10_evaluation.backtest import run_backtest  # type: ignore
except Exception:  # pragma: no cover
    run_backtest = None

try:
    from f10_evaluation.metrics import summarize_metrics  # type: ignore
except Exception:  # pragma: no cover
    summarize_metrics = None

# اجرا/استقرار
try:
    from f12_execution.executor import stage_deploy, promote_to_prod, rollback  # type: ignore
except Exception:  # pragma: no cover
    stage_deploy = None
    promote_to_prod = None
    rollback = None

# Promote API (درون‌پردازه‌ای، بدون حدس و بدون دست‌کاری config در حلقه)
try:
    from f12_execution.promote import promote_model  # type: ignore
except Exception:  # pragma: no cover
    promote_model = None


# گیت‌های ریسک
try:
    from f02_utils.risk_manager import SafetyGates  # type: ignore
except Exception:  # pragma: no cover
    SafetyGates = None

# پل و وایرینگ پرایس‌اکشن (فایل‌هایی که همین پروژه دارد)
from f14_optimization.hparam_bridge import apply_pa_hparams_to_config
from f04_features.price_action.config_wiring import build_pa_features_from_config

# YAML (fallback اگر loader داخلی نبود)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


logger = logging.getLogger("self_optimizer")



def _run_module_cli(module: str, args: list[str]) -> int:
    """
    اجرای ماژول‌های واقعی پروژه از طریق python -m <module> ...
    مثال: _run_module_cli("f03_data.mt5_data_loader", ["-c", "f01_config/config.yaml", "--symbols", "XAUUSD"])
    """
    cmd = [sys.executable, "-m", module] + list(args)
    logger.info("Running module: %s", " ".join(shlex.quote(x) for x in cmd))
    try:
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            logger.warning("Module %s exited with code %s", module, res.returncode)
        return int(res.returncode)
    except Exception as e:
        logger.exception("Failed to run module %s: %s", module, e)
        return -1



# ---------------------------------------------------------------------
# ابزارک‌های امن برای سازگاری با امضاهای موجود (بدون حدس)
# ---------------------------------------------------------------------
def _safe_call(fn: Optional[Callable], **kwargs) -> Any:
    """
    فراخوان امن یک تابع با فیلتر کردن فقط پارامترهای پشتیبانی‌شده.
    - اگر fn در دسترس نباشد → لاگ و None.
    - اگر امضا قابل بازیابی نباشد → تلاش با **kwargs و در صورت خطا، لاگ و None.
    """
    if fn is None:
        logger.warning("Skipping call: function is not available.")
        return None
    try:
        sig = inspect.signature(fn)
        accepted = {}
        for name, param in sig.parameters.items():
            if name in kwargs:
                accepted[name] = kwargs[name]
        return fn(**accepted)
    except Exception as e:  # pragma: no cover
        logger.exception("Safe call failed for %s: %s", getattr(fn, "__name__", str(fn)), e)
        return None


def _deepcopy(obj: Any) -> Any:
    try:
        return copy.deepcopy(obj)
    except Exception:
        return json.loads(json.dumps(obj))


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """نوشتن اتمیک روی دیسک؛ ابتدا فایل موقت، سپس rename."""
    tmp = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex[:8]}")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


# ---------------------------------------------------------------------
# پارامترهای سطح بالا برای اجرای Self-Optimize
# ---------------------------------------------------------------------
@dataclass
class OptimizeParams:
    """پارامترهای اجرای self-optimize (قابل توسعه)"""
    symbol: str
    horizon_days: int = 7
    do_hparam_search: bool = True
    max_trials: int = 20
    finetune_steps: int = 75_000
    canary_risk_mult: float = 0.5
    forward_check: bool = True
    forward_days: int = 3
    persist_enabled: bool = True
    persist_to_staging: bool = False
    require_validation: bool = True
    # رزرو برای توسعه آتی
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# هستهٔ Orchestrator
# ---------------------------------------------------------------------
class SelfOptimizer:
    """
    ارکستریتور نهایی:
    - load config
    - refresh datasets
    - baseline backtest
    - hyperparam search (اختیاری)
    - apply best hparams (in-memory)
    - fine-tune + candidate backtest
    - safety gates + (optional) forward check
    - stage & promote
    - persist versioned config (atomic)
    """

    def __init__(self, cfg_path: str = "f01_config/config.yaml") -> None:
        self.cfg_path = Path(cfg_path)
        # راه‌اندازی لاگ
        if setup_logging is not None:
            _safe_call(setup_logging, name="self_optimizer")
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Self-Optimizer initializing...")

        # بارگذاری پیکربندی
        self.cfg: Dict[str, Any] = self._load_cfg(self.cfg_path)
        logger.info("Config loaded from %s", self.cfg_path.as_posix())

        # برای hparam_search → CLI fallback/لاگ
        self.cfg["__cfg_path__"] = str(self.cfg_path)

    # -------------------- کمکی‌ها --------------------
    def _load_cfg(self, path: Path) -> Dict[str, Any]:
        """بارگذاری config با اولویت استفاده از loader پروژه؛ در غیر این صورت YAML خام."""
        if load_config is not None:
            cfg = _safe_call(load_config, path=str(path))
            if isinstance(cfg, dict):
                return cfg
        # fallback
        if yaml is None:
            raise RuntimeError("PyYAML is not available and project loader failed.")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _persist_cfg_versioned(self, cfg: Dict[str, Any], prefix: str = "prod_") -> Optional[Path]:
        """ذخیرهٔ نسخه‌گذاری‌شدهٔ config (اتمیک). ابتدا تلاش با save_config_versioned، سپس fallback."""
        if save_config_versioned is not None:
            out = _safe_call(save_config_versioned, cfg=cfg, prefix=prefix)
            return out if isinstance(out, Path) else None

        # fallback: فایل اصلی را بک‌آپ بگیریم و سپس بنویسیم (اتمیک)
        if yaml is None:
            logger.error("Cannot persist config: PyYAML not available.")
            return None
        backup = self.cfg_path.with_suffix(".yaml.bak")
        try:
            if self.cfg_path.exists():
                shutil.copy2(self.cfg_path, backup)
                logger.info("Backup created: %s", backup.name)
            text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
            _atomic_write_text(self.cfg_path, text, encoding="utf-8")
            logger.info("Config persisted atomically to %s", self.cfg_path.name)
            return self.cfg_path
        except Exception as e:  # pragma: no cover
            logger.exception("Persist failed: %s", e)
            return None

    def _get_hparam_space(self) -> Dict[str, Any]:
        """استخراج فضای جست‌وجوی هایپرپارامتر از config (بدون حدس)."""
        return (
            self.cfg.get("self_optimize", {})
                    .get("steps", {})
                    .get("hparam_search", {})
                    .get("space", {})
            or {}
        )

    # -------------------- فاز اجرا --------------------
    def run(self, p: OptimizeParams) -> None:
        """اجرای کامل و امن فرایند self-optimize مطابق سیاست‌های پروژه."""
        t0 = time.time()
        logger.info("=== Self-Optimization started for %s ===", p.symbol)

        # 1) تازه‌سازی دیتاست‌ها (اختیاری)
        #_safe_call(refresh_datasets, cfg=self.cfg, symbol=p.symbol, horizon_days=p.horizon_days)            horizon_days=p.horizon_days,
        # 1) تازه‌سازی دیتاست‌ها (واقعی)
        rc_dl = 0
        if refresh_datasets is not None:
            _safe_call(refresh_datasets, cfg=self.cfg, symbol=p.symbol, horizon_days=p.horizon_days)
        else:
            # فراخوانی دانلودگر واقعی MT5 (CLI): f03_data.mt5_data_loader
            rc_dl = _run_module_cli(
                "f03_data.mt5_data_loader",
                ["-c", str(self.cfg_path), "--symbols", p.symbol]
            )
            # پس از دانلود، دیتافریم یکپارچه را با DataHandler بسازیم
            # base_tf از کانفیگ: features.resample_to_base_tf یا پیش‌فرض "M5"
            base_tf = (
                self.cfg.get("features", {}).get("resample_to_base_tf")
                or self.cfg.get("features", {}).get("base_tf")
                or "M5"
            )
            rc_build = _run_module_cli(
                "f03_data.data_handler",
                ["-c", str(self.cfg_path), "--symbol", p.symbol, "--base-tf", str(base_tf)]
            )
            if rc_dl != 0 or rc_build != 0:
                logger.warning("Data refresh/build returned non-zero exit code(s): dl=%s, build=%s", rc_dl, rc_build)


        # 2) بک‌تست baseline
        baseline_report = None
        if run_backtest is not None:
            baseline_report = _safe_call(run_backtest, symbol=p.symbol, cfg=self.cfg, tag="baseline", model_path=None)
        else:
            _run_module_cli(
                "f10_evaluation.backtest",
                ["--symbol", p.symbol, "-c", str(self.cfg_path)]
            )
 

        if summarize_metrics is not None and baseline_report is not None:
            _safe_call(summarize_metrics, report=baseline_report, tag="baseline")

        # 3) جستجوی هایپرپارامتر (اختیاری)
        best_hparams: Dict[str, Any] = {}
        if p.do_hparam_search:
            search_space = self._get_hparam_space()
            if not search_space:
                logger.warning("HParam search skipped: empty space in config.")
            else:
                logger.info("Hyperparameter search started (trials=%s)", p.max_trials)
                # in-process, no fallback:
                res = hyperparam_search(cfg=self.cfg, space=search_space, max_trials=p.max_trials, symbol=p.symbol)
                best_hparams = res.get("best_hparams") or {}
                logger.info("Hparam search done. Best: %s", best_hparams if best_hparams else "{}")
        else:
            logger.info("Hyperparameter search skipped.")


        # 4) اعمال بهترین هایپرپارامترها روی cfg کاری (فقط در حافظه)
        cfg_working = _deepcopy(self.cfg)
        if best_hparams:
            apply_pa_hparams_to_config(cfg_working, best_hparams)
            logger.info("Applied best PA hyperparameters to working cfg (in-memory).")
        else:
            logger.info("No PA hyperparameters to apply; using baseline cfg for candidate steps.")

        # (اختیاری) اگر pipeline نیاز به ساخت فیچرهای PA داشته باشد:
        # - معمولاً ماژول‌های فازهای بعدی خودشان از cfg می‌خوانند؛ این خط فقط نمونه است.
        _ = build_pa_features_from_config  # silence linter (symbolic reference)

        # 5) فاین‌تیون مدل کاندید (در صورت وجود ماژول)
        #cand_ckpt = _safe_call(finetune_model, symbol=p.symbol, cfg=cfg_working, steps=p.finetune_steps)
        # 5) فاین‌تیون مدل کاندید (یا آموزش)
        cand_ckpt = None
        if finetune_model is not None:
            cand_ckpt = _safe_call(finetune_model, symbol=p.symbol, cfg=cfg_working, steps=p.finetune_steps)
        else:
            _run_module_cli(
                "f09_training.train",
                ["--symbol", p.symbol, "-c", str(self.cfg_path)]
            )


        # 6) بک‌تست کاندید
        #cand_report = _safe_call(run_backtest, symbol=p.symbol, cfg=cfg_working, tag="candidate", model_path=cand_ckpt)
        # 6) بک‌تست کاندید
        cand_report = None
        if run_backtest is not None:
            cand_report = _safe_call(run_backtest, symbol=p.symbol, cfg=cfg_working, tag="candidate", model_path=cand_ckpt)
        else:
            _run_module_cli(
                "f10_evaluation.backtest",
                ["--symbol", p.symbol, "-c", str(self.cfg_path)]
            )



        if summarize_metrics is not None and cand_report is not None:
            _safe_call(summarize_metrics, report=cand_report, tag="candidate")

        # 7) گیت‌های ریسک/عملکرد
        gates_passed = True
        if SafetyGates is not None:
            try:
                gates = SafetyGates(cfg=self.cfg)  # اگر امضای سازنده متفاوت باشد، _safe_call هم می‌توانست استفاده شود
                gates_passed = bool(gates.check(cand_report, baseline_report))
            except Exception as e:  # pragma: no cover
                logger.exception("Safety gates failed to run: %s", e)
                gates_passed = False
        logger.info("Safety gates result: %s", "PASS" if gates_passed else "FAIL")

        # 8) (اختیاری) Forward check کوتاه برای پایداری
        '''
        if p.forward_check:
            fwd_report = _safe_call(
                run_backtest,
                symbol=p.symbol,
                cfg=cfg_working,
                tag="forward",
                model_path=cand_ckpt,
                forward_days=p.forward_days,
            )
            if summarize_metrics is not None and fwd_report is not None:
                _safe_call(summarize_metrics, report=fwd_report, tag="forward")
        '''
        # 8) (اختیاری) Forward check کوتاه
        if p.forward_check:
            if run_backtest is not None:
                fwd_report = _safe_call(
                    run_backtest, symbol=p.symbol, cfg=cfg_working, tag="forward",
                    model_path=cand_ckpt, forward_days=p.forward_days
                )
                if summarize_metrics is not None and fwd_report is not None:
                    _safe_call(summarize_metrics, report=fwd_report, tag="forward")
            else:
                _run_module_cli(
                    "f10_evaluation.backtest",
                    ["--symbol", p.symbol, "-c", str(self.cfg_path)]
                )




        # اگر گیت‌ها رد شوند → rollback (در صورت وجود)
        if not gates_passed:
            _safe_call(rollback, reason="safety_gates_failed", symbol=p.symbol)
            logger.warning("Candidate rejected by safety gates; nothing persisted.")
            logger.info("=== Self-Optimization finished (REJECTED) for %s in %.2fs ===",
                        p.symbol, time.time() - t0)
            return

        # 9) استیجینگ و سپس پروموت به prod
        '''
        staged_cfg = _safe_call(
            stage_deploy,
            symbol=p.symbol,
            cfg=cfg_working,
            model_path=cand_ckpt,
            risk_multiplier=p.canary_risk_mult,
        )
        _safe_call(promote_to_prod, cfg=staged_cfg or cfg_working)
        # 9) استیجینگ/پروموت (در این نسخه ماژول f12_execution موجود نیست)
        staged_cfg = cfg_working
        logger.info("Stage/Promote skipped: f12_execution module not available in this project snapshot.")
        '''

        # 9) Promote مدل به prod (درون‌پردازه‌ای و بدون حدس)
        staged_cfg = cfg_working
        if p.persist_enabled and promote_model is not None:
            try:
                base_tf_cfg = (self.cfg.get("features") or {}).get("base_timeframe", "M5")
                base_tf_final = str(base_tf_cfg).upper()
                # Promote بر اساس جدیدترین مدل staging برای همین symbol/TF (یا Auto-detect داخل promote.py)
                _safe_call(promote_model, symbol=p.symbol, cfg_path=str(self.cfg_path), base_tf=base_tf_final)
                logger.info("Stage/Promote completed via f12_execution.promote.")
            except Exception as e:  # pragma: no cover
                logger.exception("Stage/Promote failed: %s", e)
        else:
            logger.info("Stage/Promote skipped (persist disabled or promote_model unavailable).")

        # 10) Persist نهایی (فقط پس از پذیرش)
        if p.persist_enabled:
            logger.info("Persisting tuned configuration (atomic, versioned).")
            self._persist_cfg_versioned(staged_cfg or cfg_working, prefix="prod_")
        else:
            logger.info("Persistence disabled by params; skipping on-disk write.")

        logger.info("=== Self-Optimization finished successfully for %s in %.2fs ===",
                    p.symbol, time.time() - t0)


# ---------------------------------------------------------------------
# نمونهٔ اجرای مستقیم
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # ⚠️ فقط نمونه؛ برای اجرای واقعی، مقادیر را بر اساس نیاز تغییر دهید.
    params = OptimizeParams(
        symbol="XAUUSD",
        horizon_days=7,
        do_hparam_search=False,
        max_trials=20,
        finetune_steps=75_000,
        canary_risk_mult=0.5,
        forward_check=True,
        forward_days=3,
        persist_enabled=True,
    )
    so = SelfOptimizer(cfg_path="f01_config/config.yaml")
    so.run(params)
