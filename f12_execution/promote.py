# -*- coding: utf-8 -*-
# f12_execution/promote.py   (Promote مدل از staging به prod)
# Status in (Bot-RL-3): Completed

"""
Promote مدل از staging به prod (اتمیک و نسخه‌گذاری شده).
- هیچ حدسی دربارهٔ ساختار مدل‌ها نمی‌زنیم؛ صرفاً فایل .zip را کپی/اتمیک می‌کنیم.
- اگر health-check/criteria دیگری دارید، این‌جا اضافه کنید (الان صرفاً وجود فایل و اندازه بررسی می‌شود).
"""

from __future__ import annotations

import logging
import shutil
import time, json
from pathlib import Path

from f02_utils.config_loader import load_config
from f06_env.OLD_files.OLD_utils import paths_from_cfg
# [PROMO:IMPORTS] — unique anchor
from f10_evaluation.backtest import run_backtest

LOGGER = logging.getLogger("promote")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


def _ok_file(p: Path) -> bool:
    """بررسی سادهٔ سلامت فایل (وجود و حداقل اندازه)."""
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def promote_model(symbol: str, cfg_path: str = "f01_config/config.yaml", base_tf: str = "M5") -> Path:
    """
    کپی مدل از staging به prod با نام استاندارد.
    در صورت موفقیت مسیر مقصد را برمی‌گرداند.
    """
    cfg = load_config(cfg_path, enable_env_override=True)
    paths = paths_from_cfg(cfg)

    # -- pick latest SB3 zip from staging for the symbol & TF --
    stg_dir = paths["models_dir"] / "staging"
    candidates = sorted(
        stg_dir.glob(f"*_{symbol}_{base_tf}_*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No SB3 model found in staging for {symbol}/{base_tf}")
    src = candidates[0]


    if not _ok_file(src):
        # -----
        #raise FileNotFoundError(f"Staging model not found or invalid: {src}")
        
        # تلاش برای یافتن آخرین مدل staging با هر TF (Auto-detect)
        stg_dir = paths["models_dir"] / "staging"
        

        candidates = sorted(
            list(stg_dir.glob(f"*_{symbol}_*.zip")) + list(stg_dir.glob(f"*_{symbol}.zip")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        

        if candidates:
            # اگر دقیقِ base_tf موجود نیست، جدیدترین را می‌گیریم و هشدار می‌دهیم
            LOGGER.warning("Requested base_tf %s not found; using latest staging model: %s",
                        base_tf, candidates[0].name)
            src = candidates[0]
            # TF واقعی را از نام فایل استخراج می‌کنیم تا نام مقصد صحیح باشد
            import re
            m = re.match(rf"(?P<alg>[A-Za-z0-9]+)_{symbol}_(?P<tf>[A-Za-z0-9]+).*\.zip$", src.name)

            if m:
                base_tf = m.group("tf")
        else:
            raise FileNotFoundError(f"Staging model not found or invalid: {src}")


    dst_dir = paths["models_dir"] / "prod"
    dst_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dst = dst_dir / f"{symbol}_{base_tf}_{stamp}.zip"

    # کپی اتمیک: اول tmp سپس rename
    tmp = dst.with_suffix(".zip.tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    LOGGER.info("Model promoted to: %s", dst)
    # لینک آخرین نسخه (اختیاری)
    latest = dst_dir / f"{symbol}_{base_tf}.zip"
    try:
        if latest.exists():
            latest.unlink()
        shutil.copy2(dst, latest)
        LOGGER.info("Latest copy updated: %s", latest)
    except Exception:
        LOGGER.warning("Failed to update latest copy (non-fatal).")
    # [SELFOPT:PROMOTE_LOG] — unique anchor
    try:
        ver_dir = paths["models_dir"] / "versions"
        ver_dir.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
               "symbol": symbol, "base_tf": base_tf,
               "src": str(src), "dst": str(dst), "size": int(dst.stat().st_size)}
        with open(ver_dir / "promote.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        LOGGER.warning("Failed to append promote log (non-fatal).")
    return dst


# [PROMO:CHECK] — unique anchor
def _should_promote(symbol: str, base_tf: str, cfg: dict, steps: int,
                    min_avg: float | None, min_total: float | None) -> bool:
    cfg2 = dict(cfg); (cfg2.setdefault("features", {}))["base_timeframe"] = base_tf
    rep = run_backtest(symbol=symbol, cfg=cfg2, tag="promote_check", max_steps=int(steps))
    avg, tot = float(rep.get("avg_per_step", 0.0)), float(rep.get("total_reward", 0.0))
    ok = True
    if (min_avg   is not None) and not (avg >= float(min_avg)):   ok = False
    if (min_total is not None) and not (tot >= float(min_total)): ok = False
    LOGGER.info("PromoteCheck: steps=%s avg=%.6f total=%.6f ok=%s",
                rep.get("steps"), avg, tot, ok)
    return ok


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Promote model from staging to prod")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("-c", "--config", default="f01_config/config.yaml")
    p.add_argument("--base-tf", default="M5")

    # [PROMO:CLI] — unique anchor
    p.add_argument("--check", action="store_true", help="run backtest gate before promote")
    p.add_argument("--bt-steps", type=int, default=1000, help="backtest steps for gate")
    p.add_argument("--bt-min-avg", type=float, default=None, help="min avg_per_step to pass")
    p.add_argument("--bt-min-total", type=float, default=None, help="min total_reward to pass")

    args = p.parse_args()

    # [PROMO:APPLY] — unique anchor
    # --------------------------------- start
    cfg = load_config(args.config, enable_env_override=True)
    btf = args.base_tf.upper()
    if args.check:
        ok = _should_promote(args.symbol, btf, cfg, args.bt_steps, args.bt_min_avg, args.bt_min_total)
        if not ok:
            LOGGER.warning("Promote aborted: gate not satisfied.")
            return 4
    # --------------------------------- end

    promote_model(symbol=args.symbol, cfg_path=args.config, base_tf=args.base_tf.upper())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
