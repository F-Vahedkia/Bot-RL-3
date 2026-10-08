# -*- coding: utf-8 -*-
# f03_data/symbol_specs_snapshot.py
# Status in (Bot-RL-3): Completed at 1404-09-20
# New (adds snapshot+overlay writer without altering core logic)

"""
ابزار ثبت «Snapshot مشخصات نمادها» از MT5 و تولید فایل overlay برای کانفیگ.
این فایل مشخصات نمادها را از ترمینال متاتریدر دریافت میکند،
یک فایل yaml در آدرس snapshot_out ایجاد میکند که حاوی تمام مشخصات نمادها است
یک فایل yaml در آدرس overlay_out ایجاد میکند که حاوی برخی مشخصات نمادها است
منظور از برخی از مشخضات نمادها، آنهایی هستند که در متد get_symbol_specs از کلاس MT5Connector لیست شده اند

Run:
python -m f03_data.symbol_specs_snapshot    `
    --config f01_config/config.yaml         `
    --symbols XAUUSD EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD  `
    --overlay_out   f01_config/__symbol_specs__.yaml  `
    --snapshot_out  f03_data/__specs_snapshot__.yaml
"""

# =============================================================================
# Imports & Logger
# ============================================================================= OK
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import yaml
import logging

from f02_utils.config_operations import _deep_get
#from f02_utils.config_loader import load_config   # (delayed import after bootstrap)
#from f03_data.mt5_connector import MT5Connector

# -------------------- Logger for this module ----------------------- OK
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )

# ------------------------------------------------------------------- OK
def _resolve_symbols(cfg: dict, argsymbols: Any = None) -> list[str]:
    
    # sym = مستقیم مقدار لیست یا None را از مسیر کامل می‌گیریم
    sym = _deep_get(cfg, "download_defaults.symbols", default=None)
    
    # اگر نبود یا از نوع اشتباه بود، تلاش کنیم والد را بگیریم و از آن فیلد symbols بخوانیم
    if sym is None or not isinstance(sym, list):
        parent = _deep_get(cfg, "download_defaults", default={})
        sym = parent.get("symbols") if isinstance(parent, dict) else None
    symbols = argsymbols or (sym if isinstance(sym, list) else [])

    return symbols

# ------------------------------------------------------------------- OK
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Snapshot MT5 symbol specifications and write overlay YAML.")
    ap.add_argument("-c", "--config", default="f01_config/config.yaml")
    ap.add_argument("--symbols", nargs="*", default=None, help="Optional explicit list of symbols")
    ap.add_argument("--overlay_out", default=None, help="Path to write overlay YAML (default derived from config dir)")
    ap.add_argument("--snapshot_out", default=None, help="Path to write raw snapshot YAML (default derived from config paths)")
    return ap.parse_args()

# ------------------------------------------------------------------- OK
def write_specs_yaml(args: argparse.Namespace) -> bool:

    # 1) از خود YAML خام، مسیر اوورراید را پیدا کن و اگر نبود، فایل خالی بساز (bootstrap) 
    raw = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    if not Path(args.config).exists():
        raise SystemExit(f"Config file not found: {Path(args.config)}")
    config_dir = Path(args.config).resolve().parent
    proj_root = config_dir.parent
    
    ovr = (raw.get("overlays") or [])
    if isinstance(ovr, str):
        ovr = [ovr]
    for p in ovr:
        q = (proj_root / p) if not Path(p).is_absolute() else Path(p)
        if not q.exists():
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_text("symbol_specs: {}\n", encoding="utf-8")
    

    # 2) برای فایل symbol_specs=overlay، آنچه درکانفیگ نوشته شده است، مرجح است برآنچه در CLI معرفی شده.
    overlay_out = ""; overlay_config = None
    if (len(ovr) > 0)  and (ovr[0] != ""):    
        overlay_config = (proj_root / ovr[0]) if not Path(ovr[0]).is_absolute() else Path(ovr[0])
        logger.info("Address of 'overlay.yaml' set according to confog.yaml: %s", overlay_config)
        
    else:
        logger.info("Address of 'overlay.yaml' does not set according to config: %s", args.overlay_out)
    
    overlay_out = Path(   overlay_config or args.overlay_out or (config_dir / "symbol_specs.yaml")   ).resolve()


    # 3) حالا لودر را (که ENV را هم اعمال می‌کند) صدا بزن 
    from f02_utils.config_loader import load_config
    cfg = load_config(args.config, enable_env_override=True)


    # 4) یافتن لیست نمادها، ابتدا از CLI، و درصورت فقدان، از فایلهای کانفیگ 
    symbols = _resolve_symbols(cfg=cfg, argsymbols=args.symbols)


    # 5) مسیر ذخیره برای فایل specs_snapshot.yaml 
    # اول: CLI و اگر نبود: (data_dir / "specs_snapshot.yaml") 
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    data_dir = Path(paths.get("data_dir"))   or   Path(paths.get("raw_dir")).parent   or   Path(paths.get("processed_dir")).parent
    data_dir = data_dir if data_dir.is_absolute() else (proj_root / data_dir)
    snapshot_out=(Path(args.snapshot_out) if args.snapshot_out else (data_dir / "specs_snapshot.yaml")).resolve()


    # 6) اطمینان از وجود پوشه های مقصد برای ذخیره فایلهای نتیجه این برنامه 
    overlay_out.parent.mkdir(parents=True, exist_ok=True)
    snapshot_out.parent.mkdir(parents=True, exist_ok=True)


    # 7) حالا که overlay ساخته و config با ENV لود شده، MT5Connector را import کن 
    from f03_data.mt5_connector import MT5Connector
    conn = MT5Connector(config=cfg)


    # 8) برقرار نمودن اتصال به متاتریدر 
    if not conn.initialize():
        raise SystemExit("Could not initialize MT5Connector.")


    # 9) گرفتن اطلاعات نمادها و قطع اتصال 
    try:
        snapshot = conn.get_symbol_specs(symbols)
    finally:
        conn.shutdown()


    # write raw snapshot for archive/debug (ensure parent exists already) (10
    with open(snapshot_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, allow_unicode=True, sort_keys=False)


    '''11) بدست آوردن sp بطوریکه شامل تمام کلید-مقدارهای موجود در symbol_specs باشد، بغیر از "raw" 
    دیکشنری snapshot دارای 2 آیتم است. اولی meta و دومی symbol_specs
    در دیکشنری symbol_specs، به تعداد نمادها آیتم وجود دارد. که هر آیتم بصورت symbol:spec_dict است
    تمام آیتمهای هر spec_dict به غیر از "raw" وارد دیکشنری overlay میشوند
    '''
    raw_specs = snapshot.get("symbol_specs", {}) or {}
    sp = {}
    for sym, spec in raw_specs.items():
        if not isinstance(spec, dict):
            continue
        # copy all keys except "raw"
        sp[sym] = {k: v for k, v in spec.items() if k != "raw"}


    # 12) ساختن دیکشنری overlay 
    overlay = {
                "account_currency": (snapshot.get("meta", {}) or {}).get("account_currency"),
                "symbol_specs": sp
    }

    #13) ذخیره دیکشنری overlay در مسیر args.overlay_out و یا (config_dir / "symbol_specs.yaml") 
    with open(overlay_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(overlay, f, allow_unicode=True, sort_keys=False)

    logger.info("Wrote snapshot to %s and overlay to %s", snapshot_out, overlay_out)

# ------------------------------------------------------------------- OK
def main():
    # 1) استخراج مقادیر از خط فرمان 
    args = _parse_args()
    write_specs_yaml(args)

# ------------------------------------------------------------------- OK
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())







''' Reserve functions
# -------------------------------------------------------------------
def _resolve_symbols_old(cfg: dict) -> list[str]:
    # ترجیح: download_defaults.symbols ، در غیر این صورت data.symbols یا project.symbols یا symbols
    cand_keys = [
        ("download_defaults", "symbols"),
        ("data", "symbols"),
        ("project", "symbols"),
        ("symbols",),
    ]
    for keys in cand_keys:
        cur = cfg
        ok = True
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, list) and cur:
            return [str(s) for s in cur]
    raise SystemExit("No symbols found in config (looked at download_defaults.symbols, data.symbols, project.symbols, symbols).")
'''
