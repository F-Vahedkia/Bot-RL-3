# f02_utils/config_path_funs.py

# -------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict

# -------------------------------------------------------------------
def project_root() -> Path:
    """
    پوشه ای که دارای زیرپوشه های f01, f10 باشد را به عنوان پوشه ریشه معرفی میکند
    """
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "f01_config").exists() and (p / "f02_utils").exists():
            return p
    return here.parent  # fallback

# -------------------------------------------------------------------
def resolve_raw_dir(cfg: Dict[str, Any]) -> Path:
    """
    مسیر خروجی داده‌ی خام را از config استخراج می‌کند.
    """
    paths = cfg.get("paths", {}) or {}
    raw = paths.get("raw_dir") or (Path(paths.get("data_dir", "data")) / "raw")
    raw_path = project_root() / raw
    raw_path.mkdir(parents=True, exist_ok=True)
    return raw_path

# -------------------------------------------------------------------
def resolve_process_dir(cfg: Dict[str, Any]) -> Path:
    """
    ساخت آدرس کامل دسترسی به پوشه f03_data/processed و ساخت همان پوشه
    """
    paths = cfg.get("paths", {}) or {}
    proc = paths.get("processed_dir") or (Path(paths.get("data_dir", "data")) / "processed")
    p = project_root() / proc
    p.mkdir(parents=True, exist_ok=True)
    return p

# ------------------------------------------------------------------- OK=
def full_file_path(raw_dir: Path, symbol: str, timeframe: str, fmt: str) -> Path:
    """
    مسیر فایل خروجی را بر اساس (نماد/تایم‌فریم/فرمت) می‌سازد.
    """
    sym_dir = raw_dir / symbol   #.upper()
    sym_dir.mkdir(parents=True, exist_ok=True)
    ext = ".parquet" if fmt.lower() == "parquet" else ".csv"
    return sym_dir / f"{timeframe.upper()}{ext}"

# -------------------------------------------------------------------
