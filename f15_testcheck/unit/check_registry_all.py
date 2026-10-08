# -*- coding: utf-8 -*-
"""
فایل کمکی برای استخراج و گزارش‌گیری از رجیستری اندیکاتورها.

وظایف:
- بارگذاری REGISTRY از f04_features.feature_registry
- استخراج تمام کلیدها و مشخصات تابع/کلاس متناظر
- تولید گزارش CSV در مسیر f15_testcheck/_reports/registered_features.csv

یادداشت‌ها:
- پیام‌های runtime به زبان انگلیسی چاپ می‌شوند (طبق قوانین پروژه).
- توضیحات (این بلوک) و کامنت‌ها فارسی هستند.

اجرا:
python f15_testcheck/unit/check_registry_all.py
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from datetime import datetime

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# ⚠️ واردکردن فقط REGISTRY برای جلوگیری از اجرای اضافی
from f04_features.OLD.feature_registry import REGISTRY  # noqa: E402

def _safe_str(x: object, max_len: int = 180) -> str:
    """برگرداندن رشتهٔ امن با برش طول در صورت نیاز (برای CSV)."""
    try:
        s = str(x) if x is not None else ""
    except Exception:
        s = ""
    s = s.replace("\r", " ").replace("\n", " ").strip()
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def _extract_row(key: str, obj) -> dict:
    """
    استخراج اطلاعات اصلی هر ورودی رجیستری برای ثبت در CSV.
    ستون‌ها:
      - key: نام کلید در رجیستری
      - type: نوع شیء (نام کلاس/تابع پایتونی)
      - module: نام ماژول تعریف‌کننده
      - qualname: نام صلاحیت‌دار (در صورت موجود بودن)
      - has_doc: آیا داک‌استرینگ دارد؟
      - doc: یک خط اول داک‌استرینگ (برش‌خورده)
    """
    typ = type(obj).__name__
    module = getattr(obj, "__module__", "") or ""
    qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", "")) or ""
    doc = getattr(obj, "__doc__", None)
    doc_line = ""
    has_doc = False
    if isinstance(doc, str) and doc.strip():
        has_doc = True
        # فقط خط اول توضیح برای خلاصه
        doc_line = _safe_str(doc.strip().splitlines()[0], max_len=180)

    return {
        "key": key,
        "type": _safe_str(typ, 64),
        "module": _safe_str(module, 120),
        "qualname": _safe_str(qualname, 160),
        "has_doc": "yes" if has_doc else "no",
        "doc": doc_line,
    }


def main() -> int:
    """
    نقطهٔ ورود اسکریپت.
    - ساخت مسیر گزارش
    - پیمایش REGISTRY و ساخت CSV
    - چاپ خلاصهٔ اجرای موفق
    """
    # مسیر خروجی گزارش
    out_dir = Path("f15_testcheck") / "_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "registered_features_all.csv"

    # گردآوری و مرتب‌سازی کلیدها برای پایداری خروجی
    keys = sorted(REGISTRY.keys(), key=lambda s: str(s).lower())

    # سرستون‌ها
    fieldnames = ["key", "type", "module", "qualname", "has_doc", "doc"]

    # نوشتن CSV با newline صحیح در ویندوز
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in keys:
            try:
                row = _extract_row(k, REGISTRY[k])
            except Exception as e:
                # در صورت هرگونه خطا، ردیفِ کمینه با پیام خطا را ثبت می‌کنیم
                row = {
                    "key": k,
                    "type": "ERROR",
                    "module": "",
                    "qualname": "",
                    "has_doc": "no",
                    "doc": _safe_str(f"introspection error: {e}", 160),
                }
            writer.writerow(row)

    # پیام موفقیت به زبان انگلیسی (طبق قوانین)
    print(
        f"[OK] Exported {len(keys)} registry entries to: {out_csv} "
        f"(generated at {datetime.now().isoformat(timespec='seconds')})"
    )
    # بازگشت کد موفقیت
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
