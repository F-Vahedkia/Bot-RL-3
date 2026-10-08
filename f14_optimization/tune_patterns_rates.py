# -*- coding: utf-8 -*-
# f14_optimization/tune_patterns_rates.py
# Status in (Bot-RL-3): Completed

# Optimize candlestick pattern parameters based on occurrence rates

r"""
حلقهٔ تیون کم‌بعد برای الگوهای کندلی:
- اسموکِ موجود را برای TF اجرا می‌کند (به‌صورت subprocess) و نرخ نسبی هر pat_* را از خروجی می‌خوانَد.
- اگر نرخ هر الگو بیرون از بازهٔ هدف باشد، مقدار پارامتر همان الگو را «اندکی» تنظیم می‌کند.
- سپس YAML را به‌روزرسانی و تکرار می‌کند تا همه داخل بازه بیفتند یا به سقف تکرار برسیم.

یادداشت:
- بدون افزودن کلید جدید به کانفیگ؛ فقط همان رشتهٔ spec و پارامترهای موجود همان خط تغییر می‌کنند.
- الگوهای پشتیبانی‌شده برای تیون خودکار (طبق بحث قبلی):
  tweezer(tol_k), inside_outside(min_range_k_atr),
  hammer_star(wick_ratio), pin(ratio),
  morning_evening(small_body_atr), belt(wick_frac)
- برای الگوهای دیگر یا الگوهای با پارامترهای غیرعددی، تیون خودکار انجام نمی‌شود.
- اگر چندین spec از یک الگو (مثلاً tweezer) در کانفیگ باشد، همهٔ آن‌ها با همان delta تغییر می‌کنند.
- در هر تکرار، فقط یک پارامتر از هر الگو تغییر می‌کند (اگر چندین ستون برای یک الگو وجود داشت).
- اگر پارامتر در spec وجود نداشت (مثلاً tol_k در pat_tweezer نبود)، تغییری نمی‌دهد.
- مقادیر پارامترها در بازه [0.0, 10.0] کِلَمپ می‌شوند (برای جلوگیری از مقادیر منفی یا خیلی بزرگ).
- می‌توانید با تغییر پارامترهای خط فرمان، بازه هدف و گام تغییر را تنظیم کنید.

روش اجرا توسط CLI:
python f14_optimization/tune_patterns_rates.py   `
    -c .\f01_config\config.yaml                  `
    --symbol XAUUSD                              `
    --base-tf D1                                 `
    --data-path f03_data\processed\XAUUSD\D1.parquet `
    --min-rate 0.005                             `
    --max-rate 0.030                             `
    --step 0.05                                  `
"""

import argparse, subprocess, sys, re, json, time, shutil
from pathlib import Path
import yaml

# -----------------------------
# هِلپر: اجرای اسموک و گرفتن نرخ‌ها از خروجی
# -----------------------------
def run_smoke_and_parse_rates(py: str, config: Path, symbol: str, tf: str, data_path: Path) -> dict:
    """
    اسموک را اجرا می‌کند و جدول «Top-15 flags by rate» را از stdout می‌خواند.
    خروجی: dict {flag_name: rate_float}, بعلاوه bars (اگر قابل‌استخراج باشد).
    """
    cmd = [
        py, "-m", "f15_scripts.check_patterns_smoke_real",
        "-c", str(config),
        "--symbol", symbol,
        "--base-tf", tf,
        "--data-path", str(data_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        print(out)
        raise RuntimeError("Smoke script failed")

    # استخراج bars (اختیاری)
    bars = None
    m_bars = re.search(r"\[INFO\]\s*Bars\s*\(\s*{}\s*\)\s*:\s*(\d+)".format(re.escape(tf)), out, flags=re.I)
    if m_bars:
        bars = int(m_bars.group(1))

    # استخراج جدول نرخ‌ها
    # به‌دنبال بلوک:
    # [STATS] Top-15 flags by rate (% of bars):
    # pat_xxx    1.234567%
    rates = {}
    block = re.split(r"\[STATS\]\s*Top-15 flags by rate.*?\n", out, flags=re.I)
    if len(block) >= 2:
        table = block[1]
        for line in table.strip().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            # خط‌هایی مثل: pat_name    12.345678%
            m = re.match(r"([A-Za-z0-9_\.]+)\s+([0-9\.]+)%", line.strip())
            if m:
                name = m.group(1)
                rate = float(m.group(2)) / 100.0
                rates[name] = rate
    return {"rates": rates, "bars": bars, "raw": out}

# -----------------------------
# هِلپر: جست‌وجوی رشتهٔ spec در YAML و اعمال جایگزینی
# -----------------------------
SPEC_RE = re.compile(r"^(pat_[a-z0-9_]+)(?:\((.*)\))?@([A-Z0-9]+)$", re.I)

def walk_and_collect_specs(node, parent=None, key=None, acc=None):
    """گردش روی YAML برای یافتن لیست‌های رشته‌ایِ spec؛ آدرس و ایندکس را ذخیره می‌کند."""
    if acc is None:
        acc = []
    if isinstance(node, list):
        for idx, v in enumerate(node):
            if isinstance(v, str) and SPEC_RE.match(v.strip()):
                acc.append( (parent, key, node, idx, v.strip()) )
            else:
                walk_and_collect_specs(v, parent=node, key=idx, acc=acc)
    elif isinstance(node, dict):
        for k, v in node.items():
            walk_and_collect_specs(v, parent=node, key=k, acc=acc)
    return acc

def replace_param_in_spec(spec_line: str, updates: dict) -> str:
    """
    یک spec مانند: pat_tweezer(tol_frac=null,tol_k=0.25,atr_win=14)@H1
    را می‌گیرد و فقط پارامترهایی که در updates هستند را با regex جایگزین می‌کند.
    اگر پارامتر موجود نبود، تغییری نمی‌دهد (کلید جدید اضافه نمی‌کنیم).
    """
    m = SPEC_RE.match(spec_line)
    if not m:
        return spec_line
    name, args_str, tf = m.group(1), m.group(2), m.group(3)
    if not args_str:
        return spec_line
    s = args_str

    def _repl_param(s: str, pname: str, pval_str: str) -> str:
        # جایگزینی مقدارِ موجودِ پارامتر بدون افزودن کلید جدید
        # الگو: pname = چیز
        pat = re.compile(rf"(\b{re.escape(pname)}\s*=\s*)([^,)\s]+)", flags=re.I)
        if pat.search(s):
            # Use a function replacement to avoid backref issues (\1 + digit -> \10)
            s = pat.sub(lambda m: m.group(1) + pval_str, s)
        return s

    for k, v in updates.items():
        # عدد/رشته را به شکل مناسب در spec بنویسیم
        if isinstance(v, str):
            val = f"\"{v}\""
        elif v is None:
            val = "null"
        else:
            val = f"{v}"
        s = _repl_param(s, k, val)

    return f"{name}({s})@{tf}"

# -----------------------------
# سیاست تیون برای هر الگو
# -----------------------------
def propose_param_shifts(flag_name: str, rate: float, lo: float, hi: float, step: float):
    """
    بر اساس نام ستون pat_* تعیین می‌کند کدام پارامتر باید در همان spec کمی تغییر کند.
    خروجی: dict {param_name: new_value_delta_sign} (delta با علامت؛ در YAML به مقدار جدید تبدیل می‌شود).
    """
    # نگاشت‌ها (خلاصه و کم‌بعد)
    if flag_name.startswith("pat_3soldiers") or flag_name.startswith("pat_3crows"):
        pname = "min_body_atr"
        sign = +1 if rate > hi else (-1 if rate < lo else 0)
        return {pname: sign * step}

    # برای piercing/darkcloud: اگر نام پارامتر را ندانیم، بعداً اولین پارامتر عددی spec را تغییر می‌دهیم
    if flag_name.startswith("pat_piercing") or flag_name.startswith("pat_darkcloud"):
        return {"__first_numeric__": (+step if rate > hi else (-step if rate < lo else 0))}
    if flag_name.startswith("pat_tweezer_"):
        pname = "tol_k"
        sign = -1 if rate > hi else (+1 if rate < lo else 0)
        return {pname: sign * step}
    if flag_name in ("pat_inside","pat_outside"):
        pname = "min_range_k_atr"
        sign = +1 if rate > hi else (-1 if rate < lo else 0)
        return {pname: sign * step}
    if flag_name.startswith("pat_hammer_bull") or flag_name.startswith("pat_shoot_bear") or flag_name.startswith("pat_hammer_star"):
        pname = "wick_ratio"
        sign = +1 if rate > hi else (-1 if rate < lo else 0)
        return {pname: sign * step}
    if flag_name.startswith("pat_pin_"):
        pname = "ratio"
        sign = +1 if rate > hi else (-1 if rate < lo else 0)
        return {pname: sign * step}
    if flag_name.startswith("pat_morning") or flag_name.startswith("pat_evening"):
        pname = "small_body_atr"
        # برای کاهش رخداد باید آستانه را بزرگ‌تر کنیم (برخلاف hammer/pin)
        sign = -1 if rate > hi else (+1 if rate < lo else 0)
        return {pname: sign * step}
    if flag_name.startswith("pat_belt_"):
        pname = "wick_frac"
        # نرخ زیاد -> سخت‌تر (کم‌کردن frac)
        sign = -1 if rate > hi else (+1 if rate < lo else 0)
        return {pname: sign * step}
    
    # marubozu: پارامتر تیون‌پذیر موجود در spec = wick_frac
    if flag_name.startswith("pat_marubozu_"):
        pname = "wick_frac"
        sign = -1 if rate > hi else (+1 if rate < lo else 0)
        return {pname: sign * step}
    # three soldiers / three crows: پارامتر = min_body_atr
    if flag_name.startswith("pat_3soldiers") or flag_name.startswith("pat_3crows"):
        pname = "min_body_atr"
        sign = +1 if rate > hi else (-1 if rate < lo else 0)
        return {pname: sign * step}
    # piercing / darkcloud: نام پارامتر در spec ممکن است متفاوت باشد؛
    # یکی از این‌ها را اگر موجود بود تیون کن (کلید جدید اضافه نکن):
    if flag_name.startswith("pat_piercing") or flag_name.startswith("pat_darkcloud"):
        candidate_params = ["penetration_k", "overlap_k", "cross_k"]
        # با خودِ مقدار فعلی در spec تصمیم می‌گیریم؛ sign: نرخ زیاد -> پارامتر بزرگ‌تر (سخت‌تر)
        sign = +1 if rate > hi else (-1 if rate < lo else 0)
        # مقدار delta را برای اولین پارامترِ موجود برمی‌گردانیم؛
        # resolve در مرحلهٔ replace انجام می‌شود (فقط اگر همان کلید در spec باشد).
        return {p: sign * step for p in candidate_params}    
    
    # در غیر این صورت: بدون تغییر خودکار
    return {}

# -----------------------------
# بروز کردن مقدار پارامتر در رشتهٔ spec (با توجه به مقدار فعلی)
# -----------------------------
def extract_current_param_value(spec_line: str, pname: str):
    m = SPEC_RE.match(spec_line)
    if not m or not m.group(2):
        return None
    args = m.group(2)
    pat = re.compile(rf"\b{re.escape(pname)}\s*=\s*([^,)\s]+)", flags=re.I)
    mm = pat.search(args)
    if not mm:
        return None
    raw = mm.group(1)
    try:
        if raw.lower() in ("null", "none"):
            return None
        if raw.startswith('"') or raw.startswith("'"):
            return raw.strip('"\'')
        return float(raw)
    except Exception:
        return None

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

# -----------------------------
# اصلی
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--base-tf", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--min-rate", type=float, default=0.005)  # 0.5%
    ap.add_argument("--max-rate", type=float, default=0.030)  # 3.0%
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max-iter", type=int, default=50)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    data_path = Path(args.data_path)
    py = sys.executable

    # بکاپ ساده
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = cfg_path.with_suffix(f".{ts}.bak.yaml")
    try:
        shutil.copy2(cfg_path, bak)
        print(f"[INFO] Backup created: {bak.name}")
    except Exception as e:
        print(f"[WARN] Backup failed: {e}")

    for it in range(1, args.max_iter + 1):
        print(f"[RUN] Iteration {it} ...")
        rpt = run_smoke_and_parse_rates(py, cfg_path, args.symbol, args.base_tf, data_path)
        rates = rpt["rates"]
        if not rates:
            print("[ERROR] No rates parsed. Abort.")
            print(rpt["raw"])
            sys.exit(2)

        # بارگذاری YAML و یافتن رشته‌های spec برای TF موردنظر
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        specs = walk_and_collect_specs(cfg)

        # فقط spec های همین TF
        specs_tf = [(parent, key, lst, idx, s) for (parent, key, lst, idx, s) in specs if s.endswith(f"@{args.base_tf.upper()}")]

        changed = 0
        for flag, rate in rates.items():
            # فقط اگر بیرون از بازه باشد
            if rate >= args.min_rate and rate <= args.max_rate:
                continue
            delta_map = propose_param_shifts(flag, rate, args.min_rate, args.max_rate, args.step)
            if not delta_map:
                continue

            # نام الگوی متناظر در spec (پیشوند pat_ تا قبل از اولین '_')
            base = flag.split("_")
            if base[0] != "pat":
                continue
            keyname = base[1].lower()  # مثال: tweezer, inside, belt ...
            # نگاشت تقریبی flag->spec name
            # ویژه inside/outside: هردو به spec inside_outside برمی‌گردند
            # map flag column name -> spec name in config
            mapping = {
                "inside":     "inside_outside",
                "outside":    "inside_outside",
                "morning":    "morning_evening",
                "evening":    "morning_evening",
                "hammer":     "hammer_star",
                "shoot":      "hammer_star",
                "piercing":   "piercing_dark",
                "darkcloud":  "piercing_dark",
                "3soldiers":  "3soldiers_crows",
                "3crows":     "3soldiers_crows",
            }
            spec_name = mapping.get(keyname, keyname)

            # در لیست specs همین TF، خطوط مربوط به همین الگو را بیاب
            for (parent, key, lst, idx, sline) in specs_tf:
                m = SPEC_RE.match(sline)
                if not m:
                    continue
                s_name = m.group(1).replace("pat_", "", 1).lower()
                if s_name != spec_name.lower():
                    continue

                # به‌روز کردن مقدار فعلی با delta
                updates = {}
                for pname, delta in delta_map.items():
                    cur = extract_current_param_value(sline, pname)

                    if pname == "__first_numeric__":
                        # اولین پارامتر عددی موجود در همین spec را پیدا و تیون کن
                        m = SPEC_RE.match(sline)
                        if m and m.group(2):
                            args_str = m.group(2)
                            pairs = [p.strip() for p in args_str.split(",") if p.strip()]
                            for p in pairs:
                                if "=" in p:
                                    k, v = p.split("=", 1)
                                    try:
                                        fv = float(v.strip().strip('"').strip("'"))
                                        updates[k.strip()] = clamp(fv + delta, 0.0, 10.0)
                                        break
                                    except Exception:
                                        continue
                        continue
                    if cur is None:
                        continue  # کلید جدید اضافه نکن
                    
                    new_val = clamp(cur + delta, 0.0, 10.0)
                    #updates[pname] = clamp(cur + delta, 0.0, 10.0)
                    # اگر پارامتر نامزد چندتایی بود (piercing/darkcloud)، فقط اولین کلیدِ موجود در spec اعمال شود
                    if pname in ("penetration_k", "overlap_k", "cross_k"):
                        # اینجا چک می‌کنیم آیا همان pname در spec وجود دارد
                        if extract_current_param_value(sline, pname) is not None:
                            updates = {pname: new_val}
                            break
                    else:
                        updates[pname] = new_val



                if not updates:
                    continue

                new_line = replace_param_in_spec(sline, updates)
                if new_line != sline:
                    lst[idx] = new_line
                    changed += 1
                    print(f"[TUNE] {flag}: {rate:.4%} -> {updates}")

        if changed == 0:
            print("[OK] No changes needed (all within target or unsupported). Stop.")
            break

        # ذخیرهٔ YAML
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"[OK] Config updated. Re-running smoke ...")

        # پس از ذخیره، به تکرار بعدی می‌رویم

    print("[DONE] Tuning loop finished.")

if __name__ == "__main__":
    main()
