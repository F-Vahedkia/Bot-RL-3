# f02_utils/parser.py
# Last reviewed at 1405/04/31

r"""Parser برای Spec:  <name>(args)@TF  →  (name, args[], tf)
افزودنی‌های پارس Spec (Bot-RL-3)
- parse_spec: پارسِ نام/آرگومان‌ها/کلیدواژه‌ها + @TF

مثال‌های قابل پشتیبانی:
        golden_zone(0.382,0.618)@H1
        fib_cluster(tf=[H1,H4,D1], tol_pct=0.1, prefer_ratio=0.618)
        fib_ext_targets(atr_mult=1.5)@H1
        ma_slope(window=20, method='ema')@M5
        rsi_zone(period=14)@H1
"""

# f02_utils/parser.py
# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import ast
# import inspect
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =============================================================================
# پارسر توسعه‌یافتهٔ Spec
# =============================================================================
_SPEC_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*(?:\((?P<args>.*)\))?\s*(?:@(?P<tf>[A-Za-z0-9]+))?\s*$",
    re.DOTALL
)

# =============================================================================
# مدل دادهٔ خروجی پارس
# =============================================================================
@dataclass
class ParsedSpec:
    name: str                 # نام اندیکاتور (registry key)
    args: List[Any]           # آرگومان‌های موقعیتی
    kwargs: Dict[str, Any]    # آرگومان‌های کلیدواژه‌ای
    timeframe: Optional[str]  # @TF اگر وجود داشت
    raw: str                  # متن خام Spec (برای لاگ/دیباگ)
    canonical: str             # همان متن خام اِسپک است که در ان تمام سینگل کونیشنها به دابل کوتیشن تبدیل شده اند

# =============================================================================
# کمکی‌ها: شکستن آرگومان‌ها در سطح-بالا
# =============================================================================
def _split_top_level_commas(s: str) -> List[str]:
    """تقسیم بر اساس ویرگول‌ها در سطح-بالا (داخل []/() دست‌نخورده بماند)."""
    out, buf, depth = [], [], 0
    for ch in s:
        if ch in "([{" :
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [t for t in out if t != ""]


def _parse_value(token: str) -> Any:
    """تبدیل توکن به مقدار پایتونی؛ ایمن و سازگار با TFهای بدون کوتیشن (M1,H1,...)
    ساده‌سازی: True/False/None کوچک/بزرگ
    _parse_value("true") → True
    _parse_value("14") → 14
    _parse_value("3.5") → 3.5
    _parse_value("'ema'") → "ema"
    _parse_value("[M1,H1]") → ["M1","H1"]
    _parse_value("[10,'X']") → [10,"X"]
    _parse_value("M15") → "M15"
    """
    low = token.strip()
    if low.lower() in ("true", "false", "none", "null"):
        return {"true": True, "false": False, "none": None, "null": None}[low.lower()]
    # تلاش: literal_eval
    # هر چیزی که لیترال استاندارد باشد، با literal_eval سریع و امن تبدیل می‌شود.
    try:
        return ast.literal_eval(low)
    except Exception:
        pass
    """
    # بلوک پایین:
    # تبدیل TF های بدون کوتیشن داخل لیست‌ها
    # مثالها:   "[M1,H1]" → ["M1","H1"]
    #            "[10, 20]" → [10, 20]
    #            "[M5, 10, 'X']" → ["M5", 10, "X"]
    """
    if low.startswith("[") and low.endswith("]"):
        inner = low[1:-1].strip()
        parts = _split_top_level_commas(inner)
        vals = []
        for p in parts:
            p = p.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", p):
                vals.append(p)  # TF token as string
            else:
                try:
                    vals.append(ast.literal_eval(p))
                except Exception:
                    vals.append(str(p))
        return vals
    
    # اعداد ساده int
    if re.fullmatch(r"[-+]?\d+", low):
        return int(low)
    # اعداد اعشاری
    if re.fullmatch(r"[-+]?\d*\.\d+", low):
        return float(low)
    # پیش‌فرض: رشته
    return str(low)


def _parse_args_kwargs(argstr: Optional[str]) -> Tuple[List[Any], Dict[str, Any]]:
    """پارس آرگومان‌های موقعیتی و کلیدواژه‌ای
    مثالها:
    "14, 0.5, 'ema'"                         →  args = [14, 0.5, "ema"], kwargs = {}
    "period=14, method='ema', tf=H1"         →  args = [], kwargs = {"period": 14, "method": "ema", "tf": "H1"}
    "win=(10,20), levels=[1,2,3], name='x'"  →  kwargs = {"win": (10, 20), "levels": [1,2,3], "name": "x"}
    "x=[M1,H4], 5, mode='fast'"              →  args = [5], kwargs = {"x": ["M1","H4"], "mode": "fast"}
    """
    # ساختن ظرفها
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    # اگر ورودی خالی بود، زود برگرد
    if not argstr or argstr.strip() == "":
        return args, kwargs
    # متن را روی «ویرگول‌های سطح-بالا»؛ ویرگول‌های داخل []/()/{} شکسته نمی‌شوند.
    tokens = _split_top_level_commas(argstr)
    # هر توکن را یا «کلید=مقدار» می‌گیرد، یا «موقعیتی»
    for tok in tokens:
        # اگر در توکن علامت مساوی وجود داشت، سمت چپ را کلید و سمت راست را مقدار در نظر میگیرد
        if "=" in tok:
            k, v = tok.split("=", 1)
            key = k.strip()
            if key in kwargs:
                raise ValueError(f"Duplicate parameter '{key}'")
            kwargs[key] = _parse_value(v.strip())
            
        # اگر در توکن علامت مساوی وجو نداشت، آن را مقدار در نظر میگیرد
        else:
            args.append(_parse_value(tok.strip()))
    return args, kwargs


# =============================================================================
# Registry Parameter Mapping
# =============================================================================
def _align_args_with_signature(
    ind_name: str,
    args_in: List[Any],
    kwargs_in: Dict[str, Any],
    mode: str = "train",
) -> tuple[list[Any], dict[str, Any]]:
    """
    تبدیل آرگومان‌های positional به kwargs بر اساس ParameterSpecهای رجیستری.
    مسئولیت این تابع فقط Mapping است و هیچ Validation انجام نمی‌دهد.
    مثال:
        sma(20)  →  {"period":20}
    ------------------------------
        sma(close,20)  →  {"column":"close","period":20}
    ------------------------------
        stochastic(14,3,3)  →
        {
            "k_period":14,
            "d_period":3,
            "smooth_k":3,
        }
    """

    from f04_features.feature_C_registry_1 import get_indicator
    spec = get_indicator(ind_name, mode=mode)

    if spec is None:
        return list(args_in), dict(kwargs_in)

    if not spec.parameters:
        return list(args_in), dict(kwargs_in)

    new_kwargs = dict(kwargs_in)
    positional_parameters = []
    for p in spec.parameters:
        if not p.visible:
            continue
        
        # if p.name in new_kwargs:     # for debug. کامنت شد
        #     continue                 # for debug. کامنت شد

        # اگر این پارامتر با نام اصلی یا یکی از aliasها قبلاً داده شده باشد
        if (                                                     # for debug. اضافه شد
            p.name in new_kwargs                                 # for debug. اضافه شد
            or any(alias in new_kwargs for alias in p.aliases)   # for debug. اضافه شد
        ):                                                       # for debug. اضافه شد
            continue                                             # for debug. اضافه شد

        positional_parameters.append(p)

    new_args = []
    for value in args_in:
        if positional_parameters:
            p = positional_parameters.pop(0)
            new_kwargs[p.name] = value
        else:
            if spec.accepts_var_args:
                new_args.append(value)
            else:
                new_args.append(value)

    return new_args, new_kwargs


# =============================================================================
# Signature Validation Helper
# =============================================================================
def _validate_signature(
    ind_name: str,
    args_in: list[Any],
    kwargs_in: dict[str, Any],
    mode: str = "train",
) -> tuple[list[Any], dict[str, Any]]:
    """
    Validate a ParsedSpec against IndicatorSpec.parameters.

    Registry is the Single Source of Truth.

    Responsibilities
    ----------------
    ✓ remaining positional args
    ✓ unknown kwargs
    ✓ aliases
    ✓ duplicate parameters
    ✓ required parameters
    ✓ defaults
    ✓ type checking
    ✓ choices
    ✓ minimum / maximum
    """
    from f04_features.feature_C_registry_1 import get_indicator, _NO_DEFAULT, ParameterSpec
    
    spec = get_indicator(ind_name, mode=mode)
    if spec is None:
        raise ValueError(f"Unknown indicator '{ind_name}'")
    params = spec.parameters or []

    # ------------------------------------------------------------
    # اگر metadata تعریف نشده باشد
    # ------------------------------------------------------------
    if not params:
        return args_in, kwargs_in

    # ------------------------------------------------------------
    # positional
    # ------------------------------------------------------------
    if args_in and not spec.accepts_var_args:
        raise ValueError(
            f"Too many positional arguments for '{ind_name}'."
        )

    # ------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------
    # print("PARAMETERS:")                  # for debug
    # for p in params:                      # for debug
    #     print(id(p), p.name, p.aliases)   # for debug

    lookup: dict[str, ParameterSpec] = {}
    for p in params:

        if p.name in lookup:
            raise ValueError(
                f"Duplicate parameter '{p.name}' in registry."
            )
        lookup[p.name] = p
        for alias in p.aliases:

            # if alias in lookup:
            #     raise ValueError(f"Duplicate alias '{alias}' in registry.")
            
            lookup[alias] = p

    # ------------------------------------------------------------
    # normalize aliases
    # ------------------------------------------------------------
    normalized: dict[str, Any] = {}

    for key, value in kwargs_in.items():

        p = lookup.get(key)
        if p is None:

            if spec.allow_unknown_kwargs:
                normalized[key] = value
                continue

            raise ValueError(
                f"Unknown parameter '{key}' for '{ind_name}'."
            )

        canonical_ = p.name
        if canonical_ in normalized:
            raise ValueError(
                f"Parameter '{canonical_}' specified more than once."
            )

        normalized[canonical_] = value

    # ------------------------------------------------------------
    # defaults
    # ------------------------------------------------------------
    for p in params:
        if p.name in normalized:
            continue

        if p.default is not _NO_DEFAULT:
            normalized[p.name] = p.default

    # ------------------------------------------------------------
    # required
    # ------------------------------------------------------------
    missing = [
        p.name
        for p in params
        if p.required and p.name not in normalized
    ]

    if missing:
        raise ValueError(
            f"Missing required parameter(s): {', '.join(missing)}"
        )

    # ------------------------------------------------------------
    # validate values
    # ------------------------------------------------------------
    for p in params:

        if p.name not in normalized:
            continue
        value = normalized[p.name]

        # ---------- type ----------
        if (
            p.dtype is not object
            and value is not None
            and not isinstance(value, p.dtype)
        ):
            raise TypeError(
                f"Parameter '{p.name}' must be "
                f"{p.dtype}, got {type(value).__name__}."
            )
        # ---------- choices ----------
        if (
            p.choices is not None
            and value not in p.choices
        ):
            raise ValueError(
                f"Parameter '{p.name}' must be one of "
                f"{tuple(p.choices)}."
            )
        # ---------- minimum ----------
        if (
            p.minimum is not None
            and value < p.minimum
        ):
            raise ValueError(
                f"Parameter '{p.name}' must be >= {p.minimum}."
            )
        # ---------- maximum ----------
        if (
            p.maximum is not None
            and value > p.maximum
        ):
            raise ValueError(
                f"Parameter '{p.name}' must be <= {p.maximum}."
            )
    return [], normalized


# =============================================================================
# Main Function
# =============================================================================
def parse_spec(spec: str, mode: str = "train") -> ParsedSpec:
    """
    Parse + Compile + Validate feature specification.

    Pipeline:
        parse_spec()  ->  align_signature()  ->  validate_signature()  ->  return ParsedSpec
    """

    m = _SPEC_RE.match(spec)
    if not m:
        raise ValueError(f"Invalid spec: {spec}")

    name, argstr, tf = m.group(1), m.group(2), m.group(3)

    # مرحله 1: Parse خام
    args, kwargs = _parse_args_kwargs(argstr)

    # مرحله 2: Mapping positional → kwargs
    args, kwargs = _align_args_with_signature(
        ind_name=name,
        args_in=args,
        kwargs_in=kwargs,
        mode=mode,
    )

    # print("AFTER ALIGN :", kwargs)   # for debug
    # مرحله 3: Validation + defaults + aliases
    args, kwargs = _validate_signature(
        ind_name=name,
        args_in=args,
        kwargs_in=kwargs,
        mode=mode,
    )
    # print("AFTER VALID :", kwargs)   # for debug

    # نرمال‌سازی TF
    tf_norm = tf.upper() if tf else None

    ps = ParsedSpec(
        name=name,
        args=args,
        kwargs=kwargs,
        timeframe=tf_norm,
        raw=spec,
        canonical=spec.replace("'", '"'),
    )
    logger.debug("parse_spec: %s -> %s \n", spec, ps)
    return ps


# =====================================================================================
# تست پوشش کد (برای توسعه‌دهندگان) ?????????????????????????????????????????????
# =====================================================================================
""" Func Names                           Used in Functions: ...
                                1   2   3   4   5   6
1  ParsedSpec                  --  --  --  --  --  ok
2  _split_top_level_commas     --  --  ok  ok  --  --
3  _parse_value                --  --  --  ok  --  --
4  _parse_args_kwargs          --  --  --  --  --  ok
5  _align_args_with_signature  --  --  --  --  --  ok
6  parse_spec                  --  --  --  --  --  -- USED in feature_engine.py
"""
