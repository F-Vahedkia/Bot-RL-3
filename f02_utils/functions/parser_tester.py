# فایل اصلی تستر پارسر
# Run: pytest -v -s    f02_utils/functions/parser_tester.py      برای اجرای تمام تست‌های Parser که در فایل هستند
#      pytest -q       f02_utils/functions/parser_tester.py         خلاصه نتیجه تستها را میدهد
#      pytest -x -v -s f02_utils/functions/parser_tester.py   برای آنکه در اولین خطا متوقف بشود
#      pytest -v -s    f02_utils/functions/parser_tester.py -k   test_parse_sma   برای اجرای فقط یک تست خاص (مثلاً تست test_parse_sma.)
""" Date revewed:
    1405/05/19-09:14 ==> run resulst is: OK
"""
# ==========================================================
# f02_utils/parser_tester.py
# Part 1
# ==========================================================

from __future__ import annotations
import pytest

from f02_utils.functions.parser import parse_spec

# ==========================================================
# Basic Parsing
# ==========================================================
# ----------------------------------------------- 1 OK
def test_parse_sma_keyword():
    ps = parse_spec(
        "sma(column='close',period=20)@M1"
    )
    assert ps.name == "sma"
    assert ps.timeframe == "M1"
    assert ps.kwargs["column"] == "close"
    assert ps.kwargs["period"] == 20
    assert ps.args == []

# ----------------------------------------------- 2 OK
def test_parse_sma_positional():
    ps = parse_spec(
        "sma('close',20)@H1"
    )
    assert ps.name == "sma"
    assert ps.kwargs["column"] == "close"
    assert ps.kwargs["period"] == 20

# ----------------------------------------------- 3 OK
def test_parse_rsi():
    ps = parse_spec(
        "rsi(period=14)@H4"
    )
    assert ps.name == "rsi"
    assert ps.timeframe == "H4"
    assert ps.kwargs["period"] == 14

# ----------------------------------------------- 4 OK
def test_parse_true_range():
    ps = parse_spec(
        "true_range@D1"
    )
    assert ps.name == "true_range"
    assert ps.timeframe == "D1"
    assert ps.kwargs == {
        "high_column": "high",
        "low_column": "low",
        "close_column": "close",
    }
    assert ps.args == []

# ==========================================================
# Positional Mapping
# ==========================================================
# ----------------------------------------------- 5 OK
def test_bollinger_positional():
    ps = parse_spec(
        "bollinger_bands(close,20,2)@H1"
    )
    assert ps.kwargs["column"] == "close"
    assert ps.kwargs["period"] == 20
    assert ps.kwargs["multiplier"] == 2

# ----------------------------------------------- 6 OK
def test_macd_keywords():
    ps = parse_spec(
        "macd(fast=12,slow=26,signal=9)"
    )
    assert ps.kwargs["fast"] == 12
    assert ps.kwargs["slow"] == 26
    assert ps.kwargs["signal"] == 9

# ----------------------------------------------- 7 OK
def test_stochastic_keywords():
    ps = parse_spec(
        "stochastic(k_period=14,d_period=3,smooth_k=3)"
    )
    assert ps.kwargs["k_period"] == 14
    assert ps.kwargs["d_period"] == 3
    assert ps.kwargs["smooth_k"] == 3

# ----------------------------------------------- 8 OK
def test_supertrend():
    ps = parse_spec(
        "supertrend(n=10,m=3)"
    )
    assert ps.kwargs["period"] == 10
    assert ps.kwargs["multiplier"] == 3

# =============================================================================
# Part 2
# Validation Tests
# =============================================================================

# ------------------------------------------------------------
# Unknown indicator
# ------------------------------------------------------------ 9 OK
def test_unknown_indicator():
    with pytest.raises(ValueError):
        parse_spec("abcdef(10)@H1")

# ------------------------------------------------------------
# Unknown parameter
# ------------------------------------------------------------ 10 OK
def test_unknown_parameter():
    with pytest.raises(ValueError):
        parse_spec("sma(period=20,abc=1)@H1")

# ------------------------------------------------------------
# Missing required parameter
# ------------------------------------------------------------ 11 OK, 12 OK
def test_missing_required_parameter():
    with pytest.raises(ValueError):
        parse_spec("sma(period=20)@H1")


def test_missing_period():
    with pytest.raises(ValueError):
        parse_spec("sma(column='close')@H1")

# ------------------------------------------------------------
# Duplicate parameter
# ------------------------------------------------------------ 13 OK
def test_duplicate_parameter():
    with pytest.raises(ValueError):
        parse_spec("sma(column='close',column='open',period=20)@H1")


# ------------------------------------------------------------
# Alias duplication
# (در صورتی که alias تعریف کرده باشی)
# ------------------------------------------------------------ 14 OK
def test_duplicate_alias():
    with pytest.raises(ValueError):
        parse_spec("sma(column='close',col='open',period=20)@H1")

# ------------------------------------------------------------
# Wrong type
# ------------------------------------------------------------ 15 OK, 16 OK
def test_wrong_type_period():
    with pytest.raises(TypeError):
        parse_spec("sma(column='close',period='abc')@H1")


def test_wrong_type_column():
    with pytest.raises(TypeError):
        parse_spec("sma(column=10,period=20)@H1")

# ------------------------------------------------------------
# Choice validation
# ------------------------------------------------------------ 17 OK
def test_invalid_choice():
    with pytest.raises(ValueError):
        parse_spec("sma(column='price',period=20)@H1")

# ------------------------------------------------------------
# Minimum validation
# ------------------------------------------------------------ 18 OK, 19 OK
def test_invalid_period_zero():
    with pytest.raises(ValueError):
        parse_spec("sma(column='close',period=0)@H1")


def test_invalid_period_negative():
    with pytest.raises(ValueError):
        parse_spec("sma(column='close',period=-5)@H1")

# ------------------------------------------------------------
# Maximum validation
# ------------------------------------------------------------ 20 OK
def test_maximum_validation():
    with pytest.raises(ValueError):
        parse_spec("parabolic_sar(acceleration=1000)@H1")

# ------------------------------------------------------------
# Extra positional arguments
# ------------------------------------------------------------ 21 OK
def test_too_many_positionals():
    with pytest.raises(ValueError):
        parse_spec("sma(close,20,123,456)@H1")

# ------------------------------------------------------------
# Empty spec
# ------------------------------------------------------------ 22 OK
def test_empty_spec():
    with pytest.raises(ValueError):
        parse_spec("")

# ------------------------------------------------------------
# Invalid syntax
# ------------------------------------------------------------ 23 OK, 24 OK
def test_invalid_syntax():
    with pytest.raises(ValueError):
        parse_spec("sma(period=20")


def test_invalid_name():
    with pytest.raises(ValueError):
        parse_spec("123abc()")

# ------------------------------------------------------------
# Invalid timeframe syntax
# ------------------------------------------------------------ 25 OK
def test_invalid_tf():
    with pytest.raises(ValueError):
        parse_spec("sma(period=20)@")

# ------------------------------------------------------------
# Default injection
# ------------------------------------------------------------ 26 OK
def test_default_values():
    ps = parse_spec(
        "sma(column='close',period=20)@H1"
    )
    assert ps.kwargs["min_periods"] == -1

# ------------------------------------------------------------
# Canonical names after alias resolution
# ------------------------------------------------------------ 27 OK
def test_alias_resolution():
    ps = parse_spec(
        "sma(src='close',period=20)@H1"
    )
    assert "column" in ps.kwargs
    assert "sec" not in ps.kwargs

# ------------------------------------------------------------
# ParsedSpec always returns kwargs only
# ------------------------------------------------------------ 28 OK
def test_args_are_empty_after_compile():
    ps = parse_spec(
        "sma(close,20)@H1"
    )
    assert ps.args == []

# =============================================================================
# Part 3 - One Valid Spec For Every Indicator
# =============================================================================

# ---------------------------------------------------------------------
# Every implemented indicator should be accepted by the parser.
#
# هدف این تست فقط صحت Parser و Validation است.
# اجرای خود اندیکاتورها تست نمی‌شود.
# ---------------------------------------------------------------------

VALID_SPECS = [

    # ---------------- Moving Averages ----------------
    ("sma(column='close',period=20)@M1", "sma"),
    ("wma(column='close',period=20)@H1", "wma"),
    ("ema(column='close',period=20)@D1", "ema"),
    ("dema(column='close',period=20)@H1", "dema"),
    ("tema(column='close',period=20)@H1", "tema"),
    ("kama(column='close',period=20)@H1", "kama"),
    ("hma(column='close',period=20)@H1", "hma"),

    # ---------------- Momentum ----------------
    ("roc(column='close',period=14)@H1", "roc"),
    ("rsi(period=14)@H1", "rsi"),
    ("cci(period=20)@H1", "cci"),
    ("mfi(period=14)@H1", "mfi"),
    ("williams_r(period=14)@H1", "williams_r"),

    # ---------------- Volatility ----------------
    ("true_range()@H1", "true_range"),
    ("atr(period=14)@H1", "atr"),
    ("bollinger_bands(column='close',period=20,std=2)@H1",
     "bollinger_bands"),
    ("keltner_channel(high='high', low='low', close='close',period=20,multiplier=2)@H1",
     "keltner_channel"),
    ("supertrend(period=10,multiplier=3)@H1",
     "supertrend"),

    # ---------------- Trend ----------------
    ("macd(column='close',fast=12,slow=26,signal=9)@H1",
     "macd"),
    ("aroon(period=25)@H1",
     "aroon"),

    # ---------------- Oscillators ----------------
    ("stochastic(k_period=14,d_period=3,smooth_k=3)@H1",
     "stochastic"),

    # ---------------- Volume ----------------
    ("obv()@H1",
     "obv"),

    # ---------------- Candle ----------------
    ("heikin_ashi()@H1",
     "heikin_ashi"),

    # ---------------- SAR ----------------
    ("parabolic_sar(step=0.02,max_step=0.2)@H1",
     "parabolic_sar"),
]

# ---------------------------------------------------------------------
# 
# --------------------------------------------------------------------- 29 OK
@pytest.mark.parametrize(
    "spec_text,indicator",
    VALID_SPECS,
)
def test_all_indicators(spec_text, indicator):
    ps = parse_spec(spec_text)
    assert ps.name == indicator
    assert ps.args == []
    assert ps.timeframe is not None
    assert isinstance(ps.kwargs, dict)

# ---------------------------------------------------------------------
# Raw Spec must remain unchanged
# --------------------------------------------------------------------- 30 OK
@pytest.mark.parametrize(
    "spec_text,_",
    VALID_SPECS,
)
def test_raw_spec_preserved(spec_text, _):
    ps = parse_spec(spec_text)
    assert ps.raw == spec_text

# ---------------------------------------------------------------------
# Timeframe normalization
# --------------------------------------------------------------------- 31 OK
@pytest.mark.parametrize(
    "spec_text,_",
    VALID_SPECS,
)
def test_timeframe_is_uppercase(spec_text, _):
    lower = spec_text.replace("@H1", "@h1") \
                     .replace("@M1", "@m1") \
                     .replace("@D1", "@d1")
    ps = parse_spec(lower)
    assert ps.timeframe == ps.timeframe.upper()

# ---------------------------------------------------------------------
# kwargs must exist for every indicator
# --------------------------------------------------------------------- 32 OK
@pytest.mark.parametrize(
    "spec_text,_",
    VALID_SPECS,
)
def test_kwargs_is_dictionary(spec_text, _):
    ps = parse_spec(spec_text)
    assert isinstance(ps.kwargs, dict)

# ---------------------------------------------------------------------
# args should already have been compiled away
# --------------------------------------------------------------------- 33 OK
@pytest.mark.parametrize(
    "spec_text,_",
    VALID_SPECS,
)
def test_no_remaining_positional_arguments(spec_text, _):
    ps = parse_spec(spec_text)
    assert ps.args == []

# =============================================================================
# Part 4 - Positional / Keyword / Mixed Argument Compilation
#
# هدف:
#   بررسی Compiler مرحله اول Parser
#   یعنی تمام حالت‌های مجاز ورود آرگومان‌ها باید به kwargs نهایی تبدیل شوند.
# =============================================================================

# -----------------------------------------------------------------------------
# Pure Positional
# ----------------------------------------------------------------------------- 34 OK
@pytest.mark.parametrize(
    "spec_text,expected",
    [

        (
            "sma(close,20)@H1",
            {
                "column": "close",
                "period": 20,
            },
        ),

        (
            "ema(close,55)@M15",
            {
                "column": "close",
                "period": 55,
            },
        ),

        (
            "roc(close,14)@H4",
            {
                "column": "close",
                "period": 14,
            },
        ),
    ],
)
def test_positional_arguments(spec_text, expected):
    ps = parse_spec(spec_text)
    assert ps.args == []
    for k, v in expected.items():
        assert ps.kwargs[k] == v

# -----------------------------------------------------------------------------
# Pure Keyword
# ----------------------------------------------------------------------------- 35 OK
@pytest.mark.parametrize(
    "spec_text,expected",
    [
        (
            "sma(column='close',period=20)@H1",
            {
                "column": "close",
                "period": 20,
            },
        ),
        (
            "macd(fast=12,slow=26,signal=9)@H1",
            {
                "fast": 12,
                "slow": 26,
                "signal": 9,
            },
        ),
        (
            "atr(period=14)@H1",
            {
                "period": 14,
            },
        ),
    ],
)
def test_keyword_arguments(spec_text, expected):
    ps = parse_spec(spec_text)
    assert ps.args == []
    for k, v in expected.items():
        assert ps.kwargs[k] == v

# -----------------------------------------------------------------------------
# Mixed Positional + Keyword
# ----------------------------------------------------------------------------- 36 OK
@pytest.mark.parametrize(
    "spec_text,expected",
    [
        (
            "sma(close,period=20)@H1",
            {
                "column": "close",
                "period": 20,
            },
        ),
        (
            "sma(column='close',20)@H1",
            {
                "column": "close",
                "period": 20,
            },
        ),
        (
            "bollinger_bands(close,20,std=2)@H1",
            {
                "column": "close",
                "period": 20,
                "multiplier": 2,
            },
        ),
    ],
)
def test_mixed_arguments(spec_text, expected):
    ps = parse_spec(spec_text)
    assert ps.args == []
    for k, v in expected.items():
        assert ps.kwargs[k] == v

# -----------------------------------------------------------------------------
# Order Independence
# ----------------------------------------------------------------------------- 37 OK
def test_keyword_order_independent():
    a = parse_spec(
        "macd(fast=12,slow=26,signal=9)@H1"
    )
    b = parse_spec(
        "macd(signal=9,fast=12,slow=26)@H1"
    )
    assert a.kwargs == b.kwargs

# -----------------------------------------------------------------------------
# Default Injection
# ----------------------------------------------------------------------------- 38 OK
def test_default_parameter_injected():
    ps = parse_spec(
        "sma(column='close',period=20)@H1"
    )
    assert "min_periods" in ps.kwargs

# -----------------------------------------------------------------------------
# Explicit value overrides default
# ----------------------------------------------------------------------------- 39 OK
def test_default_can_be_overridden():
    ps = parse_spec(
        "sma(column='close',period=20,min_periods=5)@H1"
    )
    assert ps.kwargs["min_periods"] == 5

# -----------------------------------------------------------------------------
# Optional parameters only
# ----------------------------------------------------------------------------- 40 OK
def test_optional_parameter_present():
    ps = parse_spec(
        "parabolic_sar(step=0.02,max_step=0.2)@H1"
    )
    assert ps.kwargs["acceleration_step"] == 0.02
    assert ps.kwargs["acceleration_max"] == 0.2
# -----------------------------------------------------------------------------
# No positional arguments should survive compilation
# ----------------------------------------------------------------------------- 41 OK
@pytest.mark.parametrize(
    "spec_text",
    [
        "sma(close,20)@H1",
        "ema(close,55)@H1",
        "roc(close,14)@H1",
        "bollinger_bands(close,20,2)@H1",
    ],
)
def test_args_compiled_into_kwargs(spec_text):
    ps = parse_spec(spec_text)
    assert ps.args == []

# -----------------------------------------------------------------------------
# Raw text must never change
# ----------------------------------------------------------------------------- 42 OK
@pytest.mark.parametrize(
    "spec_text",
    [
        "sma(close,20)@H1",
        "ema(column='close',period=20)@H1",
        "macd(fast=12,slow=26,signal=9)@H1",
    ],
)
def test_raw_text_preserved(spec_text):
    ps = parse_spec(spec_text)
    assert ps.raw == spec_text


# =============================================================================
# Part 5 - Alias / Default / Boundary / Choice / Type Validation
# هدف:
#     تست کامل Validation بر اساس ParameterSpec
# نکته:
#     تست‌های Alias فقط در صورتی Pass می‌شوند که واقعاً برای آن پارامتر
#     Alias در Registry تعریف کرده باشی.
# =============================================================================

# =============================================================================
# Alias Resolution
# ============================================================================= 43 OK
@pytest.mark.parametrize(
    "spec_text,canonical,value",
    [
        ("sma(price='close',period=20)@H1", "column", "close"),
        ("ema(src='close',period=30)@H1", "column", "close"),
    ]
)
def test_alias_resolution(spec_text, canonical, value):
    ps = parse_spec(spec_text)
    assert canonical in ps.kwargs
    assert ps.kwargs[canonical] == value

# =============================================================================
# Alias + Canonical together -> Duplicate
# ============================================================================= 44 OK
@pytest.mark.parametrize(
    "spec_text",
    [
        "sma(column='close',col='open',period=20)@H1",
        "ema(column='close',src='open',period=20)@H1",
    ]
)
def test_alias_duplicate(spec_text):
    with pytest.raises(ValueError):
        parse_spec(spec_text)

# =============================================================================
# Default Injection
# ============================================================================= 45 OK
def test_default_is_injected():
    ps = parse_spec(
        "sma(column='close',period=20)@H1"
    )
    assert "min_periods" in ps.kwargs
    assert ps.kwargs["min_periods"] == -1

# =============================================================================
# Explicit value overrides default
# ============================================================================= 46 OK
def test_default_override():
    ps = parse_spec(
        "sma(column='close',period=20,min_periods=5)@H1"
    )
    assert ps.kwargs["min_periods"] == 5

# =============================================================================
# Required Parameter
# ============================================================================= 47 OK
@pytest.mark.parametrize(
    "spec_text",
    [
        "sma(column='close')@H1",
        "sma(period=20)@H1",
        "ema(column='close')@H1",
    ]
)
def test_required_parameters(spec_text):
    with pytest.raises(ValueError):
        parse_spec(spec_text)

# =============================================================================
# Choice Validation
# ============================================================================= 48 OK, 49 OK
@pytest.mark.parametrize(
    "column",
    [
        "price",
        "median",
        "typical",
        "xxx",
    ]
)
def test_invalid_choice(column):
    with pytest.raises(ValueError):
        parse_spec(
            f"sma(column='{column}',period=20)@H1"
        )


@pytest.mark.parametrize(
    "column",
    [
        "open",
        "high",
        "low",
        "close",
    ]
)
def test_valid_choice(column):
    ps = parse_spec(
        f"sma(column='{column}',period=20)@H1"
    )
    assert ps.kwargs["column"] == column

# =============================================================================
# Minimum Boundary
# ============================================================================= 50 OK
@pytest.mark.parametrize(
    "period",
    [
        0,
        -1,
        -10,
    ]
)
def test_period_below_minimum(period):
    with pytest.raises(ValueError):
        parse_spec(
            f"sma(column='close',period={period})@H1"
        )

@pytest.mark.parametrize(
    "period",
    [
        1,
        2,
        20,
        200,
    ]
)
def test_period_valid(period):
    ps = parse_spec(
        f"sma(column='close',period={period})@H1"
    )
    assert ps.kwargs["period"] == period

# =============================================================================
# Maximum Boundary
# ============================================================================= 51 OK
@pytest.mark.parametrize(
    "step",
    [
        1.5,
        10,
        100,
    ]
)
def test_step_above_maximum(step):
    with pytest.raises(ValueError):
        parse_spec(
            f"parabolic_sar(step={step})@H1"
       )

# =============================================================================
# Type Validation
# ============================================================================= 52 OK
@pytest.mark.parametrize(
    "spec_text",
    [
        "sma(column=10,period=20)@H1",
        "sma(column='close',period='abc')@H1",
        "atr(period='xyz')@H1",
        "macd(fast='12')@H1",
    ]
)
def test_invalid_types(spec_text):
    with pytest.raises(TypeError):
        parse_spec(spec_text)

# =============================================================================
# Float Parameters
# ============================================================================= 53 OK
def test_float_parameter():
    ps = parse_spec(
        "parabolic_sar(step=0.02,max_step=0.2)@H1"
    )
    assert isinstance(ps.kwargs["acceleration_step"], float)

# =============================================================================
# Integer Parameters
# ============================================================================= 54 OK
def test_integer_parameter():
    ps = parse_spec(
        "atr(period=14)@H1"
    )
    assert isinstance(ps.kwargs["period"], int)

# =============================================================================
# Unknown Parameter
# ============================================================================= 55 OK
@pytest.mark.parametrize(
    "spec_text",
    [
        "sma(column='close',period=20,abc=1)@H1",
        "atr(period=14,test=1)@H1",
        "macd(fast=12,slow=26,hello=5)@H1",
    ]
)
def test_unknown_parameter(spec_text):
    with pytest.raises(ValueError):
        parse_spec(spec_text)

# =============================================================================
# Canonical kwargs only
# ============================================================================= 56 OK
def test_no_alias_left_after_compile():
    ps = parse_spec(
        "sma(src='close',period=20)@H1"
    )
    assert "column" in ps.kwargs
    assert "src" not in ps.kwargs

# =============================================================================
# Compiler Output Contract
# ============================================================================= 57 
def test_parser_returns_compiled_spec():
    ps = parse_spec(
        "sma(close,20)@H1"
    )
    assert ps.args == []
    assert isinstance(ps.kwargs, dict)
    assert ps.name == "sma"
    assert ps.timeframe == "H1"

# =============================================================================
# Part 6 - Regression / Edge Cases
# هدف:
#     جلوگیری از خراب شدن Parser در آینده
#     تست حالت‌های مرزی و غیرمعمول
# =============================================================================

# =============================================================================
# Whitespace
# ============================================================================= 58 OK
@pytest.mark.parametrize(
    "spec",
    [
        " sma(column='close',period=20)@H1 ",
        "sma( column='close' , period=20 )@H1",
        "sma(column='close',period=20) @H1",
        "  sma ( column='close' , period=20 ) @H1 ",
    ],
)
def test_whitespace(spec):
    ps = parse_spec(spec)
    assert ps.name == "sma"
    assert ps.timeframe == "H1"

# =============================================================================
# Case-insensitive TimeFrame
# ============================================================================= 59 OK
@pytest.mark.parametrize(
    "tf",
    [
        "h1",
        "H1",
        "m15",
        "M15",
        "d1",
        "D1",
    ],
)
def test_tf_normalization(tf):
    ps = parse_spec(
        f"sma(column='close',period=20)@{tf}"
    )
    assert ps.timeframe == tf.upper()

# =============================================================================
# Float Parsing
# ============================================================================= 60 OK
def test_float_parsing():
    ps = parse_spec(
        "parabolic_sar(step=0.02,max_step=0.2)@H1"
    )
    assert ps.kwargs["acceleration_step"] == 0.02
    assert ps.kwargs["acceleration_max"] == 0.2

# =============================================================================
# Integer Parsing
# ============================================================================= 61 OK
def test_integer_parsing():
    ps = parse_spec(
        "atr(period=14)@H1"
    )
    assert ps.kwargs["period"] == 14

# =============================================================================
# String Parsing
# ============================================================================= 62 OK
def test_string_parsing():
    ps = parse_spec(
        "sma(column='close',period=20)@H1"
    )
    assert ps.kwargs["column"] == "close"

# =============================================================================
# Empty Parentheses
# ============================================================================= 63 OK
def test_empty_parentheses():
    ps = parse_spec(
        "true_range()@H1"
    )
    assert ps.args == []
    assert isinstance(ps.kwargs, dict)

# =============================================================================
# No Parentheses
# ============================================================================= 64 OK
def test_no_parentheses():
    ps = parse_spec(
        "true_range@H1"
    )
    assert ps.name == "true_range"

# =============================================================================
# Preserve Raw Spec
# ============================================================================= 65 OK
def test_raw_preserved():
    spec = "ema(column='close',period=50)@H4"
    ps = parse_spec(spec)
    assert ps.raw == spec

# =============================================================================
# ParsedSpec Contract
# ============================================================================= 66 OK
def test_parsed_spec_contract():
    ps = parse_spec(
        "ema(column='close',period=20)@H1"
    )
    assert isinstance(ps.name, str)
    assert isinstance(ps.kwargs, dict)
    assert isinstance(ps.args, list)
    assert isinstance(ps.raw, str)
    assert ps.timeframe == "H1"

# =============================================================================
# Same Spec -> Same Result
# ============================================================================= 67 OK
def test_repeatability():
    a = parse_spec(
        "ema(column='close',period=20)@H1"
    )
    b = parse_spec(
        "ema(column='close',period=20)@H1"
    )
    assert a == b

# =============================================================================
# Regression - Positional Compilation
# ============================================================================= 68 OK
def test_regression_positional():
    ps = parse_spec(
        "sma(close,20)@H1"
    )
    assert ps.args == []
    assert ps.kwargs["column"] == "close"
    assert ps.kwargs["period"] == 20

# =============================================================================
# Regression - Default Injection
# ============================================================================= 69 OK
def test_regression_defaults():
    ps = parse_spec(
        "sma(column='close',period=20)@H1"
    )
    assert "min_periods" in ps.kwargs

# =============================================================================
# Regression - Unknown Indicator
# ============================================================================= 70 OK
def test_regression_unknown_indicator():
    with pytest.raises(ValueError):
        parse_spec("xxxx(period=10)@H1")

# =============================================================================
# Regression - Unknown Parameter
# ============================================================================= 71 OK
def test_regression_unknown_parameter():
    with pytest.raises(ValueError):
        parse_spec(
            "sma(column='close',period=20,abc=1)@H1"
        )

# =============================================================================
# Regression - Invalid Spec Syntax
# ============================================================================= 72 OK
@pytest.mark.parametrize(
    "spec",
    [
        "",
        "()",
        "@H1",
        "123",
        "sma(",
        "sma(period=20",
        "sma(period=20))",
    ],
)
def test_invalid_syntax(spec):
    with pytest.raises(ValueError):
        parse_spec(spec)

# ============================================================================= END