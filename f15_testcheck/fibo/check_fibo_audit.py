# -*- coding: utf-8 -*-
# f15_testcheck/fibo/check_fibo_audit.py
# Status in *Bot-RL-3): Completed

"""
اسکریپت ممیزی فیبوناچی (مرحله ۱ Bot-RL-3) — نسخه‌ی ساده‌شده و منطبق با ساختار پروژه

هدف:
    - فقط و فقط از روی «فولدر واقعی indicators پروژه» (بدون پشتیبانی ZIP) کدها را بخواند.
    - فایل‌های پایتون داخل indicators را با AST تحلیل کند و این موارد را استخراج نماید:
        * فهرست کلاس‌ها و متدهای هر کلاس + امضا (نام آرگومان‌ها، annotationها در صورت وجود)
        * فهرست توابع سطح ماژول + امضا (annotation خروجی در صورت وجود)
        * فراخوانی‌های احتمالی رجیستری (register/registry.register و مشابه)
        * ثابت‌ها/لیست‌های مرتبط با فیبوناچی (بر اساس نام‌های محتمل مانند FIB, FIBO, RATIO, RATIOS)
    - گزارش نهایی را «در ترمینال به انگلیسی» چاپ کند.

نکات:
    - هیچ تغییری در فایل‌های پروژه ایجاد نمی‌شود.
    - مسیر پیش‌فرض indicators همان چیزی است که در ساختار پروژه تعریف شده است (این اسکریپت «فرض» می‌گیرد
      فولدر در ریشه پروژه با نام `indicators/` وجود دارد). در صورت تفاوت، با سویچ `-i` مسیر دقیق را بدهید.
    - پیام‌های اجرایی/ترمینالی به انگلیسی هستند؛ کامنت‌ها و Docstringها فارسی.
"""

import argparse
import ast
import io
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union


# -----------------------------
# داده‌ساخت‌ها برای خروجی ممیزی
# -----------------------------

@dataclass
class ArgInfo:
    """اطلاعات هر آرگومان تابع/متد"""
    name: str
    annotation: Optional[str]
    default: Optional[str]

@dataclass
class FunctionInfo:
    """اطلاعات یک تابع سطح ماژول یا متد یک کلاس"""
    name: str
    args: List[ArgInfo]
    returns: Optional[str]
    is_async: bool

@dataclass
class ClassInfo:
    """اطلاعات یک کلاس شامل امضای متدها"""
    name: str
    bases: List[str]
    methods: List[FunctionInfo]

@dataclass
class RegistryCall:
    """اطلاعات فراخوانی‌های احتمالی رجیستری (register) برای اندیکاتورها"""
    func: str
    args: List[str]
    keywords: Dict[str, str]

@dataclass
class ConstantsInfo:
    """اطلاعات ثابت‌های مرتبط با فیبوناچی/نسبت‌ها در فایل"""
    name: str
    value_repr: str

@dataclass
class ModuleAudit:
    """نتیجه ممیزی یک فایل پایتون"""
    module_path: str
    classes: List[ClassInfo]
    functions: List[FunctionInfo]
    registry_calls: List[RegistryCall]
    fib_constants: List[ConstantsInfo]


# ------------------------------------
# ابزارک‌های کمکی برای استخراج از AST
# ------------------------------------

def _safe_unparse(node: Optional[ast.AST]) -> Optional[str]:
    """تبدیل AST node به رشته به‌صورت امن؛ اگر ممکن نبود None."""
    if node is None:
        return None
    try:
        return ast.unparse(node)  # Python 3.9+
    except Exception:
        return None

def _default_to_str(node: Optional[ast.AST]) -> Optional[str]:
    """تبدیل مقدار پیش‌فرض AST به رشته؛ اگر وجود ندارد، None."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None

def _args_from_ast(arguments: ast.arguments) -> List[ArgInfo]:
    """استخراج اطلاعات آرگومان‌ها از ast.arguments"""
    args_info: List[ArgInfo] = []

    posonly = list(getattr(arguments, "posonlyargs", []))
    normal  = list(arguments.args)
    kwonly  = list(arguments.kwonlyargs)
    vararg  = [arguments.vararg] if arguments.vararg else []
    kwarg   = [arguments.kwarg] if arguments.kwarg else []

    defaults = list(arguments.defaults)       # پیش‌فرض‌های args معمولی (انتهای لیست)
    kw_defaults = list(arguments.kw_defaults)  # پیش‌فرض‌های kwonly

    # posonly + normal
    all_simple = posonly + normal
    n_simple = len(all_simple)
    n_defaults = len(defaults)
    for i, a in enumerate(all_simple):
        default_node = None
        if n_defaults and i >= (n_simple - n_defaults):
            default_node = defaults[i - (n_simple - n_defaults)]
        args_info.append(
            ArgInfo(
                name=a.arg,
                annotation=_safe_unparse(a.annotation),
                default=_default_to_str(default_node),
            )
        )

    # vararg: *args
    for a in vararg:
        args_info.append(
            ArgInfo(
                name="*"+a.arg,
                annotation=_safe_unparse(a.annotation),
                default=None,
            )
        )

    # kwonly
    for i, a in enumerate(kwonly):
        args_info.append(
            ArgInfo(
                name=a.arg,
                annotation=_safe_unparse(a.annotation),
                default=_default_to_str(kw_defaults[i]) if i < len(kw_defaults) else None,
            )
        )

    # kwarg: **kwargs
    for a in kwarg:
        args_info.append(
            ArgInfo(
                name="**"+a.arg,
                annotation=_safe_unparse(a.annotation),
                default=None,
            )
        )

    return args_info

def _parse_function_def(fn: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> FunctionInfo:
    """استخراج اطلاعات از یک تابع/متد"""
    return FunctionInfo(
        name=fn.name,
        args=_args_from_ast(fn.args),
        returns=_safe_unparse(fn.returns),
        is_async=isinstance(fn, ast.AsyncFunctionDef),
    )

def _is_registry_call(node: ast.Call) -> bool:
    """
    تشخیص ابتدایی اینکه این Call ممکن است فراخوانی رجیستری باشد.
    الگوهای رایج: register(...), registry.register(...), indicators.registry.register(...)
    """
    def name_of(call_func: ast.AST) -> str:
        try:
            return ast.unparse(call_func)
        except Exception:
            return type(call_func).__name__
    name_repr = name_of(node.func)
    lowered = name_repr.replace(" ", "").lower()
    return lowered.endswith(".register") or lowered == "register"

def _extract_registry_call(node: ast.Call) -> RegistryCall:
    """استخراج اطلاعات از فراخوانی رجیستری در صورت وجود"""
    def safe_unparse(x: ast.AST) -> str:
        try:
            return ast.unparse(x)
        except Exception:
            return type(x).__name__
    func_name = safe_unparse(node.func)
    args = [safe_unparse(a) for a in node.args]
    keywords = {kw.arg: safe_unparse(kw.value) for kw in node.keywords if kw.arg}
    return RegistryCall(func=func_name, args=args, keywords=keywords)

def _is_fib_constant(name: str) -> bool:
    """
    تشخیص ابتدایی برای ثابت‌های مرتبط با فیبوناچی.
    صرفاً بر اساس نام متغیر قضاوت می‌شود (FIB, FIBO, RATIO, RATIOS).
    """
    n = name.upper()
    keys = ("FIB", "FIBO", "RATIO", "RATIOS")
    return any(k in n for k in keys)


# -----------------------
# موتور ممیزی یک ماژول
# -----------------------

def _attach_parents(tree: ast.AST) -> None:
    """برای تشخیص سطح (مثلاً توابع سطح ماژول)، پدر هر نود را ست می‌کنیم."""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]

def audit_python_source(module_path: str, source: str) -> ModuleAudit:
    """تحلیل AST متن پایتون و استخراج ساختارها/امضاها/فراخوانی‌های رجیستری و ثابت‌های محتمل فیبوناچی"""
    tree = ast.parse(source)
    _attach_parents(tree)

    classes: List[ClassInfo] = []
    functions: List[FunctionInfo] = []
    registry_calls: List[RegistryCall] = []
    fib_constants: List[ConstantsInfo] = []

    for node in ast.walk(tree):
        # کلاس‌ها
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                bases.append(_safe_unparse(b) or "")
            methods: List[FunctionInfo] = []
            for body_item in node.body:
                if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(_parse_function_def(body_item))
            classes.append(ClassInfo(name=node.name, bases=bases, methods=methods))

        # توابع سطح ماژول
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(getattr(node, "parent", None), ast.Module):
            functions.append(_parse_function_def(node))

        # فراخوانی‌های رجیستری
        elif isinstance(node, ast.Call) and _is_registry_call(node):
            registry_calls.append(_extract_registry_call(node))

        # ثابت‌ها در سطح ماژول
        elif isinstance(node, ast.Assign) and isinstance(getattr(node, "parent", None), ast.Module):
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_fib_constant(target.id):
                    value_repr = _default_to_str(node.value) or type(node.value).__name__
                    fib_constants.append(ConstantsInfo(name=target.id, value_repr=value_repr))

    return ModuleAudit(
        module_path=module_path,
        classes=classes,
        functions=functions,
        registry_calls=registry_calls,
        fib_constants=fib_constants,
    )


# --------------
# قالب‌بندی چاپ
# --------------

def _print_function(fn: FunctionInfo, indent: str = "  ") -> None:
    print(f"{indent}- function: {fn.name}{' [async]' if fn.is_async else ''}")
    if fn.args:
        print(f"{indent}  args:")
        for a in fn.args:
            ann = f": {a.annotation}" if a.annotation else ""
            dft = f" = {a.default}" if a.default else ""
            print(f"{indent}    - {a.name}{ann}{dft}")
    if fn.returns:
        print(f"{indent}  returns: {fn.returns}")

def _print_class(cls: ClassInfo, indent: str = "  ") -> None:
    bases = f"({', '.join([b for b in cls.bases if b])})" if cls.bases else ""
    print(f"{indent}- class: {cls.name} {bases}".rstrip())
    if cls.methods:
        print(f"{indent}  methods:")
        for m in cls.methods:
            _print_function(m, indent + "    ")

def print_audit(audit: ModuleAudit) -> None:
    print(f"\n[FILE] {audit.module_path}")
    if audit.classes:
        print(" classes:")
        for c in audit.classes:
            _print_class(c)
    if audit.functions:
        print(" functions:")
        for f in audit.functions:
            _print_function(f)
    if audit.registry_calls:
        print(" registry-calls:")
        for rc in audit.registry_calls:
            kw = f" kw={rc.keywords}" if rc.keywords else ""
            print(f"  - {rc.func}({', '.join(rc.args)}){kw}")
    if audit.fib_constants:
        print(" fib-constants:")
        for const in audit.fib_constants:
            print(f"  - {const.name} = {const.value_repr}")


# --------------
# ورود CLI
# --------------

def _gather_indicator_pyfiles(root: Path, indicators_dir: str) -> List[Path]:
    """
    گردآوری فایل‌های .py از فولدر indicators واقعی پروژه.
    * بدون هیچ پشتیبانی از ZIP *
    """
    base = (root / indicators_dir).resolve()
    if not base.exists():
        print(f"[ERROR] Indicators directory not found: {base}", file=sys.stderr)
        return []
    files = list(base.glob("*.py"))
    if not files:
        print(f"[WARN] No Python files found in indicators directory: {base}", file=sys.stderr)
        return []
    # اولویت: فایل‌های دارای کلیدواژه fibo (مثل fibonacci.py) و سپس levels.py، بعد بقیه به ترتیب نام
    def sort_key(p: Path):
        name = p.name.lower()
        if "fibo" in name:
            return (0, name)
        if name == "levels.py":
            return (1, name)
        return (2, name)
    files.sort(key=sort_key)
    return files

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Fibonacci-related modules from real project folder (no ZIP support)."
    )
    parser.add_argument(
        "-r", "--root", default=".",
        help="Project root directory (default: current directory)."
    )
    parser.add_argument(
        "-i", "--indicators-dir", default="indicators",
        help="Indicators directory relative to root (default: indicators)."
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Limit to specific filenames (e.g., fibonacci.py levels.py)."
    )
    parser.add_argument(
        "--out", default=None,
        help="Optional path to save JSON report."
    )

    args = parser.parse_args()
    print("[INFO] Starting Fibonacci audit (folder-only mode)...")

    root = Path(args.root).resolve()
    pyfiles = _gather_indicator_pyfiles(root, args.indicators_dir)

    if args.only and pyfiles:
        only_set = {name.strip() for name in args.only}
        pyfiles = [p for p in pyfiles if p.name in only_set]
        if not pyfiles:
            print("[WARN] No matching files for --only filter.", file=sys.stderr)

    audits: List[ModuleAudit] = []
    for path in pyfiles:
        try:
            with io.open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with io.open(path, "r", encoding="latin-1") as f:
                content = f.read()
        except FileNotFoundError:
            print(f"[WARN] File not found: {path}", file=sys.stderr)
            continue

        try:
            audits.append(audit_python_source(str(path), content))
        except SyntaxError as e:
            print(f"[WARN] SyntaxError while parsing {path}: {e}", file=sys.stderr)

    if audits:
        print("[INFO] Audit results:")
        for a in audits:
            print_audit(a)
    else:
        print("[INFO] No modules audited.")

    if args.out:
        try:
            serial = [asdict(a) for a in audits]
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(serial, f, ensure_ascii=False, indent=2)
            print(f"[INFO] JSON report saved to: {args.out}")
        except Exception as e:
            print(f"[WARN] Could not save JSON report: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
