from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Arg-Mapping Helper (ANCHOR: ARG_MAPPING_HELPER)
# =============================================================================
def _align_args_with_signature_old1(ind_name: str, args_in: List[Any], kwargs_in: Dict[str, Any]) -> tuple[list, dict]:
    """
    نگاشت آرگومان‌های موقعیتی Spec به نام پارامترها بر اساس امضای اندیکاتور.
    قواعد:
      - عددی‌ها (int/float) → به پارامترهای عددیِ متداول: period/n/window/length/fast/slow/signal/k/d
      - رشته‌ها (str) → به پارامترهای ستونی: col/column/field
      - باقی موارد → به ترتیب امضای تابع (fallback)
    """
    try:
        # lazy import to avoid circular import at module load time
        from f04_features.feature_B_registry_1 import get_indicator

        spec = get_indicator(ind_name, mode="train")
        if spec is None:
            return list(args_in), dict(kwargs_in)
        fn = spec.fn

        if fn is None:
            return list(args_in), dict(kwargs_in)

        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        # پارامترهای کاندید برای نگاشت: همهٔ پارامترها به‌جز اولی (df/ohlc)
        cands = [p for i, p in enumerate(params) if i > 0 and p.kind in (
            p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY
        )]
        cand_names = [p.name for p in cands]

        # اولویت نگاشت
        numeric_pref = [n for n in ("period","n","window","length","fast","slow","signal","k","d")
                        if n in cand_names and n not in kwargs_in]
        str_pref     = [n for n in ("col","column","field")
                        if n in cand_names and n not in kwargs_in]

        new_kwargs = dict(kwargs_in)
        new_args: list[Any] = []

        for v in args_in:
            assigned = False
            
            # 1) عددی‌ها → پارامترهای عددی
            if isinstance(v, (int, float)) and numeric_pref:
                tgt = numeric_pref.pop(0)
                new_kwargs[tgt] = v
                assigned = True
            
            # 2) رشته‌ها → پارامترهای ستونی
            elif isinstance(v, str) and str_pref:
                tgt = str_pref.pop(0)
                new_kwargs[tgt] = v
                assigned = True
            
            # 3) fallback: به ترتیب اولین پارامتر آزاد
            if not assigned:
                for pn in cand_names:
                    if pn in new_kwargs:
                        continue
                    new_kwargs[pn] = v
                    assigned = True
                    break
            
            # 4) اگر هیچ‌کدام نشد، عبور به‌صورت positional
            if not assigned:
                new_args.append(v)

        return new_args, new_kwargs
    except Exception:
        # هر اشکالی در تحلیل امضا → بدون تغییر
        return list(args_in), dict(kwargs_in)


