# f02_utils/config_ops.py
# Status in (Bot-RL-3): Reviewed before 1404/09/05

from typing import Any, Dict, List, Union
import yaml
import logging

""" --------------------------------------------------------------------------- OK Func1
 دسترسی امن به مسیرهای تو در توی دیکشنری با dot-notation؛ در نبود کلید، مقدار پیش‌فرض برمی‌گردد.
:مثال
a = {"k1":"a",
     "k2":{"x":1, "y":2, "z":{"aaa":10, "bbb":20}},
     "k3":"c"
     }
print(_deep_get(a, "k2.z"    ,"///")) --> {"aaa":10, "bbb":20}
print(_deep_get(a, "k2.z.bbb","///")) --> 20
print(_deep_get(a, "k3.z.bbb","///")) --> ///
"""
def _deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

""" --------------------------------------------------------------------------- OK Func2,3,4
 deep_set / yaml_set (FINAL)
توضیح: این توابع برای به‌روزرسانی امنِ کلیدهای config به‌صورت برنامه‌وار استفاده می‌شوند.
- deep_set: معکوسِ _deep_get؛ مقدار را در مسیر دلخواه قرار می‌دهد (in-place).
- yaml_set: فایل YAML را لود می‌کند، deep_set را اعمال می‌کند و ذخیره می‌کند.
ورودی path می‌تواند رشته (مانند "indicators.patterns[3].tol_k") یا لیست توکن‌ها باشد.
:مثال
path = "...aaa.bbb[  123   ].ccc"
print(_parse_path(path)) --> ['aaa', 'bbb', 123, 'ccc']
:مثال

"""
PathLike = Union[str, List[Union[str, int]]]
def _parse_path(path: PathLike) -> List[Union[str, int]]:
    """ مسیر را به توکن‌های قابل پیمایش تبدیل می‌کند.
    مثال‌ها:
        "aaa.bbb[3].ccc"  -> ["aaa","bbb",3,"ccc"]
    """
    if isinstance(path, list):
        return path[:]
    s = str(path).strip()
    out: List[Union[str, int]] = []
    buf = ""
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '.':
            if buf:
                out.append(buf)
                buf = ""
            i += 1
            continue
        if ch == '[':
            if buf:
                out.append(buf)
                buf = ""
            j = s.find(']', i+1)
            if j == -1:
                raise ValueError(f"Malformed path (missing ']'): {s}")
            token = s[i+1:j].strip().strip('"').strip("'")
            # اندیس عددی یا کلید رشته‌ای 
            if token.isdigit() or (token.startswith('-') and token[1:].isdigit()):
                out.append(int(token))
            else:
                out.append(token)
            i = j + 1
            continue
        buf += ch  # if (char!='.') and (char!='[')
        i += 1
    if buf: 
        out.append(buf) # append remaining buffer
    return out

# ---------------------------------------------------------
def deep_set(obj: Any, path: PathLike, value: Any, create_missing: bool = True) -> Any:
    """
    مقدار را در مسیر دلخواه قرار می‌دهد (in-place)
    - اگر create_missing=True باشد، شاخه‌های مفقود ساخته می‌شوند (dict یا list)
    - نوع شاخهٔ ساخته‌شده براساس توکن بعدی تعیین می‌شود (int -> list، else -> dict)

    مثال:
      deep_set(cfg, "indicators.patterns[3].tol_k", 0.35)
      deep_set(cfg, ["indicators","patterns",3,"tol_k"], 0.35)
    """
    tokens = _parse_path(path)
    if not tokens:
        raise ValueError("Empty path")

    cur = obj
    # enumerate(iterable) یک تابع پایتون است که وقتی روی یک مجموعه (مثل list) حلقه می‌زنی،
    # همزمان هم ایندکس را می‌دهد و هم مقدار را.
    # ---------- Start of for-loop --------------------------------------------
    for idx, tk in enumerate(tokens[:-1]):  # همه توکن‌ها به جز آخرین
        nxt = tokens[idx + 1]   # پیش‌نمایش توکن بعدی برای تعیین نوع شاخه جدید
        #=================================== دیکشنری ===== 
        if isinstance(tk, str):
            # اگر tk از جنس رشته باشد، نتظار داریم که cur دیکشنری باشد، پس اگر نباشد خطا می‌دهیم 
            if not isinstance(cur, dict):
                # اگر cur دیکشنری نباشد و در ضمن اگر اجازه ساخت شاخه‌های مفقود را نداشته باشیم، خطا می‌دهیم 
                if not create_missing:
                    # پیام زیر میگوید در مسیر پیمایش، به جایی رسیدم که انتظار داشتم dict باشد تا وارد کلید بعدی شوم، اما dict نبود 
                    raise TypeError(f"Cannot descend into non-dict at {tokens[:idx+1]}")
                # تبدیل اجباری
                raise TypeError(f"Expected dict at {tokens[:idx]}, got {type(cur).__name__}")
            # وقتی به این نقطه میرسیم که cur دیکشنری باشد 
            if tk not in cur or cur[tk] is None:    # اگر هردو شرط روبرو برقرار نباشند، تنها راه ادامه، ساختن کلید tk است 
                if not create_missing:              # اگر مجوز ساخت کلید نداشته باشیم، 
                    raise KeyError(f"Missing key: {tk}")   # به بن بست رسیده ایم و خطا اعلام میکنیم 
                # :در صورتی به این نقطه میرسیم که   
                #  کلید tk از جنس رشته باشد-1 
                # cut دیکشنری باشد-2 
                # tk در cur نباشد یا cur[tk] برابر با None باشد-3 
                #  مجوز ساخت کلید tk را داشته باشیم-4 
                #حالا مهم این است که مقدار این کلید باید دیکشنری باشد یا لیست؟ 
                # بنابراین باید از کلید nxt استفاده کنیم تا ببینیم که حاصل cur[tk] باید چه باشد؟ 
                cur[tk] = [] if isinstance(nxt, int) else {}
            
            cur = cur[tk]   # با استفاده از این عبارت، cur را محدود به cur[tk] میکنیم 
        #======================== لیست با اندیس عددی ===== 
        elif isinstance(tk, int):
            # اگر tk از جنس عدد باشد، نتظار داریم که cur لیست باشد، پس اگر نباشد خطا می‌دهیم 
            if not isinstance(cur, list):
                if not create_missing:
                    raise TypeError(f"Cannot index non-list at {tokens[:idx+1]}")
                raise TypeError(f"Expected list at {tokens[:idx]}, got {type(cur).__name__}")
            # :وقتی به این نقطه میرسیم که   
            # tk عدد باشد-1 
            # cur لیست باشد-2 
            # مجوز ساخت اندکس جدید tk را داشته باشیم-3 
            # توسعهٔ لیست در صورت نیاز
            need_len = tk + 1   # چون اندکس لیستها از 0 شروع میشود، و از طرفی باید لیست cur دارای اندکس tk هم باشد 
                                # بنابراین باید طول آن برابر با tk+1 باشد 
            if len(cur) < need_len:                            # اگر طول مورد نیاز برای لیست، کمتر است از طول فعلی آن 
                cur.extend([None] * (need_len - len(cur)))     # به تعداد مورد نیاز عضو جدید، با مقدار None ، به لیست cur میافزاییم 
            #حالا مهم این است که مقدار این اندکس باید دیکشنری باشد یا لیست؟ 
            # بنابراین باید از کلید nxt استفاده کنیم تا ببینیم که حاصل cur[tk] باید چه باشد؟ 
            if cur[tk] is None:
                cur[tk] = [] if isinstance(nxt, int) else {}

            cur = cur[tk]   # با استفاده از این عبارت، cur را محدود به cur[tk] میکنیم 
        #========= اگر tk نه رشته باشد و نه عدد باشد ===== 
        else:
            raise TypeError(f"Unsupported token type at {tokens[:idx+1]}: {type(tk).__name__}")
    # ---------- End of for-loop ----------------------------------------------
    
    last = tokens[-1]
    if isinstance(last, str):
        if not isinstance(cur, dict):
            raise TypeError(f"Cannot set key on non-dict at {tokens[:-1]}")
        cur[last] = value
    elif isinstance(last, int):
        if not isinstance(cur, list):
            raise TypeError(f"Cannot set index on non-list at {tokens[:-1]}")
        need_len = last + 1
        if len(cur) < need_len:
            cur.extend([None] * (need_len - len(cur)))
        cur[last] = value
    else:
        raise TypeError(f"Unsupported final token: {type(last).__name__}")

    return obj

# ---------------------------------------------------------
def yaml_set(path_to_yaml: str, key_path: PathLike, value: Any, create_missing: bool = True, save_backup: bool = True) -> None:
    """
    فایل YAML را باز می‌کند، deep_set را اعمال می‌کند و ذخیره می‌کند.
    - اگر save_backup=True باشد، از فایل یک کپی .bak می‌سازد.
    """
    import os
    if not os.path.exists(path_to_yaml):
        raise FileNotFoundError(f"YAML not found: {path_to_yaml}")

    with open(path_to_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    deep_set(cfg, key_path, value, create_missing=create_missing)

    if save_backup:
        bak = path_to_yaml + ".bak"
        try:
            if os.path.exists(bak):
                os.remove(bak)
        except Exception:
            pass
        try:
            # shutil یک ماژول استاندارد پایتون است که برای کارهای سطح بالا با فایل‌ها و پوشه‌ها استفاده می‌شود 
            # این ماژول برای کپی، جابجایی و حذف فایلها و پوشه ها و گرفتن اطلاعات آنها بکار میرود 
            import shutil
            # copy2 یک تابع در ماژول shutil است که فایل را به مقصد مشخص شده کپی می‌کند 
            # و همچنین متادیتاهای فایل (مانند زمان ایجاد و تغییر) را نیز حفظ می‌کند 
            # این تابع در واقع یک کپی کامل از فایل ایجاد می‌کند 

            # در سطر زیر قبل از تغییر فایل اصلی، یک کپی از آن بعنوان بکاپ در مسیر مشخص شده ایجاد می‌کند 
            shutil.copy2(path_to_yaml, bak)
        except Exception as e:
            logging.warning(f"[WARN] Could not create backup: {e}")

    with open(path_to_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    logging.info(f"[OK] Config updated: {key_path} -> {value}")

