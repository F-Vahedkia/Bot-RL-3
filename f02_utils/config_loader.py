# f02_utils/config_loader.py
# Status in (Bot-RL-3): Reviewed before 1404/09/05

"""
امکانات (کامل و توسعه‌پذیر):
- بارگذاری YAML از مسیر پیش‌فرض پروژه یا مسیر دلخواه
- بارگذاری متغیرهای محیطی از .env (در ریشهٔ پروژه) و اوورراید کلیدها
- پشتیبانی merge چند-فایلی: top-level keys: `extends` (Base ها) و `overlays` (اوِررایدها)
- اعتبارسنجی شِمای حداقلی و ارتقایافته (بر اساس Basic_2ok و نیازهای Self-Optimize/Executor)
- aliasing برای سازگاری نام‌های قدیمی/جدید (مثلاً max_dd_max -> max_drawdown_max)
- ابزارهای کمکی مسیرها و نسخه‌گذاری: `save_config_versioned(cfg, prefix="prod_")`
- API: get_all(copy=False), reload(), dump(path), ensure_dirs()
-      maybe: get(key, default)

نکته:
- فقط از کتابخانه‌های استاندارد + PyYAML + python-dotenv استفاده شده.
- شِما سبک است تا نیاز به نصب پکیج اضافی نباشد؛ در صورت نیاز می‌توان به Pydantic ارتقا داد.
"""

from __future__ import annotations

import os
import copy
import yaml
import logging

from typing import Any, Dict, Iterable, Literal, Optional, Union, List, Tuple
from datetime import datetime, timezone
from pathlib import Path
from f02_utils.config_path_funcs import project_root
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ============================================================
# ابزارهای کمکی سطح پایین 
# ============================================================ OK ALL
# جایگزینِ مقاوم:
def _default_config_path() -> Path:
    """  مسیر پیش‌فرض فایل کانفیگ. """
    return project_root() / "f01_config" / "config.yaml"


def _read_yaml_file(path: Union[str, Path], *, fail_on_duplicates: bool = False) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
    #with open(p, "r", encoding="utf-8") as f:     # has no difference with above line
        # بلوک try سعی می‌کند فایل YAML را بخواند و تبدیل به دیکشنری کند. 
        try:
            # توسط شرط زیر تعیین میکنیم که روی «تکراری بودن کلیدها» سخت‌گیر باشیم یا نباشیم 
            if fail_on_duplicates:
                # ----------
                class _UniqueKeyLoader(yaml.SafeLoader):
                    pass
                def _construct_mapping(loader, node, deep=False):
                    mapping = {}
                    for key_node, value_node in node.value:
                        key = loader.construct_object(key_node, deep=deep)
                        if key in mapping:
                            raise ValueError(f"Duplicate YAML key: {key} in {p}")
                        mapping[key] = loader.construct_object(value_node, deep=deep)
                    return mapping
                
                _UniqueKeyLoader.add_constructor(                     # یعنی این لودر 
                    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,   # برای دیکشنری ها 
                    _construct_mapping)                               # از این تابع استفاده کند 
                # ----------
                #  خط زیر فایل را میخواند و اگر فایل خالی بود، یک دیکشنری خالی برمیگرداند 
                data = yaml.load(f, Loader=_UniqueKeyLoader) or {}
            else:
                # در صورتیکه وجود کلیدهای تکراری مجاز باشند، خط زیر اجرا میشود 
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            # در صورتی به این نقطه میرسیم که فایل yaml ایراد داشته و خوانده نشود 
            raise ValueError(f"YAML parse error at {p}: {e}")
    
    if not isinstance(data, dict):
        raise ValueError(f"Root of YAML must be a mapping (dict). Got: {type(data)}")
    return data


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """merge عمیق دیکشنری‌ها با حفظ نوع‌ها و عدم دست‌کاری base.
    _deep_merge(low importance dict, high importance dict)
    """
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return copy.deepcopy(overlay)
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _ensure_dir(d: Union[str, Path]) -> Path:
    p = Path(d)           # تبدیل نمودن d به یک شیء path و ذخیره آن در p 
    p.mkdir(parents=True, exist_ok=True) # در مسیر شیء path پوشه را همراه با والدینش میسازد 
                                         # اگر پوشه مزبور موجود بود، خطا نمیگیرد 
    return p   # در نهایت همان شیء Path را (که حالا مطمئنیم پوشه‌اش روی دیسک وجود دارد) برمی‌گرداند. 


def _now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# ============================================================
# ConfigLoader
# ============================================================ OK ALL
class ConfigLoader:
    """   لودر کانفیگ پروژه با امکانات Merge/ENV/Validation/Versioning.  """
    def __init__(self,
                 config_path: Optional[Union[str, Path]] = None,
                 env_prefix: str = "BOT_",
                 enable_env_override: bool = True,
                 validate_mode: str = "warn",     # "warn" or "strict"
                 allow_extensions: bool = True):
        """
        config_path: مسیر YAML. اگر None باشد از f01_config/config.yaml استفاده می‌شود.
        env_prefix: پیشوند کلیدهای محیطی (مثلاً BOT_). جستجوی بدون پیشوند نیز انجام می‌شود.
        enable_env_override: اگر False، ENV نادیده گرفته می‌شود.
        """
        self.base_dir: Path = project_root()

        # بارگذاری .env از ریشهٔ پروژه (اگر موجود باشد)
        env_path = self.base_dir / ".env"
        if env_path.exists():
            try:
                load_dotenv(dotenv_path=str(env_path))
            except Exception:
                logger.exception("Failed to load .env from %s", env_path)

        self.config_path: Path = Path(config_path) if config_path else _default_config_path()
        self.env_prefix: str = env_prefix
        self.enable_env_override: bool = bool(enable_env_override)

        self.validate_mode: str = str(validate_mode).lower()
        self.allow_extensions: bool = bool(allow_extensions)
        self._fail_on_dupe_keys: bool = (self.validate_mode == "warn")  #>>>     !=

        self.config: Dict[str, Any] = {}
        # بارگذاری اولیه کانفیگ 
        self.reload()

    # ---------- بارگذاری اولیه با merge لایه‌ای ----------
    def _load_yaml_layered_old1(self) -> Dict[str, Any]:     # Reserved at 05-04-20 05/04/20
        """
        config.yaml اصلی را می‌خواند و اگر کلیدهای top-level زیر وجود داشت ادغام می‌کند:
          - extends: list[str]  → به ترتیب خوانده و به‌عنوان Base merge می‌کند (اولی کم‌اهمیت‌تر)
          - overlays: list[str] → در انتها روی نتیجه merge می‌کند (بالاترین اولویت)
        مسیرها می‌توانند نسبی به ریشهٔ پروژه یا مطلق باشند.
        """
        
        # خواندن کانفیگ و ذخیره آن در یک دیکشنری به نام root_cfg 
        root_cfg = _read_yaml_file(self.config_path, fail_on_duplicates=self._fail_on_dupe_keys)

        """ Absolute path
        یک مسیر مانند p را به مسیر مطلق تبدیل نموده و برمی گرداند
        """
        def _resolve(p: str) -> Path:
            pp = Path(p)
            return pp if pp.is_absolute() else (self.base_dir / pp)

        """ extends / bases
        اگر در کانفیگ بنویسی:
        extends:
          - f01_config/base_common.yaml
          - f01_config/base_live.yaml
         یا به‌جای extends از bases استفاده کنی،
        این لیست‌ها به‌عنوان فایل‌های Base خوانده می‌شوند و با _deep_merge روی هم merge می‌شوند.         
        extends و bases کلیدهای اختیاری top-level هستند برای لایه‌بندی کانفیگ.
        این لایه، ضعیف ترین لایه است.
        """
        merged: Dict[str, Any] = {}
        for section in ("extends", "bases"):
            files = root_cfg.get(section, []) or []
            if not isinstance(files, list):
                raise ValueError(f"'{section}' must be a list of file paths.")
            for f in files:
                # _deep_merge(low importance dict, high importance dict)
                merged = _deep_merge(merged, _read_yaml_file(_resolve(f), fail_on_duplicates=self._fail_on_dupe_keys))

        # سپس تمام بخشهای config.yaml اصلی، غیر از extends و bases و overlays را روی bases, extends بنشانیم 
        merged = _deep_merge(merged, {k: v for k, v in root_cfg.items() if k not in ("extends", "bases", "overlays")})

        # overlays روی همه قبلی ها نوشته میشود و قویترین لایه است 
        overlays = root_cfg.get("overlays", []) or []
        if not isinstance(overlays, list):
            raise ValueError("'overlays' must be a list of file paths.")
        for f in overlays:
            merged = _deep_merge(merged, _read_yaml_file(_resolve(f), fail_on_duplicates=self._fail_on_dupe_keys))

        return merged

    def _load_yaml_layered(self) -> Dict[str, Any]:          # Added at 05-04-20 05/04/20
        """
        config.yaml اصلی را می‌خواند و اگر کلیدهای top-level زیر وجود داشت ادغام می‌کند:
          - extends: list[str]  → به ترتیب خوانده و به‌عنوان Base merge می‌کند (اولی کم‌اهمیت‌تر)
          - overlays: list[str] → در انتها روی نتیجه merge می‌کند (بالاترین اولویت)
        مسیرها می‌توانند نسبی به ریشهٔ پروژه یا مطلق باشند.
        """
        
        # خواندن کانفیگ و ذخیره آن در یک دیکشنری به نام root_cfg 
        root_cfg = _read_yaml_file(self.config_path, fail_on_duplicates=self._fail_on_dupe_keys)

        """ Absolute path
        یک مسیر مانند p را به مسیر مطلق تبدیل نموده و برمی گرداند
        """
        def _resolve(p: str) -> Path:
            pp = Path(p)
            return pp if pp.is_absolute() else (self.base_dir / pp)

        """ extends / bases
        اگر در کانفیگ بنویسی:
        extends:
          - f01_config/base_common.yaml
          - f01_config/base_live.yaml
         یا به‌جای extends از bases استفاده کنی،
        این لیست‌ها به‌عنوان فایل‌های Base خوانده می‌شوند و با _deep_merge روی هم merge می‌شوند.         
        extends و bases کلیدهای اختیاری top-level هستند برای لایه‌بندی کانفیگ.
        این لایه، ضعیف ترین لایه است.
        """


        # دریافت لایه‌ها از کلید جدید config_layers
        layers = root_cfg.get("config_layers", {})
        extends = layers.get("extends", []) or []
        bases = layers.get("bases", []) or []
        overlays = layers.get("overlays", []) or []

        merged: Dict[str, Any] = {}
        # ادغام extends (کم‌اهمیت‌ترین)
        for f in extends:
            merged = _deep_merge(merged, _read_yaml_file(_resolve(f), fail_on_duplicates=self._fail_on_dupe_keys))
        # ادغام bases (اهمیت بیشتر)
        for f in bases:
            merged = _deep_merge(merged, _read_yaml_file(_resolve(f), fail_on_duplicates=self._fail_on_dupe_keys))

        # ادغام محتوای اصلی (به جز config_layers که فقط wrapper است)
        main_content = {k: v for k, v in root_cfg.items() if k != "config_layers"}
        merged = _deep_merge(merged, main_content)

        # ادغام overlays (بالاترین اولویت)



        for f in overlays:
            merged = _deep_merge(merged, _read_yaml_file(_resolve(f), fail_on_duplicates=self._fail_on_dupe_keys))

        return merged

    # ---------- ENV name helper ----------
    def _env_name_for_path(self, path: Iterable[str]) -> str:
        """ «اسم متغیر محیطی برای این مسیر» 
        env = مخفف environment variable
       در اینجا منظور از path عبارت است از (مسیرِ کلیدها داخل ساختار کانفیگ) 
        مثال مسیر: ["mt5_credentials","login"] -> BOT_MT5_CREDENTIALS_LOGIN
        """
        return f"{self.env_prefix}{'_'.join(p.upper() for p in path)}"

    # ---------- casting ----------
    def _cast_env_value(self, val_str: str, original_value: Any):
        """
        env = مخفف environment variable
        تلاش برای تبدیل مقدار ENV به نوع مناسب:
        - "true"/"false" → bool
        - int/float
        - comma-separated list
        - otherwise: string
        """
        # bool
        s = (val_str or "").strip()
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        # int
        try:
            return int(s)
        except Exception:
            pass
        # float
        try:
            return float(s)
        except Exception:
            pass
        # list
        if "," in s:
            return [item.strip() for item in s.split(",") if item.strip() != ""]
        return s

    # ---------- اعمال ENV ----------
    def _apply_env_overrides(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        این تابع کل کانفیگ را می‌گردد و هر جا برای یک مسیر، متغیر محیطی متناظر پیدا کند،
        مقدار کانفیگ را با مقدار ENV (بعد از cast) جایگزین می‌کند.

        برای هر برگ از ساختار cfg، اگر ENV متناظر وجود داشته باشد، مقدار override می‌شود.
        جستجو با دو نام انجام می‌شود: با پیشوند (BOT_...) و بدون پیشوند (برای راحتی).
        """
        def recurse(path, node):
            if isinstance(node, dict):
                return {k: recurse(path + [k], v) for k, v in node.items()}
            if not self.enable_env_override:
                return node
            # os ماژول استاندارد سیستم‌عامل پایتون است :آموزشی 
            # ------ با پیشوند ------ 
            env_name = self._env_name_for_path(path)
            env_val = os.getenv(env_name) #از متغیرهای محیطی سیستم، مقدار متغیری به نام env_name را بخوان 
            if env_val is not None:
                try:
                    casted = self._cast_env_value(env_val, node)  #برای این مسیر، مقدار جدید این است 
                    logger.debug("ENV override %s (%s) = %r", ".".join(path), env_name, casted)
                    return casted
                except Exception:
                    logger.exception("Failed to cast env var %s", env_name)
                    return node  # اگر مقدار جدید که در فضای حافظه است، قابل cast نبود، همان مقدار node قدیم برمیگردد 
            # ------ بدون پیشوند ------ 
            env_no_prefix = "_".join(p.upper() for p in path)
            env_val2 = os.getenv(env_no_prefix)
            if env_val2 is not None:
                try:
                    return self._cast_env_value(env_val2, node)  #برای این مسیر، مقدار جدید این است 
                except Exception:
                    return node  # اگر مقدار جدید که در فضای حافظه است، قابل cast نبود، همان مقدار node قدیم برمیگردد 
            return node  #اگر در حافظه، مقدار جدیدی برای آن path وجود نداشت، همان مقدار node قدیم برمیگردد 

        return recurse([], cfg)

    # ---------- سازگاری نام کلیدها (Aliases) ----------
    def _apply_compatibility_aliases(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        نگاشت برخی نام‌های قدیمی به جدید برای جلوگیری از ناسازگاری.
        مثال‌ها:
          - evaluation.acceptance_gates.max_dd_max  →  max_drawdown_max
          - executor.canary.volume_mult             →  executor.canary_deployment.volume_multiplier
          - executor.slippage_max_pips              →  executor.slippage_cap_pips
          - training.total_steps                    →  training.total_timesteps
        """
        out = copy.deepcopy(cfg)
        try:
            # 1) acceptance gates: max_dd_max → max_drawdown_max
            gates = (((out.get("evaluation") or {}).get("acceptance_gates")) or {})
            if "max_dd_max" in gates and "max_drawdown_max" not in gates:
                gates["max_drawdown_max"] = gates.pop("max_dd_max")
                out.setdefault("evaluation", {})["acceptance_gates"] = gates

            # 2) executor aliases
            ex = out.get("executor") or {}
            # slippage
            if "slippage_max_pips" in ex and "slippage_cap_pips" not in ex:
                ex["slippage_cap_pips"] = ex.pop("slippage_max_pips")
                out["executor"] = ex
            # canary: volume_mult → canary_deployment.volume_multiplier
            if "canary" in ex and isinstance(ex["canary"], dict):
                can = ex["canary"]
                if "volume_mult" in can:
                    out.setdefault("executor", {}).setdefault("canary_deployment", {})["volume_multiplier"] = can["volume_mult"]
            # ممکن است ساختار درست already در canary_deployment باشد—دست نمی‌زنیم.

            # 3) training: total_steps → total_timesteps
            tr = out.get("training") or {}
            if "total_steps" in tr and "total_timesteps" not in tr:
                tr["total_timesteps"] = tr.pop("total_steps")
                out["training"] = tr

        except Exception:
            logger.debug("Alias application skipped", exc_info=True)
        return out

    # ---------- اعتبارسنجی حداقلی/ارتقایافته ----------
    def _validate(self, cfg: Dict[str, Any]) -> None:          # Changed at 05-04-20 05/04/20
        """
        بررسی حضور کلیدهای حیاتی و نوع داده‌ها. در صورت مشکل، ValueError می‌دهد.
        هستهٔ حیاتی (الزامی):
          - evaluation.acceptance_gates: sharpe_min, winrate_min, max_drawdown_max (float)
          - risk: risk_per_trade (float)
        توصیه‌شده/عملیاتی (بهتر است باشند؛ در نبودشان خطا نمی‌دهیم اما هشدار می‌دهیم):
          - executor.canary_deployment.enabled/volume_multiplier
          - self_optimize.schedule و steps و acceptance_gates
          - rl.algorithm, env.action_space.type
          - paths.logs_dir/models_dir/config_versions_dir
        """
        # --- حالت ولیدیشن و لیست کلیدهای مجاز سطح-۱ ---
        # توضیح:
        # این پچ هیچ کلید جدیدی به کانفیگ اضافه نمی‌کند؛ فقط ولیدیشن و حالت‌ها را پیاده می‌کند.
        # حالت پیش‌فرض warn است و با strict، هم «کلید ناشناخته» و هم «کلید تکراری YAML» خطا می‌شود.
        mode = getattr(self, "validate_mode", "warn")
        def _warn(msg: str) -> None:
            if mode == "warn":
                logger.warning("Config validation warning: %s", msg)

        allowed_top = {
            "version", "logging", "project",
            # "extends", "bases", "overlays",   # old 05-04-20 05/04/20
            "config_layers",                    # new 05-04-20 05/04/20
            "paths", "connection", "download_defaults", "features", "data_quality", "env", "risk",
            "training", "evaluation", "executor", "self_optimize",
            "monitoring", "safety", "cicd", "scripts", "secrets", "per_symbol_overrides",
            
            "account_currency", "symbol_specs",   # for symbol_specs.yaml
            "extensions", "transaction_costs",    # custom extensions / transaction-cost policy
        }
        # کلیدهای ناشناختهٔ سطح-۱
        # unknown = [k for k in cfg.keys() if k not in allowed_top and k not in ("extends","bases")]   # old 05-04-20 05/04/20
        unknown = [k for k in cfg.keys() if k not in allowed_top]                                      # new 05-04-20 05/04/20
        if unknown:
            msg = f"Unknown top-level keys: {unknown}"
            if mode == "strict":
                raise ValueError(msg)
            _warn(msg)
        # extensions اجازهٔ هر ساختاری دارد (اختیاری)
        if "extensions" in cfg and not isinstance(cfg.get("extensions"), dict):
            raise ValueError("`extensions` must be a mapping (dict)")
        if getattr(self, "allow_extensions", True) is False and "extensions" in cfg:
            msg = "`extensions` section is not allowed by current settings"
            if mode == "strict":
                raise ValueError(msg)
            _warn(msg)
        # نسخهٔ شِما/کانفیگ (string/int)
        ver = cfg.get("schema_version", cfg.get("version", None))
        if ver is not None and not isinstance(ver, (str, int, float)):
            raise ValueError("`version`/`schema_version` must be str/int/float")
        # symbol_specs (اگر merge شده باشد)
        sym = cfg.get("symbol_specs")
        if sym is not None:
            if not isinstance(sym, dict):
                raise ValueError("`symbol_specs` must be a mapping of SYMBOL -> spec")
            allowed_spec_keys = {
                "digits","point","trade_tick_value","trade_tick_size","contract_size",
                "volume_min","volume_step","volume_max","stops_level","pip_value_per_lot"
            }
            for sym_name, spec in sym.items():
                if not isinstance(spec, dict):
                    raise ValueError(f"symbol_specs.{sym_name} must be a mapping")
                extra = [k for k in spec.keys() if k not in allowed_spec_keys]
                if extra:
                    msg = f"symbol_specs.{sym_name} unknown keys: {extra}"
                    if mode == "strict":
                        raise ValueError(msg)
                    _warn(msg)
                # نوع‌های اصلی عددی/None
                for k in ("digits","point","trade_tick_value","trade_tick_size","contract_size",
                        "volume_min","volume_step","volume_max","pip_value_per_lot"):
                    if k in spec and spec[k] is not None and not isinstance(spec[k], (int, float)):
                        raise ValueError(f"symbol_specs.{sym_name}.{k} must be numeric (or null)")
                if "stops_level" in spec and spec["stops_level"] is not None and not isinstance(spec["stops_level"], (int, float)):
                    raise ValueError(f"symbol_specs.{sym_name}.stops_level must be numeric (or null)")


        # ---- هستهٔ حیاتی
        eval_ = cfg.get("evaluation") or {}
        gates = eval_.get("acceptance_gates") or {}
        for k in ("sharpe_min", "winrate_min", "max_drawdown_max"):
            if k not in gates:
                raise ValueError(f"Missing evaluation.acceptance_gates.{k}")
            try:
                float(gates[k])
            except Exception:
                raise ValueError(f"evaluation.acceptance_gates.{k} must be numeric")

        risk = cfg.get("risk") or {}
        if "risk_per_trade" not in risk:
            raise ValueError("Missing risk.risk_per_trade")
        try:
            float(risk["risk_per_trade"])
        except Exception:
            raise ValueError("risk.risk_per_trade must be numeric")

        # ---- موارد توصیه‌شده (هشدار در صورت نبود)
        def _warn(msg: str) -> None:
            logger.warning("Config validation warning: %s", msg)

        # executor.canary_deployment
        ex = cfg.get("executor") or {}
        can = ex.get("canary_deployment") or {}
        if not can:
            _warn("executor.canary_deployment is missing (recommended for safe rollout).")
        else:
            if "enabled" not in can:
                _warn("executor.canary_deployment.enabled is missing (default assumed: true/false).")
            if "volume_multiplier" not in can:
                _warn("executor.canary_deployment.volume_multiplier is missing (e.g., 0.5).")
            else:
                try:
                    float(can["volume_multiplier"])
                except Exception:
                    raise ValueError("executor.canary_deployment.volume_multiplier must be numeric")

        # self_optimize
        so = cfg.get("self_optimize") or {}
        if not so:
            _warn("self_optimize block is missing (daily/weekly pipelines won't be configurable).")
        else:
            sch = so.get("schedule") or {}
            steps = so.get("steps") or {}
            if not sch:
                _warn("self_optimize.schedule is missing (e.g., daily_utc/weekly_utc).")
            if not steps:
                _warn("self_optimize.steps is missing (data_refresh/finetune/backtest/deploy).")
            # آستانه‌های پذیرش برای self_optimize: اگر نبود از evaluation می‌گیرند
            so_g = so.get("acceptance_gates")
            if so_g and "max_drawdown_max" not in so_g:
                _warn("self_optimize.acceptance_gates exists but lacks max_drawdown_max; will fall back to evaluation gates.")

        # rl & env
        rl = ((cfg.get("training") or {}).get("rl") or {})
        if "algorithm" not in rl:
            _warn("rl.algorithm is missing (default assumed: PPO).")
        env = cfg.get("env") or {}
        act = (env.get("action_space") or {}).get("type")
        if act is None:
            _warn("env.action_space.type is missing (e.g., 'discrete').")

        # paths (در صورت حضور، نوع‌ها چک می‌شود)
        paths = cfg.get("paths") or {}
        for k in ("logs_dir", "models_dir", "config_versions_dir"):
            if k in paths and paths[k] is not None and not isinstance(paths[k], (str, Path)):
                raise ValueError(f"paths.{k} must be a string path")

    # ---------- نقطهٔ ورود بازخوانی ----------
    def reload(self) -> None:
        """
        YAML لایه‌ای را بارگذاری می‌کند → aliasing → اوورراید ENV → اعتبارسنجی → ست به self.config
        """
        raw = self._load_yaml_layered()
        # اعمال نگاشت نام‌های قدیمی به جدید 
        # raw = self._apply_compatibility_aliases(raw)   
        cfg = self._apply_env_overrides(raw)
        self._validate(cfg)
        self.config = cfg
        logger.info("Config loaded: %s (env prefix=%s)", self.config_path, self.env_prefix)

    # ---------- API عمومی ----------
    #def get(self, key: str, default: Any = None) -> Any:
        """دسترسی سریع به کلیدهای سطح-اول، مثلاً get('paths').
        به نظر میرسد هنوز در پروژه اصلی استفاده نشده است.
        """
    #    return self.config.get(key, default)

    def get_all_old(self, copy_: bool = False) -> Dict[str, Any]:
        """بازگردانی کل کانفیگ؛ با copy_=True یک کپی سطحی برمی‌گرداند."""
        return dict(self.config) if copy_ else self.config
    
    def get_all(self, copy_: Literal["main", "shallow", "mutable-safe", "deep"] = "shallow") -> Dict[str, Any]:
        """
        بازگردانی کل کانفیگ
        کانفیگ اصلی برگردانده میشود         :main
        کپی مستقل از کلیدهای سطح اول، اصل کلیدهای سطح دوم و به بعد، برگردانده میشود      :shallow
        کلیدها و زیرکلیدهای حساس بصورت کپی مستقل و الباقی کلیدها بصورت اصلی برگردانده میشود :mutable-safe
        تمام کانفیگ بصورت کپی مستقل برگردانده میشود         :deep
        """
        if copy_ == "main":
            return self.config
        elif copy_ == "shallow":
            return dict(self.config)
        elif copy_ == "mutable-safe":
            out = dict(self.config)  # shallow copy for top-level
            # بلوک‌های حساس که نباید shared باشند:
            sensitive_blocks = [
                "evaluation",
                "risk",
                "executor",
                "training",
                "self_optimize",
                "paths",
                "symbol_specs",
                "features"
            ]
            for key in sensitive_blocks:
                if key in out:
                    out[key] = copy.deepcopy(out[key])  # copy safe subtree
            return out
        elif copy_ == "deep":
            return copy.deepcopy(self.config)
        else:  # در صورت عدم وجود هر چهار کلید، کپی سطحی برگردانده میشود. 
            return dict(self.config)
        
    # ---------- ابزارها ----------
    def ensure_dirs(self) -> Dict[str, Path]:
        """
        ساخت دایرکتوری‌های مهم در صورت نبودن‌شان. مسیرهای ساخته‌شده را برمی‌گرداند.
        از این متد هنوز در پروژه اصلی استفاده نشده است و در صورت استفاده میتوان به آن اصلاحاتی اعمال نمود.
        """
        made: Dict[str, Path] = {}
        paths = self.config.get("paths") or {}
        for key in ("config_versions_dir", "cache_dir", "models_dir", "logs_dir", "reports_dir", "tmp_dir"):
            p = paths.get(key)
            if p:
                made[key] = _ensure_dir(self.base_dir / p)
        return made

    def dump(self, path: Union[str, Path], *, sort_keys: bool = False) -> Path:
        """نوشتن self.config به فایل YAML (برای خروجی‌های موقت/دیباگ).
        به نظر میرسد هنوز در پروژه اصلی استفاده نشده است.
        """
        p = Path(path)
        _ensure_dir(p.parent)   # مطمئن شدن از وجود پوشه والد 
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True, sort_keys=sort_keys)
        return p

# ============================================================
# توابع سطح ماژول (سازگاری/نسخه‌گذاری) 
# ============================================================ OK ALL
# میتوان برای تابع زیر در صورت لزوم پارامتر copy_ را اضافه نمود : Future 
def load_config(path: Optional[Union[str, Path]] = None,
                env_prefix: str = "BOT_",
                enable_env_override: bool = True,
                copy_: Literal["main", "shallow", "mutable-safe", "deep"] = "shallow"
                ) -> Dict[str, Any]:
    """
    کمک‌کاربر برای بارگذاری سریع: dict کانفیگ  برمی‌گرداند.
    اگر path ندهید، از مسیر پیش‌فرض (config.yaml) استفاده می‌شود.
    """
    loader = ConfigLoader(config_path=path, env_prefix=env_prefix, enable_env_override=enable_env_override)
    return loader.get_all(copy_= copy_)

def _infer_versions_dir(cfg: Dict[str, Any]) -> Path:
    """
    این تابع مسیر فایلهای خروجی برای تابع save_config_versioned را تعیین میکند و برمیگرداند
    """
    paths = cfg.get("paths") or {}
    base = project_root()
    versions = paths.get("config_versions_dir") or "f01_config/versions"
    return base / versions

def save_config_versioned(cfg: Dict[str, Any],
                          prefix: str = "",
                          tag: Optional[str] = None,
                          *, sort_keys: bool = False) -> Path:
    """
    ذخیرهٔ نسخه‌ای از کانفیگ در پوشهٔ versions (برای بایگانی/رول‌بک)
    نام فایل: {prefix}{utc_ts}{'_tag' if tag}.yaml
    خروجی: مسیر کامل فایل ذخیره‌شده
    """
    # قبل از ذخیره یک اعتبارسنجی انجام دهیم تا فایل معیوب ذخیره نشود
    ConfigLoader()._validate(cfg)

    versions_dir = _infer_versions_dir(cfg)
    _ensure_dir(versions_dir)

    ts = _now_utc_stamp()
    suffix = f"_{tag}" if tag else ""
    fname = f"{(prefix or '')}{ts}{suffix}.yaml"
    out_path = versions_dir / fname
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=sort_keys)
    logger.info("Config version saved at %s", out_path)
    return out_path

# ============================================================
# نمونهٔ آماده برای import سریع 
# ============================================================ OK ALL
# try:
#     config: Dict[str, Any] = ConfigLoader().get_all(copy_="shallow")
# except Exception:
#     # در صورت خطا، یک dict خالی ارائه می‌کنیم تا importهای قدیمی از کار نیفتند.
#     logger.exception("Autoload of ConfigLoader failed; `config` set to {}.")
#     config = {}

# ============================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# ============================================================
""" Func Names                                                Used in Functions: ...
                                  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21
1  project_root                  --  ok  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
2  _default_config_path          --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --  --
3  _read_yaml_file               --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --
4  _deep_merge                   --  --  --  ok  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --
5  _ensure_dir                   --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  ok
6  _now_utc_stamp                --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok
7  __init__                      --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
8  _load_yaml_layered            --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --
9  _env_name_for_path            --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --
10 _cast_env_value               --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --
11 _apply_env_overrides          --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --
12 _apply_compatibility_aliases  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- NOT USED- NOT NEEDED
13 _validate                     --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  ok
14 reload                        --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --  --
15 get                 (DELETED) --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
16 get_all                       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --
17 ensure_dirs    (EXTERNAL USE) --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
18 dump           (EXTERNAL USE) --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
19 load_config    (EXTERNAL USE) --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
20 _infer_versions_dir           --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok
21 save_config_versioned(Ex Use) --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
"""

# ============================================================
# Method of Use
# ============================================================
""" 
--- 1. Using ConfigLoader class -----------------

from config_loader import ConfigLoader
cfg = ConfigLoader().config
# or
from config_loader import ConfigLoader
cfg = ConfigLoader().get_all(copy_="shallow" | "deep" | "main" | "mutable-safe")

--- 2. Using load_config function ---------------

from config_loader import load_config
cfg = load_config(path="f01_config/config.yaml", env_prefix="myBOT_",
                  enable_env_override=True, copy_="shallow")
# or
cfg = load_config()  # uses default path and settings
# or
cfg = load_config(copy_="deep")  # deep copy of the config
# or
cfg = load_config(env_prefix="MYAPP_")  # custom env prefix
# or
cfg = load_config(enable_env_override=False)  # disable env overrides
# or
cfg = load_config(path="custom_config.yaml")  # custom config path
# or
cfg = load_config("f01_config/config.yaml", "myBOT_", True, "mutable-safe")
"""