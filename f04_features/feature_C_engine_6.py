# f04_features/feature_C_engine_6.py
# Date Reviewde:
#   1405/05/22-22:09

from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import inspect
import logging

from f03_data.mtf_dataset import MTFDataset
from f04_features.feature_C_registry_1 import get_indicator, IndicatorSpec
from f04_features.feature_sys_contract import FEATURE_SYSTEM_CONTRACT
from f02_utils.functions.parser import parse_spec, ParsedSpec

# این بخش برای train و optimize است.
# هدف: اگر یک feature قبلاً محاسبه شده، دوباره محاسبه نشود.
from f04_features.feature_B_cache_6 import (
    GLOBAL_FEATURE_CACHE_MANAGER,
    ExecutionContract,
    cached_compute,
)
logger = logging.getLogger(__name__)
# logger.addHandler(logging.NullHandler())

# =============================================================================
# ENGINE CORE (PURE FEATURE TRANSFORMER)
# =============================================================================
class FeatureEngine:

    # ========================================================================= 0
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        این متد 5 کار انجام میدهد:
            1) بررسی می‌کند config معتبر است.
            2) config را ذخیره می‌کند.
            3) mode را خالی می‌گذارد.
            4) cache مربوط به اندیکاتورهای live را آماده می‌کند.
            5) قرارداد نسخه‌بندی اجرای Engine را می‌سازد.
        """
        if config is None:
            raise ValueError("config is required for FeatureEngine")

        if not isinstance(config, dict):
            raise TypeError(f"config must be dict, got {type(config).__name__}")

        self.config: Dict[str, Any] = config
        self.mode: Optional[str] = None     # تعیین واقعی این پارامتر در تابع execute() انجام میشود.
        self._run_id: int = 0               # این پارامتر در ابتدای هر execute() یک واحد زیاد میشود.

        self._live_cache: Dict[Any, Dict[str, Any]] = {}

        self._contract = ExecutionContract(
            engine_version="1",
            resolver_version="1",
            config_version="1",
            registry_version="1",
        )
    
    # ========================================================================= 1
    def reset_live_state(self) -> None:
        """
        پاک‌سازی کامل stateهای runtime مربوط به اجرای Live.

        این متد باید فقط در مرزهای اجرای مستقل Live فراخوانی شود،
        نه بین snapshotهای متوالی بازار.
        """
        self._live_cache.clear()
        self.mode = None

        logger.info("FeatureEngine live state reset.")
        
    # ========================================================================= 2
    def execute(self, dataset: MTFDataset, specs: List[str], mode: str = "train") -> MTFDataset:
        """
        ---- old docstring---------------------------------
        این متد موتور اصلی اجرای FeatureEngine است.
        کار اصلی که این متد انجام میدهد:
            1) یک MTFDataset می‌گیرد.
            2) لیست feature specification ها را می‌گیرد.
            3) هر specification را توسط Parser تبدیل می‌کند.
            4) سپس هر feature را اجرا می‌کند (batch یا live).
            5) در نهایت همان MTFDataset را با ستون‌های feature جدید برمی‌گرداند.

        ---- new docstring---------------------------------
        اجرای FeatureEngine روی یک MTFDataset.
        مراحل:
            1) اعتبارسنجی Dataset
            2) ثبت mode و run_id
            3) Parse کردن feature specifications
            4) اجرای featureها بر اساس mode
            5) بازگرداندن MTFDataset به‌روزشده
        در حالت Live:
            - mode باید در تمام فراخوانی‌های یک session برابر "live" باشد.
            - _live_cache بین فراخوانی‌های متوالی حفظ می‌شود.
            - reset کردن stateهای Live وظیفه این متد نیست و باید
            با reset_live_state() در مرز lifecycle انجام شود.
        """
        # -------------------------------------------------
        # 1. Validate input dataset
        # -------------------------------------------------

        if dataset is None:
            raise ValueError("dataset is required")
        if not isinstance(dataset, MTFDataset):
            raise TypeError(
                f"Expected MTFDataset, got {type(dataset).__name__}"
            )

        # -------------------------------------------------
        # 2. Validate / normalize execution mode
        # -------------------------------------------------

        mode = FEATURE_SYSTEM_CONTRACT.validate_mode(mode)

        if len(dataset.frames) == 0:
            logger.info("MTFDataset is empty.")
            return dataset

        # -------------------------------------------------
        # 3. Validate specs
        # -------------------------------------------------

        if specs is None:
            raise ValueError("specs is required")
        if not specs:
            return dataset

        # -------------------------------------------------
        # 4. Set execution context
        # -------------------------------------------------
        self.mode = mode
        self._run_id += 1


        # -------------------------------------------------
        # 5. Parse specifications
        # -------------------------------------------------
        parsed_specs: List[ParsedSpec] = []
        for raw_spec in specs:
            try:
                parsed_specs.append(parse_spec(spec=raw_spec, mode=mode))
            except Exception as exc:
                logger.warning("Invalid feature spec skipped: %s | %s", raw_spec, exc)

        if not parsed_specs:
            raise ValueError("No valid feature specifications were parsed.")

        # -------------------------------------------------
        # 6. Execution order
        # -------------------------------------------------
        """
        در اینجا میشود که مرتب‌سازی بر اساس dependency
        یا اجرای featureهای پایه قبل از feature های وابسته را گنجانید
        """
        ordered_specs = parsed_specs

        # -------------------------------------------------
        # 7. Execute features
        # -------------------------------------------------
        for ps in ordered_specs:
            if mode in {"train", "optimize"}:
                dataset = cached_compute(
                    # cache=GLOBAL_FEATURE_CACHE,
                    cache=GLOBAL_FEATURE_CACHE_MANAGER.get(dataset.symbol), # type: FeatureCache
                    dataset=dataset,                                        # type: 
                    ps=ps,                                                  # type: ParsedSpec
                    mode=mode,                                              # type: str
                    contract=self._contract,                                # type: ExecutionContract
                    # در سطر زیر محاسبات بچ انجام میشود
                    compute_fn=lambda ds=dataset, p=ps:                     # type: Callable[[], MTFDataset]
                        self._apply_spec(dataset=ds, ps=p, mode=mode),
                )
            else:
                dataset = self._apply_spec(dataset=dataset, ps=ps, mode=mode)

        logger.info("Feature calculation completed on %d timeframes.", len(dataset.frames))
        return dataset

    # ========================================================================= 3
    def _apply_spec(self, dataset: MTFDataset, ps: ParsedSpec, mode: str) -> MTFDataset:
        """
        Execute one feature according to the Registry contract.
            ParsedSpec
                ↓
            get_indicator(name, mode)
                ↓
            IndicatorSpec
                ↓
            dataset.get(timeframe)
                ↓
            _validate_contract()
                ↓
            Batch → _call_batch()
            Live  → _apply_live()
        """

        spec = get_indicator(ps.name, mode)
        if spec is None:
            logger.warning("Indicator not found in registry: %s (mode=%s)", ps.name, mode)
            return dataset

        tf = ps.timeframe
        if tf is None:
            logger.warning("Spec has no timeframe: %s", ps.raw)
            return dataset

        tf = tf.upper()
        try:
            df = dataset.get(tf)
        except KeyError:
            logger.warning("Dataset has no dataframe for timeframe %s (indicator=%s)", tf, ps.name)
            return dataset

        if df.empty:
            logger.warning("Dataset dataframe is empty for timeframe %s (indicator=%s)", tf, ps.name)
            return dataset

        if not self._validate_contract(
            spec=spec,
            df=df,
            mode=mode,
            timeframe=tf,
        ):
            return dataset

        if spec.is_batch_mode(mode):
            return self._call_batch(
                spec=spec,
                dataset=dataset,
                ps=ps,
            )

        if spec.is_incremental_mode(mode):
            out_df = self._apply_live(
                spec=spec,
                dataset=dataset,
                df=df,
                ps=ps,
            )
            dataset.replace(tf, out_df)
            return dataset

        logger.warning("Unsupported execution mode '%s' for indicator '%s'", mode, ps.name)
        return dataset

    # ========================================================================= 4
    def _call_batch(self, spec: IndicatorSpec, dataset: MTFDataset, ps: ParsedSpec) -> MTFDataset:
        """
        Execute one Batch indicator using the validated Registry contract.
        """

        tf = ps.timeframe

        if tf is None:
            raise ValueError(f"Batch execution requires timeframe: {ps.raw}")

        tf = tf.upper()
        df = dataset.get(tf)

        kwargs = dict(ps.kwargs)                                  # old
        # for param in spec.parameters:
        #     if param.name not in kwargs:
        #         for alias in param.aliases:
        #             if alias in kwargs:
        #                 kwargs[param.name] = kwargs.pop(alias)
        #                 break
        out = spec.fn(df, **kwargs)                               # old

        # try:                                                        # new for debug only
        #     out = spec.fn(df, **kwargs)                             # new for debug only
        #     print("DEBUG MACD OUTPUT")
        #     print(out.tail(10))
        #     print(out.isna().sum())
        # except TypeError as e:                                      # new for debug only
        #     if "unexpected keyword argument 'column'" in str(e):    # new for debug only
        #         kwargs.pop("column", None)                          # new for debug only
        #         out = spec.fn(df, **kwargs)                         # new for debug only
        #         print("DEBUG MACD OUTPUT")
        #         print(out.tail(10))
        #         print(out.isna().sum())
        #     else:                                                   # new for debug only
        #         raise                                               # new for debug only



        if out is None:
            return dataset

        if not isinstance(out, pd.DataFrame):
            raise TypeError(
                f"Batch indicator '{ps.name}' must return DataFrame, "
                f"got {type(out).__name__}."
            )

        out = out.copy()

        output_names = spec.output_names
        canonical = ps.canonical

        if output_names:
            if len(output_names) != len(out.columns):
                raise ValueError(
                    f"Output count mismatch for '{ps.name}': "
                    f"registry defines {len(output_names)} outputs, "
                    f"function returned {len(out.columns)}."
                )

            if len(output_names) == 1:
                out.columns = [canonical]
            else:
                params_part = canonical[len(ps.name):]
                out.columns = [
                    f"{name}{params_part}"
                    for name in output_names
                ]

        elif len(out.columns) == 1:
            out.columns = [canonical]
        else:
            out.columns = [
                f"{canonical}::{column}"
                for column in out.columns
            ]
        dataset.replace(tf, self._merge(df, out))
        return dataset

    # ========================================================================= 5
    def _apply_live(
        self,
        spec: IndicatorSpec,
        dataset: MTFDataset,
        df: pd.DataFrame,
        ps: ParsedSpec,
    ) -> pd.DataFrame:
        """
        اجرای incremental یک indicator stateful روی snapshot فعلی.

        رفتار:
            - اولین snapshot:
                کل پنجره موجود به عنوان warmup پردازش می‌شود.
            - snapshotهای بعدی:
                فقط ردیف‌هایی که timestamp آنها از آخرین timestamp
                پردازش‌شده جدیدتر است، به indicator داده می‌شوند.
            - اگر آخرین timestamp پردازش‌شده دیگر در snapshot فعلی
            وجود نداشته باشد، state از نو ساخته می‌شود و snapshot
            فعلی دوباره به عنوان warmup پردازش می‌شود.

        علاوه بر state خود indicator، خروجی‌های محاسبه‌شده نیز نگهداری
        می‌شوند تا featureهای موجود در پنجره فعلی قابل بازسازی باشند.
        """

        tf = ps.timeframe.upper()

        key = (
            self.mode,
            dataset.symbol,
            dataset.base_tf,
            tf,
            ps.canonical,
        )

        logger.info(
            "Applying incremental feature %s (mode=%s)",
            ps.name,
            self.mode,
        )

        # -------------------------------------------------
        # 1. Create runtime state on first use
        # -------------------------------------------------
        state = self._live_cache.get(key)

        if state is None:
            state = {
                "instance": self._build_live_instance(spec, ps),
                "initialized": False,
                "last_timestamp": None,
                "outputs": {},
            }
            self._live_cache[key] = state

        obj = state["instance"]
        last_timestamp = state["last_timestamp"]

        # -------------------------------------------------
        # 2. Ensure chronological order
        # -------------------------------------------------
        df = df.sort_index()

        # -------------------------------------------------
        # 3. Determine rows that must be fed to the indicator
        # -------------------------------------------------
        if not state["initialized"]:
            rows_to_process = df

        elif last_timestamp is None:
            rows_to_process = df
            state["initialized"] = False

        elif last_timestamp not in df.index:
            # The previously processed candle is no longer inside
            # the available warmup window.
            # Therefore incremental continuity cannot be proven.
            logger.info(
                "Live state continuity lost for %s. "
                "Reinitializing from current snapshot.",
                ps.canonical,
            )

            state["instance"] = self._build_live_instance(spec, ps)
            state["outputs"] = {}
            state["initialized"] = False
            state["last_timestamp"] = None

            obj = state["instance"]
            rows_to_process = df

        else:
            # Normal incremental case:
            # only genuinely new candles are fed to the stateful indicator.
            rows_to_process = df.loc[df.index > last_timestamp]

        # -------------------------------------------------
        # 4. Feed required rows to indicator
        # -------------------------------------------------
        for timestamp, row in rows_to_process.iterrows():
            try:
                output = self._update_live(
                    obj=obj,
                    row=row,
                    spec=spec,
                    ps=ps,
                )

                if isinstance(output, dict):
                    state["outputs"][timestamp] = output

                state["last_timestamp"] = timestamp
                state["initialized"] = True

            except Exception:
                logger.exception(
                    "Incremental update failed for %s at %s",
                    ps.raw,
                    timestamp,
                )

        # -------------------------------------------------
        # 5. Keep only outputs that belong to the current snapshot
        # -------------------------------------------------
        current_index = set(df.index)

        state["outputs"] = {
            timestamp: output
            for timestamp, output in state["outputs"].items()
            if timestamp in current_index
        }

        # -------------------------------------------------
        # 6. Reconstruct feature columns for the current snapshot
        # -------------------------------------------------
        return self._attach_live_output(
            df=df,
            ps=ps,
            spec=spec,
            outputs_by_timestamp=state["outputs"],
        )

    # ========================================================================= 6
    def _update_live(self, obj: Any, row: pd.Series, spec: IndicatorSpec, ps: ParsedSpec):
        """
        Feed one row into a stateful indicator.
        Input order is defined by the Registry's required_cols contract.
        """
        values = []
        required_cols = list(spec.required_cols)
        selected_column = ps.kwargs.get("column")

        if selected_column:
            required_cols = [selected_column]

        tf = ps.timeframe.upper() if ps.timeframe else None

        for source in required_cols:

            if source in row.index:
                values.append(row[source])
                continue

            if tf is not None:
                prefixed = f"{tf}_{source}"

                if prefixed in row.index:
                    values.append(row[prefixed])
                    continue

            logger.warning(
                "Missing live input column '%s' for %s",
                source,
                spec.name,
            )
            return None

        raw = obj.update(*values)

        return self._normalize_output(
            spec=spec,
            output=raw,
        )

    # ========================================================================= 7
    def _build_live_instance(self, spec: IndicatorSpec, ps: ParsedSpec) -> Any:
        """
        Build one stateful Live indicator instance.

        Parser has already normalized, validated and defaulted
        all Registry parameters.
        """

        kwargs = dict(ps.kwargs)

        return spec.fn(**kwargs)

    # ========================================================================= 8
    def _merge(self, df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        """
        Merge feature columns without modifying the input frame.
        """

        if out is None or out.empty:
            return df

        result = df.copy()

        for col in out.columns:
            if col not in result.columns:
                result[col] = out[col]
                continue

            if result[col].equals(out[col]):
                continue

            raise ValueError(
                f"Feature column collision: '{col}'"
            )

        return result

    # ========================================================================= 9
    def _attach_live_output(
        self,
        df: pd.DataFrame,
        ps: ParsedSpec,
        spec: IndicatorSpec,
        outputs_by_timestamp: Dict[Any, Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        اتصال خروجی‌های محاسبه‌شده‌ی Live Indicator به snapshot فعلی.

        خروجی‌های indicator بر اساس timestamp به DataFrame متصل می‌شوند
        و سپس از قرارداد عمومی _merge برای مدیریت collision عبور می‌کنند.
        """

        if not outputs_by_timestamp:
            return df

        feature_data: Dict[str, Dict[Any, Any]] = {}

        for timestamp, outputs in outputs_by_timestamp.items():
            if not isinstance(outputs, dict):
                continue
            for output_name, value in outputs.items():
                column_name = self._build_live_column_name(
                    output_name=output_name,
                    ps=ps,
                )
                if column_name not in feature_data:
                    feature_data[column_name] = {}
                feature_data[column_name][timestamp] = value

        if not feature_data:
            return df

        out = pd.DataFrame(feature_data, index=df.index)
        return self._merge(df, out)

    # ========================================================================= 10
    def _build_live_column_name(self, output_name: str, ps: ParsedSpec) -> str:
        canonical = ps.canonical

        if output_name == ps.name:
            return canonical

        params_part = canonical[len(ps.name):]

        return f"{output_name}{params_part}"

    # ========================================================================= 11
    def _normalize_output(self, spec: IndicatorSpec, output: Any) -> Dict[str, Any]:

        if isinstance(output, dict):
            return output

        names = getattr(spec, "output_names", None)

        if not names:
            return {
                spec.name: output,
            }

        if isinstance(output, (tuple, list)):
            return {
                name: output[i] if i < len(output) else None
                for i, name in enumerate(names)
            }

        if len(names) == 1:
            return {
                names[0]: output,
            }

        raise TypeError(
            f"Indicator '{spec.name}' returned scalar output "
            f"but defines multiple output_names."
        )

    # ========================================================================= 12
    def _validate_contract(self, spec: IndicatorSpec, df: pd.DataFrame, mode: str, timeframe: str) -> bool:
        if not spec.supports(mode):
            logger.warning(
                "Contract violation [%s]: mode '%s' is not supported",
                spec.name,
                mode,
            )
            return False

        missing = []

        for column in spec.required_cols:
            if column in df.columns:
                continue

            prefixed = f"{timeframe}_{column}"

            if prefixed in df.columns:
                continue

            missing.append(column)

        if missing:
            logger.warning(
                "Contract violation [%s]: missing columns %s",
                spec.name,
                missing,
            )
            return False

        return True

    # ========================================================================= END


""" Methods of FeatureEngine class:
__init__
reset_live_state
execute
_apply_spec
_call_batch
_apply_live
_update_live
_build_live_instance
_merge
_attach_live_output
_build_live_column_name
_normalize_output
_validate_contract
"""