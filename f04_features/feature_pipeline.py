# f04_features/feature_pipeline.py
# Date Reviewde:
#   1405/05/22-22:09

from __future__ import annotations
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any

from f03_data.mtf_dataset import MTFDataset
from f04_features.feature_B_graph import FeatureGraph
from f04_features.feature_C_engine_6 import FeatureEngine
from f04_features.feature_B_store import FeatureStoreV2
from f04_features.observation_B_builder import ObservationBuilder

from f04_features.time_features.time_feature_engine import apply_time_features, add_time_features_to_live_df

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """ اتصال رسمی بین لایه Data و Features.

    جریان داده برای حالت batch:
    ----------------------------
                DataHandler
                    ↓
                MTFDataset
                    ↓
                _apply_time_features()
                    ↓
                Time Features فقط روی base_tf
                    ↓
                FeatureEngine.execute()
                    ↓
                FeatureStore
                    ↓
                ObservationBuilder
                    ↓
                Observation

    و جریان داده برای حالت live:
    -----------------------------
                DataHandler
                    ↓
                MTFDataset
                    ↓
                _apply_time_features_live()
                    ↓
                Time Features فقط روی base_tf
                    ↓
                FeatureEngine.execute(mode="live")
                    ↓
                FeatureStore
                    ↓
                ObservationBuilder
                    ↓
                Observation
    """

    # ========================================================================= 1 بررسی شد و فهمیده شد
    def __init__(
        self,
        symbol: str,
        feature_engine: FeatureEngine,
        feature_store: FeatureStoreV2,
        observation_builder: ObservationBuilder,
        feature_graph: FeatureGraph,
        feature_specs: List[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        ایجاد FeaturePipeline مستقل برای یک نماد.

        FeaturePipeline یک Symbol را مدیریت می‌کند و MTFDataset همان نماد
        را از Data Layer دریافت کرده و آن را از مسیر FeatureEngine،
        FeatureStore و ObservationBuilder عبور می‌دهد.
        """

        # -----------------------------------------------------------
        # Dependencies
        # -----------------------------------------------------------
        self.feature_engine: FeatureEngine = feature_engine
        self.feature_store: FeatureStoreV2 = feature_store
        self.observation_builder: ObservationBuilder = observation_builder
        self.feature_graph: FeatureGraph = feature_graph

        # -----------------------------------------------------------
        # Symbol / Configuration
        # -----------------------------------------------------------
        self.symbol: str = symbol
        self.feature_specs: List[str] = list(feature_specs)
        self.config: Dict[str, Any] = config or {}

        # -----------------------------------------------------------
        # Runtime state
        # -----------------------------------------------------------
        self._dataset: Optional[MTFDataset] = None
        self._features: Optional[MTFDataset] = None
        self._observation = None
        self._time_feature_previous_timestamp: pd.Timestamp | None = None  # new2 کُس کِش

        logger.info(
            "FeaturePipeline initialized for symbol=%s",
            self.symbol,
        )

    # ========================================================================= 2 بررسی شد و فهمیده شد
    def reset_live_state(self) -> None:
        """
        Reset live runtime state for this Symbol.
        """
        self.feature_engine.reset_live_state()

        # -----------------------------------------------------------
        # reset live runtime state
        # -----------------------------------------------------------
        self._time_feature_previous_timestamp = None
        self._dataset = None
        self._features = None
        self._observation = None

    # ========================================================================= new added at 050617
    def _apply_time_features(
        self,
        dataset: MTFDataset,
    ) -> MTFDataset:
        """
        Apply configured time features to the current symbol.

        Configuration:
            features.time_features.enabled
            features.time_features.symbols.<SYMBOL>

        The time-feature engine intentionally applies the selected
        features only to dataset.base_tf.
        """

        if self.config is None:
            return dataset

        time_features_cfg = (
            self.config
            .get("features", {})
            .get("time_features", {})
            or {}
        )

        if not time_features_cfg.get("enabled", False):
            return dataset

        symbols_cfg = time_features_cfg.get("symbols", {}) or {}
        requested_features = symbols_cfg.get(self.symbol.upper(), [])

        if not requested_features:
            return dataset

        return apply_time_features(
            dataset=dataset,
            time_feature_config={
                self.symbol.upper(): list(requested_features),
            },
        )

    # ========================================================================= new added at 050617
    def _apply_time_features_live(
        self,
        dataset: MTFDataset,
    ) -> MTFDataset:
        """
        Apply configured time features to the base timeframe
        for live processing.

        The live API is used so stateful timestamp-derived features
        such as is_new_day can use the previous live timestamp.
        """

        time_features_cfg = (
            self.config
            .get("features", {})
            .get("time_features", {})
            or {}
        )

        if not time_features_cfg.get("enabled", False):
            return dataset

        symbols_cfg = time_features_cfg.get("symbols", {}) or {}

        requested_features = symbols_cfg.get(
            self.symbol.upper(),
            [],
        )

        if not requested_features:
            return dataset

        base_tf = dataset.base_tf.upper()
        df = dataset.get(base_tf)

        if df is None or df.empty:
            return dataset

        updated_df = add_time_features_to_live_df(
            df=df,
            features=list(requested_features),
            timeframe=base_tf,
            previous_timestamp=self._time_feature_previous_timestamp,
        )

        self._time_feature_previous_timestamp = updated_df.index[-1]

        dataset.add(
            base_tf,
            updated_df,
        )

        return dataset

    # ========================================================================= 3 بررسی شد و فهمیده شد
    def run(
        self,
        dataset: MTFDataset,
        *,
        mode: str = "train",
        save_features: bool = False,
        save_dir: Optional[str] = None,
        save_name: str = "features",
    ) -> Dict[str, Any]:
        """
        اجرای کامل Pipeline برای یک Symbol.

        جریان اجرا:
            MTFDataset
                ↓
            FeatureEngine
                ↓
            FeatureStore
                ↓
            ObservationBuilder
                ↓
            Observation

        پارامترها:
            dataset:
                دیتاست چندتایم‌فریمی مربوط به Symbol جاری.
            mode:
                حالت اجرای FeatureEngine، مانند train، optimize یا live.
            save_features:
                در صورت True، FeatureStore نیز روی دیسک ذخیره می‌شود.
            save_dir:
                مسیر ذخیره FeatureStore.
            save_name:
                نام خروجی ذخیره‌شده.

        خروجی:
            دیکشنری شامل:
                dataset
                features
                metadata
                observation
        
        کارهایی که این متد انجام میدهد:
            1) اعتبار سنجی دیتاست ورودی
            2) افزودن فیچرهای زمانی به دیتاست
            3) انجام محاسبه فیچرها توسط feature_engine.execute()
            4) ساخت فیچر استور توسط build_feature_store() که ان هم از متد feature_store.build() استفاده میکند
               ساخت matadata
            5) ذخیره نمودن فیچر استور
            6) ساخت observation
            7) ساخت و بازگرداندن دیکشنری خروجی
        """

        # -----------------------------------------------------------
        # 1. Validation of input dataset
        # -----------------------------------------------------------
        if dataset is None:
            raise ValueError("dataset is None")

        if not isinstance(dataset, MTFDataset):
            raise TypeError(
                f"Expected MTFDataset, got {type(dataset).__name__}"
            )

        if dataset.symbol != self.symbol:
            raise ValueError(
                f"Dataset symbol '{dataset.symbol}' does not match "
                f"pipeline symbol '{self.symbol}'."
            )
        logger.debug(
            "Running FeaturePipeline for symbol=%s, mode=%s",
            self.symbol,
            mode,
        )

        # -----------------------------------------------------------
        # 2. Add time features
        # -----------------------------------------------------------
        dataset = self._apply_time_features(dataset)
        
        # -----------------------------------------------------------
        # 3. Feature calculation
        # -----------------------------------------------------------
        features = self.feature_engine.execute(
            dataset=dataset,
            specs=self.feature_specs,
            mode=mode,
        )
        if features is None:
            raise RuntimeError("FeatureEngine returned None.")

        # -----------------------------------------------------------
        # 4. Feature Store                      --- OLD
        # 4. Build final FeatureStore dataset   --- NEW
        # -----------------------------------------------------------
        # feature_dataset = self.feature_store.build(      # OLD
        #     dataset=dataset,                             # OLD
        #     features=features,                           # OLD
        # )                                                # OLD
        feature_dataset = self.build_feature_store(
            raw_dataset=dataset,
            feature_dataset=features,
        )
        metadata = self.feature_store.extract_metadata(
            feature_dataset
        )

        # -----------------------------------------------------------
        # 5. Optional persistence
        # -----------------------------------------------------------
        if save_features:
            if save_dir is None:
                raise ValueError(
                    "save_dir must be provided when save_features=True"
                )
            # self.feature_store.save(       # OLD
            #     dataset=feature_dataset,   # OLD
            #     metadata=metadata,         # OLD
            #     out_dir=save_dir,          # OLD
            #     name=save_name,            # OLD
            # )                              # OLD
            self.save_feature_store(
                feature_dataset,
                out_dir=save_dir,
                name=save_name,
            )

        # -----------------------------------------------------------
        # 6. Observation
        # -----------------------------------------------------------
        # observation = self.observation_builder.build(    # OLD
        #     feature_dataset,                             # OLD
        #     self.feature_graph,                          # OLD
        # )                                                # OLD
        # if observation is None:                          # OLD
        #     raise RuntimeError(                          # OLD
        #         "ObservationBuilder returned None."      # OLD
        #     )                                            # OLD
        observation = self.build_observation(
            feature_dataset
        )

        # -----------------------------------------------------------
        # 7. Update runtime state
        # -----------------------------------------------------------
        self._dataset = dataset
        self._features = feature_dataset
        self._observation = observation

        logger.info(
            "FeaturePipeline finished successfully for symbol=%s",
            self.symbol,
        )

        return {
            "dataset": dataset,
            "features": feature_dataset,
            "metadata": metadata,
            "observation": observation,
        }

    # ========================================================================= 4 بررسی شد و فهمیده شد
    def process_live(
        self,
        dataset: MTFDataset,
    ):
        """
        اجرای کامل Pipeline برای یک آپدیت Live مربوط به یک Symbol.

        جریان:
            MTFDataset
                ↓
            Time Features
                ↓
            FeatureEngine
                ↓
            FeatureStore
                ↓
            ObservationBuilder
                ↓
            Observation
        """

        # -----------------------------------------------------------
        # 1. Validation of input dataset
        # -----------------------------------------------------------
        if dataset is None:
            raise ValueError("dataset is None")

        if not isinstance(dataset, MTFDataset):
            raise TypeError(
                f"Expected MTFDataset, got {type(dataset).__name__}"
            )

        if dataset.symbol != self.symbol:
            raise ValueError(
                f"Dataset symbol '{dataset.symbol}' does not match "
                f"pipeline symbol '{self.symbol}'."
            )

        # -----------------------------------------------------------
        # 2. Add time features
        # -----------------------------------------------------------
        dataset = self._apply_time_features_live(dataset)

        # -----------------------------------------------------------
        # 3. Incremental feature calculation
        # -----------------------------------------------------------
        features = self.feature_engine.execute(
            dataset=dataset,
            specs=self.feature_specs,
            mode="live",
        )
        
        if features is None:
            return None

        # -----------------------------------------------------------
        # 4. Build feature dataset
        # -----------------------------------------------------------
        feature_dataset = self.feature_store.build(
            dataset=dataset,
            features=features,
        )

        # -----------------------------------------------------------
        # 5. Build observation
        # -----------------------------------------------------------
        # observation = self.observation_builder.build(    # OLD
        #     feature_dataset,                             # OLD
        #     self.feature_graph,                          # OLD
        # )                                                # OLD
        # if observation is None:                          # OLD
        #     raise RuntimeError(                          # OLD
        #         "ObservationBuilder returned None."      # OLD
        #     )                                            # OLD
        observation = self.build_observation(
            feature_dataset
        )

        # -----------------------------------------------------------
        # 6. Update runtime state
        # -----------------------------------------------------------
        self._dataset = dataset
        self._features = feature_dataset
        self._observation = observation

        return observation

    # ========================================================================= 5 بررسی شد و فهمیده شد
    def build_observation(
        self,
        feature_dataset: MTFDataset,
    ):
        """
        ساخت Observation برای Symbol جاری از روی Feature Dataset.

        پارامترها:
            feature_dataset:
                دیتاست چندتایم‌فریمی حاوی Featureهای محاسبه‌شده.

        خروجی:
            Observation ساخته‌شده توسط ObservationBuilder.
        """
        # -----------------------------------------------------------
        # Validation of input
        # -----------------------------------------------------------
        if feature_dataset is None:
            raise ValueError("feature_dataset is None")

        if not isinstance(feature_dataset, MTFDataset):
            raise TypeError(
                f"Expected MTFDataset, got {type(feature_dataset).__name__}"
            )

        if feature_dataset.symbol != self.symbol:
            raise ValueError(
                f"Feature dataset symbol '{feature_dataset.symbol}' "
                f"does not match pipeline symbol '{self.symbol}'."
            )

        # -----------------------------------------------------------
        # Main calculation
        # -----------------------------------------------------------
        observation = self.observation_builder.build(
            feature_dataset,
            self.feature_graph,
        )

        # -----------------------------------------------------------
        # Validation of output
        # -----------------------------------------------------------
        if observation is None:
            raise RuntimeError(
                "ObservationBuilder returned None."
            )

        self._observation = observation

        return observation

    # ========================================================================= 6 بررسی شد و فهمیده شد
    def build_feature_store(
        self,
        raw_dataset: MTFDataset,
        feature_dataset: MTFDataset,
    ) -> MTFDataset:
        """
        ساخت Feature Dataset نهایی برای Symbol جاری.

        پارامترها:
            raw_dataset:
                MTFDataset خام مربوط به Symbol جاری.

            feature_dataset:
                MTFDataset حاصل از FeatureEngine.

        خروجی:
            MTFDataset:
                دیتاست نهایی ساخته‌شده توسط FeatureStore.
        """
        # -----------------------------------------------------------
        # Validation of inputs
        # -----------------------------------------------------------
        if raw_dataset is None:
            raise ValueError("raw_dataset is None")

        if feature_dataset is None:
            raise ValueError("feature_dataset is None")

        if not isinstance(raw_dataset, MTFDataset):
            raise TypeError(
                f"Expected raw_dataset to be MTFDataset, "
                f"got {type(raw_dataset).__name__}"
            )

        if not isinstance(feature_dataset, MTFDataset):
            raise TypeError(
                f"Expected feature_dataset to be MTFDataset, "
                f"got {type(feature_dataset).__name__}"
            )

        if raw_dataset.symbol != self.symbol:
            raise ValueError(
                f"Raw dataset symbol '{raw_dataset.symbol}' "
                f"does not match pipeline symbol '{self.symbol}'."
            )

        if feature_dataset.symbol != self.symbol:
            raise ValueError(
                f"Feature dataset symbol '{feature_dataset.symbol}' "
                f"does not match pipeline symbol '{self.symbol}'."
            )

        # -----------------------------------------------------------
        # Main calculation
        # -----------------------------------------------------------
        result = self.feature_store.build(
            raw_dataset,
            feature_dataset,
        )

        # -----------------------------------------------------------
        # Validation of output
        # -----------------------------------------------------------
        if result is None:
            raise RuntimeError(
                "FeatureStore returned None."
            )

        if not isinstance(result, MTFDataset):
            raise TypeError(
                f"FeatureStore must return MTFDataset, "
                f"got {type(result).__name__}"
            )

        if result.symbol != self.symbol:
            raise ValueError(
                f"FeatureStore result symbol '{result.symbol}' "
                f"does not match pipeline symbol '{self.symbol}'."
            )

        return result

    # ========================================================================= 7 بررسی شد و فهمیده شد
    def save_feature_store(
        self,
        dataset: MTFDataset,
        *,
        out_dir: str | Path,
        name: str,
        fmt: str = "parquet",
    ):
        """
        ذخیره Feature Dataset و metadata مربوط به Symbol جاری.

        پارامترها:
            dataset:
                MTFDataset نهایی ساخته‌شده توسط FeatureStore.
            out_dir:
                مسیر خروجی ذخیره‌سازی.
            name:
                نام فایل/مجموعه خروجی.
            fmt:
                فرمت ذخیره‌سازی، مانند parquet یا csv.

        خروجی:
            نتیجه‌ی بازگردانده‌شده توسط FeatureStore.save().
        """
        # -----------------------------------------------------------
        # Validation of input
        # -----------------------------------------------------------
        if dataset is None:
            raise ValueError("dataset is None")

        if not isinstance(dataset, MTFDataset):
            raise TypeError(
                f"Expected dataset to be MTFDataset, "
                f"got {type(dataset).__name__}"
            )

        if dataset.symbol != self.symbol:
            raise ValueError(
                f"Dataset symbol '{dataset.symbol}' "
                f"does not match pipeline symbol '{self.symbol}'."
            )

        # -----------------------------------------------------------
        # Main calculation
        # -----------------------------------------------------------
        metadata = self.feature_store.extract_metadata(dataset)

        return self.feature_store.save(
            dataset=dataset,
            metadata=metadata,
            out_dir=out_dir,
            name=name,
            fmt=fmt,
        )

    # ========================================================================= 8 بررسی شد و فهمیده شد
    def export(
        self,
        out_dir: str | Path,
        name: str,
        fmt: str = "parquet",
    ) -> Dict[str, Any]:
        """
        ساخت و ذخیره FeatureStore برای Symbol جاری.

        از آخرین raw dataset و feature dataset موجود در Pipeline
        استفاده می‌کند و FeatureStore نهایی را هم می‌سازد و هم ذخیره می‌کند.

        خروجی:
            نتیجه‌ی بازگردانده‌شده توسط FeatureStore.save().
        """

        if self._dataset is None:
            raise RuntimeError(
                "Dataset is empty. Run the pipeline first."
            )

        if self._features is None:
            raise RuntimeError(
                "Feature dataset is empty. Run the pipeline first."
            )

        feature_dataset = self.build_feature_store(
            raw_dataset=self._dataset,
            feature_dataset=self._features,
        )

        return self.save_feature_store(
            feature_dataset,
            out_dir=out_dir,
            name=name,
            fmt=fmt,
        )

    # ========================================================================= 9 بررسی شد و فهمیده شد
    def build_numpy_observation(
        self,
        feature_dataset: MTFDataset,
    ):
        """
        ساخت Observation با فرمت NumPy برای Agent مربوط به Symbol جاری.

        پارامترها:
            feature_dataset:
                MTFDataset حاوی Featureهای محاسبه‌شده.

        خروجی:
            خروجی NumPy تولیدشده توسط ObservationBuilder.
        """
        # -----------------------------------------------------------
        # Validation of input
        # -----------------------------------------------------------
        if feature_dataset is None:
            raise ValueError("feature_dataset is None")

        if not isinstance(feature_dataset, MTFDataset):
            raise TypeError(
                f"Expected feature_dataset to be MTFDataset, "
                f"got {type(feature_dataset).__name__}"
            )

        if feature_dataset.symbol != self.symbol:
            raise ValueError(
                f"Feature dataset symbol '{feature_dataset.symbol}' "
                f"does not match pipeline symbol '{self.symbol}'."
            )

        # -----------------------------------------------------------
        # Main calculations
        # -----------------------------------------------------------
        observation = self.observation_builder.build_numpy(
            feature_dataset,
            self.feature_graph,
        )

        if observation is None:
            raise RuntimeError(
                "ObservationBuilder.build_numpy() returned None."
            )

        return observation

    # ========================================================================= 10 بررسی شد و فهمیده شد
    def rebuild_graph(
        self,
        specs: Optional[List[str]] = None,
    ) -> None:
        """
        بازسازی FeatureGraph برای Symbol جاری.

        اگر specs ارائه شود، feature specifications داخلی Pipeline
        نیز با آن جایگزین می‌شود؛ در غیر این صورت specifications
        فعلی Pipeline حفظ می‌شوند.
        """

        if specs is not None:
            self.feature_specs = list(specs)

        self.feature_graph = FeatureGraph(
            self.feature_specs
        )

    # ========================================================================= 11 بررسی شد و فهمیده شد
    def reload_config(
        self,
        config: Dict[str, Any],
    ) -> None:
        """
        بارگذاری مجدد configuration و بازسازی وابستگی‌های وابسته به آن.

        توجه:
            Symbol جاری Pipeline تغییر نمی‌کند.
            Feature specifications از configuration جدید خوانده می‌شوند.
        """

        # -----------------------------------------------------------
        # Validation of input
        # -----------------------------------------------------------
        if config is None:
            raise ValueError("config is required")

        if not isinstance(config, dict):
            raise TypeError(
                f"config must be dict, got {type(config).__name__}"
            )

        self.config = config

        # -----------------------------------------------------------
        # 1- Feature specifications
        # -----------------------------------------------------------
        self.feature_specs = list(
            config.get("features", {}).get("live_specs", [])    ###       <<=== آدرس باید اصلاح شود
        )

        # -----------------------------------------------------------
        # 2- Rebuild FeatureGraph
        # -----------------------------------------------------------
        self.feature_graph = FeatureGraph(
            self.feature_specs
        )

        # -----------------------------------------------------------
        # 3- Rebuild configuration-dependent components
        # -----------------------------------------------------------
        self.feature_engine = FeatureEngine(config)
        self.observation_builder = ObservationBuilder(config)

    # ========================================================================= 12 بررسی شد و فهمیده شد
    def info(self) -> Dict[str, Any]:
        """
        بازگرداندن اطلاعات وضعیت FeaturePipeline برای Symbol جاری.
        """
        return {
            "symbol": self.symbol,
            "base_tf": self.base_tf,
            "spec_count": len(self.feature_specs),
            "specs": list(self.feature_specs),
            "timeframe_count": len(self.feature_graph.all_nodes()),
        }

    # ========================================================================= 13
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"symbol={self.symbol!r}, "
            f"base_tf={self.base_tf!r}, "
            f"specs={len(self.feature_specs)})"
        )


    # =========================================================================
    # Accessors
    # ========================================================================= 14
    @property
    def base_tf(self) -> str:
        """
        بازگرداندن تایم‌فریم مبنای نماد جاری.
        """
        base_tfs = self.config.get("__base_tfs_dict", {})
        try:
            return base_tfs[self.symbol]
        except KeyError as exc:
            raise RuntimeError(f"No base_tf found for symbol={self.symbol}") from exc
        
    # ---------------------------------------------------------------
    @property
    def dataset(self) -> Optional[MTFDataset]: 
        return self._dataset

    @property
    def engine(self) -> FeatureEngine:
        return self.feature_engine

    @property
    def graph(self) -> FeatureGraph:
        return self.feature_graph

    @property
    def features(self) -> Optional[MTFDataset]:
        return self._features

    @property
    def store(self) -> FeatureStoreV2:
        return self.feature_store

    @property
    def observation(self):
        return self._observation
    
# ============================================================================= END

"""
 1. __init__                    --> constructor
 2. reset_live_store            --> external API
 3. run                         --> external API  used in main_6_5.py
 4. process_live                --> external API  used in main_6_5.py

 5. build_observation        --> internal method  usde in 3,4
 6. build_feature_store      --> internal method  usde in 3,8
 7. save_feature_store       --> internal method  usde in 3,8

 8. export                      --> external API
 9. build_numpy_observation     --> external API
10. build_graph                 --> external API
11. reload_config               --> external API
12. info                        --> external API
"""
