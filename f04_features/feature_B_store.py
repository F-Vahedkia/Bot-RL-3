# f04_features/feature_B_store.py
# Date reviewed:
#    1405/05/23-16:00 ==> test by testet ver _A is ok for 22 tests

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from f03_data.mtf_dataset import MTFDataset
from f02_utils.config_loader import load_config
from f02_utils.functions.constants import _TF_MINUTES

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# Metadata
# =============================================================================
@dataclass(frozen=True, slots=True)
class FeatureMeta:
    """Metadata for one stored feature column."""

    timeframe: str
    column: str
    dtype: str
    first_valid_ts: Optional[str]
    nan_count: int
    coverage_ratio: float


# =============================================================================
# Timeframe alignment utility
# =============================================================================
def align_to_base(
    dataset: MTFDataset,
    *,
    include_timeframes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Align all timeframe frames onto the base timeframe without look-ahead.

    This is a pure utility: it does not mutate ``dataset`` and does not attach
    an ``aligned`` object to MTFDataset.  ObservationBuilder may use this
    utility later if its final observation contract requires aligned data.
    """
    if dataset is None:
        raise ValueError("dataset is required")

    if not isinstance(dataset, MTFDataset):
        raise TypeError(
            f"Expected MTFDataset, got {type(dataset).__name__}"
        )

    if not dataset.frames:
        return pd.DataFrame()

    base_tf = dataset.base_tf.upper()
    base_df = dataset.get(base_tf)

    if base_df is None or base_df.empty:
        return pd.DataFrame(index=base_df.index if base_df is not None else None)

    selected = (
        {tf.upper() for tf in include_timeframes}
        if include_timeframes is not None
        else set(dataset.timeframes)
    )
    selected.add(base_tf)

    left = base_df.copy()
    left["__close_time__"] = (
        left.index + pd.Timedelta(minutes=_TF_MINUTES[base_tf])
    )
    base_index_name = left.index.name
    left = left.reset_index(names="__open_time__").sort_values("__close_time__")

    for tf in sorted(selected, key=lambda x: _TF_MINUTES[x]):
        if tf == base_tf:
            continue
        if tf not in dataset.frames:
            continue

        right = dataset.get(tf)
        if right is None or right.empty:
            continue

        right = right.copy()
        right["__close_time__"] = (
            right.index + pd.Timedelta(minutes=_TF_MINUTES[tf])
        )
        right = right.reset_index(names=f"__{tf}_open_time__")
        right = right.sort_values("__close_time__")

        # Keep only source columns; the alignment helper is intentionally
        # generic and does not rename or reinterpret feature names.
        merge_cols = [
            c for c in right.columns
            if c not in {"__close_time__"}
        ]

        # Avoid accidental duplicate helper columns when multiple TFs exist.
        duplicate_cols = set(left.columns).intersection(merge_cols)
        merge_cols = [c for c in merge_cols if c not in duplicate_cols]

        right = right[["__close_time__", *merge_cols]]

        left = pd.merge_asof(
            left.sort_values("__close_time__"),
            right.sort_values("__close_time__"),
            on="__close_time__",
            direction="backward",
            allow_exact_matches=True,
        )

    left = left.set_index("__open_time__")
    left.index.name = base_index_name

    helper_cols = [
        c for c in left.columns
        if c == "__close_time__" or c.endswith("_open_time__")
    ]
    if helper_cols:
        left = left.drop(columns=helper_cols)

    return left


# =============================================================================
# FeatureStoreV2
# =============================================================================
class FeatureStoreV2:
    """
    Storage/merge boundary for MTFDataset feature results.

    Responsibilities
    ----------------
    1. Merge FeatureEngine output into a copy of the raw MTFDataset.
    2. Produce per-timeframe metadata for the resulting feature dataset.
    3. Persist each timeframe and its metadata.

    Non-responsibilities
    --------------------
    - No Registry access.
    - No FeatureEngine execution.
    - No live state/cache management.
    - No feature specification parsing.
    - No Observation construction.
    - No mutation of the source datasets.
    """
    # ------------------------------------------------------------------------- 1
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.cfg: Dict[str, Any] = config if config is not None else load_config()

    # ------------------------------------------------------------------------- 2
    def build(
        self,
        dataset: MTFDataset,
        features: MTFDataset,
    ) -> MTFDataset:
        """
        Merge FeatureEngine output into a copy of the raw MTFDataset.

        Contract
        --------
        - Both inputs must be MTFDataset.
        - Both datasets must belong to the same symbol.
        - The timeframe sets must match.
        - Raw frames are copied and never modified in place.
        - Existing equal columns are preserved.
        - A differing column with the same name is a collision and raises.
        - New feature columns are appended to the corresponding timeframe.
        """
        self._validate_datasets(dataset, features)

        result = dataset.copy()

        for tf in result.timeframes:
            raw_df = result.get(tf)
            feature_df = features.get(tf)

            if feature_df is None or feature_df.empty:
                continue

            merged_df = raw_df.copy()

            for column in feature_df.columns:
                incoming = feature_df[column]

                if column not in merged_df.columns:
                    merged_df[column] = incoming
                    continue

                existing = merged_df[column]
                if existing.equals(incoming):
                    continue

                raise ValueError(
                    f"FeatureStore column collision for symbol={dataset.symbol}, "
                    f"timeframe={tf}, column={column!r}"
                )

            result.replace(tf, merged_df)

        return result

    # ------------------------------------------------------------------------- 3
    @staticmethod
    def _validate_datasets(
        dataset: MTFDataset,
        features: MTFDataset,
    ) -> None:
        if dataset is None:
            raise ValueError("dataset is required")
        if features is None:
            raise ValueError("features is required")

        if not isinstance(dataset, MTFDataset):
            raise TypeError(
                f"Expected dataset to be MTFDataset, got {type(dataset).__name__}"
            )
        if not isinstance(features, MTFDataset):
            raise TypeError(
                f"Expected features to be MTFDataset, got {type(features).__name__}"
            )

        if dataset.symbol != features.symbol:
            raise ValueError(
                f"Dataset symbol mismatch: raw={dataset.symbol}, "
                f"features={features.symbol}"
            )

        raw_tfs = {tf.upper() for tf in dataset.timeframes}
        feature_tfs = {tf.upper() for tf in features.timeframes}

        if raw_tfs != feature_tfs:
            raise ValueError(
                f"Timeframe mismatch for symbol={dataset.symbol}: "
                f"raw={sorted(raw_tfs)}, features={sorted(feature_tfs)}"
            )

    # -------------------------------------------------------------------------4
    def extract_metadata(
        self,
        dataset: MTFDataset,
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract per-timeframe metadata from an MTFDataset.

        The store intentionally does not depend on Registry metadata.  Metadata
        therefore describes the actual stored columns and their observed data
        quality rather than reconstructing indicator parameters.
        """
        if dataset is None:
            raise ValueError("dataset is required")
        if not isinstance(dataset, MTFDataset):
            raise TypeError(
                f"Expected MTFDataset, got {type(dataset).__name__}"
            )

        result: Dict[str, pd.DataFrame] = {}

        for tf in dataset.timeframes:
            df = dataset.get(tf)
            rows: List[FeatureMeta] = []

            if df is None:
                result[tf] = pd.DataFrame(columns=self._metadata_columns())
                continue

            for column in df.columns:
                series = df[column]
                first_valid = series.first_valid_index()

                try:
                    numeric = pd.to_numeric(series, errors="coerce")
                    valid_mask = np.isfinite(numeric.to_numpy(dtype=float))
                    nan_count = int((~valid_mask).sum())
                    coverage = (
                        float(valid_mask.sum() / len(valid_mask))
                        if len(valid_mask)
                        else 0.0
                    )
                except (TypeError, ValueError):
                    valid_mask = series.notna().to_numpy()
                    nan_count = int((~valid_mask).sum())
                    coverage = (
                        float(valid_mask.sum() / len(valid_mask))
                        if len(valid_mask)
                        else 0.0
                    )

                rows.append(
                    FeatureMeta(
                        timeframe=tf,
                        column=str(column),
                        dtype=str(series.dtype),
                        first_valid_ts=(
                            None if first_valid is None else str(first_valid)
                        ),
                        nan_count=nan_count,
                        coverage_ratio=coverage,
                    )
                )

            result[tf] = pd.DataFrame(
                [asdict(row) for row in rows],
                columns=self._metadata_columns(),
            )

        return result

    # ------------------------------------------------------------------------- 5
    @staticmethod
    def _metadata_columns() -> List[str]:
        return [
            "timeframe",
            "column",
            "dtype",
            "first_valid_ts",
            "nan_count",
            "coverage_ratio",
        ]

    # ------------------------------------------------------------------------- 6
    def save(
        self,
        dataset: MTFDataset,
        metadata: Dict[str, pd.DataFrame],
        out_dir: str | Path,
        name: str,
        fmt: str = "parquet",
    ) -> Dict[str, Dict[str, str]]:
        """
        Persist each timeframe and its metadata independently.
        """
        if dataset is None:
            raise ValueError("dataset is required")
        if not isinstance(dataset, MTFDataset):
            raise TypeError(
                f"Expected MTFDataset, got {type(dataset).__name__}"
            )
        if not name:
            raise ValueError("name is required")

        fmt = fmt.lower()
        if fmt not in {"parquet", "csv"}:
            raise ValueError("Unsupported format. Use 'parquet' or 'csv'.")

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        result: Dict[str, Dict[str, str]] = {}

        for tf in dataset.timeframes:
            tf_dir = out / tf
            tf_dir.mkdir(parents=True, exist_ok=True)

            df = dataset.get(tf)
            if df is None:
                continue

            metadata_df = metadata.get(tf, pd.DataFrame())

            if fmt == "parquet":
                data_path = tf_dir / f"{name}.parquet"
                df.to_parquet(data_path, index=True)
            else:
                data_path = tf_dir / f"{name}.csv"
                df.to_csv(data_path, index=True)

            meta_csv = tf_dir / f"{name}.meta.csv"
            meta_json = tf_dir / f"{name}.meta.json"

            metadata_df.to_csv(meta_csv, index=False)
            with meta_json.open("w", encoding="utf-8") as handle:
                json.dump(
                    metadata_df.to_dict(orient="records"),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            result[tf] = {
                "data": str(data_path),
                "meta_csv": str(meta_csv),
                "meta_json": str(meta_json),
            }

        return result


# =============================================================================
# Functional API
# ============================================================================= 7
def build_feature_store(
    dataset: MTFDataset,
    features: MTFDataset,
    out_dir: str | Path,
    name: str,
    fmt: str = "parquet",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, str]]:
    """Build the merged FeatureStore dataset and persist it."""
    store = FeatureStoreV2(config=config)
    merged = store.build(dataset, features)
    metadata = store.extract_metadata(merged)
    return store.save(
        dataset=merged,
        metadata=metadata,
        out_dir=out_dir,
        name=name,
        fmt=fmt,
    )

# ============================================================================= END