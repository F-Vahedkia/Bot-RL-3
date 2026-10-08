# f04_features/feature_sys_contract.py
# Date reviewed:
#    1405/05/24-15:08 --> run result is OK for 59 tests

"""
Feature System Contract
=======================

Single source of truth for the structural contracts shared by the
Feature-layer components.

Architecture
------------
    Data Layer
        MTFDataset
            |
            v
    FeaturePipeline
        |           \
        v            v
    FeatureEngine   FeatureGraph
        |
        v
    FeatureStoreV2
        |
        v
    ObservationBuilder
        |
        v
    Observation

This module defines contracts only.

It does NOT:
- calculate features,
- manage live indicator state,
- manage cache,
- parse feature specifications,
- merge/persist datasets,
- build observations,
- orchestrate the pipeline.

Those responsibilities belong to the corresponding frozen components.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Final, FrozenSet, Mapping, Optional, Sequence, Tuple

from f03_data.mtf_dataset import MTFDataset


# =============================================================================
# Version / mode contracts
# =============================================================================

FEATURE_SYSTEM_CONTRACT_VERSION: Final[str] = "1"

TRAIN_MODES: Final[FrozenSet[str]] = frozenset({"train", "optimize"})
INCREMENTAL_MODES: Final[FrozenSet[str]] = frozenset(
    {"live", "paper", "shadow", "backtest", "replay", "eval"}
)
ALL_MODES: Final[FrozenSet[str]] = TRAIN_MODES | INCREMENTAL_MODES


# =============================================================================
# Pipeline stages
# =============================================================================

class FeatureStage(str, Enum):
    RAW = "raw"
    FEATURES = "features"
    STORED = "stored"
    OBSERVATION = "observation"


# =============================================================================
# Dataset contract
# =============================================================================

@dataclass(frozen=True, slots=True)
class DatasetContract:
    """
    Structural contract for MTFDataset instances crossing the Feature layer.
    """

    symbol: str
    base_tf: str
    timeframes: Tuple[str, ...]

    # -----------------------------------------------------
    @classmethod
    def from_dataset(cls, dataset: MTFDataset) -> "DatasetContract":
        if dataset is None:
            raise ValueError("dataset is required")

        if not isinstance(dataset, MTFDataset):
            raise TypeError(
                f"Expected MTFDataset, got {type(dataset).__name__}"
            )

        if not dataset.symbol:
            raise ValueError("MTFDataset.symbol is required")

        if not dataset.base_tf:
            raise ValueError("MTFDataset.base_tf is required")

        timeframes = tuple(str(tf).upper() for tf in dataset.timeframes)

        if not timeframes:
            raise ValueError("MTFDataset must contain at least one timeframe")

        if str(dataset.base_tf).upper() not in timeframes:
            raise ValueError(
                f"Base timeframe {dataset.base_tf!r} is not present in "
                f"dataset timeframes {timeframes!r}"
            )

        return cls(
            symbol=str(dataset.symbol),
            base_tf=str(dataset.base_tf).upper(),
            timeframes=timeframes,
        )

    # -----------------------------------------------------
    def validate_dataset(self, dataset: MTFDataset) -> None:
        actual = DatasetContract.from_dataset(dataset)

        if actual.symbol != self.symbol:
            raise ValueError(
                f"Dataset symbol mismatch: expected={self.symbol!r}, "
                f"got={actual.symbol!r}"
            )

        if actual.base_tf != self.base_tf:
            raise ValueError(
                f"Base timeframe mismatch: expected={self.base_tf!r}, "
                f"got={actual.base_tf!r}"
            )

        if set(actual.timeframes) != set(self.timeframes):
            raise ValueError(
                "Dataset timeframe mismatch: "
                f"expected={sorted(self.timeframes)!r}, "
                f"got={sorted(actual.timeframes)!r}"
            )


# =============================================================================
# Feature execution contract
# =============================================================================

@dataclass(frozen=True, slots=True)
class FeatureExecutionContract:
    """
    Contract describing one FeatureEngine execution context.

    The contract deliberately contains no live indicator instances and no cache
    state. Runtime state remains owned by FeatureEngine.
    """

    mode: str
    symbol: str
    base_tf: str
    timeframes: Tuple[str, ...]
    specs: Tuple[str, ...]

    # -----------------------------------------------------
    def __post_init__(self) -> None:
        mode = self.mode.lower()

        if mode not in ALL_MODES:
            raise ValueError(
                f"Unsupported Feature mode {self.mode!r}. "
                f"Allowed modes: {sorted(ALL_MODES)!r}"
            )

        if not self.symbol:
            raise ValueError("symbol is required")

        if not self.base_tf:
            raise ValueError("base_tf is required")

        if not self.timeframes:
            raise ValueError("At least one timeframe is required")

        normalized_base_tf = self.base_tf.upper()
        normalized_timeframes = tuple(
            tf.upper() for tf in self.timeframes
        )

        if normalized_base_tf not in normalized_timeframes:
            raise ValueError(
                f"Base timeframe {normalized_base_tf!r} is not present "
                f"in timeframes {normalized_timeframes!r}"
            )

        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "base_tf",
            # self.base_tf.upper(),
            normalized_base_tf,
        )
        object.__setattr__(
            self,
            "timeframes",
            # tuple(tf.upper() for tf in self.timeframes),
            normalized_timeframes,
        )
        object.__setattr__(
            self,
            "specs",
            tuple(self.specs),
        )


# =============================================================================
# Observation contract
# =============================================================================

@dataclass(frozen=True, slots=True)
class ObservationContract:
    """
    Structural contract for the final observation.

    DataFrame output
        - index is the ObservationBuilder-owned observation_index
        - columns are deterministic
        - feature values remain numeric for NumPy conversion

    The contract does not dictate the number of features.
    """

    symbol: str
    base_tf: str
    columns: Tuple[str, ...]
    row_count: int
    dtype: str = "float64"

    # -----------------------------------------------------
    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")

        if not self.base_tf:
            raise ValueError("base_tf is required")

        if self.row_count < 0:
            raise ValueError("row_count must be >= 0")

        object.__setattr__(self, "base_tf", self.base_tf.upper())
        object.__setattr__(self, "columns", tuple(self.columns))

    # -----------------------------------------------------
    @property
    def feature_count(self) -> int:
        return len(self.columns)


# =============================================================================
# System contract
# =============================================================================

@dataclass(frozen=True, slots=True)
class FeatureSystemContract:
    """
    Top-level immutable description of the Feature-layer contract.

    This object records architectural facts; it does not own any runtime
    component instance.
    """

    version: str = FEATURE_SYSTEM_CONTRACT_VERSION
    modes: FrozenSet[str] = ALL_MODES
    batch_modes: FrozenSet[str] = TRAIN_MODES
    incremental_modes: FrozenSet[str] = INCREMENTAL_MODES

    # Component ownership
    pipeline_owner: str = "FeaturePipeline"
    engine_owner: str = "FeatureEngine"
    cache_owner: str = "FeatureEngine"
    store_owner: str = "FeatureStoreV2"
    graph_owner: str = "FeatureGraph"
    observation_owner: str = "ObservationBuilder"
    registry_owner: str = "FeatureRegistry"

    # Dataset boundary
    input_type: str = "MTFDataset"
    feature_output_type: str = "MTFDataset"
    observation_output_type: str = "pandas.DataFrame"

    # Pipeline stage ordering
    stages: Tuple[FeatureStage, ...] = (
        FeatureStage.RAW,
        FeatureStage.FEATURES,
        FeatureStage.STORED,
        FeatureStage.OBSERVATION,
    )

    # -----------------------------------------------------
    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("Contract version is required")

        if self.batch_modes - self.modes:
            raise ValueError("batch_modes must be a subset of modes")

        if self.incremental_modes - self.modes:
            raise ValueError("incremental_modes must be a subset of modes")

        if self.batch_modes & self.incremental_modes:
            raise ValueError(
                "batch_modes and incremental_modes must be disjoint"
            )

    # -----------------------------------------------------
    def is_batch_mode(self, mode: str) -> bool:
        return mode.lower() in self.batch_modes

    # -----------------------------------------------------
    def is_incremental_mode(self, mode: str) -> bool:
        return mode.lower() in self.incremental_modes

    # -----------------------------------------------------
    def validate_mode(self, mode: str) -> str:
        normalized = mode.lower()

        if normalized not in self.modes:
            raise ValueError(
                f"Unsupported mode {mode!r}; "
                f"allowed={sorted(self.modes)!r}"
            )

        return normalized

    # -----------------------------------------------------
    def validate_stage_order(self) -> Tuple[FeatureStage, ...]:
        return self.stages


# =============================================================================
# Public immutable default
# =============================================================================

FEATURE_SYSTEM_CONTRACT: Final[FeatureSystemContract] = FeatureSystemContract()


# =============================================================================
# Contract helpers
# =============================================================================

def build_dataset_contract(dataset: MTFDataset) -> DatasetContract:
    """Build a structural contract from an MTFDataset."""
    return DatasetContract.from_dataset(dataset)

# ---------------------------------------------------------
def build_execution_contract(
    *,
    dataset: MTFDataset,
    specs: Sequence[str],
    mode: str,
) -> FeatureExecutionContract:
    """Build a FeatureEngine execution contract from a dataset and specs."""
    dataset_contract = DatasetContract.from_dataset(dataset)
    normalized_mode = FEATURE_SYSTEM_CONTRACT.validate_mode(mode)

    return FeatureExecutionContract(
        mode=normalized_mode,
        symbol=dataset_contract.symbol,
        base_tf=dataset_contract.base_tf,
        timeframes=dataset_contract.timeframes,
        specs=tuple(specs),
    )

# ---------------------------------------------------------
def summarize_system_contract() -> Dict[str, Any]:
    """
    Return a serializable summary useful for logs and tests.
    """
    contract = FEATURE_SYSTEM_CONTRACT

    return {
        "version": contract.version,
        "modes": sorted(contract.modes),
        "batch_modes": sorted(contract.batch_modes),
        "incremental_modes": sorted(contract.incremental_modes),
        "pipeline_owner": contract.pipeline_owner,
        "engine_owner": contract.engine_owner,
        "cache_owner": contract.cache_owner,
        "store_owner": contract.store_owner,
        "graph_owner": contract.graph_owner,
        "observation_owner": contract.observation_owner,
        "registry_owner": contract.registry_owner,
        "input_type": contract.input_type,
        "feature_output_type": contract.feature_output_type,
        "observation_output_type": contract.observation_output_type,
        "stages": [stage.value for stage in contract.stages],
    }


# ---------------------------------------------------------
__all__ = [
    "FEATURE_SYSTEM_CONTRACT_VERSION",
    "TRAIN_MODES",
    "INCREMENTAL_MODES",
    "ALL_MODES",
    "FeatureStage",
    "DatasetContract",
    "FeatureExecutionContract",
    "ObservationContract",
    "FeatureSystemContract",   # System contract: class
    "FEATURE_SYSTEM_CONTRACT",   # Public immutable default: constant
    "build_dataset_contract",      # Contract helpers: function
    "build_execution_contract",    # Contract helpers: function
    "summarize_system_contract",   # Contract helpers: function
]

# ============================================================================= END