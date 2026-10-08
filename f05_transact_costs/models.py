from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Protocol


class SlippageModel(Protocol):
    model_version: str

    def sample_price(self, *, pip_size: float, event_key: str) -> float: ...


@dataclass(frozen=True, slots=True)
class NormalCappedSlippageModel:
    mean_pips: float
    std_pips: float
    cap_pips: float
    seed: int
    model_version: str = "normal_capped.v1"

    def __post_init__(self) -> None:
        mean = float(self.mean_pips)
        std = float(self.std_pips)
        cap = float(self.cap_pips)
        if not all(math.isfinite(x) for x in (mean, std, cap)):
            raise ValueError("slippage parameters must be finite")
        if mean < 0.0 or std < 0.0 or cap <= 0.0:
            raise ValueError("invalid slippage parameters")
        if mean > cap or std > cap:
            raise ValueError("slippage mean/std must be <= cap")
        version = str(self.model_version).strip()
        if not version:
            raise ValueError("model_version is required")
        object.__setattr__(self, "mean_pips", mean)
        object.__setattr__(self, "std_pips", std)
        object.__setattr__(self, "cap_pips", cap)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "model_version", version)

    def sample_pips(self, *, event_key: str) -> float:
        key = str(event_key).strip()
        if not key:
            raise ValueError("event_key is required")
        digest = hashlib.sha256(f"{self.seed}|{key}".encode("utf-8")).digest()
        child_seed = int.from_bytes(digest[:8], "big", signed=False)
        if self.std_pips == 0.0:
            sample = self.mean_pips
        else:
            sample = random.Random(child_seed).gauss(self.mean_pips, self.std_pips)
        return min(self.cap_pips, max(0.0, sample))

    def sample_price(self, *, pip_size: float, event_key: str) -> float:
        pip_size = float(pip_size)
        if not math.isfinite(pip_size) or pip_size <= 0.0:
            raise ValueError("pip_size must be positive and finite")
        return self.sample_pips(event_key=event_key) * pip_size


@dataclass(frozen=True, slots=True)
class FixedSlippageModel:
    price: float
    model_version: str = "fixed.v1"

    def __post_init__(self) -> None:
        price = float(self.price)
        if not math.isfinite(price) or price < 0.0:
            raise ValueError("price slippage must be finite and >= 0")
        version = str(self.model_version).strip()
        if not version:
            raise ValueError("model_version is required")
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "model_version", version)

    def sample_price(self, *, pip_size: float, event_key: str) -> float:
        del pip_size, event_key
        return self.price
