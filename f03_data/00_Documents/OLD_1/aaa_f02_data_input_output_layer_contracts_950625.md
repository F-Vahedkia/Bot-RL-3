# f03_data input/output layer contracts

## 1. Purpose

This document is the integration contract for `f03_data` in Bot-RL-3 v8. It is intended to let a downstream layer consume batch and live market data without needing to inspect all `f03_data` source files again.

This contract is derived from the actual code supplied for the current `f03_data` core files, not from their legacy file-level docstrings.

Core files used as the basis:
- `f03_data/mt5_connector.py`
- `f03_data/mt5_data_loader_E.py`
- `f03_data/data_handler_F_3.py`
- `f03_data/data_layer_functions.py`
- `f03_data/live_market_engine.py`
- `f03_data/mtf_dataset.py`

Relevant supplied tests also confirm the currently exercised EventBus, CandleDetector, MTFDataset, `_read_raw_df`, MT5 stream worker, and live-vs-batch comparison behavior.

## 2. Scope and architectural role

`f03_data` is the market-data acquisition and preparation layer. Its downstream-facing product is a symbol-separated multi-timeframe dataset.

There are two principal data paths:

**Batch path**
`stored raw market data -> DataHandler.build(BuildParams) -> MTFDataset`

**Live path**
`MT5 -> MT5StreamWorker -> EventBus event -> DataHandler.update_live(event) -> MTFDataset`

The downstream consumer should normally integrate with `MTFDataset`, not with the internal raw-data cache, EventBus queues, or MT5 connector directly.

## 3. Primary downstream contract: MTFDataset

### 3.1 Type

`f03_data.mtf_dataset.MTFDataset`

### 3.2 Meaning

`MTFDataset` represents data for exactly one symbol and multiple independent timeframes.

It does **not** align, merge, resample, or otherwise synchronize the individual timeframe DataFrames.

### 3.3 Structure

```python
MTFDataset(
    symbol: str,
    base_tf: str,
    frames: Dict[str, pd.DataFrame]
)
```

Semantics:
- `symbol`: the symbol represented by the dataset.
- `base_tf`: the base timeframe selected by `check_tfs`; in normal operation this is the smallest valid timeframe in the set.
- `frames`: mapping `TIMEFRAME -> DataFrame`.

Timeframe keys are normalized to uppercase when inserted through `add()` and accessed through `get()` / `__getitem__()`.

### 3.4 Public operations used by consumers

```python
dataset.symbol
dataset.base_tf
dataset.frames
dataset.timeframes

df = dataset.get("M5")
df = dataset["M5"]
dataset.add("M5", df)
dataset.replace("M5", df)
copy = dataset.copy()
```

`get()` raises `KeyError` when the requested timeframe is absent. `replace()` also raises `KeyError` when the timeframe does not already exist.

### 3.5 Independence invariant

Every timeframe has its own DataFrame. A consumer must not assume that rows from different timeframes have identical indexes, identical row counts, or one-to-one positional correspondence.

Therefore downstream code should use:

```python
df_m1 = dataset["M1"]
df_h1 = dataset["H1"]
```

and should explicitly perform any cross-timeframe alignment or feature relationship it needs.

## 4. DataFrame contract inside each MTFDataset frame

The current normalization path retains the standard market columns when present:

```text
open
high
low
close
volume
spread
```

`open`, `high`, `low`, and `close` are numeric. `volume` and `spread` are normalized to pandas nullable integer dtype (`Int64`) when present.

The index is named `time` by `normalize_df`.

For data read by `DataHandler._read_raw_df`, the index is converted to a timezone-aware `DatetimeIndex` in UTC before the final normalized DataFrame is returned.

A timeframe DataFrame may contain only a subset of the standard columns if the source data did not contain all of them; consumers must not assume that every optional column is physically present unless the relevant upstream data source guarantees it.

## 5. Batch input contract

### 5.1 Public entry point

```python
from f03_data.data_handler_F_3 import DataHandler, BuildParams

data_handler = DataHandler(cfg=cfg, symbol=symbol)
dataset = data_handler.build(params)
```

### 5.2 BuildParams

The current constructor accepts:

```python
BuildParams(
    symbol,
    base_tf,
    timeframes,
    selected_tf=None,
    load_format="parquet",
    mode="number",
    start_lastrows=None,
    end_lastrows=None,
    start_time=None,
    end_time=None,
    period_size=None,
    from_last_n=None,
    to_last_n=None,
    base_time=None,
)
```

For normal downstream integration, the important fields are:
- `symbol`
- `base_tf`
- `timeframes`
- `load_format`
- `mode` plus the corresponding range parameters when a restricted historical window is required.

### 5.3 Batch modes

`DataHandler._load_raw()` currently supports the effective modes:
- `number`: selects a row-count window from the end of a stored DataFrame.
- `time`: selects `[start_time, end_time)`; the end boundary is exclusive.
- `periods`: `BuildParams` converts this into a time window using `_get_range_by_timeframe()`, after which the load path operates in time mode.

The normal downstream contract is the resulting `MTFDataset`, not the internal mode mechanics.

### 5.4 Batch result

`DataHandler.build()` returns:

```python
MTFDataset
```

The base timeframe is loaded first. Other requested timeframes are loaded independently. If an optional non-base timeframe is missing or empty, it is skipped with a warning; if the base timeframe is unavailable or has no overlap with the requested window, `build()` raises `FileNotFoundError`.

## 6. Live input contract

### 6.1 Public entry point for a consumer

```python
from f03_data.data_handler_F_3 import DataHandler

dataset = data_handler.update_live(event)
```

The current live event shape consumed by `update_live()` is:

```python
{
    "event_type": "NEW_CANDLE",
    "symbol": "XAUUSD",
    "timeframe": "M1",
    "all_dfs": {
        "XAUUSD:M1": pd.DataFrame(...),
        "XAUUSD:M5": pd.DataFrame(...),
        "XAUUSD:H1": pd.DataFrame(...),
        # ... required timeframes for the symbol
    },
}
```

`all_dfs` is symbol/timeframe keyed. The current live engine publishes the complete set fetched by `_fetch_all_tfs()` for that symbol when a new candle is detected.

### 6.2 Live result

`DataHandler.update_live()` returns an `MTFDataset` for the event symbol.

The dataset contains the cached data for the timeframes configured in the DataHandler warmup dictionary.

The live cache keeps up to:

```text
warmup_length + _cache_dict_extra_length
```

rows per configured timeframe, subject to the current update logic. Duplicate timestamps are removed with `keep='last'`.

The returned live dataset is also stored internally as `_latest_dataset`, and can be retrieved by:

```python
data_handler.get_latest_dataset()
```

### 6.3 First live update behavior

On the first event for a symbol/timeframe, `CandleDetector` initializes its state and returns no detection event. A new-candle event is emitted only after the observed closed-candle timestamp advances.

Therefore a downstream live consumer should not assume that the first polling pass necessarily produces a `NEW_CANDLE` event.

## 7. EventBus contract for live integration

The normal architecture is:

`MT5StreamWorker -> EventBus -> consumer`

The current EventBus API supports symbol-based subscription:

```python
subscriber_id = event_bus.subscribe(symbol)
queue = event_bus.get_queue(symbol)
event = event_bus.get_event(symbol, timeout=...)
event_bus.unsubscribe(symbol)
```

It also has an ID-based API (`subscribe_by_id`, `get_queue_by_id`, `get_event_by_id`, etc.), but the current `DataHandler` integration and supplied tests primarily use the symbol-based path.

`publish()` sends an event only to the queue registered under the matching symbol. `publish_by_id()` broadcasts the same event to every subscriber queue.

Queue overflow currently uses a non-blocking put and drops the event when the queue is full. This is current implementation behavior, not a future-proofed reliability guarantee.

## 8. Live candle event semantics

`CandleDetector` tracks state independently by:

```text
symbol:timeframe
```

A detected event represents a newly observed closed candle. `MT5StreamWorker._fetch_closed()` requests `n+1` candles and returns the final `n` closed candles by excluding the last/current candle from the returned MT5 snapshot.

The event payload generated by `CandleDetector.detect()` contains:

```python
{
    "symbol": str,
    "timeframe": str,
    "candle_time": datetime,
    "open": float | None,
    "high": float | None,
    "low": float | None,
    "close": float | None,
    "volume": float | None,
    "spread": float | None,
}
```

However, the EventBus payload actually delivered downstream by `MT5StreamWorker` is the larger `NEW_CANDLE` envelope described in Section 6.1; the inner detector event is used for detection/logging and is not itself the EventBus message schema.

## 9. Symbol separation contract

Symbol separation is a first-class invariant of the current layer.

A `DataHandler` instance is constructed for one `symbol`. Its live cache keys are effectively:

```text
<SYMBOL>:<TIMEFRAME>
```

A downstream layer should preserve this separation. A consumer should never silently combine two symbols into one `MTFDataset` instance.

For multi-symbol consumers, the expected higher-level structure is conceptually:

```python
{
    "XAUUSD": MTFDataset(...),
    "EURUSD": MTFDataset(...),
    "BITCOIN": MTFDataset(...),
}
```

unless a higher layer explicitly defines a different portfolio-level contract.

## 10. Timezone contract

### 10.1 Stored raw files

The current raw-data pipeline may store raw files with naive timestamps when written by `MT5DataLoader_batch` under the current configuration path.

### 10.2 DataHandler boundary

`DataHandler._read_raw_df()` treats a naive raw timestamp as being in the configured `project.broker_timezone`, then converts it to UTC.

If an incoming index is already timezone-aware, it is converted to UTC.

Therefore the downstream `DataHandler` output contract is:

**DataFrames consumed from `DataHandler.build()` and `DataHandler.update_live()` use timezone-aware UTC indexes.**

A downstream consumer should therefore treat the dataset index as UTC-aware and should not re-localize UTC timestamps as if they were naive broker-local timestamps.

### 10.3 Event candle time

The detected `candle_time` emitted by `CandleDetector` is converted to UTC and returned as a Python `datetime`.

## 11. Historical file contract

Current raw-data storage follows the symbol/timeframe layout used by `MT5DataLoader_batch` and `full_file_path`:

```text
data/raw/<SYMBOL>/<TF>.(csv|parquet)
```

The downloader can also write a metadata file beside the raw data:

```text
<SYMBOL>/<TF>.meta.json
```

The processed `DataHandler.save()` path uses:

```text
data/processed/<SYMBOL>/<TF>.(csv|parquet)
```

plus a manifest file associated with the processed dataset.

A downstream layer should prefer `DataHandler.build()` over opening these files itself when it needs an application-level batch dataset, because `build()` applies the project normalization, timezone conversion, timeframe selection, and `MTFDataset` packaging rules.

## 12. Recommended downstream connection patterns

### Batch consumer

```python
handler = DataHandler(cfg=cfg, symbol=symbol)
params = BuildParams(
    symbol=symbol,
    base_tf=base_tf,
    timeframes=timeframes,
    load_format="parquet",
    mode="number",
    start_lastrows=required_rows,
    end_lastrows=0,
)

observation_input = handler.build(params)
```

The downstream layer receives `MTFDataset` and should access each timeframe independently.

### Live consumer

```python
handler = DataHandler(cfg=cfg, symbol=symbol)

def on_new_market_data(event):
    dataset = handler.update_live(event)
    if dataset.frames:
        consume(dataset)
```

If integrating through the EventBus, subscribe the consumer to the relevant symbol queue and pass each `NEW_CANDLE` event into the symbol-specific `DataHandler`.

## 13. What a downstream layer may safely assume

A normal downstream consumer may rely on these invariants:

1. A `DataHandler` instance is symbol-specific.
2. `MTFDataset` contains independently stored DataFrames per timeframe.
3. Timeframe keys are uppercase when managed through `MTFDataset` APIs.
4. `DataHandler` output indexes are timezone-aware UTC indexes.
5. Standard market columns are `open`, `high`, `low`, `close`, `volume`, and optionally `spread`.
6. The batch and live public product is `MTFDataset`.
7. Live events are symbol-specific and carry `all_dfs` keyed by `<SYMBOL>:<TF>`.

## 14. What a downstream layer must NOT assume

Do not assume:

- that every requested non-base timeframe is present in a batch result;
- that different timeframe DataFrames have aligned indexes or equal lengths;
- that the first live polling cycle produces a new-candle event;
- that EventBus queues are lossless under overflow;
- that raw files always contain timezone-aware timestamps;
- that every frame contains every optional column;
- that the current EventBus payload schema is a final enterprise-grade event schema;
- that `MTFStreamWorker.start()` is itself an asynchronous/thread-spawning API; the current worker executes its loop directly from `start()`.

## 15. Error and empty-data behavior relevant to consumers

The current behavior includes:

- Missing/empty base timeframe during `DataHandler.build()` -> `FileNotFoundError`.
- Missing/empty optional non-base timeframe -> warning and skip.
- Invalid timeframe mode passed to `_read_raw_df()` -> `ValueError`.
- Invalid or empty time interval in `_read_raw_df(mode="time")` -> empty DataFrame.
- EventBus queue full -> event is dropped and a warning is logged.
- Missing live warmup configuration in `DataHandler.update_live()` -> an empty `MTFDataset` is returned.

Consumers should handle empty `MTFDataset.frames` and missing optional timeframe keys explicitly.

## 16. Boundary ownership

`f03_data` owns:
- MT5 connectivity and candle retrieval;
- raw-data normalization;
- raw-data persistence;
- historical window selection;
- live closed-candle detection;
- live per-symbol/timeframe caching;
- packaging into `MTFDataset`.

The downstream layer owns:
- observation construction;
- feature consumption beyond `f03_data` responsibilities;
- cross-timeframe alignment when required by its own contract;
- cross-symbol aggregation;
- trading decisions;
- portfolio/risk decisions;
- transaction-cost modeling beyond any future data-layer additions.

## 17. Current-version caveats

This document describes the current supplied v8 implementation as of the reviewed code, not an abstract future design.

In particular:
- EventBus reliability items such as overflow policy, backpressure, metrics, persistence, and shutdown semantics are not yet formalized as a stable contract.
- The live event payload schema is currently a plain dictionary and should be regarded as an implementation-level schema until a dedicated event contract is formalized.
- `DataHandler` contains legacy/unused live-consumer methods around EventBus subscription. A new downstream layer should prefer the direct `update_live(event)` boundary rather than depending on those legacy consumer-loop helpers.
- The current codebase includes some historical/development methods and old comments. They are not part of the downstream contract unless explicitly listed in this document.

## 18. Minimal integration checklist for a new downstream layer

Before connecting a new layer, confirm:

```text
[ ] One DataHandler instance per symbol.
[ ] Batch input is obtained through DataHandler.build().
[ ] Live input is obtained through DataHandler.update_live(event).
[ ] Downstream entry type is MTFDataset.
[ ] Timeframe frames are accessed independently.
[ ] UTC-aware DatetimeIndex is preserved.
[ ] Missing optional timeframes are handled.
[ ] Empty live datasets are handled.
[ ] No positional assumption is made between different TF indexes.
[ ] Symbol identity is preserved end-to-end.
```

## 19. Contract summary

The stable conceptual contract is:

```text
                 f03_data
                     |
        +------------+------------+
        |                         |
      BATCH                      LIVE
        |                         |
  DataHandler.build()      EventBus / event
        |                         |
        |                DataHandler.update_live()
        +------------+------------+
                     |
                 MTFDataset
                     |
          +----------+----------+
          |          |          |
         M1         M5         H1 ...
          |          |          |
        DataFrame  DataFrame  DataFrame
          |
      UTC-aware DatetimeIndex
      standard OHLCV(+spread)

Symbol identity remains outside and above the timeframe map.
```

This is the interface a new downstream layer should integrate against.
