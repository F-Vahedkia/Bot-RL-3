# Bot-RL-3 — Main / Composition-Root Integration Handoff

**Purpose:** This note is a handoff checklist to provide together when the active `main` / orchestration entry point is next revised. It records known integration gaps; it is not permission to patch GitHub directly.

## 1. Non-negotiable workflow constraints

- Inspect the current project tree and identify the active entry point before proposing integration changes. Do not assume that every file under `f99_main` is active.
- Never modify GitHub directly. Provide exact local patches with explicit file names and replacement boundaries.
- Do not edit existing files in `f03_data` or `f04_features` unless the user explicitly changes that restriction.
- Treat old contracts/docs as fallible. Derive behavior from the desired economic/accounting semantics plus the actual call graph; revise the contract to match the correct design.
- Preserve public APIs in `f06_env`, `f07_agents`, and `f08_risk` as far as possible. Do not broaden this task into indicator work or unrelated refactors.

## 2. Current checkpoint

- Canonical instrument contract is `f03_data.instrument_specs.InstrumentSpec`.
- `f05_transact_costs` imports/re-exports that same type via `f05_transact_costs.contracts.InstrumentSpec`.
- `MappingInstrumentResolver` accepts canonical `InstrumentSpec` instances only; it must not reconstruct specifications from raw mappings or guess currencies.
- `HistoricalQuoteResolver` and `TradingEnvironment.from_mtf_datasets` accept canonical instruments.
- Recent local patches reportedly passed all selected tests, including the combined command:

  `pytest -q f03_data/test_instrument_specs.py f05_transact_costs f06_env`

  The user reported 95 passed before the latest accounting patches and confirmed all tests passed after the latest patch set. Re-run the relevant tests after any further changes.

## 3. Composition-root responsibility

The active main/orchestrator should be the composition root: it should load project configuration once through the project's existing config-loader, construct typed transaction-cost configuration and components, resolve canonical instrument specs once, and inject dependencies into the relevant runtime/training paths. The transaction-cost layer should not read YAML itself.

Expected high-level construction (adapt this to the actual active main and loader API; do not paste blindly):

1. Read the already-loaded project mapping from the existing config loader.
2. Construct `TransactionCostConfig.from_mapping(config_mapping)`.
3. Resolve the project seed from the same config (`project.random_seed`) and call `build_components(cost_cfg, project_random_seed=seed)`.
4. Resolve a canonical `dict[str, InstrumentSpec]` from an already-loaded f03 symbol-spec snapshot using `resolve_instrument_specs(snapshot, symbols=required_symbols)`.
5. Inject the canonical instruments, selected cost components, and execution/accounting configuration into the training/backtest or live component that actually consumes them.
6. Log/validate the resolved cost policy at startup. Fail fast on incompatible/unsupported settings rather than silently ignoring them.

**Important:** Do not introduce a second config/YAML loader or construct InstrumentSpec by guessing from `digits`, `currency_profit`, symbol names, or raw mappings.

## 4. Commission: where it is configured and current gap

### Simulation/expected commission

The current global simulation commission profile is in `f01_config/config.yaml` under:

`transaction_costs.simulation.commission`

Current example profile:

```yaml
commission:
  basis: per_lot
  rate: 7.0
  currency: USD
  minimum: 0.0
```

`build_components()` turns this into `CostComponents.simulation_commission`, a `CommissionModel`.

Semantics in the current CommissionModel:
- `per_lot`: `rate * lots`, then apply `minimum`.
- `percent_notional`: `rate * notional`, then apply `minimum`.
- `fixed`: currently one fixed amount **per call to `CommissionModel.amount()`**, not automatically once per logical execution with many partial fills. Do not describe it as once-per-trade until aggregation/idempotency is implemented and tested.

### Live/actual commission

`live.commission.source: broker` means the live execution adapter is expected to pass the actual observed commission to `LiveObservedExecutionCostEngine.observe()` through `observed_commission` and `observed_commission_currency`, alongside the correct conversion rate. Do not replace a broker-reported actual value with the simulation estimate.

### Integration gap to close

The current `f06_env.ExecutionSimulator` accepts `ExecutionCost(commission=...)`, which is a fixed monetary amount supplied to a transition; it does not itself accept a CommissionModel or calculate `rate * lots` / percent-notional commission. The fact that `f05` builds `CostComponents.simulation_commission` does not automatically wire that model into `f06_env`. A main-level composition change that merely passes the literal `7.0` is not sufficient for arbitrary lot sizes or percent-notional commission.

Before main integration, choose and implement one explicit route:
- Preferred if maintaining the current f06 transition architecture: introduce a narrow adapter/policy at the execution boundary that receives lots + notional, applies the configured `CommissionModel`, applies the entry/exit flags, and returns the exact monetary cost for that transition; or
- Refactor the execution boundary to consume an `ExecutionCost`/cost object that can compute per-fill cost with lots/notional and preserve accounting breakdown. Avoid duplicating the full f05 calculator inside f06.

The composition root must also support different brokers/symbols if the project needs different schedules. The current YAML has one global simulation commission profile; it is not yet a broker-keyed or symbol-keyed commission catalog. Decide whether to support named broker/account profiles and per-symbol overrides, with deterministic precedence, instead of adding silent symbol-based inference.

## 5. `charge_on_entry` / `charge_on_exit` policy

The latest intended layout is under `transaction_costs.simulation.cost_application`, and the typed `SimulationConfig` passes both booleans to the simulation cost calculator/engine and round-trip estimator.

Intended meaning for expected/simulated costs:
- `charge_on_entry=False`: skip modeled commission/additional-fee charging for an ENTRY leg; do **not** remove spread/slippage embodied in execution prices.
- `charge_on_exit=False`: skip modeled commission/additional-fee charging for an EXIT leg; do **not** remove spread/slippage embodied in execution prices.
- Both `True`: charge modeled cash costs on their respective legs.

Live observed costs are different: once the broker has actually charged commission/fees, the actual amount must remain recorded in the actual fill/audit trail. Do not use a false expected-cost policy to erase real broker charges. If reporting needs a separate policy for realized cash posting, define that as a separate concept rather than overloading these two expected-cost controls.

The current executable-fill architecture represents spread/slippage in bid/ask and fill price; therefore `price_costs_embedded_in_fill=False` should not be exposed as a supported toggle unless the architecture is deliberately redesigned to prevent both double-charging and under-charging. Prefer fail-fast validation for `False` in the current version.

## 6. Additional fee basis — still unresolved

The active `f01_config/config.yaml` contains:

```yaml
additional_fee:
  basis: per_trade  # per_trade | per_lot
  value: 0.0
  currency: USD
```

Before declaring f05 frozen, the parser and typed config must retain/validate `basis`; the earlier implementation read `value` and `currency` but did not model `basis` in `SimulationConfig` or Factory. The engine currently receives `additional_fee` as an already-computed monetary amount per request/fill.

Define these semantics explicitly:
- `per_lot`: fee = configured rate/value times the lots for that fill (respect currency conversion).
- `per_trade`: one fee for one logical execution/order identified by `execution_id`, regardless of the number of partial fills. Do not charge the full fee once per partial fill.

For `per_trade`, choose an explicit owner for once-only application. Recommended: an execution-scoped fee allocator/accumulator keyed by `execution_id`, or apply the fee at an aggregation/settlement boundary where the complete logical execution is known. Define idempotency, partial-fill behavior, order cancellation/remainder behavior, and reconciliation of the allocated fee. Do not implement `per_trade` as a stateless multiplication inside a per-fill calculator.

Live additional fees should use actual broker-observed fee amounts where available, not the simulation estimate.

## 7. Overnight swap / financing — not accounted for in the reviewed f05/f06 path

The reviewed `f05_transact_costs` request/breakdown/engine contracts have no swap/financing component, and `f06_env.ExecutionSimulator` does not accrue overnight financing when a position crosses a rollover boundary. Therefore swap is not represented in the currently reviewed execution-cost/PnL path; do not claim that train/backtest net returns include it.

Future work should add a separate financing/swap model, not mislabel swap as commission or `additional_fee`. It needs to define:
- long and short swap values/rules per instrument;
- broker swap calculation mode/unit (points, account/profit currency, percentage/interest, etc.);
- rollover timezone and daily rollover cut-off;
- triple-swap weekday and non-trading-day handling;
- open lots held across each rollover, including partial close/reversal;
- conversion into account currency when the swap is not already expressed in it;
- actual live swap from broker deal/account records and a separate expected simulation estimate;
- how swap enters balance/equity, reward, audit, and expected-vs-actual reconciliation without double charging.

Do not patch f03_data to add swap data under the current restriction. When this feature is scheduled, first inspect the existing snapshot/raw metadata and define a provider-neutral input contract; keep MT5 discovery changes outside this f05/f06 patch unless the user explicitly authorizes them.

## 8. Train/backtest path: commission/accounting gap

`f06_env.TradingEnvironment` constructs/uses `ExecutionSimulator` (f06), which is distinct from `f05_transact_costs.SimulationExecutionCostEngine`. Simply constructing `CostComponents` in main will not alter f06 behavior unless those outputs are explicitly connected to the execution environment.

The latest local patch makes `ExecutionSimulator` accept `charge_on_entry` and `charge_on_exit`, and returns `accounting_realized_delta` so `TradingEnvironment` can credit the net realized change (gross realized PnL minus commission) to `PortfolioState`. Verify this remains true after future merges:
- `PositionState.realized_pnl` is position-local; do not accidentally count it again in portfolio balance.
- portfolio accounting must use the net `accounting_realized_delta`, not gross `realized_pnl`.
- reversal has two legs (exit old position + enter new position): apply exit and entry fee policies to the respective legs, not simply one undifferentiated flat fee.
- reduce and close should charge exit policy only; open and increase should charge entry policy only.
- slippage/spread remain reflected in executable fill prices; never subtract them a second time as cash commission.
- `ExecutionCost.commission` as a fixed amount is not a replacement for a configured `CommissionModel` on arbitrary lots/notional.

The main/orchestrator must inject a properly configured simulation cost policy into the actual environment construction call. Tests that manually pass `ExecutionCost(commission=7.0)` do not prove that production main reads `transaction_costs.simulation.commission`.

## 9. Live fill safety and policies

- Broker `fill_price` is authoritative and must never be clamped/rewritten after execution.
- Once a broker Fill exists, a slippage-cap violation should be recorded and reconciled, not cause the Fill record to be dropped. Rejection belongs at the pre-trade/order decision boundary, before broker execution.
- `LiveObservedExecutionCostEngine.observe()` should receive the observed broker commission/fee and exact currencies/rates; simulation expected-cost switches must not suppress actual observations.
- Partial fills need unique `fill_id`, shared `execution_id`, and once-only execution-level fees where applicable.

## 10. Required tests before f05/f06 functional freeze

1. Commission basis tests: `fixed`, `per_lot`, and `percent_notional`, with more than one lot size and a non-zero minimum.
2. Broker/profile selection tests if multiple broker/accounts or symbol overrides are supported.
3. Entry/exit policy matrix tests for `True/False`, verifying spread/slippage remain in price costs while only simulated cash commissions/additional fees change.
4. Factory integration test proving config settings reach the actual components used by the caller.
5. f06 portfolio test proving commission reduces portfolio balance/equity once, for OPEN, INCREASE, REDUCE, CLOSE, and REVERSE.
6. Partial-fill `execution_id` tests proving `per_trade` additional fee is charged exactly once, including duplicate/replayed fills.
7. Live tests proving observed broker costs remain recorded regardless of simulation charge flags and that cap breaches retain authoritative Fill records.
8. `price_costs_embedded_in_fill=False` is rejected unless a fully tested alternate accounting design is intentionally supported.
9. Overnight swap accrual tests across rollover time, triple-swap day, weekend gaps, partial close, reversal, and long/short directions before claiming swap is included.
10. End-to-end parity test proving configured costs flow from config-loader → main/composition root → execution path → portfolio equity/reward/audit.

## 11. Active-main review checklist

When the user asks to revise main:

- First list main/orchestrator candidates and inspect which one is the actual entry point used for `train`, `backtest`, and `live`.
- Trace config loading, `project.random_seed`, `transaction_costs`, symbol-spec snapshot loading, instrument resolution, `TradingEnvironment(...)` / `from_mtf_datasets(...)`, and live broker-fill adapter.
- Confirm whether a current main already creates `CostComponents`; do not assume imports mean runtime use.
- Supply exact patches for only the relevant active files; do not edit GitHub.
- Run tests and report each integration gap honestly. Do not announce functional freeze solely because unit tests pass.

## 12. Suggested order of work

1. Finish `additional_fee.basis` semantics and implementation, including execution-scope behavior for `per_trade`.
2. Connect configured commission model to the actual f06 training/backtest execution path; test lot- and notional-dependent commission.
3. Inspect and wire the active main/composition root to the canonical instruments and cost policy.
4. Confirm live fill + actual commission/fee path and ensure realized fills cannot be lost after execution.
5. Design/implement overnight swap/financing in a separate scoped task.
6. Update the f05 Word/Markdown contract and change log to reflect the final implementation, not the prior ZIP.
7. Only then declare f05 functional freeze, with the exact tested scope recorded.
