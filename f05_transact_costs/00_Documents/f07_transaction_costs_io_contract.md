# Bot-RL-3 V8 — f05_transact_costs Interface & I/O Contract

**Source basis:** `f05_transact_costs_live_slippage_tolerance_patched.zip` (package 1.0.0)
**Verified:** 19/19 tests passed on the source ZIP.

## Purpose
This document is the reusable interface/data contract for `f05_transact_costs`. Future work that needs transaction costs can use this contract instead of receiving the full layer source, provided no internal implementation change has altered the public contract.

## Runtime boundary
- `f05_transact_costs` consumes an already-loaded `Mapping` from the project config loader.
- It never opens/parses YAML itself.
- Account currency resolves from `project.base_currency`; deterministic seed comes from `project.random_seed`.

## Public contracts
### Enums
`FillRole`: `ENTRY`, `EXIT`

`CommissionBasis`: `FIXED`, `PER_LOT`, `PERCENT_NOTIONAL`

`CostRealization`: `EXPECTED`, `ACTUAL`

`CostSource`: `SIMULATION`, `BROKER`

### MarketQuote
Fields: `symbol`, `bid`, `ask`, `timestamp`, `source`.
Rules: bid/ask positive finite, `bid <= ask`, timestamp timezone-aware. Derived: `mid=(bid+ask)/2`, `spread_price=ask-bid`, `half_spread_price=(ask-bid)/2`; executable price is ask for BUY (+1) and bid for SELL (-1).

### InstrumentSpec
Fields: `symbol`, `pip_size`, `tick_size`, `tick_value`, `tick_value_currency`, optional `contract_size`, `volume_min`, `volume_step`, `volume_max`.
Derived: `value_per_price_unit_per_lot=tick_value/tick_size`.

### CommissionModel
Fields: `basis`, `rate`, `currency`, `minimum`.
Formula: fixed=`rate`; per_lot=`rate*lots`; percent_notional=`rate*notional`; final native commission=`max(raw,minimum)`.

### TransactionCostRequest
Required fields:
- `symbol: str`
- `side: int` (-1/+1)
- `role: FillRole`
- `lots: float > 0`
- `quote: MarketQuote`
- `instrument: InstrumentSpec`
- `tick_value_to_account_rate: float > 0`
- `commission_model: CommissionModel`
- `account_currency: str`
- `commission_to_account_rate: float > 0`
- `commission_notional: float >= 0`
- `additional_fee: float >= 0`
- `additional_fee_currency: str`
- `additional_fee_to_account_rate: float > 0`
- `slippage_price: float >= 0`

Derived: `market_price`, `reference_price`, `spread_price`, `price_value_per_unit`.

## Outputs
### TransactionCostBreakdown
Fields: `realization`, `spread_price`, `slippage_price`, `spread_cost`, `slippage_cost`, `commission`, `additional_fee`, `price_cost_total`, `cash_cost_total`, `total_cost`, `price_costs_embedded_in_fill`, `account_currency`.

Formula:
```text
price_cost_total = spread_cost + slippage_cost
total_cost = price_cost_total + commission + additional_fee
if price_costs_embedded_in_fill:
    cash_cost_total = commission + additional_fee
else:
    cash_cost_total = total_cost
```

When `price_costs_embedded_in_fill=true`, spread/slippage must not be charged a second time in realized-PnL cash accounting.

### ExecutionFill
Fields: `execution_id`, `fill_id`, `symbol`, `side`, `role`, `lots`, `bid`, `ask`, `reference_price`, `market_price`, `fill_price`, `quote_timestamp`, `execution_timestamp`, `cost`, `source`, optional `broker_order_id`, `model_version`, `slippage_cap_exceeded`.

Invariant: `source=SIMULATION` implies `cost.realization=EXPECTED`; `source=BROKER` implies `cost.realization=ACTUAL`. For broker fills `broker_order_id` is required. `execution_timestamp >= quote_timestamp`.

**Live authoritative-fill rule:** broker `fill_price` is authoritative and immutable. It is never clamped/overwritten. Actual adverse slippage is derived from the broker fill.

BUY: `max(0, fill_price-market_price)`
SELL: `max(0, market_price-fill_price)`

Live cap: `cap_price=cap_pips*pip_size`; `slippage_cap_exceeded = adverse_slippage > cap_price`.
Current policy: `cap_policy=tolerance`, `on_exceed=flag_and_reconcile`. A breach flags the fill and propagates to reconciliation; it does not change `fill_price`. `reject` is also supported by the API/config parser.

### CostReconciliation
Fields: `execution_id`, `expected_lots`, `actual_lots`, `expected_total`, `actual_total`, `variance`, `account_currency`, `slippage_cap_exceeded`.

`variance = actual_total - expected_total`. Expected fills must be simulation/expected; actual fills must be broker/actual. Partial fills share `execution_id` and use unique `fill_id`; reconciliation sums all fills.

### RoundTripCostEstimate
Fields: `entry`, `exit`; properties `total_cost`, `account_currency`. Entry/exit must match on lots, symbol and account currency. Used for downstream Risk pre-trade economics.

## Main APIs
- `TransactionCostConfig.from_mapping(raw) -> TransactionCostConfig`
- `TransactionCostConfig.resolve_account_currency(project_config) -> str`
- `TransactionCostCalculator.calculate(request, realization, price_costs_embedded_in_fill) -> TransactionCostBreakdown`
- `TransactionCostCalculator.build_fill(...) -> ExecutionFill`
- `SimulationExecutionCostEngine.execute(...) -> ExecutionFill`
- `LiveObservedExecutionCostEngine.observe(...) -> ExecutionFill`
- `TransactionCostLedger.record(fill)`
- `TransactionCostLedger.reconcile_execution(expected, actual) -> CostReconciliation`
- `RoundTripCostEstimator.estimate(entry, exit) -> RoundTripCostEstimate`
- `build_components(config, project_random_seed) -> CostComponents`

## External data expected by f07
- config-loader `Mapping`
- historical `MarketQuote` for Simulation
- broker `MarketQuote` for Live
- `InstrumentSpec`
- currency conversion rates to account currency
- actual broker `fill_price`, `execution_id`, `fill_id`, broker order ID
- observed commission and optional observed fees plus conversion rates

## Integration map
- `f01_config`: supplies loaded config only.
- `f06_env`: simulation execution/accounting consumes `ExecutionFill` and `TransactionCostBreakdown`.
- `f07_agents`: may use the public cost outputs, but must not duplicate f07 implementation.
- `f08_risk`: uses `RoundTripCostEstimator` for expected entry+exit cost as a downstream economic input; Risk policy remains owned by f06.
- broker/transaction layer: supplies live quote and actual fill; f07 observes and accounts for costs.
- audit/monitoring: consumes `ExecutionFill` and `CostReconciliation`.

## Non-negotiable invariants
1. f07 does not read YAML.
2. BUY uses ask; SELL uses bid.
3. `spread_price` is half the bid/ask width.
4. `slippage_price` is non-negative adverse slippage.
5. Live broker fill is authoritative and never clamped/overwritten.
6. Live cap is tolerance; breach is flagged/reconciled.
7. Source and realization must agree.
8. Broker fills require `broker_order_id`.
9. Partial fills share `execution_id` and have unique `fill_id`.
10. No double charging of spread/slippage when embedded in fill price.
11. Do not invent hidden currency/quote/conversion defaults.
12. Downstream layers consume the contract rather than duplicating implementation.

## Handoff text
```text
Bot-RL-3 f05_transact_costs contract reference.
Use TransactionCostRequest as the resolved pre-cost input.
Primary outputs: TransactionCostBreakdown and ExecutionFill.
Reconciliation: CostReconciliation via TransactionCostLedger.
Pre-trade expected round trip: RoundTripCostEstimator.
Simulation: historical bid/ask + deterministic slippage model; source=simulation; realization=expected.
Live: broker quote + actual broker fill; source=broker; realization=actual.
Live cap: tolerance; on_exceed=flag_and_reconcile; actual fill_price is authoritative and must never be clamped/overwritten.
Accounting: price_costs_embedded_in_fill=true; do not double-charge spread/slippage.
Partial fills: same execution_id, unique fill_id.
Do not invent parallel APIs or hidden defaults.
```
