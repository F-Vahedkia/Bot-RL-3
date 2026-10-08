# f07_agents — Input / Output Layer

## Contract

## نقش لایه

`f07_agents` لایه‌ی **Decision / Agent** سیستم است.

این لایه `Observation` مربوط به هر `Symbol` را دریافت می‌کند، توسط `Symbol-Agent` برای هر `Symbol` یک تصمیم محلی تولید می‌کند، سپس `Meta-Agent` تصمیم‌های `Symbol`ها را همراه با وضعیت `Portfolio` در سطح Portfolio ترکیب و محدود می‌کند و در نهایت یک `PortfolioDecision` تولید می‌نماید.

### جریان مفهومی اصلی

```text
f04_features
      ↓
Observation
      ↓
AgentObservation
      ↓
Symbol-Agent(s)
      ↓
SymbolAgentOutput
      ↓
Meta-Agent
      +
PortfolioContext
      ↓
PortfolioDecision
      ↓
Risk Layer
```

### f07_agents

* مستقیماً Broker یا MT5 را صدا نمی‌زند.
* Execution واقعی انجام نمی‌دهد.
* Accounting نهایی را انجام نمی‌دهد.
* محاسبه‌ی نهایی Risk متعلق به این لایه نیست.
* Position Sizing نهایی متعلق به این لایه نیست.
* Portfolio-wide allocation و risk توسط Meta-Agent در سطح تصمیم انجام می‌شود، اما **enforcement نهایی Risk متعلق به لایه Risk است**.
* خروجی اصلی این لایه برای downstream برابر `PortfolioDecision` است.

---

# 1. ورودی‌های f07_agents

## 1.1. Observation از f04_features

برای هر `Symbol`، خروجی نهایی `ObservationBuilder` از `f04_features` وارد مرز `f07_agents` می‌شود.

خروجی Feature Layer در ابتدا یک:

```python
pandas.DataFrame
```

است.

برای عبور از مرز Feature → Agent از:

```text
FeatureObservationAdapter
```

استفاده می‌شود.

### جریان

```text
f04_features
    ↓
ObservationBuilder
    ↓
Observation DataFrame
    ↓
FeatureObservationAdapter
    ↓
AgentObservation
    ↓
Symbol-Agent
```

`FeatureObservationAdapter` فقط semantic translation و boundary validation انجام می‌دهد.

### مسئولیت‌های FeatureObservationAdapter

* دریافت Observation نهایی.
* حفظ ترتیب دقیق feature columns.
* استخراج timestamp.
* تبدیل آخرین ردیف به `AgentObservation`.
* کنترل timezone-aware بودن timestamp.
* کنترل numeric و finite بودن مقادیر.
* حفظ Symbol isolation.

### FeatureObservationAdapter انجام نمی‌دهد

* Feature محاسبه نمی‌کند.
* Feature alignment انجام نمی‌دهد.
* Feature selection انجام نمی‌دهد.
* Feature shift انجام نمی‌دهد.
* `SymbolContext` نمی‌سازد.
* `PortfolioContext` نمی‌سازد.
* RL inference انجام نمی‌دهد.

## قرارداد AgentObservation

```text
AgentObservation

    symbol
    base_tf
    timestamp
    values
    feature_names
    metadata
```

### ویژگی‌های مهم

* هر Observation فقط متعلق به یک Symbol است.
* Observationها بین Symbolها مخلوط نمی‌شوند.
* `timestamp` باید timezone-aware باشد.
* `values` باید non-empty و finite باشند.
* `feature_names` در صورت وجود باید با `values` هم‌اندازه باشد.
* ترتیب `values` باید دقیقاً مطابق ترتیب `feature_names` باشد.

برای `train / optimize / replay / evaluation` امکان تبدیل کل Sequence نیز وجود دارد، ولی قرارداد Decision Layer در نقطه‌ی inference همان `AgentObservation` است.

---

# 1.2. وضعیت محلی هر Symbol

هر `Symbol-Agent` یک `SymbolContext` دریافت می‌کند:

```text
SymbolContext

    symbol
    current_side
    current_lots
    exposure
    drawdown
    volatility
    local_risk_score
    metadata
```

این Context فقط مربوط به همان Symbol است.

`Symbol-Agent` نباید اطلاعات Portfolio-wide را مستقیماً دریافت کند.

اطلاعاتی مانند:

* equity کل Portfolio
* correlation بین Symbolها
* concentration کل Portfolio
* margin allocation کل
* Portfolio risk

متعلق به `PortfolioContext` و `Meta-Agent` هستند.

`f07_agents` این runtime state را از لایه‌های پایین‌تر دریافت می‌کند و مالک Data / Accounting نیست.

---

# 1.3. وضعیت Portfolio

`Meta-Agent` یک `PortfolioContext` دریافت می‌کند:

```text
PortfolioContext

    timestamp
    equity
    balance
    used_margin
    free_margin
    margin_level
    drawdown
    daily_drawdown
    exposure
    concentration
    correlation
    risk_blocked
    mode
```

این Context می‌تواند از مسیر زیر ساخته شود:

```text
f06_env.PortfolioState
        ↓
PortfolioContextAdapter
        ↓
PortfolioContext
```

### PortfolioContextAdapter

`PortfolioContextAdapter`:

* accounting انجام نمی‌دهد.
* PnL را محاسبه نمی‌کند.
* margin را محاسبه نمی‌کند.
* risk را محاسبه نمی‌کند.
* exposure را حدس نمی‌زند.
* correlation را حدس نمی‌زند.

بلکه فقط state موجود را به قرارداد `PortfolioContext` منتقل می‌کند.

`exposure`، `concentration` و `correlation` از خارج Adapter دریافت می‌شوند.

---

# 1.4. Configuration و Model Identity

هر `Symbol-Agent` و `Meta-Agent` دارای `Configuration` و `ModelIdentity` هستند.

### Modeهای رسمی

```text
TRAIN
OPTIMIZE
BACKTEST
REPLAY
EVAL
SHADOW
PAPER
LIVE
```

### ModelIdentity

```text
ModelIdentity

    model_name
    model_version
    policy_version
    config_version
    experiment_id
```

این اطلاعات برای موارد زیر حفظ می‌شوند:

* audit
* replay
* evaluation
* experiment lineage
* self-optimization
* model versioning

---

# 1.5. Policy / RL Backend

هسته‌ی Agentها نباید به یک RL framework یا الگوریتم خاص وابسته باشد.

مرز RL به شکل زیر است:

```text
RL Backend
    ↓
RLModelPolicyAdapter
    ↓
SymbolPolicy
    ↓
Symbol-Agent
```

و برای `Meta-Agent`:

```text
RL Backend
    ↓
RLMetaPolicyAdapter
    ↓
MetaPolicy
    ↓
Meta-Agent
```

## RLModelPolicyAdapter

`RLModelPolicyAdapter`:

* Observation را از `AgentObservation` دریافت می‌کند.
* `values` را به vector عددی finite تبدیل می‌کند.
* `model.predict(...)` را فراخوانی می‌کند.
* backend-native output را از طریق `SymbolDecoder` به `PolicyOutput` تبدیل می‌کند.

## RLMetaPolicyAdapter

`RLMetaPolicyAdapter`:

* خروجی‌های `Symbol-Agent` را همراه `PortfolioContext` دریافت می‌کند.
* از طریق یک `observation_builder` آن‌ها را به observation مربوط به `Meta-Agent` تبدیل می‌کند.
* prediction مدل RL را اجرا می‌کند.
* از طریق `MetaDecoder` خروجی را به `MetaPolicyOutput` تبدیل می‌کند.

در نتیجه:

```text
PPO / SAC / DQN / Transformer / ...
```

نباید مستقیماً در هسته‌ی `Symbol-Agent` یا `Meta-Agent` hard-code شوند.

---

# 1.6. Policy Factory

ساخت Policyها توسط `PolicyFactory` انجام می‌شود.

`PolicyFactory` مسئول این مسیر است:

```text
PolicySpec
    ↓
Backend Provider
    ↓
RL Backend Model
    ↓
RLModelPolicyAdapter / RLMetaPolicyAdapter
```

### PolicySpec

```text
PolicySpec

    role
    backend
    algorithm
    policy
    deterministic
    metadata
```

Factory اجازه می‌دهد backendهای مختلف بدون تغییر هسته‌ی Agent ثبت شوند:

```python
register_backend(...)
```

بنابراین وابستگی به backend واقعی از طریق `ModelProvider` تزریق می‌شود.

`f07_agents` مستقیماً config file مانند YAML را نمی‌خواند.

Configuration باید از لایه بالاتر به صورت object / config / spec به Factory و Agentها تزریق شود.

---

# 1.7. ساخت Symbol-Agent و Meta-Agent

ساخت Agentها توسط `AgentFactory` انجام می‌شود.

## Symbol-Agent

```text
symbols
+
SymbolPolicy
+
ModelIdentity
+
mode
        ↓
SymbolAgent
```

## Meta-Agent

```text
MetaPolicy
+
ModelIdentity
+
mode
+
allocation limits
        ↓
MetaAgent
```

### AgentFactory تضمین می‌کند

* Symbolها normalize شوند.
* Symbolها unique باشند.
* برای هر Symbol دقیقاً یک Policy وجود داشته باشد.
* برای هر Symbol دقیقاً یک ModelIdentity وجود داشته باشد.
* Symbol-Agent و Model مربوط به همان Symbol باشند.
* configuration مربوط به Meta-Agent به صورت validated تزریق شود.

---

# 2. جریان داخلی f07_agents

## Symbol Layer

```text
AgentObservation
        +
SymbolContext
        ↓
Symbol-Agent
        ↓
Policy
        ↓
PolicyOutput
        ↓
SymbolAgentOutput
```

## Portfolio Layer

```text
SymbolAgentOutput
        +
PortfolioContext
        ↓
Meta-Agent
        ↓
MetaPolicy
        ↓
MetaPolicyOutput
        ↓
PortfolioDecision
```

## Multi-Symbol

```text
Observation[XAUUSD]
      ↓
Symbol-Agent[XAUUSD]
      ↓
SymbolAgentOutput[XAUUSD]
                         \
                          \
Observation[EURUSD]  → Meta-Agent → PortfolioDecision
      ↓                   /
Symbol-Agent[EURUSD]     /
      ↓
SymbolAgentOutput[EURUSD]

Observation[GBPUSD]
      ↓
Symbol-Agent[GBPUSD]
      ↓
SymbolAgentOutput[GBPUSD]
```

`MultiSymbolDecisionEngine` orchestrator این چرخه را مدیریت می‌کند.

### تضمین‌های MultiSymbolDecisionEngine

* تمام Symbolهای مورد انتظار حاضر باشند.
* هر Observation متعلق به Agent صحیح باشد.
* هر Context متعلق به Agent صحیح باشد.
* timestamp تمام Observationها یکسان باشد.
* timestamp آن‌ها با `PortfolioContext` یکسان باشد.
* mode کل چرخه یکسان باشد.
* تمام خروجی‌های Symbol-Agent دارای همان `decision_id` باشند.
* خروجی‌های آخرین cycle برای audit / replay نگهداری شوند.

---

# 3. خروجی Symbol-Agent

هر `Symbol-Agent` یک `SymbolAgentOutput` تولید می‌کند:

```text
SymbolAgentOutput

    symbol
    timestamp
    mode
    signal
    confidence
    expected_return
    risk_score
    desired_exposure
    stop_price
    model
    decision_id
    metadata
```

## معنای فیلدهای اصلی

### signal

جهت تصمیم:

```text
-1 / 0 / +1
```

### confidence

اطمینان Policy.

### expected_return

بازده مورد انتظار.

### risk_score

امتیاز ریسک محلی.

### desired_exposure

Exposure هدف برای Symbol.

### stop_price

Stop Loss پیشنهادی Symbol-Agent.

## قواعد

```text
signal != 0
    →
desired_exposure != 0

stop_price != None
```

و:

```text
signal != 0
    →
stop_price != None
```

`stop_price` متعلق به Symbol-Agent یک **پیشنهاد خام** است و Meta-Agent می‌تواند آن را override یا حذف کند.

---

# 4. خروجی Meta-Agent

Meta-Agent خروجی خام Policy خود را در قالب `MetaPolicyOutput` دریافت/تولید می‌کند:

```text
MetaPolicyOutput

    capital_allocation
    target_signals
    target_exposure
    margin_allocation
    portfolio_risk
    reason_codes
    target_stop_price
```

سپس Meta-Agent آن را به:

```text
PortfolioDecision
```

تبدیل می‌کند.

---

# 5. PortfolioDecision

خروجی رسمی `f07_agents` برای downstream:

```text
PortfolioDecision

    approved
    timestamp
    mode
    decision_id
    capital_allocation
    margin_allocation
    target_exposure
    target_signals
    portfolio_risk
    reason_codes
    model
    target_stop_price
    metadata
```

## فیلدهای اصلی

### target_signals

جهت هدف هر Symbol.

### target_exposure

Exposure هدف هر Symbol.

### capital_allocation

Allocation پیشنهادی سرمایه.

### margin_allocation

Allocation پیشنهادی Margin.

### portfolio_risk

ارزیابی/پیشنهاد ریسک Portfolio.

### target_stop_price

Stop Price نهایی در سطح Portfolio.

---

# 6. قرارداد Stop Price

Symbol-Agent:

```text
SymbolAgentOutput.stop_price
```

را پیشنهاد می‌دهد.

Meta-Agent می‌تواند:

```text
symbol absent
    →
inherit Symbol-Agent stop_price
```

یا:

```text
symbol → float
    →
override Symbol-Agent stop_price
```

یا:

```text
symbol → None
    →
explicitly remove stop price
```

در نتیجه:

```text
Symbol-Agent stop
        ↓
Meta-Agent override / inherit / remove
        ↓
PortfolioDecision.target_stop_price
```

این Stop Price هنوز بخشی از Decision Layer است و به معنی ارسال مستقیم Stop Order به Broker نیست.

---

# 7. مرز خروجی به Risk Layer

خروجی اصلی `f05`:

```text
PortfolioDecision
```

است.

### مسیر معماری مقصد

```text
f04_features
      ↓
Observation
      ↓
FeatureObservationAdapter
      ↓
f07_agents
      ↓
PortfolioDecision
      ↓
f08_risk
      ↓
Risk Decision
      ↓
Action / Execution boundary
      ↓
f06_env / Execution
```

`f07_agents` نباید Risk Enforcement نهایی را انجام دهد.

Risk Layer می‌تواند بر اساس موارد زیر تصمیم Meta-Agent را بررسی، محدود، رد یا اصلاح کند:

* Portfolio constraints
* margin constraints
* exposure limits
* correlation
* concentration
* drawdown
* risk budgets
* سایر risk policies

---

# 8. Position Sizing و Lots

`desired_exposure` و `target_exposure` به معنی lots نهایی نیستند.

تبدیل:

```text
exposure
    ↓
lots
```

به یک `Position Sizing Policy` نیاز دارد.

بنابراین sizing نباید در `Symbol-Agent` یا `Meta-Agent` hard-code شود.

در معماری فعلی:

```text
PortfolioDecision
      ↓
PortfolioActionBuilder
      ↓
PositionSizer
      ↓
PortfolioAction
```

وجود `PortfolioActionBuilder` در `f07_agents` یک **compatibility / boundary component** است و Decision Layer را به Broker متصل نمی‌کند.

برای معماری مقصد، تبدیل نهایی پس از Risk باید در مسیر مناسب Risk / Environment انجام شود.

---

# 9. Action Layer Boundary

`PortfolioActionBuilder`:

* Broker را نمی‌شناسد.
* Order ارسال نمی‌کند.
* Execution انجام نمی‌دهد.
* Accounting انجام نمی‌دهد.

بلکه فقط `PortfolioDecision` را به semantic action تبدیل می‌کند.

### ساختار

```text
PortfolioDecision
      ↓
PortfolioActionBuilder
      ↓
PortfolioAction
      ↓
Environment / Execution
```

`PositionSizer` از طریق dependency injection وارد می‌شود.

---

# 10. Audit / Replay

`f07_agents` باید decision cycleها را برای audit و replay قابل مشاهده و بازتولید نگه دارد.

`decision_id` شناسه‌ی یک Decision Cycle است.

این شناسه باید در مسیر زیر حفظ شود:

```text
Observation
    ↓
SymbolAgentOutput
    ↓
PortfolioDecision
```

`MultiSymbolDecisionEngine` آخرین خروجی‌های Symbol-Agent و آخرین `PortfolioDecision` را نگه می‌دارد.

لایه Audit نیز می‌تواند:

```text
SymbolAgentOutput

Meta / PortfolioDecision
```

را به صورت ordered و bounded نگهداری کند.

### هدف

* traceability
* replay
* debugging
* model lineage
* deterministic validation

است.

Decision Layer باید در شرایط ورودی یکسان، رفتار deterministic و قابل replay داشته باشد؛ البته deterministic بودن backend RL باید توسط configuration / policy backend کنترل شود.

---

# 11. Model Lineage

هر `SymbolAgentOutput` و `PortfolioDecision` باید `Model Identity` مربوط به مدل تصمیم‌گیرنده را حفظ کند.

این امکان را فراهم می‌کند که مشخص شود:

```text
کدام model

کدام model_version

کدام policy_version

کدام config_version

کدام experiment
```

تصمیم را تولید کرده است.

این اطلاعات برای موارد زیر ضروری هستند:

* Audit
* Evaluation
* Replay
* Experiment comparison
* Self-Optimization
* Model Promotion

---

# 12. Symbol Isolation

اصل مهم `f07_agents`:

```text
One Symbol-Agent
        =
One Symbol
```

هر `Symbol-Agent` فقط این موارد را دریافت می‌کند:

```text
Observation همان Symbol

+

SymbolContext همان Symbol
```

اطلاعات global نباید وارد `SymbolContext` شوند.

اطلاعات Portfolio-wide فقط در:

```text
PortfolioContext
```

برای `Meta-Agent` قرار می‌گیرند.

`MultiSymbolDecisionEngine` نیز باید این isolation را حفظ کند.

---

# 13. Timestamp Contract

یک Decision Cycle چند Symbolی باید timestamp مشترک داشته باشد:

```text
Observation[XAUUSD].timestamp
=
Observation[EURUSD].timestamp
=
Observation[GBPUSD].timestamp
=
PortfolioContext.timestamp
```

و خروجی همه Symbol-Agentها نیز باید همان timestamp را حفظ کنند.

در نتیجه یک Decision Cycle از نظر زمانی atomic است.

---

# 14. Mode Contract

تمام اجزای یک چرخه Decision باید mode سازگار داشته باشند.

### Modeهای رسمی

```text
train
optimize
backtest
replay
eval
shadow
paper
live
```

Mode بخشی از قرارداد Decision Layer است و در موارد زیر حفظ می‌شود:

```text
Agent
Policy
SymbolAgentOutput
PortfolioContext
PortfolioDecision
```

---

# 15. Configuration Boundary

`f07_agents` مالک فایل Config نیست.

یعنی موارد زیر نباید داخل هسته‌ی این لایه قرار گیرند:

```text
YAML reader

JSON reader

Config file parser
```

لایه بالاتر باید configuration را تولید و به صورت:

```text
Config Object

Spec

Factory Config

Agent Config

Policy Config
```

به `f05` تزریق کند.

---

# 16. خروجی‌های اصلی قابل مصرف توسط لایه‌های دیگر

## ورودی Symbol-Agent

```text
AgentObservation
+
SymbolContext
```

## خروجی Symbol-Agent

```text
SymbolAgentOutput
```

## ورودی Meta-Agent

```text
SymbolAgentOutput(s)
+
PortfolioContext
```

## خروجی Meta-Agent

```text
PortfolioDecision
```

## خروجی اختیاری semantic boundary

```text
PortfolioDecision
    ↓
PortfolioActionBuilder
    ↓
PortfolioAction
```

این مسیر باید با معماری Risk/Environment مقصد سازگار نگه داشته شود.

---

# 17. قرارداد کلی f07_agents با سایر لایه‌ها

## ورودی از f04_features

```text
f04_features
    ↓
Observation DataFrame
    ↓
FeatureObservationAdapter
    ↓
AgentObservation
```

## ورودی از f06_env / runtime state

```text
PortfolioState
    ↓
PortfolioContextAdapter
    ↓
PortfolioContext
```

و برای وضعیت محلی Symbol:

```text
Symbol runtime state
    ↓
SymbolContext
```

## خروجی به downstream

```text
f07_agents
    ↓
PortfolioDecision
```

و در مسیر semantic action:

```text
PortfolioDecision
    ↓
PortfolioAction
```

---

# 18. مسئولیت‌های f07_agents نیست

این لایه نباید مسئول موارد زیر باشد:

* Broker connection
* MT5 communication
* Order submission
* Order execution
* Fill handling
* Final accounting
* Final PnL calculation
* Final margin calculation
* Final risk enforcement
* Final position sizing
* Market data acquisition
* Feature calculation
* Feature alignment
* Feature persistence

---

# 19. اجزای کلیدی f07_agents

اجزای اصلی فعلی:

```text
contracts.py
    ↓
Data Contracts
```

```text
observation_adapter.py
    ↓
f04_features → f07_agents boundary
```

```text
portfolio_context_adapter.py
    ↓
f06_env → PortfolioContext
```

```text
symbol_agent.py
    ↓
Local Symbol Decision
```

```text
meta_agent.py
    ↓
Portfolio Decision
```

```text
decision_engine.py
    ↓
Multi-Symbol orchestration
```

```text
rl_policy_adapter.py
    ↓
RL backend boundary
```

```text
policy_factory.py
    ↓
Policy construction
```

```text
agent_factory.py
    ↓
Agent construction
```

```text
agent_state.py
    ↓
Runtime state
```

```text
audit.py
    ↓
Audit / replay support
```

```text
action_builder.py
    ↓
Semantic Decision → Action compatibility boundary
```

---

# 20. قرارداد نهایی معماری

قرارداد فشرده‌ی کل سیستم:

```text
                         f04_features
                              │
                              ▼
                   Observation DataFrame
                              │
                              ▼
                 FeatureObservationAdapter
                              │
                              ▼
                      AgentObservation
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          SymbolContext              SymbolContext
                 │                         │
                 ▼                         ▼
          Symbol-Agent               Symbol-Agent
                 │                         │
                 ▼                         ▼
      SymbolAgentOutput         SymbolAgentOutput
                 │                         │
                 └────────────┬────────────┘
                              │
                              │
                       PortfolioContext
                              │
                              ▼
                         Meta-Agent
                              │
                              ▼
                     PortfolioDecision
                              │
                              ▼
                          f08_risk
                              │
                              ▼
                    Risk-controlled action
                              │
                              ▼
                       f06_env / execution
```

---

# اصل نهایی

`f07_agents` یک **Decision Layer مستقل، چند-Symbol، RL-compatible، deterministic/replayable و audit-able** است.

مرزهای اصلی آن عبارت‌اند از:

## INPUT

```text
AgentObservation
SymbolContext
PortfolioContext
Policy / Model dependencies
Configuration / Specs
```

## OUTPUT

```text
SymbolAgentOutput
PortfolioDecision
```

و قرارداد اصلی برای downstream:

```text
PortfolioDecision
```

است.

هر لایه‌ی دیگر باید از طریق همین Contractها با `f07_agents` ارتباط برقرار کند و نباید برای دسترسی مستقیم به implementation داخلی `Symbol-Agent` یا `Meta-Agent` وابسته شود.
