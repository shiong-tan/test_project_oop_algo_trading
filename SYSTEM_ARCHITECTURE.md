# Risk Management System Architecture

## System Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      ML-BASED TRADING SYSTEM                              │
│                    WITH RISK MANAGEMENT FRAMEWORK                          │
└────────────────────────────────────────────────────────────────────────────┘

                           ┌─────────────────────┐
                           │   Market Data       │
                           │  (AAPL daily bars)  │
                           └──────────┬──────────┘
                                      │
                           ┌──────────▼──────────┐
                           │  Feature Engineer   │
                           │  (SMA, EWMA, Vol)   │
                           └──────────┬──────────┘
                                      │
                           ┌──────────▼──────────┐
                           │  ML Model Training  │
                           │ (GaussianNB, etc)   │
                           └──────────┬──────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
         ┌──────────▼──────────┐           ┌──────────▼──────────┐
         │ Model Predictions   │           │ Prediction Proba    │
         │ (0=down, 1=up)      │           │ (Confidence 0-1)    │
         └──────────┬──────────┘           └──────────┬──────────┘
                    │                                   │
                    └─────────────┬─────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  TRADING ENGINE           │
                    │  (Entry/Exit Logic)       │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────────────────────────┐
                    │           PRE-TRADE VALIDATION               │
                    ├─────────────────────────────────────────────┤
                    │  1. Signal Quality Check (Confidence)       │
                    │  2. Position Sizing Calculation             │
                    │  3. Risk/Reward Ratio Validation            │
                    │  4. Portfolio Heat Check                    │
                    │  5. Drawdown Zone Check                     │
                    │  6. Account Status Verification             │
                    └─────────────┬─────────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────────────────────────┐
                    │         TRADE ACCEPTANCE DECISION             │
                    ├─────────────────────────────────────────────┤
                    │  ALL checks pass? → ACCEPT TRADE            │
                    │  ANY check fails? → SKIP TRADE              │
                    └─────────────┬─────────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────────────────────────┐
                    │    POSITION SIZING CALCULATION              │
                    ├─────────────────────────────────────────────┤
                    │  Method 1: Fixed Fractional (RECOMMENDED)   │
                    │    Units = (Account Risk %) / Risk Per Unit │
                    │                                             │
                    │  Method 2: Kelly Criterion (ADVANCED)       │
                    │    Fraction = (WinRate × B - LossRate) / B │
                    │                                             │
                    │  Method 3: Confidence-Based (ADJUSTMENT)    │
                    │    Actual Units = Base Units × Confidence  │
                    │    Multiplier                               │
                    └─────────────┬─────────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────────────────────────┐
                    │   STOP-LOSS & TAKE-PROFIT CALCULATION       │
                    ├─────────────────────────────────────────────┤
                    │  Entry Price: Determined by market          │
                    │  Hard Stop:   Entry × (1 - 0.03) = -3%     │
                    │  Take Profit: Entry × (1 + 0.05) = +5%     │
                    │  Trailing:    Activated at +2% profit       │
                    │  Time Stop:   Close after 10 days           │
                    └─────────────┬─────────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────────────────────────┐
                    │         TRADE EXECUTION                      │
                    ├─────────────────────────────────────────────┤
                    │  1. Create TradeExecution object             │
                    │  2. Update account balance                  │
                    │  3. Add to portfolio heat tracking           │
                    │  4. Record in trade history                 │
                    │  5. Log to console (if verbose)             │
                    └─────────────┬─────────────────────────────────┘
                                  │
                                  │
        ┌─────────────────────────▼─────────────────────────────┐
        │              DURING POSITION (EACH BAR)                │
        └─────────────────────────┬─────────────────────────────┘
                                  │
        ┌─────────────┬───────────┴───────────┬──────────────┐
        │             │                       │              │
   ┌────▼────┐  ┌────▼─────┐  ┌────────▼──┐  ┌────────▼──┐
   │ Check   │  │ Check    │  │ Check    │  │ Check     │
   │ Hard    │  │ Take-    │  │ Trailing │  │ Time Stop │
   │ Stop    │  │ Profit   │  │ Stop     │  │ (10 days) │
   └────┬────┘  └────┬─────┘  └────┬─────┘  └────┬──────┘
        │             │             │             │
        │ If HIT      │ If HIT      │ If HIT      │ If HIT
        │             │             │             │
        ├─────────────┴─────────────┴─────────────┴──────────┐
        │                                                    │
        │            ┌───────────────────────────────┐       │
        │            │ Close Trade                   │       │
        │            │ - Record exit price           │       │
        │            │ - Calculate P&L               │       │
        │            │ - Calculate R-Multiple        │       │
        │            │ - Record exit type            │       │
        │            │ - Remove from portfolio heat  │       │
        │            │ - Update balance              │       │
        │            └───────────┬───────────────────┘       │
        │                        │                          │
        └────────────────────────┼──────────────────────────┘
                                 │
                    ┌────────────▼──────────────┐
                    │  EQUITY & METRICS UPDATE   │
                    ├────────────────────────────┤
                    │ Current Equity = Balance + │
                    │   (Units Held × Price)    │
                    │                            │
                    │ Update Drawdown Tracker    │
                    │ Check Drawdown Zones:      │
                    │  - Normal (0-5%)          │
                    │  - Caution (5-7.5%)       │
                    │  - Alert (7.5-10%)        │
                    │  - Emergency (>10%)       │
                    │                            │
                    │ Calculate Daily Return     │
                    │ Track Equity Curve         │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────▼──────────────┐
                    │  RISK METRICS CALCULATION  │
                    ├────────────────────────────┤
                    │ For Completed Trades:      │
                    │  - Win Rate (%)            │
                    │  - Avg Win / Avg Loss      │
                    │  - Profit Factor           │
                    │  - R Expectancy            │
                    │                            │
                    │ For Portfolio:             │
                    │  - Max Drawdown            │
                    │  - Sharpe Ratio            │
                    │  - Sortino Ratio           │
                    │  - VaR (95%)               │
                    │  - CVaR (95%)              │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────▼──────────────┐
                    │  DECISION: CONTINUE?      │
                    ├────────────────────────────┤
                    │ Check Drawdown Zone        │
                    │                            │
                    │ If NORMAL (0-5%):         │
                    │   Position Multiplier=1.0 │
                    │   Trade normally          │
                    │                            │
                    │ If CAUTION (5-7.5%):      │
                    │   Position Multiplier=0.5 │
                    │   Reduce sizes 50%        │
                    │   Tighten confidence      │
                    │                            │
                    │ If ALERT (7.5-10%):       │
                    │   Position Multiplier=0.25│
                    │   Only best signals       │
                    │   Close weak positions    │
                    │                            │
                    │ If EMERGENCY (>10%):      │
                    │   Position Multiplier=0.0 │
                    │   STOP ALL TRADING        │
                    │   Review system           │
                    └────────────┬───────────────┘
                                 │
                                 └─────────────┐
                                               │
                                    ┌──────────▼──────────┐
                                    │   END OF BACKTEST   │
                                    │  Generate Reports   │
                                    │  Save Metrics       │
                                    └─────────────────────┘
```

---

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BACKTESTING ENGINE (BacktestingBase)             │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  DATA PROCESSING (FinancialData Class)                           │
  ├──────────────────────────────────────────────────────────────────┤
  │  • Load market data                                              │
  │  • Calculate technical features (SMA, EWMA, Vol)                │
  │  • Create lagged features                                        │
  │  • Scale features                                                │
  │  • Train ML model                                                │
  │  • Generate predictions + probabilities                          │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  POSITION SIZING (PositionSizer Classes)                         │
  ├──────────────────────────────────────────────────────────────────┤
  │  FixedFractionalSizer                                            │
  │  ├─ Input: Entry, Stop, Account Value, Risk%                   │
  │  └─ Output: Number of units                                     │
  │                                                                  │
  │  ConfidenceBasedSizer (wraps FixedFractional)                   │
  │  ├─ Input: Confidence (0-1)                                     │
  │  └─ Output: Position multiplier (0-1)                           │
  │                                                                  │
  │  KellyCriterionSizer (for advanced usage)                       │
  │  ├─ Input: Win rate, Avg win, Avg loss                         │
  │  └─ Output: Fraction to risk (with safety factor)               │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  STOP-LOSS MANAGEMENT (StopLossManager)                          │
  ├──────────────────────────────────────────────────────────────────┤
  │  • Calculate stop-loss levels                                    │
  │  • Calculate take-profit levels                                  │
  │  • Validate risk/reward ratio                                    │
  │  • Calculate trailing stop levels                                │
  │  • Check stop hits at each bar                                   │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  TRADE EXECUTION (TradeExecution Objects)                        │
  ├──────────────────────────────────────────────────────────────────┤
  │  Per Trade:                                                      │
  │  ├─ trade_id: unique identifier                                 │
  │  ├─ symbol: stock being traded                                  │
  │  ├─ entry_price: price at entry                                 │
  │  ├─ stop_loss: hard stop price                                  │
  │  ├─ take_profit: target price                                   │
  │  ├─ units: position size                                        │
  │  ├─ confidence: model confidence                                │
  │  ├─ entry_date: when entered                                    │
  │  ├─ exit_price: where exited                                    │
  │  ├─ exit_date: when exited                                      │
  │  ├─ exit_type: (stop/target/time/signal)                       │
  │  ├─ pnl: profit/loss in dollars                                 │
  │  └─ r_multiple: outcome in R-multiples                          │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  DRAWDOWN TRACKING (DrawdownTracker)                             │
  ├──────────────────────────────────────────────────────────────────┤
  │  • Track peak equity (running maximum)                           │
  │  • Calculate current drawdown %                                  │
  │  • Calculate current drawdown $                                  │
  │  • Determine drawdown zone (NORMAL/CAUTION/ALERT/EMERGENCY)    │
  │  • Get position sizing multiplier based on zone                 │
  │  • Enforce max drawdown limit (stop trading if exceeded)        │
  │  • Calculate maximum drawdown encountered                        │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  PORTFOLIO HEAT (PortfolioHeat)                                  │
  ├──────────────────────────────────────────────────────────────────┤
  │  • Track all open positions                                      │
  │  • Calculate total risk (heat) across positions                 │
  │  • Prevent new positions if heat would exceed limit             │
  │  • Remove positions from heat tracking when closed              │
  │  • Report current heat usage                                     │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  TRADE VALIDATION (TradeValidator)                               │
  ├──────────────────────────────────────────────────────────────────┤
  │  Pre-Trade Checks:                                               │
  │  ├─ Confidence within range?                                    │
  │  ├─ Position size acceptable?                                   │
  │  ├─ Risk/reward ratio sufficient?                               │
  │  ├─ Portfolio heat allows?                                      │
  │  ├─ Drawdown not in emergency?                                  │
  │  └─ Account status OK?                                          │
  │                                                                  │
  │  Returns: (is_valid, list_of_rejection_reasons)                │
  └──────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │  RISK METRICS (RiskMetrics Static Methods)                       │
  ├──────────────────────────────────────────────────────────────────┤
  │  • calculate_var_95(): Value at Risk                             │
  │  • calculate_cvar_95(): Conditional VaR                          │
  │  • calculate_sharpe_ratio(): Risk-adjusted return                │
  │  • calculate_sortino_ratio(): Downside risk only                │
  │  • calculate_max_drawdown(): Peak-to-trough decline              │
  │  • calculate_r_expectancy(): Expected return per trade           │
  │  • calculate_profit_factor(): Ratio of wins to losses            │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
MARKET DATA (Bars)
        │
        ▼
┌────────────────────────────────┐
│ Feature Engineering            │
│ - Calculate SMA, EWMA, Vol     │
│ - Create lagged features       │
│ - Normalize features           │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│ ML Model                       │
│ - Predict: 0 (down) or 1 (up) │
│ - Predict_proba: confidence    │
└────────────────────────────────┘
        │
        ├─ prediction (0/1)
        ├─ confidence (0-1)
        └─ date, price
                │
                ▼
        ┌───────────────────────┐
        │ Trade Signal          │
        │ (Buy/Sell/Hold)       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────────────────────┐
        │ PRE-TRADE VALIDATION                  │
        │ ├─ TradeValidator.validate_trade()   │
        │ ├─ Return: (valid, rejection_reasons)│
        │ └─ If invalid, skip trade            │
        └───────────┬───────────────────────────┘
                    │
                    ├─ YES ──┬─────────────────────────┐
                    │        │                         │
                    │        ▼                         │
                    │   ┌──────────────────────────────────┐
                    │   │ Position Sizing                  │
                    │   │ ├─ FixedFractionalSizer         │
                    │   │ │  (Entry - Stop) / Account Risk│
                    │   │ ├─ ConfidenceBasedSizer         │
                    │   │ │  Adjust by confidence        │
                    │   │ └─ Result: units               │
                    │   └───────────┬────────────────────┘
                    │               │
                    │               ▼
                    │   ┌──────────────────────────────────┐
                    │   │ Stop Loss Calculation            │
                    │   │ ├─ Hard Stop: Entry × (1 - 0.03)│
                    │   │ ├─ Take Profit: Entry × (1 +0.05│
                    │   │ └─ Validate Risk/Reward >= 1:2  │
                    │   └───────────┬────────────────────┘
                    │               │
                    │               ▼
                    │   ┌──────────────────────────────────┐
                    │   │ Portfolio Heat Check             │
                    │   │ ├─ Current heat + new position  │
                    │   │ ├─ Check <= 3% limit            │
                    │   │ └─ If exceeds, reject           │
                    │   └───────────┬────────────────────┘
                    │               │
                    │       YES     │      NO
                    │               ▼
                    │           ┌───────────┐
                    │           │ Trade     │
                    │           │ Accepted  │
                    │           └─────┬─────┘
                    │                 │
                    └─────── NO ──────┴──────────────┐
                                                    │
                                                    ▼
                                        ┌──────────────────────┐
                                        │ Trade Rejected       │
                                        │ Skip to next bar     │
                                        └──────────────────────┘
                                                    │
                                                    ▼
                        ┌───────────────────────────┴────────────────────┐
                        │                                                │
            ┌───────────▼──────────────┐              ┌────────────────▼─┐
            │ TRADE EXECUTION          │              │ MONITORING      │
            │ ├─ Create TradeExecution │              │ ├─ Check stops  │
            │ ├─ Debit account balance │              │ ├─ Check target │
            │ ├─ Add to heat tracker   │              │ ├─ Check trail  │
            │ └─ Log to history        │              │ ├─ Check time   │
            └───────────┬──────────────┘              │ └─ Update equity│
                        │                             └────────┬────────┘
                        │                                      │
                        │                                      │
                        ├──────────────────────┬───────────────┤
                        │                      │               │
        ┌───────────────▼───┐      ┌──────────▼──┐    ┌──────▼─────┐
        │ Update Account    │      │ Daily Equity│    │ Calculate  │
        │ ├─ Current balance│      │ Curve       │    │ Metrics    │
        │ ├─ Units held     │      │ Daily       │    │ ├─ Return  │
        │ └─ Position value │      │ Returns     │    │ ├─ Sharpe  │
        └───────────────────┘      └─────────────┘    │ ├─ VaR     │
                                                       │ └─ R-Mult  │
                                                       └────────────┘
                                                             │
                                                             ▼
                                                    ┌─────────────────┐
                                                    │ End of Backtest │
                                                    │ Generate Report │
                                                    └─────────────────┘
```

---

## Key Classes and Their Relationships

```
                          FinancialData (Base)
                                  │
                                  │ inherits
                                  │
                              BacktestingBase
                                  │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
        creates   │               uses               │
                  ▼                 ▼                 │
        ┌─────────────────┐ ┌──────────────────┐    │
        │ PositionSizer   │ │ TradeExecution   │    │
        │                 │ │ Objects          │    │
        ├─ Fixed          │ ├─ per trade       │    │
        │   Fractional    │ ├─ entry/exit      │    │
        │ ├─ Kelly        │ │ prices           │    │
        │ └─ Confidence   │ └─ P&L tracking    │    │
        └────────┬────────┘                         │
                 │                                  │
        invokes  │                                  │
                 ▼                                  │
        ┌─────────────────┐                        │
        │StopLossManager  │                        │
        ├─ calc stops     │                        │
        ├─ calc targets   │                        │
        ├─ validate R:R   │                        │
        └─ trailing stop  │                        │
                                                  │
        ┌────────────────────────────────┐        │
        │ DrawdownTracker                │        │
        ├─ track equity peak             │        │
        ├─ calc current DD               │        │
        ├─ determine zones               │        │
        └─ enforce max DD limit          │        │
                                         │        │
        ┌────────────────────────────────┐        │
        │ PortfolioHeat                  │        │
        ├─ track open positions          │        │
        ├─ sum total risk                │        │
        ├─ enforce heat limits           │        │
        └─ manage multi-position risk    │        │
                                         │        │
        ┌────────────────────────────────┐        │
        │ TradeValidator                 │        │
        ├─ validate signal quality       │◄───────┘
        ├─ validate risk/reward          │
        ├─ validate position size        │
        ├─ validate portfolio status     │
        └─ reject invalid trades         │
                                         │
        ┌────────────────────────────────┐
        │ RiskMetrics (Static)           │
        ├─ calculate VaR                 │
        ├─ calculate Sharpe              │
        ├─ calculate max DD              │
        ├─ calculate R expectancy        │
        └─ calculate profit factor       │
```

---

## Execution Flow: Single Trade Cycle

```
STEP 1: Get Market Signal
    ├─ Market bar arrives
    ├─ ML model predicts: 0 (down) or 1 (up)
    ├─ Model predicts probability: 0.52-0.70 (confidence)
    └─ Trader receives: (signal, confidence, price, date)

STEP 2: Pre-Trade Validation
    ├─ TradeValidator.validate_trade()
    │   ├─ Check confidence >= 52%?
    │   ├─ Check confidence <= 70%?
    │   ├─ Check position size <= 5%?
    │   ├─ Check risk/reward >= 1:2?
    │   ├─ Check heat + new position <= 3%?
    │   ├─ Check drawdown not in EMERGENCY?
    │   └─ Return: (valid, rejection_reasons)
    │
    ├─ If NOT valid: SKIP TRADE, go to STEP 10

STEP 3: Position Sizing
    ├─ FixedFractionalSizer.calculate_position_size()
    │   ├─ Account risk = $10,000 × 1% = $100
    │   ├─ Entry price = $80.00
    │   ├─ Stop price = $77.60 (3% below)
    │   ├─ Risk per unit = $80 - $77.60 = $2.40
    │   ├─ Base units = $100 / $2.40 = 41.67 ≈ 41
    │   └─ Return: 41 units
    │
    ├─ ConfidenceBasedSizer.get_confidence_multiplier()
    │   ├─ Confidence = 0.65
    │   ├─ Multiplier = 0.75 (strong signal)
    │   ├─ Adjusted units = 41 × 0.75 = 30.75 ≈ 30
    │   └─ Return: 30 units

STEP 4: Stop-Loss and Take-Profit Calculation
    ├─ StopLossManager.calculate_stop_and_target()
    │   ├─ Entry price: $80.00
    │   ├─ Stop loss: $80 × (1 - 0.03) = $77.60
    │   ├─ Take profit: $80 × (1 + 0.05) = $84.00
    │   └─ Return: {entry: 80, stop: 77.60, target: 84}
    │
    ├─ StopLossManager.validate_risk_reward()
    │   ├─ Risk = $80 - $77.60 = $2.40
    │   ├─ Reward = $84 - $80 = $4.00
    │   ├─ Ratio = $4 / $2.40 = 1.67 < 2.0 ❌ FAIL
    │   └─ Return: (False, 1.67, "Below minimum 2.0")

STEP 5: Reject Trade
    └─ Trade rejected due to R:R
    └─ Go to STEP 10

[If all validations pass, continue...]

STEP 6: Execute Trade
    ├─ Create TradeExecution object
    │   ├─ trade_id = 42
    │   ├─ symbol = 'AAPL'
    │   ├─ entry_price = 80.00
    │   ├─ units = 30
    │   ├─ stop_loss = 77.60
    │   ├─ take_profit = 84.00
    │   ├─ confidence = 0.65
    │   └─ entry_date = 2024-01-15
    │
    ├─ Update account
    │   ├─ Cost = 30 × $80 × (1 + 0.01 commission) = $2,424
    │   ├─ Balance = $10,000 - $2,424 = $7,576
    │   └─ Units held = 30
    │
    ├─ Update heat tracking
    │   ├─ Position heat = 30 × $2.40 = $72
    │   ├─ Total heat = $72
    │   ├─ Heat % = $72 / $10,000 = 0.72%
    │   └─ Log: "Heat OK, 2.28% remaining"
    │
    ├─ Add to trade history
    │   └─ append(trade_execution_object)

STEP 7-9: Each Bar While Position is Open
    ├─ STEP 7: Update Current Equity
    │   ├─ Current price = $81.50
    │   ├─ Position value = 30 × $81.50 = $2,445
    │   ├─ Current equity = $7,576 + $2,445 = $10,021
    │   └─ Track: equity_history.append($10,021)
    │
    ├─ STEP 8: Check Stops and Targets
    │   ├─ Is price <= $77.60 (stop loss)?
    │   │  └─ NO, continue
    │   ├─ Is price >= $84.00 (take profit)?
    │   │  └─ NO, continue
    │   ├─ Trailing stop active? Price > $81.60?
    │   │  ├─ Current price $81.50 < $81.60 threshold
    │   │  └─ NO, trailing stop not yet activated
    │   └─ Days open = 1 < 10 (time stop)?
    │      └─ NO, continue holding
    │
    ├─ STEP 9: Update Metrics
    │   ├─ Daily return = ($10,021 - $10,000) / $10,000 = +0.21%
    │   ├─ Log daily_returns.append(0.0021)
    │   ├─ Drawdown tracker update
    │   │  ├─ Peak = $10,021
    │   │  ├─ Current DD = 0% (at peak)
    │   │  └─ Zone = NORMAL
    │   └─ Continue to next bar

STEP 10: Position Close (All Scenarios)
    ├─ SCENARIO A: Stop Loss Hit
    │   ├─ Price fell to $77.50
    │   ├─ Close at $77.60 (hard stop)
    │   ├─ Proceeds = 30 × $77.60 × (1 - 0.01 commission) = $2,299
    │   ├─ P&L = $2,299 - $2,424 = -$125
    │   ├─ R-Multiple = -$125 / $72 = -1.74R
    │   ├─ Exit type = 'stop'
    │   └─ Log: "STOP HIT at $77.50, exited at $77.60, loss -1.74R"
    │
    ├─ SCENARIO B: Take Profit Hit
    │   ├─ Price rose to $84.50
    │   ├─ Close at $84.00 (take profit target)
    │   ├─ Proceeds = 30 × $84.00 × (1 - 0.01) = $2,494
    │   ├─ P&L = $2,494 - $2,424 = +$70
    │   ├─ R-Multiple = +$70 / $72 = +0.97R
    │   ├─ Exit type = 'target'
    │   └─ Log: "TARGET HIT at $84.50, exited at $84.00, win +0.97R"
    │
    ├─ SCENARIO C: Trailing Stop Hit
    │   ├─ Price rose to $86, trailing stop at $85.12
    │   ├─ Price fell to $85.00
    │   ├─ Close at $85.12 (trailing stop)
    │   ├─ Proceeds = 30 × $85.12 × (1 - 0.01) = $2,530
    │   ├─ P&L = $2,530 - $2,424 = +$106
    │   ├─ R-Multiple = +$106 / $72 = +1.47R
    │   ├─ Exit type = 'manual'
    │   └─ Log: "TRAILING STOP HIT at $85.00, exited at $85.12, win +1.47R"
    │
    ├─ SCENARIO D: Time Stop Hit
    │   ├─ 10 trading days have passed
    │   ├─ Price is currently $82.00
    │   ├─ Close at market = $82.00
    │   ├─ Proceeds = 30 × $82.00 × (1 - 0.01) = $2,434
    │   ├─ P&L = $2,434 - $2,424 = +$10
    │   ├─ R-Multiple = +$10 / $72 = +0.14R
    │   ├─ Exit type = 'time'
    │   └─ Log: "TIME STOP (10 days), exited at $82.00, win +0.14R"

STEP 11: Post-Close Updates
    ├─ Update balance with proceeds
    ├─ Record P&L in trade object
    ├─ Record R-Multiple in trade object
    ├─ Remove position from heat tracker
    ├─ Update drawdown tracker with new equity
    ├─ Calculate new daily return
    ├─ Log trade to history
    └─ Go to STEP 1 (wait for next signal)

STEP 12: End of Backtest
    ├─ Calculate all metrics
    │   ├─ Win rate: # winners / # trades
    │   ├─ Profit factor: sum(wins) / sum(losses)
    │   ├─ R expectancy: (win% × avg_win_r) - (loss% × avg_loss_r)
    │   ├─ Sharpe: (avg return - rf) / volatility
    │   ├─ Max DD: (peak - trough) / peak
    │   └─ VaR/CVaR: percentile analysis
    │
    ├─ Generate report
    │   ├─ Total return %
    │   ├─ Number of trades
    │   ├─ Win/loss statistics
    │   ├─ Risk metrics
    │   └─ Drawdown analysis
    │
    └─ Export trade log and metrics

```

---

## Summary

This architecture provides:

1. **Separation of Concerns**
   - Data loading (FinancialData)
   - Position sizing (PositionSizer)
   - Risk management (DrawdownTracker, PortfolioHeat)
   - Trade tracking (TradeExecution)
   - Metrics (RiskMetrics)

2. **Validation at Every Step**
   - Pre-trade validation before entry
   - Stop/target checking during hold
   - Heat limits enforced continuously
   - Drawdown zones monitored

3. **Complete Trade Lifecycle**
   - Entry with full risk parameters
   - Monitoring with multiple exit conditions
   - Tracking with detailed metrics
   - Analysis with comprehensive reporting

4. **Scalability**
   - Single position to multiple positions
   - Single symbol to multiple symbols
   - Fixed sizing to dynamic sizing
   - Backtest to live trading

The system is designed to be robust, measurable, and adaptable.
