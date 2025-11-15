# Risk Management Parameter Tuning Guide

## Overview

After implementing the risk management framework, systematically test parameter combinations to optimize your strategy while maintaining capital preservation.

---

## Phase 1: Baseline Testing (Conservative Defaults)

Start with conservative parameters that should work across all market conditions.

### Baseline Configuration
```python
# Position Sizing
risk_per_trade_pct = 0.01          # 1% of account per trade

# Stop-Loss & Take-Profit
stop_loss_pct = 0.03               # 3% below entry
take_profit_pct = 0.05             # 5% above entry
min_risk_reward_ratio = 2.0        # Minimum 1:2 ratio

# Drawdown Controls
max_drawdown_pct = 0.10            # 10% maximum

# Portfolio Heat
max_heat_pct = 0.03                # 3% maximum

# Position Limits
max_position_pct = 0.05            # 5% max per position

# Signal Quality
min_confidence = 0.52              # Minimum 52%
max_confidence = 0.70              # Maximum before overfit
```

### Expected Results with Conservative Settings
```
For models with 45-50% raw win rate:
- Win Rate (after risk management): ~40-45%
- Max Drawdown: 3-8%
- Return: Likely neutral to slightly negative
- Reason: Overly restrictive for weak models

Purpose: Verify system doesn't blow up
This baseline establishes that capital preservation works
```

---

## Phase 2: Confidence Threshold Optimization

Test different minimum confidence thresholds to find sweet spot.

### Test Matrix

```
Test Case 1: min_confidence = 0.50 (very permissive)
  Configuration: 50-70% confidence range
  Expected: Higher win rate, more trades, higher returns
  Risk: More whipsaws and false signals

Test Case 2: min_confidence = 0.52 (default)
  Configuration: 52-70% confidence range
  Expected: Baseline behavior
  Reason: Slightly better than random (50%)

Test Case 3: min_confidence = 0.55 (moderate)
  Configuration: 55-70% confidence range
  Expected: Fewer trades, better quality signals
  Impact: ~30-40% fewer trades

Test Case 4: min_confidence = 0.60 (strict)
  Configuration: 60-70% confidence range
  Expected: High-quality signals only
  Impact: ~50-60% fewer trades, better win rate

Test Case 5: min_confidence = 0.65 (very strict)
  Configuration: 65-70% confidence range
  Expected: Only best signals
  Impact: ~70-80% fewer trades, much lower commission drag
```

### How to Test

```python
# In your backtesting loop
confidence_thresholds = [0.50, 0.52, 0.55, 0.60, 0.65]

for min_conf in confidence_thresholds:
    validator.min_confidence = min_conf

    results = run_backtest()

    results_matrix = {
        'min_confidence': min_conf,
        'total_trades': results['num_trades'],
        'win_rate': results['win_rate'],
        'profit_factor': results['profit_factor'],
        'sharpe': results['sharpe_ratio'],
        'max_drawdown': results['max_drawdown'],
        'total_return': results['total_return']
    }

    print(f"Confidence {min_conf:.0%}: {results_matrix}")
```

### Decision Framework

```
Choose confidence threshold based on:

IF Model Win Rate < 45%:
  Use min_confidence = 0.65
  → Need only highest quality signals

IF Model Win Rate 45-50%:
  Use min_confidence = 0.60
  → Filter to best 30-40% of signals

IF Model Win Rate 50-55%:
  Use min_confidence = 0.55
  → Balanced approach

IF Model Win Rate > 55%:
  Use min_confidence = 0.52
  → Can use more signals

MONITOR: After 20-30 trades, check actual win rate
If actual < expected, increase threshold
If actual > expected, can decrease threshold
```

---

## Phase 3: Stop-Loss Distance Optimization

Test different stop-loss percentages.

### Test Matrix

```
Test Case 1: stop_loss_pct = 0.01 (very tight)
  Configuration: 1% below entry
  Expected: More frequent stops, smaller losses
  Risk: Whipsawed on normal volatility
  Best for: High-confidence signals in calm markets

Test Case 2: stop_loss_pct = 0.02 (tight)
  Configuration: 2% below entry
  Expected: Balanced approach
  Impact: Reduces max DD, improves Sharpe

Test Case 3: stop_loss_pct = 0.03 (default)
  Configuration: 3% below entry
  Expected: Baseline, captures most signals
  Reason: Most reliable level

Test Case 4: stop_loss_pct = 0.05 (loose)
  Configuration: 5% below entry
  Expected: Fewer stops, larger losses
  Risk: Larger drawdowns when wrong
  Best for: Low-confidence signals needing room
```

### How to Test

```python
stop_loss_percentages = [0.01, 0.02, 0.03, 0.05]

for sl_pct in stop_loss_percentages:
    stop_loss_manager.stop_loss_pct = sl_pct

    results = run_backtest()

    results_matrix = {
        'stop_loss_pct': sl_pct,
        'trades_from_stops': results['exits_from_stops'],
        'avg_loss_pct': results['avg_loss_pct'],
        'max_drawdown': results['max_drawdown'],
        'win_rate': results['win_rate'],
        'total_return': results['total_return']
    }

    print(f"Stop Loss {sl_pct:.0%}: {results_matrix}")
```

### Decision Framework

```
Optimal stop loss depends on:

Symbol Volatility (measure ATR):
  High volatility (ATR > 3%):  Use stop_loss_pct = 0.03-0.05
  Normal volatility (ATR 2%):  Use stop_loss_pct = 0.03
  Low volatility (ATR < 1%):   Use stop_loss_pct = 0.02

Model Confidence:
  High confidence (65%+):      Use stop_loss_pct = 0.02
  Medium confidence (55-60%):  Use stop_loss_pct = 0.03
  Low confidence (52-55%):     Use stop_loss_pct = 0.05

Historical Drawdown Pattern:
  If existing DD hits 10%, increase stops to 0.05
  If winning trades get stopped out, increase stops
  If losing trades too large, decrease stops
```

---

## Phase 4: Risk Per Trade Optimization

Test different risk percentages per trade.

### Test Matrix

```
Test Case 1: risk_per_trade_pct = 0.005 (0.5%)
  Configuration: Micro-position sizing
  Expected: Very slow account growth, very safe
  Best for: Building confidence, proving edge
  Drawback: Leaves capital on sidelines

Test Case 2: risk_per_trade_pct = 0.01 (1%)
  Configuration: Conservative (recommended start)
  Expected: Steady growth with volatility
  Best for: Most traders, most market conditions

Test Case 3: risk_per_trade_pct = 0.015 (1.5%)
  Configuration: Moderate
  Expected: Faster growth but larger drawdowns
  Risk: DD increases by ~50%
  Best for: After proving edge with 50+ trades

Test Case 4: risk_per_trade_pct = 0.02 (2%)
  Configuration: Aggressive
  Expected: Higher volatility and growth
  Risk: Much larger drawdowns
  Best for: Only after 100+ profitable trades
```

### How to Test

```python
# Warning: Test in order, don't jump to 2%

risk_percentages = [0.005, 0.01, 0.015, 0.02]

for risk_pct in risk_percentages:
    position_sizer.risk_per_trade_pct = risk_pct

    results = run_backtest()

    # Calculate growth vs. risk
    growth_per_trade = results['total_return'] / results['num_trades']
    growth_per_risk = results['total_return'] / risk_pct
    sharpe_adjusted = results['sharpe_ratio'] * np.sqrt(results['num_trades'])

    results_matrix = {
        'risk_pct': risk_pct,
        'avg_position_size_pct': results['avg_position_pct'],
        'total_return': results['total_return'],
        'max_drawdown': results['max_drawdown'],
        'return_per_trade': growth_per_trade,
        'sharpe_ratio': results['sharpe_ratio']
    }

    print(f"Risk {risk_pct:.1%}: {results_matrix}")
```

### Decision Framework

```
Risk Per Trade Selection:

Stage 1: Unproven Model (0-50 trades)
  Use: 0.5% risk per trade
  Why: Capital preservation while testing
  Goal: Prove >45% win rate

Stage 2: Developing Edge (50-100 trades)
  Use: 1.0% risk per trade
  Why: Standard level, proven sustainable
  Goal: Achieve 50%+ win rate consistently

Stage 3: Validated Edge (100+ trades)
  Use: 1.0-1.5% based on win rate
  Why: Can increase slightly with proven track record
  Condition: Only if win rate consistently > 50%

Stage 4: Optimized (200+ trades)
  Use: Up to 2.0% if win rate > 55%
  Why: Proven edge allows higher leverage
  Caution: Risk scales non-linearly

CRITICAL: If win rate drops below 45%, reduce to 0.5%
Track weekly: If 5 consecutive losers, reduce risk
```

---

## Phase 5: Take-Profit Target Optimization

Test different take-profit distances.

### Test Matrix

```
Test Case 1: take_profit_pct = 0.02 (2%)
  Configuration: Very tight
  Expected: High win rate, small wins, many trades
  Risk: Leaves money on table in strong trends
  Best for: Choppy/ranging markets

Test Case 2: take_profit_pct = 0.03 (3%)
  Configuration: Tight
  Expected: Moderate wins, frequent targets hit
  Ratio with 3% stop: 1:1 (too tight, skip this)

Test Case 3: take_profit_pct = 0.05 (5%)
  Configuration: Default
  Expected: Balanced approach
  Ratio with 3% stop: 1:1.67 (good baseline)

Test Case 4: take_profit_pct = 0.08 (8%)
  Configuration: Wide
  Expected: Fewer targets hit, larger wins when hit
  Ratio with 3% stop: 1:2.67 (excellent odds)
  Risk: More positions don't reach target

Test Case 5: take_profit_pct = 0.10 (10%)
  Configuration: Very wide
  Expected: Very few targets hit, huge wins when they do
  Ratio with 3% stop: 1:3.33 (ideal odds)
  Challenge: Need very good signals
```

### How to Test (Must Fix stop_loss_pct First)

```python
# Keep stop loss at optimal value, vary take profit
stop_loss_pct = 0.03  # Fixed at optimal

take_profit_percentages = [0.02, 0.03, 0.05, 0.08, 0.10]

for tp_pct in take_profit_percentages:
    stop_loss_manager.take_profit_pct = tp_pct

    results = run_backtest()

    # Calculate risk/reward
    risk_reward = tp_pct / stop_loss_pct

    results_matrix = {
        'take_profit_pct': tp_pct,
        'risk_reward_ratio': risk_reward,
        'targets_hit': results['targets_hit'],
        'targets_hit_pct': results['targets_hit'] / results['num_trades'],
        'avg_win_pct': results['avg_win_pct'],
        'total_return': results['total_return'],
        'sharpe_ratio': results['sharpe_ratio']
    }

    print(f"TP {tp_pct:.0%}: {results_matrix}")
```

### Decision Framework

```
Take-Profit Distance Selection:

Based on Historical Win Rate:

IF win rate > 60%:
  Use take_profit_pct = 0.05 or wider
  → You're right often, let winners run

IF win rate 50-60%:
  Use take_profit_pct = 0.05
  → Balanced approach

IF win rate 45-50%:
  Use take_profit_pct = 0.08
  → Need bigger wins to offset losses

IF win rate < 45%:
  STOP and improve model first
  Taking bigger targets won't help if you're wrong too much

Adjustment Rule:
  If too many trades don't reach target (>70% stop out):
    → Increase target width

  If most trades hit target quickly (<2 bars):
    → Target is too generous
    → Tighten to lock in more profits

Trailing Stop Alternative:
  Instead of fixed take-profit, use trailing stop:
  - Hits at target normally
  - But keeps running if momentum continues
  - Captures outlier winning trades
  - Better long-term performance
```

---

## Phase 6: Maximum Drawdown Limit Optimization

Test different max drawdown thresholds.

### Test Matrix

```
Test Case 1: max_drawdown_pct = 0.05 (5%)
  Configuration: Very conservative
  Expected: Stops trading frequently (CAUTION/ALERT)
  Impact: Misses recovery opportunities
  Best for: Risk-averse, small accounts

Test Case 2: max_drawdown_pct = 0.10 (10%)
  Configuration: Default (RECOMMENDED)
  Expected: Good capital preservation
  Impact: Provides room while preventing disaster
  Best for: Most traders

Test Case 3: max_drawdown_pct = 0.15 (15%)
  Configuration: Moderate risk
  Expected: More trading allowed in drawdowns
  Impact: Some recovery opportunities, more risk
  Best for: Proven traders with 100+ trades

Test Case 4: max_drawdown_pct = 0.20 (20%)
  Configuration: Aggressive
  Expected: Allows significant drawdowns
  Risk: Can turn drawdown into disaster
  Best for: Institutional traders with hedges
```

### How to Test

```python
drawdown_limits = [0.05, 0.10, 0.15, 0.20]

for dd_limit in drawdown_limits:
    drawdown_tracker.max_allowed_dd = dd_limit

    results = run_backtest()

    # Track zone management
    times_in_caution = count_times_in_zone(results, 'CAUTION')
    times_in_alert = count_times_in_zone(results, 'ALERT')
    times_emergency_stop = count_times_dd_exceeded(results)

    results_matrix = {
        'max_dd_limit': dd_limit,
        'actual_max_dd': results['max_drawdown'],
        'times_caution': times_in_caution,
        'times_alert': times_in_alert,
        'trades_blocked': times_emergency_stop,
        'total_return': results['total_return']
    }

    print(f"Max DD {dd_limit:.0%}: {results_matrix}")
```

### Decision Framework

```
Maximum Drawdown Selection:

Account Size Factor:
  Small (<$10k):    max_dd = 5-10%
  Medium ($10-50k): max_dd = 10-15%
  Large (>$50k):    max_dd = 15-20%

Experience Factor:
  Beginner (<50 trades):       max_dd = 5%
  Intermediate (50-100):       max_dd = 10%
  Advanced (100-500):          max_dd = 15%
  Institutional (500+):        max_dd = 20%

Risk Tolerance Factor:
  Very conservative:  max_dd = 5%
  Conservative:       max_dd = 10%
  Moderate:          max_dd = 15%
  Aggressive:        max_dd = 20%

RECOMMENDED STARTING POINT: 10%

Adjustment Strategy:
  If you hit emergency stop often:
    → You're trading too aggressively
    → Either increase max_dd OR reduce risk_per_trade
    → Prefer: reduce risk_per_trade

  If you never hit CAUTION zone:
    → Models are very consistent
    → Can safely increase to 15%

  If you hit EMERGENCY multiple times:
    → Models not reliable enough
    → Reduce to 5%, work on improving model
```

---

## Phase 7: Portfolio Heat Limit Optimization

Test different portfolio heat percentages.

### Test Matrix

```
Test Case 1: max_heat_pct = 0.02 (2%)
  Configuration: Very conservative
  Expected: Max 2-3 positions open at once
  Impact: Limits concurrent risk
  Best for: Single-position traders evolving to multi-position

Test Case 2: max_heat_pct = 0.03 (3%)
  Configuration: Default
  Expected: Max 3-4 positions open
  Impact: Balanced diversification and risk
  Best for: Most traders

Test Case 3: max_heat_pct = 0.05 (5%)
  Configuration: Moderate
  Expected: Max 5-6 positions open
  Impact: More diversification, higher total risk
  Best for: Portfolio approach with correlation analysis

Test Case 4: max_heat_pct = 0.08 (8%)
  Configuration: Aggressive
  Expected: Many concurrent positions possible
  Impact: Portfolio effect, but complex to manage
  Best for: Institutional traders managing multiple strategies
```

### How to Test

```python
heat_limits = [0.02, 0.03, 0.05, 0.08]

for heat_limit in heat_limits:
    portfolio_heat.max_heat_pct = heat_limit
    portfolio_heat.max_heat_amount = initial_capital * heat_limit

    results = run_backtest()

    # Track concurrent positions
    avg_concurrent = results['avg_concurrent_positions']
    max_concurrent = results['max_concurrent_positions']
    times_heat_rejected = count_heat_rejections(results)

    results_matrix = {
        'max_heat_pct': heat_limit,
        'avg_positions': avg_concurrent,
        'max_positions': max_concurrent,
        'trades_rejected': times_heat_rejected,
        'total_return': results['total_return'],
        'max_drawdown': results['max_drawdown']
    }

    print(f"Heat {heat_limit:.0%}: {results_matrix}")
```

### Decision Framework

```
Portfolio Heat Selection:

Single Symbol (Stock):
  max_heat_pct = 0.02-0.03
  → Usually holding 1 position, max 2
  → Lower heat makes sense

Multiple Symbols (Basket):
  max_heat_pct = 0.03-0.05
  → Can diversify across 3-5 positions
  → Correlation analysis important

Portfolio Approach:
  max_heat_pct = 0.05-0.08
  → Large number of lower-risk positions
  → Requires sophisticated position sizing

RECOMMENDED: 0.03 (3%) for equity swing trading

Why 3%?
  - With 1% risk per trade and 3 positions
  - Total portfolio heat = 3%
  - Matches max drawdown tolerance
  - Proven across many traders
```

---

## Phase 8: Comprehensive Test Plan

### Week 1-2: Establish Baseline
```
Config: Conservative defaults
- risk_per_trade = 1%
- stop_loss = 3%
- take_profit = 5%
- max_dd = 10%
- max_heat = 3%

Goal: Verify system works without blowing up
Metric: Any positive or small negative return acceptable
Output: Baseline metrics for comparison
```

### Week 3-4: Optimize Confidence
```
Test 5 confidence levels
- min_confidence: 0.50, 0.52, 0.55, 0.60, 0.65

Goal: Find sweet spot for your model
Metric: Win rate vs. trade count trade-off
Output: Recommended min_confidence setting
```

### Week 5-6: Optimize Stops
```
Test 4 stop-loss distances
- stop_loss_pct: 0.01, 0.02, 0.03, 0.05

Goal: Minimize stops while maintaining R:R ratio
Metric: Stop-out frequency vs. max drawdown
Output: Recommended stop_loss_pct setting
```

### Week 7-8: Optimize Take-Profit
```
Test 5 take-profit distances
- take_profit_pct: 0.02, 0.03, 0.05, 0.08, 0.10

Goal: Maximize wins while maintaining good odds
Metric: Target hit frequency vs. return per trade
Output: Recommended take_profit_pct setting
```

### Week 9-10: Optimize Risk Per Trade
```
Test 4 risk percentages
- risk_per_trade: 0.005, 0.01, 0.015, 0.02

Goal: Find optimal growth rate with acceptable DD
Metric: Return vs. Sharpe ratio vs. max drawdown
Output: Recommended risk_per_trade_pct setting
```

### Week 11-12: Final Validation
```
Run full backtest with optimized parameters
- All settings from previous testing
- Track all metrics
- Generate final report

Goal: Validate chosen parameters
Metric: All viability checks pass (win rate, PF, etc.)
Output: Production-ready configuration
```

---

## Parameter Optimization Results Template

Create a tracking spreadsheet:

```
OPTIMIZATION RESULTS SUMMARY
═══════════════════════════════════════════════════════════════

CONFIDENCE THRESHOLD TEST:
  Min_Conf | Trades | Win_Rate | Profit_Factor | Max_DD | Return%
  ────────────────────────────────────────────────────────────
  0.50     |   85   |  44%     |     1.32      |  8.2%  |  +3.2%
  0.52     |   78   |  46%     |     1.41      |  7.1%  |  +4.1%
  0.55     |   62   |  48%     |     1.53      |  5.8%  |  +3.8%
  0.60     |   41   |  51%     |     1.67      |  4.2%  |  +2.1%
  0.65     |   22   |  55%     |     1.84      |  2.1%  |  +0.8%
  WINNER:  0.55 (best balance)

STOP LOSS TEST:
  Stop_Loss | Stops | Avg_Loss% | Max_DD | Return%
  ────────────────────────────────────────
  0.01      |  42   |   0.9%    |  7.8%  |  +2.1%
  0.02      |  28   |   1.8%    |  6.2%  |  +4.3%
  0.03      |  18   |   2.8%    |  4.5%  |  +5.1%
  0.05      |   8   |   4.6%    |  3.2%  |  +3.8%
  WINNER:   0.03 (best return and DD)

...continue for other parameters...

FINAL OPTIMAL CONFIGURATION:
  min_confidence = 0.55
  stop_loss_pct = 0.03
  take_profit_pct = 0.05
  risk_per_trade = 0.01
  max_drawdown = 0.10
  max_heat = 0.03

EXPECTED PERFORMANCE:
  Win Rate: ~48%
  Profit Factor: ~1.5
  Sharpe: ~1.2
  Max DD: ~6-8%
  Return (annual): ~8-12% (depending on market)
```

---

## Key Takeaways

1. **Don't Optimize in Isolation**
   - Stop loss affects take-profit optimization
   - Risk per trade affects drawdown limits
   - Test in phases, not randomly

2. **More Trades Isn't Better**
   - Fewer high-confidence trades beats more weak ones
   - Aim for 50+ trades to establish pattern
   - Not 1000 trades in 3 months

3. **Sharpe Ratio Is Key Metric**
   - Higher returns with lower risk = good parameter
   - Maximize Sharpe, not just returns
   - 1.0+ Sharpe = meaningful edge

4. **Document Everything**
   - Track parameter combinations tested
   - Save results for comparison
   - Create reproducible configuration

5. **Validate Before Production**
   - Test on new out-of-sample data
   - Confirm results hold on different time period
   - Challenge optimized parameters with worst-case

This systematic approach ensures you find optimal parameters without curve-fitting to historical data.
