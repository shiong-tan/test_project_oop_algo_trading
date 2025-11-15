# Risk Management Cheat Sheet - Quick Reference

## Position Sizing Rules (Choose One)

### Fixed Fractional (RECOMMENDED for beginners)
```
Risk per Trade = 1% of account
Position Size = (Account Risk $) / (Entry Price - Stop Loss Price)

Example: $10,000 account
  Account Risk: $100 (1%)
  Entry: $80, Stop: $77.60 (3%)
  Risk per Unit: $2.40
  Position Size: $100 / $2.40 = 41.67 units ≈ 41 units

Capital Deployed: 41 × $80 = $3,280 (32.8%)
Max Loss: 41 × $2.40 = $98.40 (1% of account)
```

### Kelly Criterion (ADVANCED - need 50+ trades)
```
f* = (Win% × (Avg Win/Avg Loss) - Loss%) / (Avg Win/Avg Loss)

Example: 50% win rate, 1.5R average win, 1.0R average loss
  f* = (0.50 × 1.5 - 0.50) / 1.5 = 8.3%
  With 25% safety factor: 8.3% × 0.25 = 2.1% per trade

Use only 25-50% of Kelly result to reduce volatility
```

### Confidence-Based (COMBINE with above)
```
Confidence < 55%:  0.25× position size (weak signal)
Confidence 55-60%: 0.50× position size (moderate)
Confidence 60-70%: 0.75× position size (strong)
Confidence 70-80%: 1.00× position size (very strong)
Confidence > 80%:  0.50× position size (suspicious overfit)

Skip trade if confidence < 52%
```

---

## Stop-Loss and Take-Profit Defaults

```
Entry Price:                $80.00
Hard Stop-Loss Distance:    3% below entry = $77.60
Initial Take-Profit:        5% above entry = $84.00

Risk Per Trade:             $80.00 - $77.60 = $2.40 per unit
Reward Per Trade:           $84.00 - $80.00 = $4.00 per unit
Risk/Reward Ratio:          $4.00 / $2.40 = 1.67

Minimum acceptable: 1:2 ratio (at least 2x reward for 1x risk)
If ratio < 1:2, SKIP TRADE
```

### Trailing Stop (Once Profitable)
```
Entry: $80
Price reaches: $82 (up 2.5%)
Trailing Stop Distance: 2% below current price
Trailing Stop Level: $82 × 0.98 = $80.36

If price drops to $80.36, exit automatically
If price continues up to $85, trailing stop moves to $83.30
Follows price up while protecting downside
```

### Time-Based Exit
```
7 days:   Review position thesis
10 days:  Close position regardless of profit/loss
30 days:  Never hold positions longer than 30 days
```

---

## Drawdown Management Zones

```
Current Drawdown        Zone          Actions
─────────────────────────────────────────────────────
0% to 5%               NORMAL         Trade normally
5% to 7.5%             CAUTION        Reduce position sizes 50%
7.5% to 10%            ALERT          Reduce position sizes 25%
> 10%                  EMERGENCY      STOP ALL TRADING

Position Size Multiplier:
  NORMAL:     1.00× (full size)
  CAUTION:    0.50× (half size)
  ALERT:      0.25× (quarter size)
  EMERGENCY:  0.00× (no new trades)
```

---

## Portfolio Heat Calculation

```
Portfolio Heat = Sum of all (Position Size × Risk Per Unit)

Example: 3 open positions
  Position 1: 41 units × $2.40 risk = $98.40
  Position 2: 35 units × $2.50 risk = $87.50
  Position 3: 30 units × $2.00 risk = $60.00
  ─────────────────────────────────────
  Total Heat:                          $246.00

Heat as % of Account: $246 / $10,000 = 2.46%
Max Allowed Heat: 3% = $300

Can add more positions? YES - have $54 heat remaining

Rule: Heat should never exceed 3% of account
```

---

## Pre-Trade Validation Checklist

Before EVERY trade, verify ALL of these:

```
SIGNAL QUALITY
  [ ] Confidence >= 52%
  [ ] Confidence <= 70%
  [ ] Model signal is clearly 1 (buy) or 0 (sell)

POSITION SIZING
  [ ] Position size calculated correctly
  [ ] Position size adjusted for confidence
  [ ] Position size <= 5% of account

RISK/REWARD
  [ ] Risk/Reward >= 1:2 minimum
  [ ] Stop-loss is 2-3% below entry
  [ ] Take-profit is 5-8% above entry

PORTFOLIO HEAT
  [ ] Current heat + new position <= 3% of account
  [ ] Enough cash for stop-loss if hit

DRAWDOWN STATUS
  [ ] Not in EMERGENCY mode (> 10% DD)
  [ ] Drawdown zone allows this trade

ACCOUNT STATUS
  [ ] Sufficient buying power
  [ ] Account not underwater
```

If ANY check fails, SKIP THE TRADE.

---

## Risk Metrics Quick Reference

### Sharpe Ratio (Overall Risk-Adjusted Return)
```
Formula: (Average Return - Risk-Free Rate) / Std Dev

Target: > 1.0
Good:   > 1.5
Great:  > 2.0

Interpretation:
  > 1.0:  Returns compensate for risk taken
  < 1.0:  Returns don't justify risk (inefficient)
```

### Sortino Ratio (Downside Risk Only)
```
Formula: (Average Return - Risk-Free Rate) / Downside Std Dev
Note: Only counts downside volatility, ignores upside

Target: > 1.0
Good:   > 1.5
Great:  > 2.0

Better metric than Sharpe for trading strategies
(Sharpe penalizes good upside volatility)
```

### Max Drawdown
```
Worst peak-to-trough decline in account value

Acceptable: < 15%
Good:       < 10%
Great:      < 5%

Example: Peak $10,500 → Trough $9,100
Max DD = ($10,500 - $9,100) / $10,500 = 13.3%
```

### Value at Risk (VaR) - 95%
```
Maximum expected loss in 95% of situations

Example: -2.5% VaR on $10,000 account
Meaning: 95 out of 100 days, won't lose more than $250

Use to set emergency stop levels
```

### R-Multiple (The Most Important Metric)
```
1R = Initial Risk Per Trade (Entry - Stop Loss Price)

Trade Outcomes:
  -1R = Hit stop loss (maximum loss)
  +2R = Won 2x the risk amount
  +5R = Won 5x the risk amount

Better than dollars because normalized to risk

Expected Value:
  (Win% × Avg Win R) - (Loss% × Avg Loss R)
  Example: (50% × 2R) - (50% × 1R) = +0.5R per trade
  = Expected profit of 0.5R on each trade
```

---

## Trade Rejection Reasons (Don't Trade If)

```
Confidence Issues:
  • Confidence < 52%                    (too weak)
  • Confidence > 70%                    (suspicious overfit)

Risk/Reward Issues:
  • Risk/Reward < 1:2                   (unfavorable odds)
  • Stop-loss above entry               (invalid setup)

Position Sizing:
  • Position > 5% of account            (too concentrated)
  • Position size = 0 after calculation (too small)

Portfolio:
  • Heat + new position > 3%            (over-leveraged)
  • Drawdown in EMERGENCY (>10%)        (protection mode)

Account:
  • Insufficient buying power           (not enough cash)
  • Insufficient funds for worst case   (stop-loss risk)
```

---

## Performance Viability Check (Monthly)

### Continue Trading If:
```
✓ Win Rate >= 40%
✓ Profit Factor >= 1.2
✓ R Expectancy > 0
✓ Sharpe Ratio > 0.5
✓ Max Drawdown < 15%
✓ No pattern of violations
```

### Modify Strategy If:
```
⚠ Win Rate 35-40%:        Tighten entry rules (confidence threshold)
⚠ Profit Factor 1.0-1.2:  Adjust stop loss or take profit
⚠ R Expectancy ~0:        Reduce position sizes
⚠ Sharpe Ratio 0.5-1.0:   Add more filters/rules
⚠ Max Drawdown 15-20%:    Reduce risk per trade to 0.5%
```

### Stop Trading If:
```
✗ Win Rate < 35%          (losing more than winning)
✗ Profit Factor < 1.0     (losing money overall)
✗ R Expectancy < -0.5     (losing 0.5R per trade on average)
✗ Sharpe Ratio < 0.5      (poor risk-adjusted returns)
✗ Max Drawdown > 25%      (unacceptable risk)
```

---

## Daily Monitoring Checklist

```
At market open:
  [ ] Check current drawdown percentage
  [ ] Review open positions and their stops
  [ ] Verify portfolio heat usage
  [ ] Check for any trades hitting stops overnight

During trading:
  [ ] Monitor open position for stop/target hit
  [ ] Track daily equity
  [ ] Count trades (max 3 per day)
  [ ] Verify all new trades meet validation criteria

At market close:
  [ ] Update equity curve
  [ ] Calculate daily return
  [ ] Log any trades closed
  [ ] Review any violations or alerts
  [ ] Check if in warning zones (5-7.5% or 7.5-10% DD)
```

---

## Position Sizing Calculator Quick Math

```
For $10,000 account, 1% risk per trade:

Entry   Stop    Risk/Unit   Units Calc      Units   Deployed   Max Loss
────────────────────────────────────────────────────────────────────────
$100    $97     $3.00       $100/$3 = 33    33      $3,300     $99
$80     $77.60  $2.40       $100/$2.4 = 42  42      $3,360     $101
$50     $48.50  $1.50       $100/$1.5 = 67  67      $3,350     $101
$150    $145.50 $4.50       $100/$4.5 = 22  22      $3,300     $99
$200    $194    $6.00       $100/$6 = 17    17      $3,400     $102

Consistency: Most trades deploy similar % of capital (32-34%)
by using this fixed fractional sizing method
```

---

## Capital Preservation Rules

```
Minimum Cash Reserve: 10% of account ($1,000 for $10k)

If Cash > 20%:     Can trade normally
If Cash 10-20%:    Can trade but monitor
If Cash < 10%:     DO NOT TAKE NEW TRADES
                   Close weakest positions until cash > 15%

Why?
  - Flexibility for high-opportunity trades
  - Psychological buffer
  - Emergency liquidity
  - Opportunity to add to winners
```

---

## Common Mistakes to Avoid

```
POSITION SIZING
  ✗ Using 50% of capital per trade        (too concentrated)
  ✗ Same size regardless of confidence    (no signal quality adjustment)
  ✗ Ignoring portfolio heat               (over-leveraged)
  → Use fixed fractional at 1% with confidence adjustment

STOP-LOSS
  ✗ Not using hard stops                  (unlimited downside)
  ✗ Moving stops after entry              (protecting losers)
  ✗ Too tight stops (1-2%)                (whipsawed out)
  → Hard stop at 3%, no moving

RISK/REWARD
  ✗ Trading with R:R < 1:2                (poor odds)
  ✗ Skipping validation                   (bad discipline)
  ✗ Entering on weak signals              (low win rate)
  → Validate every trade, minimum 1:2 R:R

MANAGEMENT
  ✗ Over-trading (>3 trades/day)          (commission drag)
  ✗ Revenge trading after losses          (losing discipline)
  ✗ Holding losers hoping to break even   (thesis creep)
  → Trade the plan, close at stops, limit daily trades
```

---

## When Things Go Wrong

```
Drawdown Enters CAUTION Zone (5-7.5%):
  1. Reduce position sizes to 50% immediately
  2. Tighten stops to 2% instead of 3%
  3. Increase minimum confidence to 55%
  4. Document what went wrong
  5. Continue trading with caution

Drawdown Enters ALERT Zone (7.5-10%):
  1. Reduce position sizes to 25% immediately
  2. Only take trades with 65%+ confidence
  3. Close any questionable positions
  4. Increase trade spacing (30 min between)
  5. Prepare to stop trading if DD hits 10%

Drawdown Hits EMERGENCY (>10%):
  1. STOP ALL NEW TRADES immediately
  2. Let open positions close at stops/targets
  3. Hold no positions overnight
  4. Investigate root cause (1 week minimum)
  5. Restart only after cause identified and fixed
  6. Reset max DD tracker when resuming

Root Causes to Investigate:
  • Model performance degradation (retrain needed?)
  • Market regime change (new patterns)
  • Data quality issues (gaps, splits)
  • Overfitting detected (test on new data)
  • Over-trading in caution zone (discipline break)
```

---

## Excel Tracking Template Structure

```
DAILY LOG:
Date | Time | Symbol | Signal | Confidence | Entry | Stop | Target |
Units | Position% | Heat% | Exit Price | Exit Type | P&L | R Multiple | Notes

TRADE SUMMARY:
Trade ID | Entry Date | Exit Date | Days | Entry | Exit | Units |
P&L $ | R Multiple | Win/Loss | Confidence | Exit Reason

METRICS:
Total Trades | Winners | Losers | Win Rate | Avg Win $ | Avg Loss $ |
Profit Factor | Total P&L | Max DD | Sharpe | Sortino | R Expectancy

DAILY EQUITY:
Date | Opening Equity | Daily Return % | Max/Min | Closing Equity |
Cumulative Return % | Current DD % | In Position?
```

---

## Remember: Three Core Principles

```
1. POSITION SIZING
   Never risk more than 1% per trade
   Adjust for confidence and heat

2. STOPS & TARGETS
   Hard stop loss on every trade (3% default)
   Take-profit target (5% default)
   No exceptions

3. MONITORING & DISCIPLINE
   Track drawdown constantly
   Stop trading if DD > 10%
   Follow the rules without emotion
```

These three principles will protect your capital while you improve your trading system.
