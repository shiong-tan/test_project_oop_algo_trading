# Comprehensive Risk Management Framework for ML-Based Trading Strategy

## Executive Summary

The current trading strategy has critical risk management gaps:
- Single order size limited to 50% of capital (overly aggressive for unproven models)
- No stop-loss mechanisms (unlimited downside per trade)
- No drawdown controls (can lose entire capital across multiple trades)
- No position limits or portfolio heat tracking
- No trade validation rules (enters on weak signals)
- All models showing negative returns (model risk + lack of risk control)

This framework provides battle-tested risk management rules suitable for prediction-based ML trading on equities.

---

## 1. POSITION SIZING FRAMEWORK

### 1.1 Fixed Fractional Position Sizing (Recommended Starting Point)

**Formula:**
```
Position Size (units) = (Account Risk / Risk Per Unit)
Position Size (%) = Account Risk % / Risk Per Unit %

Account Risk $ = Initial Capital × Risk % per Trade
Risk % per Trade = Stop Loss Distance (%) from Entry

Risk Per Unit = Entry Price - Stop Loss Price
```

**Implementation Parameters:**
```
Initial Capital:           $10,000
Risk per Trade:           1-2% of Account (START AT 1%)
Stop Loss Distance:       2-3% below entry price
Account Risk per Trade:   $100-$200 (1-2% of $10,000)
```

**Example Calculation:**
```
Entry Price:              $80.00
Stop Loss Price:          $77.60 (3% below entry)
Risk per Unit:            $2.40

Account Risk (1%):        $100
Position Size:            100 / 2.40 = 41.67 units (round to 41)
Capital Deployed:         41 × $80 = $3,280 (32.8% of account)
Maximum Loss:             41 × $2.40 = $98.40 (approximately 1%)
```

**Key Rules:**
- Never risk more than 2% per trade
- Start conservatively at 1% per trade
- Adjust based on win rate:
  - Win Rate > 60%: Can increase to 1.5-2%
  - Win Rate 50-60%: Keep at 1%
  - Win Rate < 50%: Reduce to 0.5% or stop trading

---

### 1.2 Kelly Criterion Position Sizing (Advanced)

**Formula:**
```
Optimal Position Size (%) = (Win% × Avg Win$) - (Loss% × Avg Loss$) / Avg Win$

Or simplified:
f* = (p × b - q) / b

Where:
f* = fraction of capital to wager
p = probability of win
q = probability of loss (1 - p)
b = ratio of win size to loss size (Avg Win / Avg Loss)
```

**Implementation:**
```
Win Rate (p):            45%
Loss Rate (q):           55%
Average Win (in R):      1.5R
Average Loss (in R):     1.0R
Win/Loss Ratio (b):      1.5

f* = (0.45 × 1.5 - 0.55) / 1.5
   = (0.675 - 0.55) / 1.5
   = 0.125 / 1.5
   = 0.083 or 8.3%

Recommended Position Size: 8.3% × 25% (Kelly safety factor) = 2.1% of account
```

**Safety Considerations:**
```
Full Kelly:          Use only after 100+ trades with proven edge
Fractional Kelly:    Use 25-50% of Kelly fraction (recommended)
  - 25% Kelly:       Most conservative, slow growth
  - 50% Kelly:       Balanced approach
Confidence Level:    Need 50+ trades before using Kelly
```

**Decision Matrix:**
```
Scenario 1: Early Stage (< 50 trades)
  Use Fixed Fractional at 1% risk per trade
  Ignore Kelly criterion results

Scenario 2: Established Strategy (50-100 trades)
  Compare Kelly with Fixed Fractional
  Use whichever is more conservative
  Apply 50% Kelly safety factor

Scenario 3: Validated Strategy (> 100 trades)
  Use 25-50% Kelly with confirmed win rate
  Monitor for regime changes
  Reduce if win rate drops below 45%
```

---

### 1.3 Confidence-Based Position Sizing

**Formula:**
```
Position Size = Base Position × Confidence Multiplier

Confidence Multiplier = Model Prediction Probability
  - Probability 0.50-0.55: 0.25x (weak signal)
  - Probability 0.55-0.60: 0.50x (moderate signal)
  - Probability 0.60-0.70: 0.75x (strong signal)
  - Probability 0.70-0.80: 1.00x (very strong signal)
  - Probability > 0.80:    0.50x (potential overfit, reduce exposure)
```

**Implementation (Using predict_proba):**
```python
# Get prediction probabilities instead of just class
prediction_proba = model.predict_proba(features)
confidence = max(prediction_proba)  # Probability of predicted class

# Confidence-based sizing
base_units = 41  # From fixed fractional calculation
confidence_multiplier = get_multiplier(confidence)
actual_units = int(base_units * confidence_multiplier)

def get_multiplier(confidence):
    if confidence <= 0.55:
        return 0.25
    elif confidence <= 0.60:
        return 0.50
    elif confidence <= 0.70:
        return 0.75
    elif confidence <= 0.80:
        return 1.00
    else:  # > 0.80, suspicious overfit
        return 0.50
```

**Rules for Confidence-Based Sizing:**
```
Threshold Requirements:
  - Minimum confidence: 52% (barely above random)
  - Skip trade if confidence < 52%
  - Maximum trade: confidence 70-80%
  - Above 80%: Likely overfitting, reduce size

Portfolio Optimization:
  - High confidence + high R-multiple: Full size
  - Low confidence + low R-multiple: Skip trade
  - High confidence + low R-multiple: Reduce to 25%
  - Low confidence + high R-multiple: Reduce to 50%
```

---

## 2. STOP-LOSS AND TAKE-PROFIT MECHANISMS

### 2.1 Hard Stop-Loss Rules

**Physical Stop-Loss (Mandatory):**
```
For EVERY trade:

Stop Loss Distance:      2-3% below entry (initial setup)
Take Profit Distance:    5-8% above entry (initial setup)
Trade Risk/Reward Ratio: Minimum 1:2 (1R loss, 2R+ gain potential)

Example Setup:
Entry Price:             $80.00
Hard Stop Loss:          $77.60 (3% below)
Initial Take Profit:     $84.00 (5% above)
Risk/Reward Ratio:       $2.40 loss / $4.00 gain = 1:1.67

If R/R < 1:2, DO NOT TAKE TRADE
```

**Implementation Logic:**
```python
def validate_trade_setup(entry_price, model_signal, confidence):
    # Calculate stop loss and take profit
    stop_loss = entry_price * (1 - stop_loss_pct)  # 2-3%
    take_profit = entry_price * (1 + take_profit_pct)  # 5-8%

    risk_per_trade = entry_price - stop_loss
    reward_per_trade = take_profit - entry_price

    risk_reward_ratio = reward_per_trade / risk_per_trade

    # Validation
    if risk_reward_ratio < 2.0:  # Minimum 1:2
        return False, "Risk/Reward ratio too low"

    if confidence < 0.52:  # Minimum confidence
        return False, "Confidence below threshold"

    return True, {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward_ratio': risk_reward_ratio
    }
```

### 2.2 Trailing Stop-Loss

**Dynamic Stop After Profit:**
```
Once trade is profitable:

Trailing Stop Distance:  1-2% below current price
Triggers when:          Price rises by 2-3% above entry

Example:
Entry:                   $80.00
Price reaches:           $82.40 (3% above entry)
Activate trailing stop:  $81.08 (2% below $82.40)

If price drops to $81.08, position automatically closes
If price continues up:   Trailing stop continues following at 1-2% below
Maximum profit capture:  Prevents giving back large gains
```

**Decision Logic:**
```
Initial Trade (0-5 days):
  Use hard stop loss only
  No trailing stop

Profitable Trade (>2% gain):
  Activate trailing stop
  Distance: 1-2% below current price

Running Profit Threshold:
  +2% to +5%: Trailing stop 2% below
  +5% to +10%: Trailing stop 2% below + lock in 3% minimum
  >10%: Trailing stop 3% below + lock in 5% minimum
```

### 2.3 Time-Based Stops

**Maximum Trade Duration:**
```
If position is still open after:

7 trading days:  Review thesis, consider closing if thesis broken
10 trading days: Close position regardless (avoid thesis creep)
30 trading days: Close all positions (institutional best practice)

Exceptions:
  - Position stopped out (hard loss)
  - Position taken profit (full gain)
  - Position size increased (restart timer)
```

---

## 3. MAXIMUM DRAWDOWN LIMITS

### 3.1 Portfolio Drawdown Control

**Definitions:**
```
Drawdown = (Peak Equity - Current Equity) / Peak Equity × 100%

Maximum Drawdown Limits:
  Small Account (<$10K):     10% max drawdown
  Medium Account ($10-50K):  15% max drawdown
  Large Account (>$50K):     20% max drawdown

Recommended Starting: 10% max drawdown regardless of account size
```

**Calculation Method:**
```python
class DrawdownTracker:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.peak_equity = initial_capital
        self.current_equity = initial_capital
        self.max_drawdown = 0
        self.max_dd_limit = initial_capital * 0.10  # 10% limit

    def update_equity(self, new_equity):
        self.current_equity = new_equity

        # Update peak
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity

        # Calculate current drawdown
        dd = (self.peak_equity - new_equity) / self.peak_equity

        if dd > self.max_drawdown:
            self.max_drawdown = dd

        # Check drawdown limit
        dd_amount = self.peak_equity - new_equity
        if dd_amount > self.max_dd_limit:
            return False, f"Max drawdown exceeded: {dd:.2%}"

        return True, f"Drawdown: {dd:.2%}, Limit: {self.max_dd_limit:.0f}"

    def get_stats(self):
        return {
            'current_dd': (self.peak_equity - self.current_equity) / self.peak_equity,
            'max_dd': self.max_drawdown,
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity
        }
```

**Drawdown Management Rules:**
```
Zone 1: 0-5% Drawdown
  Status: NORMAL
  Action: Continue normal trading
  Monitoring: Standard

Zone 2: 5-7.5% Drawdown
  Status: CAUTION
  Action: Reduce position sizes to 50% normal
  Action: Tighten stops to 2% instead of 3%
  Action: Increase minimum confidence to 60%
  Monitoring: Every trade

Zone 3: 7.5-10% Drawdown
  Status: ALERT
  Action: Reduce position sizes to 25% normal
  Action: Only take highest probability trades (>65% confidence)
  Action: Close losers after 5 days max
  Monitoring: Real-time

Zone 4: > 10% Drawdown
  Status: EMERGENCY
  Action: STOP ALL TRADING immediately
  Action: Review strategy performance
  Action: Do not resume until root cause found
  Action: Require strategy restart with new max DD reset
```

### 3.2 Equity Curve Monitoring

**Track Daily Equity:**
```
Daily Equity = Current Cash + (Position Size × Current Price)

Monitor:
  - Daily equity change
  - Equity vs. risk exposure
  - Win/loss ratio trend
  - Largest consecutive losses
```

---

## 4. POSITION LIMITS AND PORTFOLIO HEAT

### 4.1 Position Concentration Limits

**Single Position Limits:**
```
Maximum size of any one position: 5-10% of total capital
  Conservative: 5% (start here)
  Moderate:     7.5%
  Aggressive:   10% (only for proven high-confidence traders)

For $10,000 account:
  Conservative: $500 maximum per position
  Moderate:     $750 maximum per position
  Aggressive:   $1,000 maximum per position
```

**Implementation:**
```python
def validate_position_size(units, price, account_value, max_pct=0.05):
    position_value = units * price
    position_pct = position_value / account_value

    if position_pct > max_pct:
        max_units = int((account_value * max_pct) / price)
        return False, f"Position too large. Max {max_units} units"

    return True, f"Position OK: {position_pct:.1%} of account"
```

### 4.2 Portfolio Heat (Total Risk Exposure)

**Definition:**
```
Portfolio Heat = Sum of all potential losses if all stops hit simultaneously

Formula:
Heat = Sum(Position Size × Risk Per Unit) for all open positions
Heat % = Heat / Account Value
```

**Heat Limits:**
```
Maximum Portfolio Heat: 3-5% of account
Recommended:          3% (conservative)
Aggressive:           5% (only experienced traders)

Example:
Account Value:        $10,000
Max Heat Limit:       3% = $300

If 3 positions open:
Position 1: 41 units × $2.40 risk = $98.40
Position 2: 35 units × $2.50 risk = $87.50
Position 3: 30 units × $2.00 risk = $60.00
Total Heat: $246 (2.46% - OK)

Position 4 would add $95, bringing heat to $341 (3.41% - REJECT)
```

**Portfolio Heat Management:**
```python
class PortfolioHeat:
    def __init__(self, account_value, max_heat_pct=0.03):
        self.account_value = account_value
        self.max_heat = account_value * max_heat_pct
        self.positions = {}  # symbol: {'units': X, 'risk': Y}
        self.total_heat = 0

    def can_add_position(self, symbol, units, risk_per_unit):
        potential_heat = self.total_heat + (units * risk_per_unit)

        if potential_heat > self.max_heat:
            available_heat = self.max_heat - self.total_heat
            max_units = int(available_heat / risk_per_unit)
            return False, f"Heat limit exceeded. Can add max {max_units} units"

        return True, f"Position OK. Heat after: {potential_heat:.2f}"

    def add_position(self, symbol, units, risk_per_unit):
        heat = units * risk_per_unit
        self.positions[symbol] = {'units': units, 'risk': heat}
        self.total_heat += heat

        return {
            'position_heat': heat,
            'total_heat': self.total_heat,
            'heat_pct': self.total_heat / self.account_value,
            'heat_remaining': self.max_heat - self.total_heat
        }

    def close_position(self, symbol):
        if symbol in self.positions:
            self.total_heat -= self.positions[symbol]['risk']
            del self.positions[symbol]
```

### 4.3 Correlation-Based Position Limits

**Position Diversification Rules:**
```
No more than 2-3 highly correlated positions (correlation > 0.7)
No single sector > 30% of portfolio
No single country > 50% of portfolio

For ML trading (single stock):
Keep to 1 stock initially
Only add 2nd stock if:
  - Correlation < 0.6 with first stock
  - Both have independent trading signals
  - Portfolio heat allows
```

---

## 5. RISK METRICS TRACKING AND MONITORING

### 5.1 Essential Risk Metrics

**Daily Tracking:**

**1. Value at Risk (VaR) - 95% confidence**
```
Definition: Maximum expected loss in 95% of cases

Calculation Method (Historical Simulation):
  1. Calculate daily returns: r_t = (equity_t - equity_{t-1}) / equity_{t-1}
  2. Sort returns from worst to best
  3. VaR = 5th percentile of historical returns × Current Equity

Example for $10,000 account:
  Worst 5% of days: -2.5% average
  VaR (95%) = -2.5% × $10,000 = -$250

This means in 95 out of 100 days, we won't lose more than $250
```

**Implementation:**
```python
def calculate_var_95(daily_returns, confidence=0.95):
    """Calculate Value at Risk at given confidence level"""
    percentile = (1 - confidence) * 100
    var = np.percentile(daily_returns, percentile)
    return var

# Example
daily_returns = [0.01, -0.02, 0.015, -0.035, 0.02, -0.01, ...]
var_95 = calculate_var_95(daily_returns)
account_value = 10000
var_dollar = var_95 * account_value
# Result: Daily VaR = -2.1% = -$210
```

**2. Conditional Value at Risk (CVaR) - Expected Shortfall**
```
Definition: Average loss on the worst 5% of days

Calculation:
  1. Calculate daily returns
  2. Sort from worst to best
  3. Average the worst 5% of returns
  4. CVaR = Average of worst 5% × Current Equity

Example:
  Worst 5% days: [-4.5%, -3.2%, -2.8%, -2.1%, -1.5%]
  Average: -2.82%
  CVaR (95%) = -2.82% × $10,000 = -$282

Interpretation: On really bad days (bottom 5%), expect -$282 loss
```

**Implementation:**
```python
def calculate_cvar_95(daily_returns, confidence=0.95):
    """Calculate Conditional VaR (Expected Shortfall)"""
    percentile = (1 - confidence) * 100
    cutoff = np.percentile(daily_returns, percentile)
    return daily_returns[daily_returns <= cutoff].mean()

# Example
cvar_95 = calculate_cvar_95(daily_returns)
account_value = 10000
cvar_dollar = cvar_95 * account_value
# Result: CVaR = -2.82% = -$282
```

**3. Sharpe Ratio - Risk-Adjusted Return**
```
Formula:
Sharpe = (Average Daily Return - Risk-Free Rate) / Std Dev of Daily Returns

Interpretation:
  Sharpe > 1.0:  Good risk-adjusted returns
  Sharpe > 2.0:  Excellent risk-adjusted returns
  Sharpe < 0:    Losing money on a risk-adjusted basis

Calculation:
  Risk-Free Rate: 5% annually = 0.019% daily (5% / 252 days)

Example:
  Average Daily Return: 0.05%
  Daily Std Dev: 0.8%
  Risk-Free: 0.019%

  Sharpe = (0.05% - 0.019%) / 0.8% = 0.039 (POOR - below 1.0)
```

**Implementation:**
```python
def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.0005):
    """Calculate Sharpe ratio"""
    excess_return = daily_returns.mean() - risk_free_rate
    volatility = daily_returns.std()

    if volatility == 0:
        return 0

    return excess_return / volatility * np.sqrt(252)  # Annualized

# Example
daily_returns = np.array([...])
sharpe = calculate_sharpe_ratio(daily_returns)
# Interpret: If sharpe < 1.0, strategy is not risk-efficient
```

**4. Sortino Ratio - Downside Risk Only**
```
Formula:
Sortino = (Average Return - Risk-Free Rate) / Downside Std Dev

Key Difference from Sharpe:
  Sharpe uses all volatility (up and down)
  Sortino uses only downside volatility (losses)

Interpretation:
  Sortino > 1.0:  Good downside risk control
  Sortino > 2.0:  Excellent downside protection

Calculation:
  Downside Returns: Only negative days
  Downside Std Dev: Std Dev of negative returns only

Example:
  Average Daily Return: 0.05%
  Downside Std Dev (losses only): 0.5%
  Risk-Free: 0.019%

  Sortino = (0.05% - 0.019%) / 0.5% = 0.062 (POOR)
```

**Implementation:**
```python
def calculate_sortino_ratio(daily_returns, risk_free_rate=0.0005):
    """Calculate Sortino ratio (downside risk only)"""
    excess_return = daily_returns.mean() - risk_free_rate

    # Only negative returns
    downside_returns = daily_returns[daily_returns < 0]
    downside_volatility = downside_returns.std()

    if downside_volatility == 0:
        return 0

    return excess_return / downside_volatility * np.sqrt(252)

# Example
sortino = calculate_sortino_ratio(daily_returns)
# Compare to sharpe: If sortino > sharpe, good - means upside outliers exist
```

**5. Maximum Drawdown**
```
Definition: Largest peak-to-trough decline in account equity

Calculation:
  1. Track equity over time
  2. Find running maximum (peak)
  3. Calculate (Peak - Current) / Peak
  4. Find maximum drawdown across entire period

Example:
  Peak Equity: $10,500
  Trough: $9,100
  Max DD = ($10,500 - $9,100) / $10,500 = 13.33%

Rule of Thumb:
  Max DD < 10%:  Good
  Max DD 10-20%: Acceptable
  Max DD > 20%:  Concerning (need better risk control)
```

**Implementation:**
```python
def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve"""
    peak = equity_curve.expanding().max()
    drawdown = (peak - equity_curve) / peak
    return drawdown.max()

# Example
import pandas as pd
equity_curve = pd.Series([10000, 10200, 10100, 9800, 10300, ...])
max_dd = calculate_max_drawdown(equity_curve)
print(f"Max Drawdown: {max_dd:.2%}")
```

### 5.2 Trade-Level Metrics (R-Multiples)

**R-Multiple Definition:**
```
1R = Initial Risk per Trade (from entry to stop loss)

Example:
Entry Price: $80
Stop Loss:   $77.60 (3% below)
1R = $2.40 per unit

If position is 41 units:
1R loss = -41 × $2.40 = -$98.40 (1R)
Trade outcome:
  - If loses at stop: -1R
  - If gains 2× risk amount: +2R
  - If gains 4× risk amount: +4R
```

**R-Multiple Tracking:**
```python
class TradeRMultiple:
    def __init__(self, entry_price, stop_loss_price, units):
        self.entry_price = entry_price
        self.stop_loss_price = stop_loss_price
        self.units = units
        self.risk_per_unit = entry_price - stop_loss_price
        self.total_risk = self.risk_per_unit * units
        self.one_r = self.total_risk

    def calculate_r_multiple(self, exit_price):
        """Calculate trade outcome in R-multiples"""
        pnl = (exit_price - self.entry_price) * self.units
        r_multiple = pnl / self.one_r
        return r_multiple

    def get_trade_stats(self, exit_price):
        r_multiple = self.calculate_r_multiple(exit_price)
        return {
            'entry': self.entry_price,
            'exit': exit_price,
            'units': self.units,
            '1r_loss': -self.one_r,
            'pnl': (exit_price - self.entry_price) * self.units,
            'r_multiple': r_multiple
        }

# Example Usage
trade = TradeRMultiple(entry_price=80, stop_loss_price=77.6, units=41)

# Trade closes at $84
result = trade.get_trade_stats(exit_price=84)
# 'r_multiple': +1.67 (gained 1.67 times the risk amount)

# Trade closes at $77.60
result = trade.get_trade_stats(exit_price=77.6)
# 'r_multiple': -1.0 (lost exactly 1R)
```

### 5.3 Portfolio-Level Performance

**Win Rate & Expectancy**
```
Win Rate = Number of Winning Trades / Total Trades
Avg Win = Average profit on winning trades
Avg Loss = Average loss on losing trades
R Expectancy = (Win% × Avg Win R) - (Loss% × Avg Loss R)

Example:
Trades: 20 total
Winners: 10 (50%)
Losers: 10 (50%)
Avg Winner: +2.5R
Avg Loser: -1.0R

R Expectancy = (0.50 × 2.5R) - (0.50 × 1.0R)
             = 1.25R - 0.50R
             = +0.75R per trade

Interpretation: Each trade expected to return 0.75R on average
```

**Implementation:**
```python
def calculate_r_expectancy(trades_list):
    """
    trades_list: list of r_multiple values from all closed trades
    """
    if not trades_list:
        return 0

    wins = [r for r in trades_list if r > 0]
    losses = [r for r in trades_list if r <= 0]

    if len(trades_list) == 0:
        return 0

    win_rate = len(wins) / len(trades_list)
    loss_rate = len(losses) / len(trades_list)

    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))

    return {
        'expectancy': expectancy,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(sum(wins) / sum(losses)) if losses else 0,
        'total_trades': len(trades_list)
    }

# Example
trades = [2.5, -1.0, 3.2, -1.0, 2.1, -1.0, 1.8, -1.0, 2.3, -1.0]
stats = calculate_r_expectancy(trades)
# expectancy: +0.75R per trade (profitable at 0.75R per trade)
```

**Profit Factor:**
```
Profit Factor = Sum of All Wins / Sum of All Losses (absolute value)

Interpretation:
  PF > 2.0:  Excellent (2 dollars win for every 1 dollar loss)
  PF > 1.5:  Good
  PF 1.0-1.5: Acceptable
  PF < 1.0:  Losing money

Example:
Total Wins: $2,500
Total Losses: $1,200
Profit Factor = $2,500 / $1,200 = 2.08

Meaning: For every $1 lost, we made $2.08
```

---

## 6. TRADE VALIDATION RULES

### 6.1 Pre-Trade Validation Checklist

**Before entering ANY trade, ALL of these must pass:**

```
1. SIGNAL QUALITY CHECK
   [ ] Model confidence >= 52% (minimum)
   [ ] Model confidence <= 70% (avoid overfit)
   [ ] Not in emergency drawdown mode (>10% DD)

2. POSITION SIZING CHECK
   [ ] Position size calculated correctly (fixed fractional or Kelly)
   [ ] Position size adjusted for confidence level
   [ ] Position size <= 5% of account (concentration limit)
   [ ] Position size allows for hard stop loss

3. PORTFOLIO HEAT CHECK
   [ ] Total portfolio heat <= 3% of account
   [ ] Adding this position keeps heat under limit
   [ ] Enough capital to cover stop loss

4. RISK/REWARD CHECK
   [ ] Risk/Reward ratio >= 1:2 (minimum)
   [ ] Take profit > 5% above entry (default)
   [ ] Stop loss = 2-3% below entry (default)
   [ ] Hard stops are set BEFORE entering

5. MARKET CONDITIONS CHECK
   [ ] Not in pre-earnings blackout (7 days before earnings)
   [ ] Not during high volatility spikes (VIX > 25)
   [ ] Trading during liquid hours (9:30-16:00 ET)

6. ACCOUNT STATUS CHECK
   [ ] Account not in drawdown > 7.5% (caution zone)
   [ ] Sufficient buying power available
   [ ] Account can cover worst-case losses

7. THESIS VALIDATION
   [ ] Can articulate WHY the trade is good
   [ ] Thesis is based on model signal, not emotion
   [ ] No revenge trading (trying to recover losses quickly)
```

**Code Implementation:**
```python
def validate_trade(entry_price, exit_price, confidence,
                   account_value, current_equity, open_positions,
                   model_signal):
    """Comprehensive trade validation"""

    checks = {
        'signal_quality': True,
        'position_sizing': True,
        'portfolio_heat': True,
        'risk_reward': True,
        'market_conditions': True,
        'account_status': True
    }

    reasons_to_skip = []

    # 1. Signal Quality
    if confidence < 0.52 or confidence > 0.70:
        checks['signal_quality'] = False
        reasons_to_skip.append("Confidence outside valid range")

    # 2. Position Sizing (would calculate here)
    # ...

    # 3. Portfolio Heat
    current_heat = sum([pos['risk'] for pos in open_positions.values()])
    max_heat = account_value * 0.03
    if current_heat > max_heat:
        checks['portfolio_heat'] = False
        reasons_to_skip.append("Portfolio heat limit exceeded")

    # 4. Risk/Reward
    risk = entry_price - (entry_price * 0.03)  # 3% stop loss
    reward = (entry_price * 0.05) - entry_price  # 5% target
    if reward / risk < 2.0:
        checks['risk_reward'] = False
        reasons_to_skip.append("Risk/Reward ratio too low")

    # 5-7: Other checks...

    all_passed = all(checks.values())

    return {
        'trade_approved': all_passed,
        'checks': checks,
        'reasons_to_skip': reasons_to_skip
    }
```

### 6.2 Over-Trading Prevention

**Rules to Prevent Excessive Trading:**

```
Daily Trade Limits:
  Maximum trades per day: 3
  Minimum time between trades: 15 minutes

Weekly Trade Limits:
  Maximum trades per week: 10
  Maximum trades on same symbol per week: 3

Monthly Trade Limits:
  Review if:
    - More than 30 trades per month (too much activity)
    - Win rate drops below 40%
    - Profit factor below 1.0
```

**Implementation:**
```python
class TradeRateLimiter:
    def __init__(self):
        self.trades_today = 0
        self.trades_this_week = 0
        self.last_trade_time = None
        self.last_trade_on_symbol = {}

    def can_trade_now(self, symbol, current_time):
        # Daily limit
        if self.trades_today >= 3:
            return False, "Daily trade limit (3) reached"

        # Time between trades
        if self.last_trade_time:
            time_since_last = (current_time - self.last_trade_time).total_seconds()
            if time_since_last < 15 * 60:  # 15 minutes
                return False, "Too soon since last trade"

        # Same symbol limit
        if symbol in self.last_trade_on_symbol:
            trades_on_symbol = self.last_trade_on_symbol[symbol]
            if trades_on_symbol >= 3:
                return False, "3 trades on this symbol this week"

        return True, "OK to trade"

    def record_trade(self, symbol, trade_time):
        self.trades_today += 1
        self.trades_this_week += 1
        self.last_trade_time = trade_time

        if symbol not in self.last_trade_on_symbol:
            self.last_trade_on_symbol[symbol] = 0
        self.last_trade_on_symbol[symbol] += 1
```

---

## 7. CAPITAL PRESERVATION STRATEGIES

### 7.1 Cash Reserve Rules

**Maintain Cash Buffer:**
```
Minimum Cash Reserve: 10-20% of account
Rule: Never deploy more than 80-90% of account simultaneously

Example:
Account: $10,000
Minimum Cash: $1,000-$2,000
Max Deployed: $8,000-$9,000

Benefits:
  - Flexibility to add to winning trades
  - Ability to take new high-confidence opportunities
  - Psychological buffer (not "all-in")
  - Emergency liquidity
```

**Decision Tree:**
```
Account Size = $10,000

If Cash > $2,000 (>20%):
  -> Can take new trades normally

If Cash $1,000-$2,000 (10-20%):
  -> Can take trades but monitor closely
  -> Reduce position sizes by 25%

If Cash < $1,000 (<10%):
  -> DO NOT TAKE NEW TRADES
  -> Only exit existing positions
  -> Close smallest/most questionable positions first
  -> Rebuild cash to 15% minimum
```

### 7.2 Drawdown Recovery Plan

**Three-Stage Recovery Process:**

**Stage 1: Immediate Response (First -5% drawdown)**
```
Actions:
  1. Reduce position sizes to 50% of normal
  2. Tighten stops to 2% (instead of 3%)
  3. Increase minimum confidence to 55%
  4. Document what went wrong
  5. Continue trading with caution
```

**Stage 2: Caution Mode (First -7.5% drawdown)**
```
Actions:
  1. Reduce position sizes to 25% of normal
  2. Only trade highest confidence signals (65%+)
  3. Increase time between trades (30 min instead of 15)
  4. Review model performance metrics
  5. Consider model retraining with new data
```

**Stage 3: Recovery Mode (At -10% drawdown)**
```
Actions:
  1. STOP ALL NEW TRADES immediately
  2. Hold existing positions, don't add
  3. Close winning positions first (lock in profits)
  4. Let losers run to stop loss (don't hold underwater)
  5. Investigate root cause:
     - Model performance degradation?
     - Market regime change?
     - Data quality issues?
     - Overfitting detected?

Restart Conditions:
  Recovery back to peak equity, OR
  Investigation shows specific, fixable issue that's been addressed, OR
  Wait 30 days minimum before resuming
```

### 7.3 Position Closeout Rules

**Mandatory Close Scenarios:**

```
1. HARD STOP LOSS HIT
   Close immediately, no questions asked
   Record loss and move on

2. TAKE PROFIT HIT
   Close position
   Take the win

3. TIME STOP REACHED
   After 10 trading days: close regardless of profit/loss
   Prevents thesis creep and dead capital

4. THESIS BROKEN
   Model signal reverses (1 -> 0)
   Close immediately
   Don't hold for additional confirmation

5. EMERGENCY STOP (> 10% DD)
   All positions must close
   Preserve remaining capital

6. OPPORTUNITY COST
   Position not moving for 5 days
   Close and redeploy capital to higher-confidence trades
```

---

## 8. RISK-ADJUSTED PERFORMANCE MEASUREMENT

### 8.1 Monthly Performance Review

**Key Metrics to Track:**

```
PROFITABILITY
  [ ] Total Return (%)
  [ ] Return vs. Buy-and-Hold
  [ ] Monthly Win Rate (%)
  [ ] Avg Win vs. Avg Loss (R multiples)
  [ ] Profit Factor
  [ ] R Expectancy

RISK MANAGEMENT
  [ ] Max Drawdown (%)
  [ ] Largest losing trade (R)
  [ ] Consecutive losers (max count)
  [ ] Largest losing day
  [ ] % of trades that hit hard stops

EFFICIENCY
  [ ] Sharpe Ratio
  [ ] Sortino Ratio
  [ ] Win Rate (should be > 40%)
  [ ] Winning trades (%) - target 45-55%
  [ ] Losers within expected size

EXECUTION
  [ ] Number of trades
  [ ] Average trade duration
  [ ] Times over-traded (> 3/day)
  [ ] Times violated position limits
  [ ] Heat violations (exceeded 3%)
```

**Example Monthly Report:**

```
MONTH: November 2025
Starting Equity: $10,000
Ending Equity: $10,425
Return: +4.25%

TRADES:
Total Trades: 18
Winners: 9 (50%)
Losers: 9 (50%)

Avg Winner: +2.3R
Avg Loser: -1.0R
Profit Factor: 1.87
R Expectancy: +0.65R per trade

RISK METRICS:
Max Drawdown: -3.2%
Sharpe Ratio: 1.45
Sortino Ratio: 1.82
Largest Win: +4.2R
Largest Loss: -1.0R

ASSESSMENT: GOOD
- Return positive and in line with expectancy
- Risk controlled (drawdown < 5%)
- Win rate stable at 50%
- Profit factor > 1.5 (healthy)
- No heat violations
- Sharpe > 1.0 (efficient)

ACTION ITEMS:
- Investigate 2 over-trading days (should have been 1 trade each)
- Model confidence still accurate - no retraining needed
```

### 8.2 Strategy Viability Checklist

**Decision: Continue, Modify, or Stop Trading**

**Continue If:**
```
[ ] Win Rate >= 40%
[ ] Profit Factor >= 1.2
[ ] R Expectancy > 0
[ ] Sharpe Ratio > 0.5
[ ] Max Drawdown < 15%
[ ] Risk stays within limits
[ ] No pattern of major violations
```

**Modify If:**
```
[ ] Win Rate 35-40%: Increase position confidence threshold
[ ] Profit Factor 1.0-1.2: Tighten take-profit or widen stops
[ ] R Expectancy near 0: Reduce position sizes
[ ] Sharpe Ratio 0.5-1.0: Add position sizing rules
[ ] Max Drawdown 15-20%: Reduce risk per trade to 0.5%
[ ] Repeated violations: Strengthen validation rules
```

**Stop If:**
```
[ ] Win Rate < 35%
[ ] Profit Factor < 1.0 (losing money)
[ ] R Expectancy < -0.5 (losing 0.5R per trade)
[ ] Sharpe Ratio < 0.5 (poor risk-adjusted returns)
[ ] Max Drawdown > 25%
[ ] Model confidence becoming unreliable
[ ] Multiple strategy violations unresolved
```

---

## 9. IMPLEMENTATION ROADMAP FOR BACKTESTING ENGINE

### Phase 1: Foundation (Week 1)
```
[ ] Implement basic position sizing (fixed fractional 1%)
[ ] Add hard stop-loss mechanics (3% below entry)
[ ] Add take-profit mechanics (5% above entry)
[ ] Track individual trade P&L
[ ] Calculate daily equity curve
```

### Phase 2: Core Risk Controls (Week 2)
```
[ ] Implement maximum drawdown tracker (10% limit)
[ ] Add portfolio heat calculation (3% limit)
[ ] Add position concentration check (5% max)
[ ] Implement time-based stops (10 days max)
[ ] Add R-multiple tracking per trade
```

### Phase 3: Advanced Metrics (Week 3)
```
[ ] Calculate daily VaR (95%)
[ ] Calculate Sharpe and Sortino ratios
[ ] Calculate profit factor and R expectancy
[ ] Add confidence-based position sizing
[ ] Implement Kelly criterion calculator (with safety factor)
```

### Phase 4: Trade Validation (Week 4)
```
[ ] Add pre-trade validation checklist
[ ] Implement over-trading prevention
[ ] Add market condition filters
[ ] Add emergency stop protocols
[ ] Create trade rejection logging
```

### Phase 5: Monitoring & Reporting (Week 5)
```
[ ] Monthly performance reporting
[ ] Strategy viability assessment
[ ] Drawdown zone management
[ ] Position monitoring dashboard
[ ] Violation alerts and logging
```

---

## 10. RECOMMENDED INITIAL PARAMETERS FOR BACKTESTING

```
POSITION SIZING:
  Method:                Fixed Fractional
  Risk per Trade:        1% of account
  Stop Loss Distance:    3% below entry
  Take Profit Distance:  5% above entry
  Min Risk/Reward:       1:2

DRAWDOWN LIMITS:
  Maximum Drawdown:      10% of account
  Caution Zone:          5-7.5% drawdown
  Alert Zone:            7.5-10% drawdown
  Emergency Stop:        > 10% drawdown

PORTFOLIO HEAT:
  Maximum Heat:          3% of account
  Heat Limit:            Sum of all stop-loss losses

POSITION LIMITS:
  Max Position Size:     5% of account
  Max Concentration:     1 position per symbol
  Max Simultaneous:      3-5 open positions

TRADE VALIDATION:
  Min Confidence:        52%
  Max Confidence:        70%
  Min Trades Spacing:    15 minutes
  Max Trades Daily:      3
  Max Trade Duration:    10 trading days

CASH RESERVES:
  Minimum Cash:          10% of account
  Normal Operations:     80-90% deployed max

PERFORMANCE TARGETS:
  Target Win Rate:       50%+ (at minimum)
  Target Profit Factor:  1.5+
  Target Sharpe:         > 1.0
  Target Max Drawdown:   < 15%
```

---

## APPENDIX: Formula Reference

### Position Sizing
```
Fixed Fractional:
  Units = (Account Risk $) / (Entry Price - Stop Loss Price)

Kelly Criterion:
  f* = (Win% × B - Loss%) / B
  where B = Avg Win / Avg Loss
  Recommended: Use 25-50% of Kelly result

Confidence-Based:
  Adjusted Units = Base Units × Confidence Multiplier
```

### Stop-Loss & Take-Profit
```
Hard Stop Loss:        Entry × (1 - Stop % ranging 2-3%)
Trailing Stop:         Current Price × (1 - Trail % ranging 1-2%)
Take Profit Target:    Entry × (1 + Target % ranging 5-8%)
Risk/Reward Ratio:     (TP - Entry) / (Entry - SL)
```

### Drawdown
```
Current Drawdown:      (Peak Equity - Current Equity) / Peak Equity
Running Maximum:       Maximum equity seen year-to-date
Running Minimum:       Lowest equity seen year-to-date
```

### Risk Metrics
```
VaR (95%):            5th percentile of daily returns
CVaR (95%):           Average of worst 5% days
Sharpe Ratio:         (Avg Return - RFR) / Std Dev
Sortino Ratio:        (Avg Return - RFR) / Downside Std Dev
Max Drawdown:         (Peak - Trough) / Peak
Profit Factor:        Sum of Wins / Sum of Losses
R Expectancy:         (WinRate × AvgWin) - (LossRate × AvgLoss)
Win Rate:             # Winners / Total Trades
```

### R-Multiples
```
1R (Risk Per Trade):    Entry - Stop Loss
Trade Outcome (R):      (Exit Price - Entry) / 1R
Portfolio R Return:     (Final Equity - Initial) / 1R per trade
```

---

## CONCLUSION

This framework provides:

1. **Clear position sizing rules** - No more guessing at 50% capital
2. **Mandatory stop-loss mechanics** - Downside protection on every trade
3. **Drawdown controls** - Prevents catastrophic losses
4. **Portfolio heat limits** - Prevents over-concentration
5. **Risk metrics tracking** - Objective performance assessment
6. **Trade validation rules** - Prevents poor-quality signals
7. **Capital preservation** - Cash reserves and recovery plans
8. **Performance measurement** - Monthly reviews and viability checks

The framework should reduce losses on the current negative-return models by:
- Limiting single-trade exposure from 50% to 5% of capital
- Closing trades quickly when thesis breaks (currently only exits on opposite signal)
- Controlling total portfolio risk across multiple positions
- Validating model confidence before entering (avoiding weak 50-51% signals)
- Enforcing risk/reward ratios (minimum 1:2)

Expected Outcomes with This Framework:
- Max loss per trade: ~1% (not 50%)
- Max portfolio loss: ~10% controlled (not unlimited)
- Trading only on >52% confidence (not all signals)
- Better performance due to smaller, controlled position sizes
- Measurable risk metrics for continuous improvement
